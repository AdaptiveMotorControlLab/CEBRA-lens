
# run using (e.g): python -m GithubFolder.src.scripts.layer_activation_retrieval --layer_type conv --session_id 3 --filename offset10alllayers
# attention: need to be one step above the GithubFolder to have data and finalmodels

import os
import pickle
import torch
import torch.nn as nn

from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(model_name,layer_type, session_id,filename):
    
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('BEGINNING OF SCRIPT')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    ######################
    ####### LOADING ######
    ######################

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading Data and models...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    # LOAD DATA
    train_datas, valid_datas, discrete_labels_train, discrete_labels_val = get_single_session_datasets()
    


    train_data = train_datas[session_id].neural
    valid_data = valid_datas[session_id].neural
    train_label = discrete_labels_train[session_id]
    valid_label = discrete_labels_val[session_id]

    # LOAD MODELS
    models_folder_path = f'FinalModels/VISION/{model_name}'
    files_list = os.listdir(models_folder_path)

    models_list = []
    for file in files_list: # load only the torch models for cpu usage
        if file.endswith("torch.pt"):
            models_list.append(file)

    models_list
    print('Number of models: ',len(models_list))
    print(models_list)

    models_untrained = [] # will be multi_ut, single_ut
    models_single = [] # will be all the singles trained
    models_multi = [] # will be all the multi trained


    for model in models_list:

        loaded_model = cebra.CEBRA.load(os.path.join(models_folder_path,model), backend = 'torch', map_location=torch.device('cpu')).to('cpu')
        if 'UT' in model:
            models_untrained.append(loaded_model)

        elif 'multi' in model:
            models_multi.append(loaded_model)

        else:
            models_single.append(loaded_model)
    
    models = {
        'UT': models_untrained,
        'single': models_single,
        'multi': models_multi
    }

    # check the models
    print('# of Untrained models: ',len(models["UT"]))
    print('# of Single Trained models: ',len(models["single"]))
    print('# of Single Trained models: ',len(models["multi"]))

    print('Solver Untrained model 1: ',models["UT"][0].solver_name_) # HERE IT'S SINGLE SESSION FIRST
    print('Solver Untrained model 2: ',models["UT"][1].solver_name_)
    print('key single: ', models["single"][0].solver_name_)
    print('key multi: ',models["multi"][0].solver_name_)


    ###########################
    ####### ATTACH HOOKS ######
    ###########################

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Retrieving layer activations...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    # Dictionary to store activations of all models
    activations = {}

    # Function to create a hook that stores the activations in the dictionary
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().squeeze().numpy()
        return hook


    def attach_hooks(model,name,bool_train = False,layer_type = 'conv'): # only attaches hooks on convolutional layers
        
        valid_layer_types = ['all', 'conv'] # TODO: add more layer types to have more specificity
    
        if layer_type not in valid_layer_types:
            raise ValueError(f"Invalid layer_type: {layer_type}. Expected one of {valid_layer_types}")
        
        
        if not(bool_train): # attach _UT when it's not a trained model
            string_ut = 'UT'
        else:
            string_ut = ''
        num_layer = 1

        if layer_type == 'conv':
            for i in range (len(model.net)):
                if isinstance(model.net[i], nn.Conv1d):
                    model.net[i].register_forward_hook(get_activation(f'{name}{string_ut}_layer{num_layer}'))
                    num_layer += 1

                elif bool(model.net[i]._modules): # empty dict evaluate to false. here we go in the _Skip connection where some conv may be stored
                    for j in range (len(model.net[i].module)):
                        if isinstance(model.net[i].module[j], nn.Conv1d):
                            model.net[i].module[j].register_forward_hook(get_activation(f'{name}{string_ut}_layer{i}'))
                            num_layer += 1

        elif layer_type == 'all':
            for i in range (len(model.net)):
                if bool(model.net[i]._modules): # empty dict evaluate to false. here we go in the _Skip connection where some conv may be stored
                    for j in range (len(model.net[i].module)):
                        if isinstance(model.net[i].module[j], nn.Conv1d):
                            model.net[i].module[j].register_forward_hook(get_activation(f'{name}{string_ut}_layer{i}'))
                            num_layer += 1

                else:
                    model.net[i].register_forward_hook(get_activation(f'{name}{string_ut}_layer{num_layer}'))
                    num_layer += 1


                
    output_embeddings = {
        'UT': [],
        'single' : [],
        'multi' : []
    }
    # UNTRAINED
    attach_hooks(models['UT'][1].model_[session_id],'multi',False, layer_type=layer_type)
    attach_hooks(models['UT'][0].model_,'single',False, layer_type=layer_type)

    # do a forward pass
    output_embeddings['UT'].append(models['UT'][0].transform(train_data).T)
    output_embeddings['UT'].append(models['UT'][1].transform(train_data,session_id = session_id).T)

    # SINGLE
    for i,model in enumerate(models['single']):
        attach_hooks(model.model_,f'single{i}',True, layer_type=layer_type)
        output_embeddings['single'].append(model.transform(train_data).T)

    # MULTI
    for i,model in enumerate(models['multi']):
        attach_hooks(model.model_[session_id],f'multi{i}',True, layer_type=layer_type)
        output_embeddings['multi'].append(model.transform(train_data,session_id = session_id).T)
    
    activations_dict = process_activations(activations,output_embeddings) #TODO: change process_activations so it takes the output embedding as well

   

    print(activations_dict['act_single'])
    print(activations_dict['labels_single'])
    print(len(activations_dict['labels_single']))


    # Save activations to a pickle file
    with open(os.path.join('data/activations', f'{filename}.pkl'), 'wb') as f:
        pickle.dump(activations_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        '--model_name', type=str, default='offset10', help="name of the folder where the models (assuming they are under FinalModels/VISION)"
    )
    parser.add_argument(
        '--layer_type', type=str, default='conv', help="Type of layer to process ('all' or 'conv')"
    )
    parser.add_argument(
        '--session_id', type=int, default=3, help="session id for the analysis, used to retrieve the correct data and multi-session model"
    )
    parser.add_argument(
        '--filename', type=str, default='offset10', help="filename of the activations"
    )
    args = parser.parse_args()
    print(args)
    main(args.model_name,args.layer_type, args.session_id, args.filename)
=======
import pickle
import argparse
import cebra
from GithubFolder.src.cebra_lens import cebra_lens as lens
import os
import logging


def setup_logging():

    # Get directory and filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_filename = os.path.splitext(os.path.basename(__file__))[0]

    logs_dir = os.path.join(script_dir, "logs")

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_path = os.path.join(logs_dir, f"{script_filename}.log")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(
    model_name, session_id, activations_filepath, bool_plot_embeddings, layer_type
):
    logging.info("Script started with arguments:")
    for arg, value in locals().items():
        logging.info(f"{arg}: {value}")

    # LOAD DATA
    train_datas, _, discrete_labels_train, _ = (
        lens.utils_allen.get_single_session_datasets()
    )

    train_data = train_datas[session_id].neural
    train_label = discrete_labels_train[session_id]

    # LOAD MODELS
    models = lens.model.model_loader(model_name=model_name)

    if bool_plot_embeddings:

        X = train_data
        y = train_label
        embeddings_single = []
        embeddings_multi = []

        # Go to 5 max for plotting clarity (works even if there are less than 5 models)
        for model in models["multi_TR"][:5]:
            embeddings_multi.append(model.transform(X, session_id=session_id))
        for model in models["single_TR"][:5]:
            embeddings_single.append(model.transform(X))

        # Align the single session embeddings to the first rat
        alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

        for i in range(len(embeddings_single)):
            embeddings_single[i] = alignment.fit_transform(
                embeddings_single[0], embeddings_single[i], y, y
            )

        for i in range(len(embeddings_multi)):
            embeddings_multi[i] = alignment.fit_transform(
                embeddings_multi[0], embeddings_multi[i], y, y
            )

        embeddings_untrained_single = models["single_UT"][0].transform(
            X
        )  # only take the first untrained model for plotting
        embeddings_untrained_multi = models["multi_UT"][0].transform(
            X, session_id=session_id
        )  # only take the first untrained model for plotting

        fig = lens.plotting.plot_embeddings_singlevmulti(
            embeddings_single,
            embeddings_multi,
            embeddings_untrained_single,
            embeddings_untrained_multi,
            y,
        )
        fig.show()

    activations = {}
    activations = lens.activations.get_activations_multi_model(
        models=models,
        data=train_data,
        session_id=session_id,
        activations=activations,
        layer_type=layer_type,
    )

    activations_dict = lens.activations.process_activations(activations)

    with open(activations_filepath, "wb") as f:
        pickle.dump(activations_dict, f)


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="offset10",
        help="name of the folder where the models (assuming they are under FinalModels/VISION)",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="filename of the activations",
    )
    parser.add_argument(
        "--bool_plot_embeddings",
        type=int,
        default=0,
        help="Plots the output embeddings of the models (0 or 1)",
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="conv",
        help="Type of layer: e.g. 'conv', 'all'",
    )

    args = parser.parse_args()
    main(
        args.model_name,
        args.session_id,
        args.activations_filepath,
        args.bool_plot_embeddings,
        args.layer_type,
    )

