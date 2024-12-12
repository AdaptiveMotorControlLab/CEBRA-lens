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

def main(model_name,layer_type, session_id,filename, bool_plot_embeddings):
    
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('BEGINNING OF SCRIPT')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')


    ######################
    ####### LOADING ######
    ######################

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading Data and models...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    # LOAD DATA
    train_datas, valid_datas, discrete_labels_train, discrete_labels_val = get_single_session_datasets()
    


    train_data = train_datas[session_id].neural
    valid_data = valid_datas[session_id].neural
    train_label = discrete_labels_train[session_id]
    valid_label = discrete_labels_val[session_id]

    # LOAD MODELS
    models = model_loader(model_name= model_name)

    #############################
    ####### PLOT EMBEDDINGS #####
    #############################
    if bool_plot_embeddings:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Calculating output embeddings...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

        X = train_data
        y = train_label
        embeddings_single = []
        embeddings_multi = []

        for i in range(len(models["multi"])): # assumes that the same number of single sessions and multi sessions were trained

            embeddings_single.append(models['single'][i].transform(X))
            embeddings_multi.append(models['multi'][i].transform(X,session_id = session_id))



        # Align the single session embeddings to the first rat
        alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

        for j in range(len(models["multi"])):
            embeddings_single[j] = alignment.fit_transform(
                embeddings_single[0], embeddings_single[j], y, y)
            
        for j in range(len(models["multi"])):
            embeddings_multi[j] = alignment.fit_transform(
                embeddings_multi[0], embeddings_multi[j], y, y)
            
        embeddings_untrained_single = models['UT'][0].transform(X) # this assumes that single will always come first. This is true if they are named in the same convention but this should be changed by a reg expression of UT + single.
        embeddings_untrained_multi = models['UT'][1].transform(X,session_id = session_id)

        plot_embeddings_singlevmulti(embeddings_single, embeddings_multi, embeddings_untrained_single, embeddings_untrained_multi, y)

    ###########################
    ####### ATTACH HOOKS ######
    ###########################

    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Retrieving layer activations...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')


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
    
    activations_dict = process_activations(activations,output_embeddings) 

    # Save activations to a pickle file
    with open(os.path.join('data/activations', f'{filename}.pkl'), 'wb') as f:
        pickle.dump(activations_dict, f)
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Layer activations saved!')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')


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
    parser.add_argument(
        '--bool_plot_embeddings', type=int, default=1, help="Plots the output embeddings of the models (0 or 1)"
    )

    args = parser.parse_args()
    print(args)
    main(args.model_name,args.layer_type, args.session_id, args.filename, args.bool_plot_embeddings)
