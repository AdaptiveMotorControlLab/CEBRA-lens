# run using: python -m GithubFolder.src.scripts.layer_activation_retrieval      
# attention: need to be one step above the GithubFolder to have data and finalmodels

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.spatial.distance import cosine, correlation,cdist,pdist, squareform
from random import sample

from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(layer_type, session_id,filename):
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
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
    models_folder_path = 'FinalModels/VISION/offset10'
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

   


    # Save activations to a pickle file
    with open(os.path.join('data/activations', f'{filename}.pkl'), 'wb') as f:
        pickle.dump(activations_dict, f)
    print(activations_dict)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        '--layer_type', type=str, default='all', help="Type of layer to process ('all' or 'conv')"
    )
    parser.add_argument(
        '--session_id', type=int, default=3, help="session id for the analysis, used to retrieve the correct data and multi-session model"
    )
    parser.add_argument(
        '--filename', type=str, default='offset10', help="filename of the activations"
    )
    args = parser.parse_args()
    print(args)
    main(args.layer_type, args.session_id, args.filename)
