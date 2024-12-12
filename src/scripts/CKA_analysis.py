# run using python -m GithubFolder.src.scripts.CKA_analysis --filename offset10-mse --bool_comput True
import os
import pickle
import numpy as np
from tqdm import tqdm
from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(filename,bool_comput,saving_foldername,num_trained_models,debugging):
    
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('BEGINNING OF SCRIPT')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    ######################
    ####### LOADING ######
    ######################

    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading activations...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    with open(os.path.join('data/activations', f'{filename}.pkl'),'rb') as f:
        activations_dict = pickle.load(f)

    #####################################
    ####### CKA MATRIX CALCULATION ######
    #####################################

   
    directory = f'data/CKA/{saving_foldername}'

    if bool_comput:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Calculating CKA matrices...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


        activations_UT = activations_dict['act_UT']
        activations_multi =  activations_dict['act_multi']
        activations_single = activations_dict['act_single']

        num_layers = (len(activations_UT)-2)//2 # This assumes that there will always be only 1 instance of single UT and 1 multi UT.
        output_embeddings_idxs = [i for i in range(1,num_trained_models+1)][::-1]

        embeddings_untrained_single = activations_UT[:num_layers] + [activations_UT[-2]]
        embeddings_untrained_multi = activations_UT[num_layers:-2] + [activations_UT[-1]]

        # define cka matrices
        cka_matrix_single = np.zeros((num_trained_models,len(embeddings_untrained_single))) # number of Models (num_trained_models) X number of layers
        cka_matrix_multi = np.zeros((num_trained_models,len(embeddings_untrained_single))) # number of Models (num_trained_models) X number of layers
        cka_matrix_sm = np.zeros((num_trained_models,len(embeddings_untrained_single)))
        cka_matrix_singlevsingle = np.zeros((num_trained_models,len(embeddings_untrained_single))) # number of Models (num_trained_models) X number of layers
        cka_matrix_multivmulti = np.zeros((num_trained_models,len(embeddings_untrained_single))) # number of Models (num_trained_models) X number of layers
        # MULTI
        embeddings_trained_multi_0 = activations_multi[:num_layers] + [activations_multi[-output_embeddings_idxs[0]]] # first multi model from 0-num_layers and add embedding

        # SINGLE
        embeddings_trained_single_0 = activations_single[:num_layers] + [activations_single[-output_embeddings_idxs[0]]]

        for i in tqdm(range(num_trained_models)): # per model
            print('---------------------------------------')
            print('---------------------------------------')

            print('I: ',i)
            # MULTI
            embeddings_trained_multi = activations_multi[i*num_layers:(i+1)*num_layers] + [activations_multi[-output_embeddings_idxs[i]]] # first multi model from 0-num_layers and add embedding

            # SINGLE
            embeddings_trained_single = activations_single[i*num_layers:(i+1)*num_layers] + [activations_single[-output_embeddings_idxs[i]]]
            print('LEN EMBEDDINGS TRAINED SINGLE (num layers): ',len(embeddings_trained_single))
            print('LEN EMBEDDINGS UT SINGLE (num layers): ',len(embeddings_untrained_single))
            for j in tqdm(range(len(embeddings_trained_single))):
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('J: ', j)
                print('SHAPE EMBEDDINGS UT SINGLE: ',embeddings_untrained_single[j].shape)
                print('SHAPE EMBEDDINGS TRAINED SINGLE: ',embeddings_trained_single[j].shape)


                cka_matrix_single[i,j] = cka(gram_linear(embeddings_untrained_single[j].T), gram_linear(embeddings_trained_single[j].T))
                cka_matrix_multi[i,j] = cka(gram_linear(embeddings_untrained_multi[j].T), gram_linear(embeddings_trained_multi[j].T))
                cka_matrix_sm[i,j] = cka(gram_linear(embeddings_trained_single[j].T), gram_linear(embeddings_trained_multi[j].T))
                cka_matrix_singlevsingle[i,j] = cka(gram_linear(embeddings_trained_single_0[j].T), gram_linear(embeddings_trained_single[j].T))
                cka_matrix_multivmulti[i,j] = cka(gram_linear(embeddings_trained_multi_0[j].T), gram_linear(embeddings_trained_multi[j].T))
        
        if not debugging:
            # SAVE MATRICES
            if os.path.exists(directory):
                print(f"Directory '{directory}' already exists. Overwriting previous results.")
            else:
                os.makedirs(directory)

            
            np.save(f'{directory}/cka_matrix_single.npy', cka_matrix_single)
            np.save(f'{directory}/cka_matrix_multi.npy', cka_matrix_multi)
            np.save(f'{directory}/cka_matrix_sm.npy', cka_matrix_sm)
            np.save(f'{directory}/cka_matrix_singlevsingle.npy', cka_matrix_singlevsingle)
            np.save(f'{directory}/cka_matrix_multivmulti.npy',cka_matrix_multivmulti)

    else:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Loading CKA matrices...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


        # LOAD MATRICES
        cka_matrix_single = np.load(f'{directory}/cka_matrix_single.npy')
        cka_matrix_multi = np.load(f'{directory}/cka_matrix_multi.npy')
        cka_matrix_sm = np.load(f'{directory}/cka_matrix_sm.npy')
        cka_matrix_singlevsingle = np.load(f'{directory}/cka_matrix_singlevsingle.npy')
        cka_matrix_multivmulti = np.load(f'{directory}/cka_matrix_multivmulti.npy')

    if not debugging:
        plot_cka_heatmaps(
            cka_matrices=[cka_matrix_single, cka_matrix_multi, cka_matrix_sm],
            titles=['Single-session', 'Multi-session', 'Single vs Multi'],
            annot = False
        )
        plot_cka_heatmaps(
            cka_matrices=[cka_matrix_sm, cka_matrix_singlevsingle, cka_matrix_multivmulti],
            titles=['Single vs Multi', 'Single vs Single', 'Multi vs Multi'],
            annot = False
        )



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        '--filename', type=str, default='offset10', help="name of the activations (assuming they are under data/activations)"
    )
    
    parser.add_argument(
        '--bool_comput', type=int, default=0, help="If True, will recompute and overwrite the cka matrices (0 or 1)"
    )

    parser.add_argument(
        '--saving_foldername', type=str, default=None, help="name of the folder where to save the CKA matrices (it will be under data/CKA/saving_foldername)"
    )
    parser.add_argument(
        '--num_trained_models', type=int, default=5, help="number of trained instances per solver (e.g. 5)"
    )
    parser.add_argument(
        '--debugging', type=int, default=0, help="Debugging mode (0 or 1)"
    )
    
    args = parser.parse_args()

    if args.saving_foldername is None:
        args.saving_foldername = args.filename

    main(args.filename,args.bool_comput,args.saving_foldername,args.num_trained_models,args.debugging)
