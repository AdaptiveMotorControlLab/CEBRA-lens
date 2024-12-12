# run using python -m GithubFolder.src.scripts.RDM_analysis --filename offset10 --bool_comput 1
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import block_diag
from scipy.spatial.distance import cosine, correlation,cdist,pdist, squareform
from random import sample

from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(filename,bool_comput,saving_filename,num_trained_models,session_id):
    
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('BEGINNING OF SCRIPT')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


    ######################
    ####### LOADING ######
    ######################

    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Loading Data and activations...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    # LOAD DATA
    train_datas, valid_datas, discrete_labels_train, discrete_labels_val = get_single_session_datasets()
    


    train_data = train_datas[session_id].neural
    valid_data = valid_datas[session_id].neural
    train_label = discrete_labels_train[session_id]
    valid_label = discrete_labels_val[session_id]


    with open(os.path.join('data/activations', f'{filename}.pkl'),'rb') as f:
        activations_dict = pickle.load(f)

    #####################################
    ####### RDM MATRIX CALCULATION ######
    #####################################

    directory = f'data/RDM/'

    if bool_comput:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Calculating RDM matrices...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # BINNING
        num_bins = 30
        num_samples = 200 if len(train_data)/30 >= 200 else int(len(train_data)/30) 
        step_distance = 30
        idxs = np.zeros((num_bins,num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where((train_label[:] >= j*step_distance) & (train_label[:] < (j+1)*step_distance))[0]
            idxs[i,:] = sample(list(full_idxs), num_samples)
            j = j + 1

        idxs = idxs.astype(int)



        ########################################
        ####### Neural v MultiSessino E.G ######
        ########################################

        # Neural
        neural_data = train_data[list(idxs.flatten()),:]

        # Activations
        activations_UT = activations_dict['act_UT']
        activations_multi =  activations_dict['act_multi']
        activations_single = activations_dict['act_single']

        num_layers = (len(activations_UT)-2)//2 # This assumes that there will always be only 1 instance of single UT and 1 multi UT.
        output_embeddings_idxs = [i for i in range(1,num_trained_models+1)][::-1]

        # Output Embedding of first multi-session model instance
        embeddings_trained_multi_all = activations_multi[:num_layers] + [activations_multi[-output_embeddings_idxs[0]]]
        
        embeddings_trained_multi = activations_multi[-output_embeddings_idxs[0]][:,idxs.flatten()].T # .T to keep consistency between neural data and this

        neural_data_rdm = squareform(pdist(neural_data,metric='euclidean'))
        embedding_rdm = squareform(pdist(embeddings_trained_multi,metric='euclidean'))
        # just to show that with correlation it doesn't work with neural input
        neural_data_rdm_corr = squareform(pdist(neural_data,metric='correlation'))
        embedding_rdm_corr = squareform(pdist(embeddings_trained_multi,metric='correlation'))
        # Normalize the RDMs using Min-Max normalization
        rdm1_normalized = normalize_minmax(neural_data_rdm)
        rdm2_normalized = normalize_minmax(embedding_rdm)

        plot_rdm([rdm1_normalized,rdm2_normalized],['Neural input', 'Output Layer'],metric = 'Normalized Euclidean distance')
        plot_rdm([neural_data_rdm_corr, embedding_rdm_corr],['Neural input', 'Output Layer'],metric = 'Dissimilarity')



        ####################################
        ####### ORACLE RDM COMPARISON ######
        ####################################

        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Comparing to Oracle RDM...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # Create Oracle RDM.
        one_class = np.ones((200, 200))
        all_classes = [one_class for _ in range(30)]
        block_rdm_sqform = 1 - block_diag(*all_classes)
        oracle_rdm = squareform(block_rdm_sqform)

        all_corrs_multi = []
        all_corrs_single = []

        for j in tqdm(range(num_trained_models)):
            embeddings_trained_multi_all = activations_multi[j*num_layers:(j+1)*num_layers] + [activations_multi[-output_embeddings_idxs[j]]]
            embeddings_trained_single_all = activations_single[j*num_layers:(j+1)*num_layers] + [activations_single[-output_embeddings_idxs[j]]]

            correlation_RDM_multi = []
            correlation_RDM_single = []

            for i in tqdm(range(len(embeddings_trained_multi_all))):
                rmd_corr_multi = pdist(embeddings_trained_multi_all[i][:,idxs.flatten()].T,metric='correlation')
                correlation_RDM_multi.append(1-correlation(oracle_rdm,rmd_corr_multi))

                rmd_corr_single = pdist(embeddings_trained_single_all[i][:,idxs.flatten()].T,metric='correlation')
                correlation_RDM_single.append(1-correlation(oracle_rdm,rmd_corr_single))

            all_corrs_multi.append(correlation_RDM_multi)
            all_corrs_single.append(correlation_RDM_single)


        all_corrs_UT = []

        embeddings_untrained_single_all = activations_UT[:num_layers] + [activations_UT[-2]]
        embeddings_untrained_multi_all = activations_UT[num_layers:-2] + [activations_UT[-1]]

        correlation_RDM_multi = []
        correlation_RDM_single = []

        for i in tqdm(range(len(embeddings_untrained_multi_all))):

            rmd_corr_multi_UT = pdist(embeddings_untrained_multi_all[i][:,idxs.flatten()].T,metric='correlation')
            correlation_RDM_multi.append(1-correlation(oracle_rdm,rmd_corr_multi_UT))

            rmd_corr_single_UT = pdist(embeddings_untrained_single_all[i][:,idxs.flatten()].T,metric='correlation')
            correlation_RDM_single.append(1-correlation(oracle_rdm,rmd_corr_single_UT))

        all_corrs_UT.append(correlation_RDM_multi)
        all_corrs_UT.append(correlation_RDM_single)

        all_corrs = {
            'UT': all_corrs_UT,
            'single': all_corrs_single,
            'multi': all_corrs_multi
                     }
        
        with open(os.path.join(directory, f'{saving_filename}.pkl'), 'wb') as f:
            pickle.dump(all_corrs, f)
            print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('RDM Matrices saved!')
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    else:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Loading RDM matrices...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        with open(os.path.join(directory, f'{saving_filename}.pkl'), 'rb') as f:
            all_corrs = pickle.load(f)

    # PLOTTING 
    # Calculate means
    mean_multi = np.mean(all_corrs['multi'], axis=0)
    mean_single = np.mean(all_corrs['single'], axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    sns.set(style="white")
    # Define pastel colors
    colors = sns.color_palette("hls",8)
    pastel_purple = colors[6]
    pastel_blue = colors[4]
    grey = sns.color_palette("Greys")[5]

    #  UT
    for i in range(2):
        sns.lineplot(x=np.arange(1,len(all_corrs['single'][i])+1), y=all_corrs['UT'][i],
                    linestyle='--', marker='D', color=grey, alpha = 0.5,label='Untrained' if i == 0 else "")
    #  MULTI
    for i in range(num_trained_models):
        sns.lineplot(x=np.arange(1,len(all_corrs['single'][i])+1), y=all_corrs['multi'][i],
                    linestyle='-', marker='D', color=pastel_purple,alpha = 0.5, label='Multi' if i == 0 else "")
    #  SINGLE
    for i in range(num_trained_models):
        sns.lineplot(x=np.arange(1,len(all_corrs['single'][i])+1), y=all_corrs['single'][i],
                    linestyle='-', marker='D', color=pastel_blue,alpha = 0.5, label='Single' if i == 0 else "")

    sns.lineplot(x=np.arange(1,len(all_corrs['single'][i])+1), y=mean_multi,
                linestyle='-', color=pastel_purple, linewidth=2.5, label='Mean Multi')
    sns.lineplot(x=np.arange(1,len(all_corrs['single'][i])+1), y=mean_single,
                linestyle='-', color=pastel_blue, linewidth=2.5, label='Mean Single')

    plt.xlabel('Depth of layer')
    plt.ylabel('Correlation')
    plt.title(f'Correlation to Oracle RDM for {filename}')
    sns.despine()
    plt.legend()
    plt.show()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        '--filename', type=str, default='offset10', help="name of the activations (assuming they are under data/activations)"
    )
    
    parser.add_argument(
        '--bool_comput', type=int, default=0, help="If True, will recompute and overwrite the cka matrices (0 or 1)"
    )

    parser.add_argument(
        '--saving_filename', type=str, default=None, help="name of the file where to save the RDM matrices (it will be under data/RDM/saving_filename)"
    )
    parser.add_argument(
        '--num_trained_models', type=int, default=5, help="number of trained instances per solver (e.g. 5)"
    )
    parser.add_argument(
        '--debugging', type=int, default=0, help="Debugging mode (0 or 1)"
    )
    parser.add_argument(
        '--session_id', type=int, default=3, help="session id for the analysis, used to retrieve the correct data and multi-session model"
    )
    args = parser.parse_args()

    if args.saving_filename is None:
        args.saving_filename = args.filename
    print('INPUT: ',args)
    main(args.filename,args.bool_comput,args.saving_filename,args.num_trained_models,args.session_id)
