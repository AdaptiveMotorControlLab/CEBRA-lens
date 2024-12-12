# run using python -m GithubFolder.src.scripts.distance_analysis --filename offset10 --bool_comput 1 --num_repetitions 9
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from random import sample

from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(filename,bool_comput,saving_filename,num_trained_models,session_id,num_repetitions):
    
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
    ####### DISTANCE  CALCULATION #######
    #####################################

    directory = f'data/distances/'
    num_layers = (len(activations_dict['act_UT'])-2)//2 # This assumes that there will always be only 1 instance of single UT and 1 multi UT.

    if bool_comput:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Calculating Distances...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # BINNING
        num_bins = 30
        num_samples = int(len(train_data)/num_bins)

        step_distance = 30
        idxs = np.zeros((num_bins,num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where((train_label[:] >= j*step_distance) & (train_label[:] < (j+1)*step_distance))[0]
            idxs[i,:] = list(full_idxs)
            j = j + 1

        idxs = idxs.astype(int)

        output_embeddings_idxs = [i for i in range(1,num_trained_models+1)][::-1]

        distances_inter = {'multi': [], 'single': [], 'UT' : []}  
        distances_intra = {'multi': [], 'single': [], 'UT' : []}  
        distances_rep = {'multi': [], 'single': [], 'UT' : []}  


        embeddings_untrained_single = activations_dict['act_UT'][:num_layers] + [activations_dict['act_UT'][-2]] 
        embeddings_untrained_multi = activations_dict['act_UT'][num_layers:-2] + [activations_dict['act_UT'][-1]] 

        distances_inter['UT'].append(compute_inter_bin_distances(embeddings_untrained_single,idxs,num_bins))
        distances_inter['UT'].append(compute_inter_bin_distances(embeddings_untrained_multi,idxs,num_bins))

        distances_intra['UT'].append(compute_intra_bin_distances(embeddings_untrained_single,idxs,num_bins)[0])
        distances_intra['UT'].append(compute_intra_bin_distances(embeddings_untrained_multi,idxs,num_bins)[0])

        distances_rep['UT'].append(compute_mean_repetition_distances(embeddings_untrained_single,idxs,num_bins,num_repetitions))
        distances_rep['UT'].append(compute_mean_repetition_distances(embeddings_untrained_multi,idxs,num_bins,num_repetitions))



        # Loop over the trained instances of each model and compute distances
        for j in range(num_trained_models):
            embeddings_trained_multi_all = activations_dict['act_multi'][j*num_layers:(j+1)*num_layers] +[activations_dict['act_multi'][-output_embeddings_idxs[j]]]
            embeddings_trained_single_all = activations_dict['act_single'][j*num_layers:(j+1)*num_layers] + [activations_dict['act_single'][-output_embeddings_idxs[j]]]


            # Compute inter-bin distances
            interbin_multi = compute_inter_bin_distances(embeddings_trained_multi_all,idxs,num_bins)
            interbin_single = compute_inter_bin_distances(embeddings_trained_single_all,idxs,num_bins)
            
            # Compute intra-bin distances
            intrabin_multi,_ = compute_intra_bin_distances(embeddings_trained_multi_all,idxs,num_bins)
            intrabin_single,_ = compute_intra_bin_distances(embeddings_trained_single_all,idxs,num_bins)

            # Compute interrep distances
            rep_multi = compute_mean_repetition_distances(embeddings_trained_multi_all,idxs,num_bins,num_repetitions)
            rep_single = compute_mean_repetition_distances(embeddings_trained_single_all,idxs,num_bins,num_repetitions)

            # Store all the distances
            distances_inter['multi'].append(interbin_multi)
            distances_inter['single'].append(interbin_single)

            distances_intra['multi'].append(intrabin_multi)
            distances_intra['single'].append(intrabin_single)

            distances_rep['multi'].append(rep_multi)
            distances_rep['single'].append(rep_single)


        distances = {'inter': distances_inter, 'intra': distances_intra, 'rep': distances_rep}

        with open(os.path.join(directory, f'{saving_filename}.pkl'), 'wb') as f:
                pickle.dump(distances, f)
                print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
                print('Distances saved!')
                print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
            
    else:
        print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Loading Distances...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        with open(os.path.join(directory, f'{saving_filename}.pkl'), 'rb') as f:
            distances = pickle.load(f)


    #####################################
    ############# PLOTTING ##############
    #####################################
    print('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Plotting...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    
    #TODO: encapsulate all of this into a function that takes a dictionnnary of distances
    layers = list(range(1, num_layers + 2))
    sns.set_theme(style="white")
    # Define pastel colors
    colors = sns.color_palette("hls",8)
    pastel_purple = colors[6]
    pastel_blue = colors[4]
    grey = sns.color_palette("Greys")[5]

    for key,dist in distances.items():
        plt.figure(figsize=(10, 6))

        # Plot for Multi model instances
        for dist_multi in dist['multi']:
            plt.plot(layers, dist_multi, alpha=0.5, color=pastel_purple)  # Plot with transparency for individual instances

        # Plot the mean of Multi model instances
        mean_multi_intra = np.mean(dist['multi'], axis=0)
        plt.plot(layers, mean_multi_intra, marker='o', color=pastel_purple, label=f'Multi {key} Mean', linewidth=2)

        # Plot for Single model instances
        for dist_single in dist['single']:
            plt.plot(layers, dist_single, alpha=0.5, color=pastel_blue)  # Plot with transparency for individual instances

        # Plot the mean of Single model instances
        mean_single_intra = np.mean(dist['single'], axis=0)
        plt.plot(layers, mean_single_intra, marker='s', color=pastel_blue, label=f'Single {key} Mean', linewidth=2)

        plt.plot(layers,dist['UT'][1], linestyle='--', marker='D', color=grey, alpha = 0.5, label = 'Untrained')
        plt.plot(layers,dist['UT'][0],linestyle='--', marker='D', color=grey, alpha = 0.5, label = '')

        # Customize the plot
        plt.xlabel('Layer')
        plt.ylabel(f'Mean {key} Distance (Cosine)')
        plt.title(f'Evolution of Mean {key} Distance Across Layers (Multi vs Single)')
        plt.legend(title='Model Type')
        sns.despine()

        # Show the plot
        plt.tight_layout()
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
    parser.add_argument(
        '--num_repetitions', type=int, default=9, help="number of repetitions in your training data on which to do the repetition distance"
    )
    args = parser.parse_args()

    if args.saving_filename is None:
        args.saving_filename = args.filename
    print('INPUT: ',args)
    main(args.filename,args.bool_comput,args.saving_filename,args.num_trained_models,args.session_id,args.num_repetitions)
