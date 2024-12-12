#run using % python -m GithubFolder.src.scripts.model_decoding --model_name offset10 --session_id 3

import torch.nn as nn
import matplotlib as plt
from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(model_name, session_id,bool_plot_loss):
    
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

    if bool_plot_loss:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Plotting lossess...')
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')


        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Plot for single models
        for i in range(len(models['single'])):
            axs[0].plot(models['single'][i].state_dict_['loss'], c='blue', alpha=0.6)
        axs[0].set_xlabel("Steps", fontsize=15)
        axs[0].set_ylabel("Loss", fontsize=15)
        axs[0].set_title("Single Models", fontsize=20)

        # Plot for multi models
        for i in range(len(models['multi'])):
            axs[1].plot(models['multi'][i].state_dict_['loss'], c='orange', alpha=0.6)
        axs[1].set_xlabel("Steps", fontsize=15)
        axs[1].set_ylabel("Loss", fontsize=15)
        axs[1].set_title("Multi Models", fontsize=20)

        plt.tight_layout()
        plt.show()



    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Decoding models...')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

    results_untrained, results_single, results_multi = decoding_models(
    models['UT'], 
    models['single'], 
    models['multi'], 
    train_data, 
    train_label, 
    valid_data, 
    valid_label
    )
   
    mean_results_untrained = np.mean(results_untrained, axis = 0)
    mean_results_single = np.mean(results_single, axis = 0)
    mean_results_multi = np.mean(results_multi, axis = 0)

    print('UNTRAINED: ', 'Mean test score (R2): ', round(mean_results_untrained[0],4),'Mean test acc: ', round(mean_results_untrained[2],2),'%')
    print('Mean test acc Single: ',  round(mean_results_single[2],2),'%')
    print('Mean test acc Multi: ', round(mean_results_multi[2],2),'%')

    plot_accuracy_comparison(results_untrained, results_single, results_multi)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        '--model_name', type=str, default='offset10', help="name of the folder where the models (assuming they are under FinalModels/VISION)"
    )
    parser.add_argument(
        '--session_id', type=int, default=3, help="session id for the analysis, used to retrieve the correct data and multi-session model"
    )
    parser.add_argument(
        '--bool_plot_loss', type=int, default=1, help="Plots losses of the models"
    )

    args = parser.parse_args()
    print('INPUT PARAMETERS: ', args)
    main(args.model_name, args.session_id,args.bool_plot_loss)
