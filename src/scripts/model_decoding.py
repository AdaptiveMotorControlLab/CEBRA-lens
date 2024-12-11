#run using % python -m GithubFolder.src.scripts.model_decoding --model_name offset10 --session_id 3

import torch.nn as nn

from ..preprocessing.CEBRA_preprocessing.plotting_utils import *
from ..preprocessing.CEBRA_preprocessing.data_utils import *
from ..preprocessing.CEBRA_preprocessing.quantification_utils import *
import argparse

def main(model_name, session_id):
    
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
 

    args = parser.parse_args()
    print('INPUT PARAMETERS: ', args)
    main(args.model_name, args.session_id)
