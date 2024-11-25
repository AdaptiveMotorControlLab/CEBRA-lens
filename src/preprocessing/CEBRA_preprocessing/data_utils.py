import pandas as pd
import os
import numpy as np
import torch
import cebra
import cebra.datasets
import copy


# NOT USED IN THE NOTEBOOKS (TO BE REMOVED)
def multisession_preparation(stimuli: list, split: float):
    """
    This function loads the Open scope data and formats it for multisession training.

    Input:
    - stimuli (list): list of stimuli to load.
    - split (float): split value for train/test splitting.

    Output:
    - data_train, data_test: The Open scope neural data.
    - labels_train,labels_test: The Openscope data labels.
    - embeddings: corresponding DINOv2 embeddings
    """
    
    # make sure it's in a list format
    if isinstance(stimuli,list):
        print('list of stimuli:', stimuli)
        #stimuli = set(stimuli)
    elif isinstance(stimuli,str):
        stimuli = [stimuli]

    try:
        data_dir = os.environ["DATA_PATH"]
    except KeyError:
        raise ValueError("DATA_PATH environment variable is not set")


    data_stimulus_path =  os.path.join(os.environ["DATA_PATH"],'df_stimulus.csv')
    data_dff_path = os.path.join(os.environ["DATA_PATH"],'dff_trace.npy')

    df_stimulus = pd.read_csv(data_stimulus_path)
    dff_trace = np.load(data_dff_path)

    data_train = []
    data_test = []
    labels_train =[]
    labels_test = []
    embeddings_train = []
    embeddings_test =[]
    embeddings_simple = []

    print(stimuli)

    for stimulus in stimuli:

        print('Processing stimulus: ',stimulus)

        print('type stimulus', type(stimulus))
        load_path = os.path.join(os.environ["DATA_PATH"],stimulus,'Dinov2_embeddings/vitb14.pt')
        print('OS PATH', os.environ["DATA_PATH"])
        print('load path', load_path)
        embeddings = torch.load(load_path,map_location=torch.device('cpu'))

        single_stimulus_df = df_stimulus[df_stimulus['stim_type'] == stimulus].reset_index()

        # Create embeddingsExtended to align with filtered_df
        embeddingsExtended = torch.empty(len(single_stimulus_df), embeddings.size(1))

        # Map the frame numbers to their respective indices in the embeddings tensor
        for idx, frame in enumerate(single_stimulus_df['frame'].values):

            embeddingsExtended[idx] = embeddings[frame]

        dff_trace_stimulus_all_repetitions = dff_trace[single_stimulus_df['index'].values,:]    

        segments = _get_training_repeat_indices(single_stimulus_df) # list of tuples 
        
       #training sets split*10
        train_idx = int(split*len(segments)) - 1

        train_dff = dff_trace_stimulus_all_repetitions[segments[0][0]:segments[train_idx][1],:]
        train_embeddings_stimulus = embeddingsExtended[segments[0][0]:segments[train_idx][1],:]
        train_labels = single_stimulus_df['frame'].values[segments[0][0]:segments[train_idx][1]]

        test_dff = dff_trace_stimulus_all_repetitions[segments[train_idx+1][0]:segments[-1][1],:]
        test_embeddings_stimulus = embeddingsExtended[segments[train_idx+1][0]:segments[-1][1],:]
        test_labels = single_stimulus_df['frame'].values[segments[train_idx+1][0]:segments[-1][1]]
            
        ##########################################
        # Filter out the frame 899

        idx_no_899_train = np.where(train_labels != 899)[0]
        idx_no_899_test = np.where(test_labels != 899)[0]

        train_labels = train_labels[idx_no_899_train]
        test_labels = test_labels[idx_no_899_test]

        train_dff = train_dff[idx_no_899_train,:]
        test_dff = test_dff[idx_no_899_test,:]

        train_embeddings_stimulus = train_embeddings_stimulus[idx_no_899_train,:]
        test_embeddings_stimulus = test_embeddings_stimulus[idx_no_899_test,:]

        ##########################################
        data_train.append(train_dff)
        data_test.append(test_dff)

        labels_train.append(train_labels)
        labels_test.append(test_labels)

        embeddings_train.append(train_embeddings_stimulus)
        embeddings_test.append(test_embeddings_stimulus)
        embeddings_simple.append(embeddings)

    return data_train,data_test,labels_train,labels_test, embeddings_train,embeddings_test,embeddings_simple


def _get_training_repeat_indices(df):
  segments = [] # to store the tuples (start_idx,end_idx)

  start_idx = None
  for idx, row in df.iterrows():
      if row['frame'] == 0:
          if start_idx is not None:
              # When we find a new 0, the previous segment ends here
              segments.append((start_idx, idx - 1))
          start_idx = idx

  # Append the last segment
  if start_idx is not None:
      segments.append((start_idx, len(df) - 1))

  return segments



def add_function_at_line(file_path, new_function, line_number):
  """
    This function adds a line to a file. Was meant to be used to add a line to the CEBRA package when loading on COLAB.

    Input:
    - file_path: where the file to modify is.
    - new_function (string): the function text to insert.
    - line_number: line where the insertion should be made. Manually found by looking in the files.
  """
  
  # Read the existing file content into a list of lines
  with open(file_path, 'r') as file:
      lines = file.readlines()

  indent = 4 # manually found in the file

  indented_function = '\n'.join([(' ' * indent) + line for line in new_function.strip().split('\n')])

  # Insert the new function at the desired line number
  lines.insert(line_number, indented_function + '\n')  # Add a newline at the end

  # Write the modified content back to the file
  with open(file_path, 'w') as file:
      file.writelines(lines)

def split_data_HPC(data, test_ratio):
    """
    This function splits the hippocampus data into test and train.

    Input:
    - data: hippocampus data to split.
    - test_ratio (float): split value for train/test splitting.

    Output:
    - neural_train, neural_test: neural data.
    - labels_train,labels_test: data labels.
    """

    split_idx = int(len(data)* (1-test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()

def separate_activations(activations):
    """
    This function separates the layer activations into trained and untrained.

    Input:
    - activations: dictionnary of activations. Each key has _UT if the activations come from untrained models

    Output:
    - embeddings_untrained,embeddings_trained: splitted activations.
    """
    # put everything in a list
    embeddings_untrained = []
    embeddings_trained = []

    for key, value in activations.items():
      if 'UT' in key:
        embeddings_untrained.append(value.squeeze().cpu().numpy())
      else:
        embeddings_trained.append(value.squeeze().cpu().numpy())
    return embeddings_untrained,embeddings_trained

########################################################################################################################
########################################################################################################################
######################################## TAKEN FROM utils_allen.py of Célia ############################################
########################################################################################################################
########################################################################################################################


def get_single_session_datasets(
    test_session=9,
    corrupted=False,
    pseudomice=False,
    mice=4,
    shot_noise: float = None,
    gaussian_noise: float = None,
):
    """
    Args: 
        test_session: The session ID to consider as the test session. NOTE(celia): this will need
            to be changed if we want to test smaller training set number of repeats.
        corrupted: If True, loads the corrupted dataset, see `datasets/allen/single_session_ca.py` in CEBRA 
            codebase.
        pseudomice: If True, uses pseudomice rather than full, with default number of neurons per 
            mouse.
        mice: Number of mice to use (max is 4, for now). NOTE(celia): this could be increased by
            pre-processing more mice. 
        shot_noise: Level of shot noise (Poisson noise) to apply on the dataset. Default is None, 
            and that means that no noise is applied.
        gaussian_noise: Value of the standard deviaiton of the Gaussian noise to add on the data.
            Default is None, and that means that no noise is applied.
    
    Returns: 
        The train and valid datasets, and the train and valid frame IDs.
    """
    train_datas, valid_datas = [], []
    if corrupted:
        for i in range(mice):
            train_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-corrupt-{i}-repeat-{test_session}-train"
                )
            )
            valid_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-corrupt-{i}-repeat-{test_session}-test"
                )
            )
    else:
        for i in range(4):
            train_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-{i}-repeat-{test_session}-train"
                )
            )
            valid_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-{i}-repeat-{test_session}-test"
                )
            )
    if pseudomice:
        for i in range(len(train_datas)):
            train_datas[i].neural = torch.from_numpy(
                obtain_pseudomice(
                    [train_datas[i].neural for i in range(len(train_datas))]
                )
            )
            valid_datas[i].neural = torch.from_numpy(
                obtain_pseudomice(
                    [valid_datas[i].neural for i in range(len(train_datas))]
                )
            )

    # Add noise to the 4th mouse only
    if shot_noise is not None:
        # train_datas[0].neural = _add_shot_noise(train_datas[0].neural, scale_factor=shot_noise)
        valid_datas[3].neural = _add_shot_noise(
            valid_datas[3].neural, scale_factor=shot_noise
        )
    elif gaussian_noise is not None:
        # train_datas[0].neural = _add_gaussian_noise(train_datas[0].neural, sigma=gaussian_noise)
        valid_datas[3].neural = _add_gaussian_noise(
            valid_datas[3].neural, sigma=gaussian_noise
        )

    # discrete_labels = [np.tile(np.arange(900), 10) for i in range(len(mice))]
    discrete_labels_train = [np.tile(np.arange(900), 9) for i in range(mice)]
    discrete_labels_val = [np.tile(np.arange(900), 1) for i in range(mice)]

    return train_datas, valid_datas, discrete_labels_train, discrete_labels_val


def obtain_pseudomice(mice, num_neurons_per_mouse=80):
    """
    Creates a pseudomouse by selecting a random subset of neurons from each mouse's neural data.

    Parameters:
    mice (list): List of neural data for different mice
    num_neurons_per_mouse (int): the number of neurons to select from each mouse

    Returns:
    Mouse: a pseudomouse object with the concatenated neural data from the selected neurons
    """
    neuron_ids = []
    pseudomice_matrix = None
    for i, session in enumerate(mice):
        session_length = session.shape[1]
        selected_neurons = np.random.choice(
            session_length, replace=False, size=num_neurons_per_mouse
        )
        neuron_ids.append(selected_neurons)
        if pseudomice_matrix is None:
            pseudomice_matrix = session[:, selected_neurons]
        else:
            pseudomice_matrix = np.concatenate(
                (pseudomice_matrix, session[:, selected_neurons]), axis=1
            )

    pseudomouse = copy.deepcopy(mice[0])
    pseudomouse = pseudomice_matrix
    return pseudomouse


def _add_gaussian_noise(neural_data, sigma: float = 2):
    gaussian_noise = torch.normal(mean=0.0, std=sigma, size=neural_data.size())
    return neural_data + gaussian_noise


def _add_shot_noise(neural_data, scale_factor: float = 1.0):
    # Neural data * scale_factor = Poisson lambda
    return torch.poisson(neural_data * scale_factor) / scale_factor
