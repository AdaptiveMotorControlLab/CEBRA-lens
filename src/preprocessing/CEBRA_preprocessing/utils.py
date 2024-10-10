import pandas as pd
import os
import numpy as np
import torch


def multisession_preparation(stimuli: list):
    
    print(type(stimuli))
    
    # make sure it's in a list format
    if isinstance(stimuli,list):
        stimuli = set(stimuli)
    elif isinstance(stimuli,str):
        stimuli = [stimuli]
        print('jher', stimuli)

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

        segments = _get_training_repeat_indices(single_stimulus_df)
        
        #training sets 9/10
        train_dff = dff_trace_stimulus_all_repetitions[segments[0][0]:segments[8][1],:]
        train_embeddings_stimulus = embeddingsExtended[segments[0][0]:segments[8][1],:]
        train_labels = single_stimulus_df['frame'].values[segments[0][0]:segments[8][1]]

        #testing sets 1/10
        test_dff = dff_trace_stimulus_all_repetitions[segments[9][0]:segments[9][1],:]
        test_embeddings_stimulus = embeddingsExtended[segments[9][0]:segments[9][1],:]
        test_labels = single_stimulus_df['frame'].values[segments[9][0]:segments[9][1]]

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