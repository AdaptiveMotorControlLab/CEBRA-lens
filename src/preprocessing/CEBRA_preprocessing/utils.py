import pandas as pd
import os
import numpy as np
import torch


def multisession_preparation(stimuli: list, split: float):
    
    print(type(stimuli))
    
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
        
        print('TRAIN INDEX', train_idx)
        print('END TRAIN:' ,segments[train_idx][1])
        print('BEGIN TEST:' ,segments[train_idx+1][0])
        print('END TEST: ',segments[-1][1])
        print('FULL SIZE: ', dff_trace_stimulus_all_repetitions.shape)

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
  #line_number manually found by looking in the files.
  
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
