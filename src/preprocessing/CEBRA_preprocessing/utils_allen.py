import numpy as np

import cebra
import cebra.datasets
import copy
import torch
import sklearn.metrics


def _quantize_acc(frame_diff, time_window=1):

    true = (abs(frame_diff) < (time_window * 30)).sum()
    return true / len(frame_diff) * 100


def create_sequences(embedding, labels, seq_len=10):
    sequences = []
    sequence_labels = []
    for i in range(len(embedding) - seq_len):
        seq = embedding[i : i + seq_len]
        # Label is the frame number following the sequence
        label = labels[i + seq_len]
        sequences.append(seq)
        sequence_labels.append(label)
    return np.array(sequences), np.array(sequence_labels)


def decoding_frames(
    embedding_train, embedding_test, label_train, label_test, time_window=1, seq_len=1
):
    """ 1-frame decoding.
    
    TODO(celia): Implement n-frames decoding. Started but not functional yet.
    """
    if seq_len > 1:
        embedding_train, label_train = create_sequences(
            embedding_train, label_train, seq_len
        )
        embedding_test, label_test = create_sequences(
            embedding_test, label_test, seq_len
        )

    params = np.power(np.linspace(1, 10, 10, dtype=int), 2)
    errs = []
    for n in params:
        train_decoder = cebra.KNNDecoder(n_neighbors=n, metric="cosine")
        train_valid_idx = int(len(embedding_train) / 9 * 8)
        if seq_len > 1:
            train_decoder.fit(
                embedding_train[:train_valid_idx].reshape(
                    -1, seq_len * embedding_train.shape[2]
                ),
                label_train[:train_valid_idx],
            )
            pred = train_decoder.predict(
                embedding_train[train_valid_idx:].reshape(
                    -1, seq_len * embedding_train.shape[2]
                )
            )
        else:
            train_decoder.fit(
                embedding_train[:train_valid_idx], label_train[:train_valid_idx]
            )
            pred = train_decoder.predict(embedding_train[train_valid_idx:])
        err = label_train[train_valid_idx:] - pred
        errs.append(abs(err).sum())

    test_decoder = cebra.KNNDecoder(
        n_neighbors=params[np.argmin(errs)], metric="cosine"
    )
    if seq_len > 1:
        test_decoder.fit(
            embedding_train.reshape(-1, seq_len * embedding_train.shape[2]), label_train
        )
        frame_pred = test_decoder.predict(
            embedding_test.reshape(-1, seq_len * embedding_test.shape[2])
        )
    else:
        test_decoder.fit(embedding_train, label_train)
        frame_pred = test_decoder.predict(embedding_test)

    frame_errors = frame_pred - label_test
    test_score = sklearn.metrics.r2_score(label_test, frame_pred)
    test_err = np.median(abs(frame_errors))
    test_acc = _quantize_acc(frame_errors, time_window=1)

    return test_score, test_err, test_acc


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

    # Add noise to the first mouse only
    if shot_noise is not None:
        # train_datas[0].neural = _add_shot_noise(train_datas[0].neural, scale_factor=shot_noise)
        valid_datas[0].neural = _add_shot_noise(
            valid_datas[0].neural, scale_factor=shot_noise
        )
    elif gaussian_noise is not None:
        # train_datas[0].neural = _add_gaussian_noise(train_datas[0].neural, sigma=gaussian_noise)
        valid_datas[0].neural = _add_gaussian_noise(
            valid_datas[0].neural, sigma=gaussian_noise
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
