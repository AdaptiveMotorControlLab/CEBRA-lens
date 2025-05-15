import cebra
import numpy as np
import sklearn
import cebra.datasets
import numpy.typing as npt
from typing import List


def get_single_session_datasets(
    rats: List[str] = ["achilles", "buddy", "cicero", "gatsby"]
):
    """
    Args:
        rats:  List of names of the different sessions

    Returns:
      The train and valid datasets, and the train and valid labels.
    """
    train_datas, valid_datas, continuous_labels_train, continuous_labels_val = (
        [],
        [],
        [],
        [],
    )

    for i in rats:
        data = cebra.datasets.init(f"rat-hippocampus-single-{i}")
        neural_train, neural_valid, label_train, label_valid = split_data_HPC(data)
        train_datas.append(neural_train)
        valid_datas.append(neural_valid)
        continuous_labels_train.append(label_train)
        continuous_labels_val.append(label_valid)

    return train_datas, valid_datas, continuous_labels_train, continuous_labels_val


def decoding_pos_dir(
    embedding_train: npt.NDArray,
    embedding_test: npt.NDArray,
    label_train: npt.NDArray,
    label_test: npt.NDArray,
):
    """
    Decodes position and direction from embeddings using K-Nearest Neighbors (KNN) and evaluates the performance.
    Parameters:
    ----------
    embedding_train : npt.NDArray
    embedding_test : npt.NDArray
    label_train : npt.NDArray
        Training labels, where the first column represents position and the second column represents direction.
    label_test : npt.NDArray
        Testing labels, where the first column represents position and the second column represents direction.
    Returns:
    -------
    test_score : float
        R^2 score for the combined position and direction predictions.
    pos_test_err : float
        Median absolute error for the position prediction.
    pos_test_score : float
        R^2 score for the position prediction.
    """

    pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
    dir_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")

    pos_decoder.fit(embedding_train, label_train[:, 0])
    dir_decoder.fit(embedding_train, label_train[:, 1])

    pos_pred = pos_decoder.predict(embedding_test)
    dir_pred = dir_decoder.predict(embedding_test)

    prediction = np.stack([pos_pred, dir_pred], axis=1)

    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])

    return test_score, pos_test_err, pos_test_score


def split_data_HPC(data: object, test_ratio: float = 0.2):
    """
    Splits the given data into training and testing sets based on the specified test ratio.
    Args:
        data (object): The data object containing 'neural' and 'continuous_index' attributes.
        test_ratio (float): The ratio of the data to be used for testing. Should be a value between 0 and 1.
    Returns:
        tuple: A tuple containing four elements:
            - neural_train (numpy.ndarray): The training set for neural data.
            - neural_test (numpy.ndarray): The testing set for neural data.
            - label_train (numpy.ndarray): The training set for continuous index labels.
            - label_test (numpy.ndarray): The testing set for continuous index labels.
    """

    split_idx = int(len(data) * (1 - test_ratio))
    neural_train = data.neural[:split_idx]
    neural_test = data.neural[split_idx:]
    label_train = data.continuous_index[:split_idx]
    label_test = data.continuous_index[split_idx:]

    return (
        neural_train.numpy(),
        neural_test.numpy(),
        label_train.numpy(),
        label_test.numpy(),
    )
