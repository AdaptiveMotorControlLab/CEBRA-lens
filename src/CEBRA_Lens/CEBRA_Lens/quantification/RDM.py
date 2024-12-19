import numpy as np
from random import sample
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, pdist, squareform
from tqdm import tqdm
import torch


def _rdm_binning(
    train_data: torch.Tensor, train_label: torch.Tensor, dataset_label: str = "Visual"
) -> np.ndarray:
    """
    Bins the training data based on the provided labels, creating indices for sampling. Used to discretize a continuous input for RDM.

    Parameters:
    -----------
    train_data : torch.Tensor
        The training data array of shape (num_samples, num_features).
    train_label : torch.Tensor
        The array of labels corresponding to the training data.
    dataset_label : str, optional
        The dataset type, either 'Visual' or 'HPC'. Default is 'Visual'.

    Returns:
    --------
    np.ndarray
        An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
    """

    # BINNING
    if dataset_label == "Visual":
        num_bins = 30
        num_samples = 200 if len(train_data) / 30 >= 200 else int(len(train_data) / 30)
        step_distance = 30
        idxs = np.zeros((num_bins, num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where(
                (train_label[:] >= j * step_distance)
                & (train_label[:] < (j + 1) * step_distance)
            )[0]
            idxs[i, :] = sample(list(full_idxs), num_samples)
            j = j + 1
    elif dataset_label == "HPC":
        # TODO: implement this
        raise NotImplementedError("not implemented. A FAIRE")
    else:
        raise NotImplementedError(
            f"Binning not implemented for {dataset_label}. Use 'Visual' or 'HPC'."
        )

    return idxs.astype(int)


def create_oracle_rdm(dataset_label: str = "Visual") -> np.ndarray:
    """
    Creates the Oracle RDM for the specified dataset.

    Parameters:
    -----------
    dataset_label : str, optional
        The dataset type, either 'Visual' or 'HPC'. Default is 'Visual'.

    Returns:
    --------
    np.ndarray
        The Oracle RDM as a squareform distance matrix.
    """

    if dataset_label == "Visual":
        # Create Oracle RDM.
        one_class = np.ones((200, 200))
        all_classes = [one_class for _ in range(30)]
        block_rdm_sqform = 1 - block_diag(*all_classes)
        oracle_rdm = squareform(block_rdm_sqform)
    elif dataset_label == "HPC":
        # TODO: complete this
        raise NotImplementedError(
            f"Oracle RDM not defined for {dataset_label}. Please use 'Visual'"
        )

    else:
        raise NotImplementedError(
            f"Oracle RDM not defined for {dataset_label}. Please use 'Visual'"
        )

    return oracle_rdm


def compute_single_RDM_layers(
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    activations: list,
    dataset_label: str = "Visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> list:
    """
    Computes the RDMs (Representational Dissimilarity Matrices) for each layer's activations.

    Parameters:
    -----------
    train_data : torch.Tensor
        The training data array of shape (num_samples, num_features).
    train_label :torch.Tensor
        The array of labels corresponding to the training data.
    activations : list
        A list of activations, each being an array of shape (num_neurons, num_samples).
    dataset_label : str, optional
        The dataset type, either 'Visual' or 'HPC'. Default is 'Visual'.
    metric : str, optional
        The distance metric to use for computing the RDMs. Default is 'correlation'.
    bool_oracle : bool, optional
        Whether to compute and compare with the Oracle RDM. Default is True.

    Returns:
    --------
    list
        A list of tuples, where each tuple contains the RDM for the layer and the correlation with the Oracle RDM (if computed).
    """

    if isinstance(
        activations, (np.ndarray, torch.Tensor)
    ):  # if only one activation is passed instead of a list of arrays
        activations = [activations]

    idxs = _rdm_binning(
        train_data=train_data, train_label=train_label, dataset_label=dataset_label
    )

    layer_rdm = []

    for layer in activations:
        # to ensure the right shape: numSamples X numNeurons
        if layer.shape[0] < layer.shape[1]:
            layer = layer.T

        rdm = pdist(layer[idxs.flatten(), :], metric=metric)
        if bool_oracle:
            oracle_rdm = create_oracle_rdm(dataset_label="Visual")
            correlation = compare_RDM(rdm_1=oracle_rdm, rdm_2=rdm, metric="correlation")
        else:
            correlation = None

        layer_rdm.append((squareform(rdm), correlation))
    return layer_rdm


def compare_RDM(
    rdm_1: np.ndarray, rdm_2: np.ndarray, metric: str = "correlation"
) -> float:
    """
    Compares two RDMs using the specified metric.

    Parameters:
    -----------
    rdm_1 : np.ndarray
        The first RDM to compare.
    rdm_2 : np.ndarray
        The second RDM to compare.
    metric : str, optional
        The distance metric to use for comparison. Default is 'correlation'.

    Returns:
    --------
    float
        The similarity score between the two RDMs, based on the specified metric.
    """

    if metric == "correlation":
        comparison = 1 - correlation(rdm_1, rdm_2)
    else:
        raise NotImplementedError(
            f"The metric {metric} is not yet implemented. Please use 'correlation'."
        )

    return comparison


def compute_multi_RDM_layers(
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    activations_dict: dict,
    dataset_label: str = "Visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> dict:
    """
    Computes RDMs for multiple layers of activations across multiple models.

    Parameters:
    -----------
    train_data : torch.Tensor
        The training data array of shape (num_samples, num_features).
    train_label : torch.Tensor
        The array of labels corresponding to the training data.
    activations_dict : dict
        A dictionary containing activations for different models.
    dataset_label : str, optional
        The dataset type, either 'Visual' or 'HPC'. Default is 'Visual'.
    metric : str, optional
        The distance metric to use for computing the RDMs. Default is 'correlation'.
    bool_oracle : bool, optional
        Whether to compute and compare with the Oracle RDM. Default is True.

    Returns:
    --------
    dict
        A dictionary containing RDMs for each model, with each entry being a list of RDMs for the corresponding layer.
    """

    rdm_dict = {}

    for outer_key, outer_value in activations_dict.items():
        rdm_dict[outer_key] = {}
        for inner_key, outer_list in tqdm(
            outer_value.items(), desc=f"Processing {outer_key}"
        ):
            rdm_dict[outer_key][inner_key] = []
            for inner_list in tqdm(
                outer_list, desc=f"Processing {outer_key} {inner_key}"
            ):
                processed_inner_list = compute_single_RDM_layers(
                    train_data=train_data,
                    train_label=train_label,
                    activations=inner_list,
                    dataset_label=dataset_label,
                    metric=metric,
                    bool_oracle=bool_oracle,
                )

                rdm_dict[outer_key][inner_key].append(processed_inner_list)

    return rdm_dict
