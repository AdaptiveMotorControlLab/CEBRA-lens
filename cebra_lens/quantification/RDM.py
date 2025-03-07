"""All the functions relative to the Representation Dissimilarity Matrix (RDM) calculation"""

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, pdist, squareform
from tqdm import tqdm
from .misc import discrete_binning
import torch


def create_oracle_rdm(dataset_label: str = "visual") -> np.ndarray:
    """
    Creates the Oracle RDM for the specified dataset.

    Parameters:
    -----------
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.

    Returns:
    --------
    np.ndarray
        The Oracle RDM as a squareform distance matrix.
    """

    if dataset_label == "visual":
        # Create Oracle RDM.
        one_class = np.ones((200, 200))
        all_classes = [one_class for _ in range(30)]
        block_rdm_sqform = 1 - block_diag(*all_classes)
        oracle_rdm = squareform(block_rdm_sqform)
    elif dataset_label == "HPC":
        # TODO: complete this
        raise NotImplementedError(
            f"Oracle RDM not defined for {dataset_label}. Please use 'visual'"
        )

    else:
        raise NotImplementedError(
            f"Oracle RDM not defined for {dataset_label}. Please use 'visual'"
        )

    return oracle_rdm


def compute_RDM_model(
    data: torch.Tensor,
    label: torch.Tensor,
    activations: list,
    dataset_label: str = "visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> list:
    """
    Computes the RDMs (Representational Dissimilarity Matrices) for each layer's activations.

    Parameters:
    -----------
    data : torch.Tensor
        The data array of shape (num_samples, num_features).
    label :torch.Tensor
        The array of labels corresponding to the data.
    activations : list
        A list of activations, each being an array of shape (num_neurons, num_samples).
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
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

    idxs = discrete_binning(data=data, label=label, dataset_label=dataset_label)

    layer_rdm = []

    for layer in activations:
        # to ensure the right shape: numSamples X numNeurons
        if layer.shape[0] < layer.shape[1]:
            layer = layer.T

        rdm = pdist(layer[idxs.flatten(), :], metric=metric)
        if bool_oracle:
            oracle_rdm = create_oracle_rdm(dataset_label="visual")
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


def compute_RDM_models(
    data: torch.Tensor,
    label: torch.Tensor,
    activations_dict: dict,
    dataset_label: str = "visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> dict:
    """
    Computes RDMs for multiple layers of activations across multiple models.

    Parameters:
    -----------
    data : torch.Tensor
        The data array of shape (num_samples, num_features).
    label : torch.Tensor
        The array of labels corresponding to the data.
    activations_dict : dict
        A dictionary containing activations for different models.
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
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

    for model_label, activations in activations_dict.items():
        rdm_dict[model_label] = []
        for activation in tqdm(activations, desc=f"Processing {model_label}"):
            processed_inner_list = compute_RDM_model(
                data=data,
                label=label,
                activations=activation,
                dataset_label=dataset_label,
                metric=metric,
                bool_oracle=bool_oracle,
            )

            rdm_dict[model_label].append(processed_inner_list)

    return rdm_dict
