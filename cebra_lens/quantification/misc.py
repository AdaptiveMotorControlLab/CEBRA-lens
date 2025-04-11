"""misc functions like normalization and possibly others"""

from random import sample
import numpy as np
import torch
import numpy.typing as npt
from typing import List


def normalize_minmax(rdm: npt.NDArray) -> npt.NDArray:
    """
    Normalizes a given array using Min-Max normalization.

    Parameters:
    -----------
    rdm : npt.NDArray
        A NumPy array to be normalized. This can be any numeric array, such as an RDM (Representational
        Dissimilarity Matrix), where values are normalized to the range [0, 1].

    Returns:
    --------
    npt.NDArray
        A normalized NumPy array where the minimum value is scaled to 0 and the maximum value is scaled to 1.
    """

    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)


def discrete_binning(
    data: torch.Tensor,
    label: torch.Tensor,
    dataset_label: str = "visual",
    sample_mode: str = "sub_sample",
) -> npt.NDArray:
    """
    Bins the training data based on the provided labels, creating indices for sampling. Used to discretize a continuous input for RDM.

    Parameters:
    -----------
    data : torch.Tensor
        The data array of shape (num_samples, num_features).
    label : torch.Tensor
        The array of labels corresponding to the data.
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
    sample_mode : str, optional
        If set to "sub" it will sample of subset of data (e.g. 200 samples per class as used in RDM), if "all" it will take all the training data (e.g. distance analysis).
    Returns:
    --------
    npt.NDArray
        An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
    """

    # BINNING
    if dataset_label == "visual":
        num_bins = 30

        if sample_mode == "sub_sample":
            num_samples = 200 if len(data) / 30 >= 200 else int(len(data) / 30)
        elif sample_mode == "all":
            num_samples = int(len(data) / 30)
        else:
            raise NotImplementedError(
                f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
            )

        step_distance = 30
        idxs = np.zeros((num_bins, num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where(
                (label[:] >= j * step_distance) & (label[:] < (j + 1) * step_distance)
            )[0]

            if sample_mode == "sub_sample":
                idxs[i, :] = sample(list(full_idxs), num_samples)
            elif sample_mode == "all":
                idxs[i, :] = list(full_idxs)
            else:
                raise NotImplementedError(
                    f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
                )

            j = j + 1
    elif dataset_label == "HPC":

        num_samples = 200
        num_bins = 16
        step_distance = 1.6 / num_bins * 2  # *2 for direction only
        idxs = np.zeros((num_bins, num_samples))

        j = 0
        for i in range(num_bins):
            if i < num_bins / 2:
                direction = 1
            else:
                direction = 2
            if i == num_bins / 2:
                j = 0

            full_idxs = np.where(
                (label[:, 0] >= j * step_distance)
                & (label[:, 0] < (j + 1) * step_distance)
                & (label[:, direction] == 1)
            )[0]

            if sample_mode == "sub_sample":
                idxs[i, :] = sample(list(full_idxs), num_samples)
            elif sample_mode == "all":
                idxs[i, :] = list(full_idxs)
            else:
                raise NotImplementedError(
                    f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
                )

            j = j + 1

    else:
        raise NotImplementedError(
            f"Binning not implemented for {dataset_label}. Use 'visual' or 'HPC'."
        )

    return idxs.astype(int)


def repetition_binning(
    indices: npt.NDArray, data, dataset_label: str = "visual"
) -> List[np.int64]:

    if dataset_label == "visual":
        samples_per_rep = 900
        step = 30
    elif dataset_label == "HPC":
        raise NotImplementedError("Not yet implemented for HPC.")
    else:
        raise NotImplementedError(f"Not yet implemented for {dataset_label}.")

    num_repetitions = data.shape[0] // samples_per_rep

    repetition_idxs = []
    for i in range(indices.shape[0]):
        repetition_bin_idxs = []

        for j in range(num_repetitions):

            repetition_bin_idxs.append(indices[i][j * step : (j + 1) * step])

        repetition_idxs.append(repetition_bin_idxs)
    return repetition_idxs
