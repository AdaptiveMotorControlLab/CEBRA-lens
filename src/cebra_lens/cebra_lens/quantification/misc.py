"""misc functions like normalization and possibly others"""

from random import sample
import numpy as np
import torch


def normalize_minmax(rdm: np.ndarray) -> np.ndarray:
    """
    Normalizes a given array using Min-Max normalization.

    Parameters:
    -----------
    rdm : np.ndarray
        A NumPy array to be normalized. This can be any numeric array, such as an RDM (Representational
        Dissimilarity Matrix), where values are normalized to the range [0, 1].

    Returns:
    --------
    np.ndarray
        A normalized NumPy array where the minimum value is scaled to 0 and the maximum value is scaled to 1.
    """

    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)


def discrete_binning(
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    dataset_label: str = "Visual",
    sample_mode: str = "sub_sample",
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
    sample_mode : str, optional
        If set to "sub" it will sample of subset of data (e.g. 200 samples per class as used in RDM), if "all" it will take all the training data (e.g. distance analysis).
    Returns:
    --------
    np.ndarray
        An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
    """

    # BINNING
    if dataset_label == "Visual":
        num_bins = 30

        if sample_mode == "sub_sample":
            num_samples = (
                200 if len(train_data) / 30 >= 200 else int(len(train_data) / 30)
            )
        elif sample_mode == "all":
            num_samples = int(len(train_data) / 30)
        else:
            raise NotImplementedError(
                f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
            )

        step_distance = 30
        idxs = np.zeros((num_bins, num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where(
                (train_label[:] >= j * step_distance)
                & (train_label[:] < (j + 1) * step_distance)
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
        # TODO: implement this

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
                (train_label[:, 0] >= j * step_distance)
                & (train_label[:, 0] < (j + 1) * step_distance)
                & (train_label[:, direction] == 1)
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
            f"Binning not implemented for {dataset_label}. Use 'Visual' or 'HPC'."
        )

    return idxs.astype(int)


def repetition_binning(
    indices: np.ndarray, train_data, dataset_label: str = "Visual"
) -> list:

    if dataset_label == "Visual":
        samples_per_rep = 900
        step = 30
    elif dataset_label == "HPC":
        raise NotImplementedError("Not yet implemented for HPC.")
    else:
        raise NotImplementedError(f"Not yet implemented for {dataset_label}.")

    num_repetitions = train_data.shape[0] // samples_per_rep

    repetition_idxs = []
    for i in range(indices.shape[0]):
        repetition_bin_idxs = []

        for j in range(num_repetitions):

            repetition_bin_idxs.append(indices[i][j * step : (j + 1) * step])

        repetition_idxs.append(repetition_bin_idxs)
    return repetition_idxs
