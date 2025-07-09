"""misc functions like normalization and possibly others"""

import warnings
from random import sample
from typing import List

import numpy as np
import numpy.typing as npt
import torch


def normalize_minmax(rdm: npt.NDArray) -> npt.NDArray:
    """Normalizes a given array using Min-Max normalization.

    Args:
        rdm : npt.NDArray
            A NumPy array to be normalized. This can be any numeric array, such as an RDM (Representational
            Dissimilarity Matrix), where values are normalized to the range [0, 1].

    Returns:
        npt.NDArray
            A normalized NumPy array where the minimum value is scaled to 0 and the maximum value is scaled to 1.
    """

    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)


def discrete_binning(labels: npt.NDArray) -> npt.NDArray:
    """Defines bins for discrete labels and the indices of the samples in each bin.
    
    This function is used to create bins for discrete labels in RDM analysis and distance analysis.

    Args:
        labels : npt.NDArray
            A NumPy array containing discrete labels, which can be of any numeric type.

    Returns:
        npt.NDArray
            An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
    """
    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    idxs_dict = {
        unique_labels[i]: np.where(inverse_indices == i)[0]
        for i in range(len(unique_labels))
    }

    min_count = min(len(idxs) for idxs in idxs_dict.values())
    idxs = []
    for label, ind in idxs_dict.items():
        idxs.append(np.random.choice(ind, min_count, replace=False))

    return np.array(idxs)


def continuous_binning(
    data: torch.Tensor,
    label: torch.Tensor,
    dataset_label: str = None,
    sample_mode: str = "sub_sample",
    max_num_samples: int = 200,
) -> npt.NDArray:
    """Bins the training data based on the provided labels, creating indices for sampling. 
    
    Used to discretize a continuous input for RDM.
    For non-specific datasets, the number of bins is determined empirically based on the data size, based on a heuristic of 0.005 * num_samples.

    Args:
        data : torch.Tensor
            The data array of shape (num_samples, num_features).
        label : torch.Tensor
            The array of labels corresponding to the data.
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'. If None, it will do an empirical binning based on the data.
        sample_mode : str, optional
            If set to "sub" it will sample of subset of data (e.g. 200 samples per class as used in RDM), if "all" it will take all the training data (e.g. distance analysis).
        max_num_samples : int
            The maximum number of samples per bin allowed if the number of labels divided by the num_bins is bigger than 200

    Returns:
        npt.NDArray
            An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
    """

    num_bins = None
    if dataset_label == "visual":
        num_bins = 30

        if sample_mode == "sub_sample":
            num_samples = (max_num_samples if len(data) / num_bins
                           >= max_num_samples else int(len(data) / num_bins))
        elif sample_mode == "all":
            num_samples = int(len(data) / num_bins)
        else:
            raise NotImplementedError(
                f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
            )

        step_distance = 30
        idxs = np.zeros((num_bins, num_samples))

        j = 0
        for i in range(num_bins):

            full_idxs = np.where((label[:] >= j * step_distance)
                                 & (label[:] < (j + 1) * step_distance))[0]

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

            full_idxs = np.where((label[:, 0] >= j * step_distance)
                                 & (label[:, 0] < (j + 1) * step_distance)
                                 & (label[:, direction] == 1))[0]

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
        if len(data) < 1000:
            warnings.warn(
                "Continuous binning is not recommended for datasets with less than 1000 samples. "
                "Consider using discrete binning instead.", UserWarning)
        num_bins = int(
            0.005 * len(data)
        )  # 0.005 is a heuristic to get a reasonable number of bins for continuous data
        if sample_mode == "sub_sample":
            num_samples = (max_num_samples if len(data) / num_bins
                           >= max_num_samples else int(len(data) / num_bins))
        elif sample_mode == "all":
            num_samples = int(len(data) / num_bins)
        else:
            raise NotImplementedError(
                f"Sample mode {sample_mode} not yet implemented. Please use 'all' or 'sub_sample'."
            )

        max_value = max(label).item()
        min_value = min(label).item()
        step_distance = (max_value - min_value) / num_bins

        print("Binning continuous data for non-specific dataset:")
        print(f"Number of bins: {num_bins}")
        print(f"Step size between bins: {round(step_distance,2)}")
        print("-----------------------------")
        indices = []
        for i in range(num_bins):
            lower_bin_border = round(min_value + i * step_distance, 2)
            higher_bin_border = round(min_value + (i + 1) * step_distance, 2)
            full_idxs = np.where((label[:] >= lower_bin_border)
                                 & (label[:] < higher_bin_border))[0]
            indices.append(full_idxs)

        # Due do uneven number of samples in each bin, we will take the minimum number of samples from each bin, maybe need to discuss this further
        min_ind = np.argmin([len(i) for i in indices])
        num_samples = len(indices[min_ind])
        print("Number of samples per bin:", num_samples)
        idxs = np.zeros((num_bins, num_samples))
        for i in range(num_bins):
            idxs[i, :] = np.random.choice(indices[i],
                                          num_samples,
                                          replace=False)
    print(num_bins)
    return idxs.astype(int), num_bins


def repetition_binning(indices: npt.NDArray,
                       data,
                       dataset_label: str = "visual") -> List[npt.NDArray]:
    """Creates a list of indices for each repetition based on the provided indices and dataset label.

    This is relevant for datasets where the labels are repeated over multiple samples, such as in the Allen visual dataset.

    Args:
        indices : npt.NDArray
            An array of shape (num_bins, num_samples) representing the indices of samples in each bin.
        data : npt.NDArray
            The data array of shape (num_samples, num_features).
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'. If None, it will do an empirical binning based on the data.

    Returns:
        List[npt.NDArray]
            A list of indices for each repetition, where each element is an array of shape (num_bins, num_samples_per_rep).
    """

    if dataset_label == "visual":
        samples_per_rep = 900
        step = 30
    elif dataset_label == "HPC":
        raise NotImplementedError("Not yet implemented for HPC.")
    else:
        raise NotImplementedError(f"Not yet implemented for {dataset_label}.")

    num_repetitions = data.shape[0] // samples_per_rep

    repetition_idxs = []
    # indices.shape = (num_bins, num_samples)
    for i in range(indices.shape[0]):
        repetition_bin_idxs = []

        for j in range(num_repetitions):

            repetition_bin_idxs.append(indices[i][j * step:(j + 1) * step])

        repetition_idxs.append(repetition_bin_idxs)
    return repetition_idxs
