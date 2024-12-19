import numpy as np
from random import sample
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, cdist, pdist, squareform
from tqdm import tqdm


def _rdm_binning(
    train_data: np.ndarray, train_label: np.ndarray, dataset_label: str = "Visual"
) -> np.ndarray:

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
    train_data: np.ndarray,
    train_label: np.ndarray,
    activations: list,
    dataset_label: str = "Visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> list:
    # activations should be a list of arrayts of shape numNeuronsXnumSAmples
    # make sure its a list, otherwise turn it into one. if type(activations) == np.ndarray or torch.Tensor: activations = [activations]
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

    if metric == "correlation":
        comparison = 1 - correlation(rdm_1, rdm_2)
    else:
        raise NotImplementedError(
            f"The metric {metric} is not yet implemented. Please use 'correlation'."
        )

    return comparison


def compute_multi_RDM_layers(
    train_data: np.ndarray,
    train_label: np.ndarray,
    activations_dict: dict,
    dataset_label: str = "Visual",
    metric: str = "correlation",
    bool_oracle: bool = "True",
) -> dict:

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
