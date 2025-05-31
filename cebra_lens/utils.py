import pathlib
import cebra
import torch
from typing import Dict, List, Any, Union
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch import nn
from .quantification.decoding import Decoding
from .quantification.rdm_metric import RDM
from .quantification.cka_metric import CKA
from .utils_hpc import get_datasets as get_datasets_hpc
from .utils_allen import get_datasets as get_datasets_visual


def get_data(
    dataset_label: str = None, session_id: int = None
) -> list[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    if dataset_label == "visual":
        return get_datasets_visual(session_id=session_id)
    elif dataset_label == "HPC":
        return get_datasets_hpc(session_id=session_id)
    else:
        raise ValueError(
            f"Dataset label {dataset_label} is not recognized. Please use 'visual' or 'HPC', or add data loading function here."
        )


def extract_label(labels: npt.NDArray, label_ind: int) -> List:
    """
    Extracts unique labels from a NumPy array of labels.
    Parameters:
    -----------
    labels : npt.NDArray
        A NumPy array containing labels, which can be of any numeric type.
    label_ind : int
        The index of the label to extract from the array.
    Returns:
    --------
    List
        A list of unique labels extracted from the input array.
    """
    try:
        num_labels = labels.shape[1]
    except:
        num_labels = 1
        labels = labels.reshape(-1, 1)
    if label_ind > num_labels - 1:
        raise ValueError(
            f"label_ind {label_ind} is out of range for labels with shape {labels.shape}"
        )
    labels = labels[:, label_ind]

    return labels


def compute_metric(
    model_data: Dict[str, List[npt.NDArray[Any]]],
    metric_class: Any,
    output_only: bool = False,
    max_num_samples: int = None,
) -> Dict[str, npt.NDArray[Any]]:
    """
    Computes metrics for each model using a provided metric class.

    Parameters:
    -----------
    model_data : Dict[str, List[npt.NDArray]]
        Dictionary mapping model labels to lists of data samples (e.g., activations).

    metric_class : object
        Object with a `compute` method that takes a single data sample and returns a computed metric.

    Returns:
    --------
    Dict[str, npt.NDArray]
        Dictionary mapping model labels to arrays of computed metric values.
    """
    result_dict = {}

    if isinstance(metric_class, CKA):
        for comparison in tqdm(metric_class.comparisons):
            cka_matrix = metric_class.compute(model_data, comparison)
            result_dict[f"{comparison[0]}_v_{comparison[1]}"] = cka_matrix

    else:
        for group_name, samples in model_data.items():
            if isinstance(metric_class, Decoding):
                metric_class.set_output_only(output_only)

            computed_values = [
                metric_class.compute(sample)
                for sample in tqdm(samples, desc=f"Processing {group_name}")
            ]
            if not isinstance(metric_class, RDM):
                computed_values = np.array(computed_values)

            result_dict[group_name] = computed_values

    return result_dict


def plot_metric(
    data_dict: Union[Dict[str, npt.NDArray[Any]], npt.NDArray],
    metric_class: object,
    group_name: str = "Model group",
    **kwargs,
) -> None:
    """
    Plots metrics for each model using a provided metric class.

    Parameters:
    -----------
    data_dict : Union[Dict[str, npt.NDArray[Any]], npt.NDArray]
        Dictionary mapping model labels to arrays of computed metric values or metric values for a single model.

    metric_class : object
        Object with a `plot` method that takes a single data sample and returns a plot.
    """

    if not isinstance(data_dict, Dict) and isinstance(metric_class, RDM):
        data_dict = {group_name: data_dict}
    return metric_class.plot(data_dict, **kwargs)


def model_loader(
    model_dir: str, labels: Dict[str, str] = {}
) -> Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]:
    """
    Loads and categorizes CEBRA models from a given directory.

    Parameters:
    -----------
    model_dir : str
        Path to the directory containing model `.pt` or `.pth` files.

    labels : Dict[str, str], optional
        Dictionary mapping model file names (without extensions) to category labels.
        If not provided, the model's own filename will be used as its label.

    Returns:
    --------
    Dict[str, List[CEBRA]]
        Dictionary with model category labels as keys and lists of loaded CEBRA models as values.
    """
    models_folder_path = pathlib.Path(model_dir)
    if not models_folder_path.exists():
        raise FileNotFoundError(f"Folder {models_folder_path} not found.")

    models = {}
    for file in models_folder_path.iterdir():
        if str(file).endswith((".pt", ".pth")):
            loaded_model = cebra.CEBRA.load(
                file, backend="torch", map_location=torch.device("cpu")
            ).to("cpu")
            key = labels.get(file.stem, file.stem)
            models.setdefault(key, []).append(loaded_model)
            print(f"Model {file.stem} loaded successfully.")

    return models
