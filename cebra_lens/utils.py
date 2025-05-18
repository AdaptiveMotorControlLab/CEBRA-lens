import pathlib
import cebra
import torch
from typing import Dict, List, Any
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch import nn
from .quantification.decoding import Decoding
from .quantification.rdm_metric import RDM
from .quantification.cka_metric import CKA


def compute_metric(
    model_data: Dict[str, List[npt.NDArray[Any]]],
    metric_class: Any,
    output_only: bool = False,
    max_num_samples: int = None,
    num_bins: int = None,
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
        for model_label, samples in model_data.items():
            if isinstance(metric_class, Decoding):
                metric_class.set_output_only(output_only)
            elif isinstance(metric_class, RDM):
                metric_class.set_num_bins(num_bins)
                metric_class.set_num_samples(max_num_samples)

            computed_values = np.array(
                [
                    metric_class.compute(sample)
                    for sample in tqdm(samples, desc=f"Processing {model_label}")
                ]
            )
            result_dict[model_label] = computed_values

    return result_dict

def plot_metric(
    data_dict: Dict[str, npt.NDArray[Any]],
    metric_class: Any,
    **kwargs
) -> None:
    """
    Plots metrics for each model using a provided metric class.

    Parameters:
    -----------
    data_dict : Dict[str, npt.NDArray]
        Dictionary mapping model labels to arrays of computed metric values.

    metric_class : object
        Object with a `plot` method that takes a single data sample and returns a plot.

    """
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
