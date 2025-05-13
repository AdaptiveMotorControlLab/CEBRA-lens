import pathlib
import cebra
import torch
from typing import Dict, List, Any
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch import nn


def compute(
    model_data: Dict[str, List[npt.NDArray[Any]]],
    metric_class: Any,
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
    result_dict: Dict[str, npt.NDArray[Any]] = {}

    for model_label, samples in model_data.items():
        computed_values = [
            metric_class.compute(sample)
            for sample in tqdm(samples, desc=f"Processing {model_label}")
        ]
        result_dict[model_label] = computed_values

    return result_dict


def load(
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
