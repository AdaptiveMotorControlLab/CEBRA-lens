import pathlib
import cebra
import torch
from typing import Dict, List, Union, Type, Optional
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from torch import nn
from .activations import process_activations, aggregate_activations
from .quantification.decoding import DecodeModel, Decoding


class CEBRALens:
    """
    Provides functions to load, analyze, and evaluate CEBRA models.
    """

    def compute(
        model_data: Dict[str, List[npt.NDArray]],
        metric_class: object,
    ) -> Dict[str, npt.NDArray]:
        """
        Computes metrics for each model using a provided metric class.

        Parameters:
        -----------
        model_data : Dict[str, List[npt.NDArray]]
            Dictionary with model labels as keys and lists of activation arrays or other model outputs as values.

        metric_class : object
            An object that implements a `compute` method which accepts individual data samples.

        Returns:
        --------
        Dict[str, npt.NDArray]
            Dictionary with model labels as keys and arrays of computed metric values as values.
        """
        result_dict = {}
        for model_label, data_list in model_data.items():
            result_dict[model_label] = []
            for data in tqdm(data_list, desc=f"Processing {model_label}"):
                result_dict[model_label].append(metric_class.compute(data))

            if isinstance(metric_class, DecodeModel):
                result_dict[model_label] = np.array(result_dict[model_label])

        return result_dict

    def evaluate_decoding(
        models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
    ) -> Dict[str, npt.NDArray]:
        """
        Computes decoding evaluation scores for CEBRA models.

        Parameters:
        -----------
        models : Dict[str, List[CEBRA]]
            Dictionary where the keys are model labels and values are lists of CEBRA models.

        train_data : torch.Tensor
            Input tensor for training, shape (samples, features).

        train_label : npt.NDArray
            Ground truth labels for training data.

        test_data : torch.Tensor
            Input tensor for testing, shape (samples, features).

        test_label : npt.NDArray
            Ground truth labels for test data.

        session_id : int, optional
            Session identifier used for multi-session models (default is -1).

        dataset_label : str, optional
            Label for the dataset being evaluated (default is "visual").

        Returns:
        --------
        Dict[str, npt.NDArray]
            Dictionary where keys are model labels and values are decoding score arrays.
        """
        decoding_metric = DecodeModel(
            train_data, train_label, test_data, test_label, session_id, dataset_label
        )
        return CEBRALens.compute(models, decoding_metric)

    def evaluate_decoding_per_layer(
        models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
        layer_type: Optional[Type[nn.Module]] = None,
    ) -> Dict[str, npt.NDArray]:
        """
        Computes decoding evaluation scores for each layer of CEBRA models.

        Parameters:
        -----------
        models : Dict[str, List[CEBRA]]
            Dictionary where the keys are model labels and values are lists of CEBRA models.

        train_data : torch.Tensor
            Input tensor for training, shape (samples, features).

        train_label : npt.NDArray
            Ground truth labels for training data.

        test_data : torch.Tensor
            Input tensor for testing, shape (samples, features).

        test_label : npt.NDArray
            Ground truth labels for test data.

        session_id : int, optional
            Session identifier used for multi-session models (default is -1).

        dataset_label : str, optional
            Label for the dataset being evaluated (default is "visual").

        layer_type : Type[nn.Module], optional
            If provided, specifies the type of layer (e.g. nn.Conv1d) from which to extract activations.

        Returns:
        --------
        Dict[str, npt.NDArray]
            Dictionary where keys are model labels and values are arrays of decoding scores per layer.
        """
        decoding_metric = Decoding(
            train_data, train_label, test_data, test_label,
            session_id, dataset_label, layer_type
        )
        return CEBRALens.compute(models, decoding_metric)

    @staticmethod
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

    @staticmethod
    def get_activations(
        models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
        data: torch.Tensor,
        session_id: int,
        activations: Optional[Dict[str, npt.NDArray]] = None,
        layer_type: Optional[Type[nn.Module]] = None,
    ) -> Dict[str, npt.NDArray]:
        """
        Extracts and organizes activations from models.

        Parameters:
        -----------
        models : Dict[str, List[CEBRA]]
            Dictionary of models categorized by label.

        data : torch.Tensor
            Input tensor for the models, shape (samples, features).

        session_id : int
            Session identifier used for selecting the appropriate model.

        activations : Dict[str, npt.NDArray], optional
            Optional dictionary to store activations.

        layer_type : Type[nn.Module], optional
            Optional layer type (e.g., nn.Conv1d) to extract specific activations.

        Returns:
        --------
        Dict[str, npt.NDArray]
            Dictionary with model label prefixes as keys and lists of activation arrays as values.
        """
        activations = activations or {}


        aggregated_activations = aggregate_activations(
            process_activations(models, data, session_id, activations, layer_type)
        )

        activations_dict = {}
        for key, value in aggregated_activations.items():
            prefix = "_".join(key.split("_")[:-1])
            activations_dict.setdefault(prefix, []).append(value)

        return activations_dict
