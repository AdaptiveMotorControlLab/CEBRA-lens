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
        self,
        model_data: Dict[str, List[npt.NDArray]],
        metric_class: object,
    ) -> Dict[str, npt.NDArray]:
        """
        Computes metrics for each model using a provided metric class.
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
        self,
        models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
    ) -> Dict[str, npt.NDArray]:
        """
        Evaluate decoding across models using DecodeModel.
        """
        decoding_metric = DecodeModel(
            train_data, train_label, test_data, test_label, session_id, dataset_label
        )
        return self.compute(models, decoding_metric)

    def evaluate_decoding_per_layer(
        self,
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
        Evaluate decoding scores per layer using Decoding.
        """
        decoding_metric = Decoding(
            train_data, train_label, test_data, test_label,
            session_id, dataset_label, layer_type
        )
        return self.compute(models, decoding_metric)

    @staticmethod
    def load(
        model_dir: str, labels: Dict[str, str] = {}
    ) -> Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]:
        """
        Load and categorize CEBRA models from directory.
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
        Extract and organize activations from given models.
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
