"Model handling. For now only loading is used."

import pathlib
import cebra
import torch
from typing import Dict, List, Union
from cebra import CEBRA
import numpy.typing as npt
from .activations import *
from tqdm import tqdm
from .quantification.decoding import DecodeModel, Decoding


class CEBRALens(CEBRA):
    """
    Compute the same metric across multiple models.
    Contains essential CEBRA-Lens functions to load CEBRA models,
    to get activations of layers and to get evaluation scores for the decoding of models.

    Parameters:
    ----------
    metric_class : object
        The metric class to be used for computation.
    """

    def __init__(self, metric_class: object):
        self.metric_class = metric_class
        self.results_dict = {}

    def compute(
        self,
        model_data: Union[
            Dict[str, npt.NDArray], List[cebra.integrations.sklearn.cebra.CEBRA]
        ],
    ) -> Dict[str, npt.NDArray]:
        """
        Computes the metric based on metric_class given for each model and stores the results in a dictionary.

        Parameters:
        -----------
        data : Union[Dict[str, npt.NDArray], List[cebra.integrations.sklearn.cebra.CEBRA]]
            A dictionary where keys are strings which represent the model label and values are 2d lists with the corresponding activations per layer or a list of CEBRA models to decode.

        Returns:
        --------
        Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are 2d lists with the calculated metric per layer or decoding scores of models.

        """
        self.result_dict = {}
        for model_label, data_list in model_data.items():
            self.result_dict[model_label] = []
            for data in tqdm(data_list, desc=f"Processing {model_label}"):
                self.result_dict[model_label].append(self.metric_class.compute(data))
            if isinstance(self.metric_class, DecodeModel):
                self.result_dict[model_label] = np.array(self.result_dict[model_label])
        return self.result_dict

    def plot(self, *args, **kwargs):
        """
        Plots the results of the metric computation.
        """
        self.metric_class.plot(*args, **kwargs)

    def load(
        model_dir: str, labels: Dict = {}
    ) -> Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]:
        """
        Load and categorize models based on their training status and session type.
        Parameters:
        -----------
        model_dir : str
            The path of the models: e.g. FinalModels/VISION

        labels : Dict, optional
            A dictionary containing the labels for the models. The keys should be the model file names and the values should be the model category labels. Default is an empty dictionary:
        e.g. {'model1': 'single_UT',
            'model2': 'single_UT',
            'model3': 'multi_UT',
            'model4': 'multi_UT',
            'model5': 'single_TR',
            'model6': 'single_TR'}
        Returns:
            Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]: A dictionary containing the loaded models (label, model) where label is taken from the input dictionary given by user or if not given, the model file name.
        """

        # LOAD MODELS

        models_folder_path = pathlib.Path(model_dir)
        if not pathlib.Path.exists(models_folder_path):
            raise FileNotFoundError(f"Folder {models_folder_path} not found.")
        models = {}
        for file in pathlib.Path.iterdir(models_folder_path):
            if str(file).endswith((".pt", ".pth")):
                model_path = models_folder_path / file
                loaded_model = cebra.CEBRA.load(
                    model_path,
                    backend="torch",
                    map_location=torch.device("cpu"),
                ).to("cpu")
                key = labels.get(file.stem, None)
                if key is None:
                    models[file.stem] = [loaded_model]
                else:
                    if key not in models:
                        models[key] = [loaded_model]
                    else:
                        models[key].append(loaded_model)
                print(f"Model {file.stem} loaded succesfully.")

        return models

    def get_activations(
        models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
        data: torch.Tensor,
        session_id: int,
        activations: Dict[str, npt.NDArray] = {},
        layer_type: Type[nn.Module] = None,
    ) -> Dict[str, npt.NDArray]:
        """
        Retrieves the activations and formats them into a structured dictionary.

        Parameters:
        -----------
        models : Dict[str, List[ cebra.integrations.sklearn.cebra.CEBRA]]
            A dictionary containing different sets of models.
        data : torch.Tensor
            The input data for which activations are to be extracted. Shape of samples X channels (neurons).
        session_id : int
            The session identifier used for selecting the appropriate model in multi-session solvers.
        activations : Dict[str, npt.NDArray]
            A dictionary to store the activations. If passed as an argument, the new keys will be concatenated to the existing dictionary.
        layer_type : Type[nn.Module]
            The type of layer from which to extract activations (e.g., nn.Conv1d).

        Returns:
        --------
        activations_dict : Dict[str, npt.NDArray]
            A dictionary where the keys are the model category names, and the values are arrays of activation values for each instance:
            e.g.{'single_UT': [[instance1_activations], [instance2_activations], ...], 'single_TR': [[instance1_activations], [instance2_activations], ...]}'
        """
        aggregated_activations = aggregate_activations(
            activations=process_activations(
                models, data, session_id, activations, layer_type
            )
        )

        activations_dict = {}

        for key, value in aggregated_activations.items():
            prefix = "_".join(key.split("_")[:-1])
            if prefix not in activations_dict.keys():
                activations_dict[prefix] = []
            activations_dict[prefix].append(value)

        return activations_dict

    def evaluate_decoding(
        self,
        models: List[cebra.integrations.sklearn.cebra.CEBRA],
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
    ) -> Dict[str, npt.NDArray]:
        """
        Computes evaluation scores of decoding CEBRA models.

        Parameters:
        -----------
        models : List [cebra.integrations.sklearn.cebra.CEBRA]
            The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
        train_data : torch.Tensor
            The training data used for model transformation.
        train_label : npt.NDArray
            The true labels corresponding to the training data.
        test_data : torch.Tensor
            The validation data used for testing the model.
        test_label : npt.NDArray
            The true labels corresponding to the validation data.
        session_id : int, optional
            The session ID for multi-session models. For single-session no need to input it.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").
        Returns:
        --------
        Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are decoding scores of models.
        """

        decoding_class = DecodeModel(
            train_data,
            train_label,
            test_data,
            test_label,
            session_id,
            dataset_label,
        )

        self.metric_class = decoding_class

        scores = self.compute(models)

        return scores

    def evaluate_decoding_per_layer(
        self,
        models,
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
        layer_type: Type[nn.Module] = None,
    ):
        """
        Computes evaluation scores of decoding CEBRA models per layer.

        Parameters:
        -----------
        models : List [cebra.integrations.sklearn.cebra.CEBRA]
            The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
        train_data : torch.Tensor
            The training data used for model transformation.
        train_label : npt.NDArray
            The true labels corresponding to the training data.
        test_data : torch.Tensor
            The validation data used for testing the model.
        test_label : npt.NDArray
            The true labels corresponding to the validation data.
        session_id : int, optional
            The session ID for multi-session models. For single-session no need to input it.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").
        layer_type : Type[nn.Module]
            The type of layer to extract activations from. Defaults to None, meaning activations will be extracted from all layers.

        Returns:
        --------
        Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are decoding scores of models per layer.
        """

        decode_per_layer = Decoding(
            train_data,
            train_label,
            test_data,
            test_label,
            session_id,
            dataset_label,
            layer_type,
        )
        self.metric_class = decode_per_layer
        scores = self.compute(models)

        return scores
