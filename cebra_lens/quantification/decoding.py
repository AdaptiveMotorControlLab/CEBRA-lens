import torch
import numpy as np
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from ..activations import get_activations_model
from .base import _BaseMetric, _MultiMetric


class MultiDecoding(_MultiMetric):
    def __init__(self, models_dict: dict):
        self.models_dict = models_dict
        self.base = Decoding
        self.data = super().transform(self.models_dict, self.base)

    def compute(
        self,
        train_data: torch.Tensor,
        train_label: np.ndarray,
        test_data: torch.Tensor,
        test_label: np.ndarray,
        session_id: int,
        dataset_label: str = "visual",
        layer_type: str = "conv",
    ):
        """Equivalent to the method decode_layer_models"""
        return super().compute(
            self.data,
            train_data,
            train_label,
            test_data,
            test_label,
            session_id,
            dataset_label,
            layer_type,
        )

    def decode(
        self,
        train_data: torch.Tensor,
        train_label: torch.Tensor,
        test_data: torch.Tensor,
        test_label: torch.Tensor,
        session_id: int = -1,
        dataset_label: str = "visual",
    ) -> dict:
        """
        Decodes multiple models and stores their results in a dictionary.

        Parameters:
        -----------
        models : dict
            A dictionary where keys are model category labels or model file names and values are lists of model objects to be decoded.
        train_data : torch.Tensor
            The training data used for model transformation.
        train_label : torch.Tensor
            The true labels corresponding to the training data.
        test_data : torch.Tensor
            The test data used for model transformation.
        test_label : torch.Tensor
            The true labels corresponding to the test data.
        session_id : int, optional
            The session ID for multi-session models (default is -1 for single-session models).
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").

        Returns:
        --------
        results_dict : dict
            A dictionary where the keys are the model category labels or model names, and the values are the corresponding decoding results.
        """

        return super().decode(
            self.data,
            train_data,
            train_label,
            test_data,
            test_label,
            session_id,
            dataset_label,
        )


class Decoding(_BaseMetric):
    def __init__(self, model):
        self.model = model

    def _decoding_function_selection(
        # figure out what to do about the arguments and parameters
        self,
        embedding_train: np.ndarray,
        label_train: np.ndarray,
        embedding_test: np.ndarray,
        label_test: np.ndarray,
        dataset_label: str = "visual",
    ):
        """
        Decodes a model by choosing the appropriate function.

        Parameters:
        -----------
        embedding_train : np.ndarray
            The part of the output embedding to use as training for the decoding.
        train_label : np.ndarray
            The true labels corresponding to the training data.
        embedding_test : np.ndarray
            The part of the output embedding to use as testing for the decoding.
        test_label : np.ndarray
            The true labels corresponding to the validation data.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").

        Returns:
        --------
        np.ndarray : Array containing the results. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset.
        """
        if (
            embedding_train.shape[0] < embedding_train.shape[1]
        ):  # should be samples X neurons
            embedding_train = embedding_train.T
        if (
            embedding_test.shape[0] < embedding_test.shape[1]
        ):  # should be samples X neurons
            embedding_test = embedding_test.T

        if dataset_label == "visual":

            results = decoding_frames(
                embedding_train=embedding_train,
                label_train=label_train,
                embedding_test=embedding_test,
                label_test=label_test,
            )
        elif dataset_label == "HPC":
            results = decoding_pos_dir(
                embedding_train=embedding_train,
                label_train=label_train,
                embedding_test=embedding_test,
                label_test=label_test,
            )
        else:
            raise NotImplementedError(
                f"Decoding not implemented for {dataset_label}. Please use 'visual' or 'HPC'."
            )
        return results

    def compute(
        self,
        train_data: torch.Tensor,
        train_label: np.ndarray,
        test_data: torch.Tensor,
        test_label: np.ndarray,
        session_id: int,
        dataset_label: str = "visual",
        layer_type: str = "conv",
    ):
        """
        Decode neural data by layer using a given CEBRA model.

        Parameters:
        ----------
        model : cebra.integrations.sklearn.cebra.CEBRA
            The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
        train_data : torch.Tensor
            The training data used for model transformation.
        train_label : np.ndarray
            The true labels corresponding to the training data.
        test_data : torch.Tensor
            The validation data used for testing the model.
        test_label : np.ndarray
            The true labels corresponding to the validation data.
        session_id : int, optional
            The session ID for multi-session models. For single-session no need to input it.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").
        layer_type : str, optional
            The type of layer to extract activations from. Defaults to 'conv'.

        Returns:
        -------
        np.ndarray
            A numpy array containing the decoding results for each layer and the neural input baseline.
        """

        activations_train = get_activations_model(
            model=self.model,
            data=train_data,
            name=self.model.solver_name_,
            session_id=session_id,
            layer_type=layer_type,
        )

        activations_test = get_activations_model(
            model=self.model,
            data=test_data,
            name=self.model.solver_name_,
            session_id=session_id,
            layer_type=layer_type,
        )

        num_layers = len(activations_train)

        if dataset_label in ["HPC", "visual"]:
            results = np.zeros((num_layers + 1, 3))
        else:
            raise NotImplementedError(
                f"Decoding not implemented for {dataset_label}. Please use 'visual' or 'HPC'."
            )
        keys = list(activations_train.keys())
        for i in range(num_layers + 1):

            if i == 0:
                results[i, :] = (
                    self._decoding_function_selection()
                )  # neural input baseline
            else:
                results[i, :] = self._decoding_function_selection(
                    activations_train[keys[i - 1]],
                    train_label,
                    activations_test[keys[i - 1]],
                    test_label,
                    dataset_label,
                )  # layer decoding

        return results

    def decode(
        self,
        train_data: torch.Tensor,
        train_label: np.ndarray,
        test_data: torch.Tensor,
        test_label: np.ndarray,
        session_id: int = -1,
        dataset_label: str = "visual",
    ) -> np.ndarray:
        """
        Decodes a single model.

        Parameters:
        -----------
        model : cebra.integrations.sklearn.cebra.CEBRA
            The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
        train_data : torch.Tensor
            The training data used for model transformation.
        train_label : np.ndarray
            The true labels corresponding to the training data.
        test_data : torch.Tensor
            The validation data used for testing the model.
        test_label : np.ndarray
            The true labels corresponding to the validation data.
        session_id : int, optional
            The session ID for multi-session models. For single-session no need to input it.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").

        Returns:
        --------
        np.ndarray : Array containing the results. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset.
        """

        if self.model.solver_name_ == "multi-session":

            embedding_train = self.model.transform(train_data, session_id)
            embedding_test = self.model.transform(test_data, session_id)

        elif self.model.solver_name_ == "single-session":

            embedding_train = self.model.transform(train_data)
            embedding_test = self.model.transform(test_data)

        else:
            raise NotImplementedError(
                f"Solver {self.model.solver_name_} is not yet implemented."
            )

        results = self._decoding_function_selection(
            embedding_train, train_label, embedding_test, test_label, dataset_label
        )
        return np.array(results)
