import torch
import numpy as np
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from ..activations import get_activations_model


class Decoding:
    def __init__(self, train_data: torch.Tensor,
        train_label: np.ndarray,
        test_data: torch.Tensor,
        test_label: np.ndarray,
        session_id: int = -1,
        dataset_label: str = "visual"):

        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        self.session_id = session_id
        self.dataset_label = dataset_label

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

    def decode_by_layer(
        self,
        model,
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
            model=model,
            data=self.train_data,
            name=model.solver_name_,
            session_id=self.session_id,
            layer_type=layer_type,
        )

        activations_test = get_activations_model(
            model=model,
            data=self.test_data,
            name=model.solver_name_,
            session_id=self.session_id,
            layer_type=layer_type,
        )

        num_layers = len(activations_train)

        if self.dataset_label in ["HPC", "visual"]:
            results = np.zeros((num_layers + 1, 3))
        else:
            raise NotImplementedError(
                f"Decoding not implemented for {self.dataset_label}. Please use 'visual' or 'HPC'."
            )
        keys = list(activations_train.keys())
        for i in range(num_layers + 1):

            if i == 0:
                results[i, :] = self._decoding_function_selection(
                    self.train_data, self.train_label, self.test_data, self.test_label, self.dataset_label
                )  # neural input baseline
            else:
                results[i, :] = self._decoding_function_selection(
                    activations_train[keys[i - 1]],
                    self.train_label,
                    activations_test[keys[i - 1]],
                    self.test_label,
                    self.dataset_label,
                )  # layer decoding

        return results

    def decode(
        self,model

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

        if model.solver_name_ == "multi-session":

            embedding_train = model.transform(self.train_data, self.session_id)
            embedding_test = model.transform(self.test_data, self.session_id)

        elif model.solver_name_ == "single-session":

            embedding_train = model.transform(self.train_data)
            embedding_test = model.transform(self.test_data)

        else:
            raise NotImplementedError(
                f"Solver {model.solver_name_} is not yet implemented.")

        results = self._decoding_function_selection(
            embedding_train, self.train_label, embedding_test, self.test_label, self.dataset_label
        )
        return np.array(results)
