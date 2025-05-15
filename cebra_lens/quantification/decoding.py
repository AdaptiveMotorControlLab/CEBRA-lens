import cebra
import torch
import numpy as np
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from ..activations import get_activations_model
from .base import _BaseMetric
from ..matplotlib import *
import numpy.typing as npt
from typing import Dict, Type
import torch.nn as nn


class Decoding(_BaseMetric):
    """
    Decoding class for decoding neural data by layer using a given CEBRA model.

    Parameter:
    ----------
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
    output_only: bool
        A bool which defines whether to calculation decoding scores for the activations layers of a model, or for the embeddings of the model. Default: True.
    """

    def __init__(
        self,
        train_data: torch.Tensor,
        train_label: npt.NDArray,
        test_data: torch.Tensor,
        test_label: npt.NDArray,
        session_id: int = -1,
        dataset_label: str = "visual",
        layer_type: Optional[Type[nn.Module]] = None,
        output_only: bool = True,
    ):

        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        self.session_id = session_id
        self.dataset_label = dataset_label
        self.layer_type = layer_type
        self.output_only = output_only

    def _decode(
        self,
        embedding_train: npt.NDArray,
        label_train: npt.NDArray,
        embedding_test: npt.NDArray,
        label_test: npt.NDArray,
        dataset_label: str = "visual",
    ) -> npt.NDArray:
        """
        Decodes a model by choosing the appropriate function base on the dataset.
        Currently compatible with multi-session and single-session data only.

        Parameters:
        -----------
        embedding_train : npt.NDArray
            The part of the output embedding to use as training for the decoding.
        train_label : npt.NDArray
            The true labels corresponding to the training data.
        embedding_test : npt.NDArray
            The part of the output embedding to use as testing for the decoding.
        test_label : npt.NDArray
            The true labels corresponding to the validation data.
        dataset_label : str, optional
            The type of dataset being used for decoding (default is "visual").

        Returns:
        --------
        npt.NDArray
            Array containing the decoding results based on the given embeddings and labels. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset, or test_score, test_err, test_acc for Allen visual dataset.

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
        model: cebra.integrations.sklearn.cebra.CEBRA,
    ) -> npt.NDArray:
        """
        Decode neural data by layer using a given CEBRA model.

        Parameters:
        ----------
        model : cebra.integrations.sklearn.cebra.CEBRA
            The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).

        Returns:
        -------
        npt.NDArray
            A numpy array containing the decoding results for each layer and the neural input baseline.
        """


        if self.output_only:

            num_layers = 0

            if model.solver_name_ == "multi-session":

                train_embedding = model.transform(self.train_data, self.session_id)
                test_embedding = model.transform(self.test_data, self.session_id)

            elif model.solver_name_ == "single-session":

                train_embedding = model.transform(self.train_data)
                test_embedding = model.transform(self.test_data)

            else:
                raise NotImplementedError(
                    f"Solver {model.solver_name_} is not yet implemented."
                )
        else:

            activations_train = get_activations_model(
                model=model,
                data=self.train_data,
                name=model.solver_name_,
                session_id=self.session_id,
                layer_type=self.layer_type,
            )

            activations_test = get_activations_model(
                model=model,
                data=self.test_data,
                name=model.solver_name_,
                session_id=self.session_id,
                layer_type=self.layer_type,
            )
            num_layers = len(activations_train)
            keys = list(activations_train.keys())

        if self.dataset_label in ["HPC", "visual"]:
            results = np.zeros((num_layers + 1, 3))
        else:
            raise NotImplementedError(
                f"Decoding not implemented for {self.dataset_label}. Please use 'visual' or 'HPC'."
            )

        for i in range(num_layers + 1):

            # if output_only == True, then it will only do this loop and for train_data it will take in the embeddings
            if i == 0:
                if not self.output_only:
                    train_embedding = self.train_data
                    test_embedding = self.test_data

                results[i, :] = self._decode(
                    train_embedding,
                    self.train_label,
                    test_embedding,
                    self.test_label,
                    self.dataset_label,
                )

            else:

                results[i, :] = self._decode(
                    activations_train[keys[i - 1]],
                    self.train_label,
                    activations_test[keys[i - 1]],
                    self.test_label,
                    self.dataset_label,
                )
        if self.output_only:
            results = results[0]
        return results

    @property
    def __name__(self):
        return "decode_by_layer"
    
    def set_output_only(self, output_only):
        self.output_only = output_only


    def plot(
        self,
        results_dict: Dict[str, npt.NDArray],
        title: str = "Decoding by layer",
        figsize: tuple = (15, 5),
        palette: str = "hls",
        dataset_label="visual",
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        if self.output_only:
            return plot_decoding(results_dict, palette, dataset_label, ax)
        else:
            return plot_layer_decoding(results_dict, title, figsize)
