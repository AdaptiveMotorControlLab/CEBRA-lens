import cebra
import torch
import numpy as np
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from ..activations import get_activations_model
from .base import _BaseMetric
from ..matplotlib import *
import numpy.typing as npt
from typing import Dict, Type, Tuple
import torch.nn as nn
import sklearn.metrics
import torch as pt


def decoding(
    embedding_train: pt.tensor,
    embedding_test: pt.tensor,
    label_train: npt.NDArray,
    label_test: npt.NDArray,
) -> Tuple[np.float64, list, list]:
    """
    Function to decode the embeddings using KNNDecoder from CEBRA. The decoding scores are returned in the form of average R^2 score across all labels, R^2 scores per label and error per label.

    Parameters:
    ----------
    embedding_train : pt.tensor
        The part of the output embedding to use as training for the decoding.
    embedding_test : pt.tensor
        The part of the output embedding to use as testing for the decoding.
    label_train : npt.NDArray
        The true labels corresponding to the training data.
    label_test : npt.NDArray
        The true labels corresponding to the validation data.

    Returns:
    -------
    Tuple[np.float64, list, list]
        A tuple containing the overall test score (R^2), a list of median errors for each label, and a list of R^2 scores for each label.
    """
    try:
        num_labels = label_train.shape[1]
    except:
        num_labels = 1
        label_train = label_train.reshape(-1, 1)
        label_test = label_test.reshape(-1, 1)

    # resampling, subsampling and supervised model architecture is still not supported
    # checked via '''supported_model_architectures()''' function

    # for each label find another K
    predictions, labels_test_err, labels_test_score = [], [], []
    for i in range(num_labels):
        params = np.power(np.linspace(1, 10, 10, dtype=int), 2)
        errs = []
        for n in params:
            train_decoder = cebra.KNNDecoder(n_neighbors=n, metric="cosine")
            train_valid_idx = int(len(embedding_train) / 9 * 8)
            train_decoder.fit(
                embedding_train[:train_valid_idx], label_train[:train_valid_idx, i]
            )
            pred = train_decoder.predict(embedding_train[train_valid_idx:])
            err = label_train[train_valid_idx:, i] - pred
            errs.append(abs(err).sum())

        test_decoder = cebra.KNNDecoder(
            n_neighbors=params[np.argmin(errs)], metric="cosine"
        )

        test_decoder.fit(embedding_train, label_train[:, i])
        label_pred = test_decoder.predict(embedding_test)

        predictions.append(label_pred)
        label_test_err = np.median(abs(label_pred - label_test[:, i]))
        labels_test_err.append(label_test_err)
        label_test_score = sklearn.metrics.r2_score(label_test[:, i], label_pred)
        labels_test_score.append(label_test_score)

    # transform it into an appropriate shape
    predictions = np.stack(np.array(predictions), axis=1)
    # difference between classification error and regression error -> here we are only taking into account regression style labels

    test_score = sklearn.metrics.r2_score(label_test, predictions)

    # always plot the test_score in R2 for overall labels, if wanted you can choose a label and plot its error, but I need to add a parameter

    return test_score, labels_test_err, labels_test_score


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
        The type of dataset being used for decoding.
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
        dataset_label: str = None,
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
        dataset_label: str = None,
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
            The type of dataset being used for decoding.

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
            results = decoding(
                embedding_train=embedding_train,
                label_train=label_train,
                embedding_test=embedding_test,
                label_test=label_test,
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
        transform_kwargs = {}
        if self.output_only:

            num_layers = 0

            if model.solver_name_ not in [
                "single-session",
                "single-session-aux",
                "single-session-hybrid",
                "single-session-full",
                "multi-session",
                "multi-session-aux",
                "multiobjective-solver",
            ]:
                raise NotImplementedError(
                    f"Solver {model.solver_name_} is not yet implemented."
                )
            elif model.solver_name_ in [
                "multi-session",
                "multi-session-aux",
                "multiobjective-solver",
            ]:
                transform_kwargs.update({"session_id": self.session_id})

            train_embedding = model.transform(self.train_data, **transform_kwargs)
            test_embedding = model.transform(self.test_data, **transform_kwargs)
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

        results = {}
        for i in range(num_layers + 1):

            # if output_only == True, then it will only do this loop and for train_data it will take in the embeddings
            if i == 0:
                if not self.output_only:
                    train_embedding = self.train_data
                    test_embedding = self.test_data

                results.update(
                    {
                        i: self._decode(
                            train_embedding,
                            self.train_label,
                            test_embedding,
                            self.test_label,
                            self.dataset_label,
                        )
                    }
                )

            else:

                results.update(
                    {
                        i: self._decode(
                            activations_train[keys[i - 1]],
                            self.train_label,
                            activations_test[keys[i - 1]],
                            self.test_label,
                            self.dataset_label,
                        )
                    }
                )

        return results

    @property
    def __name__(self):
        return "decode_by_layer"

    def set_output_only(self, output_only):
        self.output_only = output_only

    def plot(
        self,
        results_dict: Dict[str, Dict[int, Tuple[np.float64, list, list]]],
        title: str = "Decoding by layer",
        label: int = None,
        figsize: tuple = (15, 5),
        palette: str = "hls",
        plot_error: bool = False,
        ax: Optional[matplotlib.axes.Axes] = None,
    )-> matplotlib.axes.Axes:
        """
        Plot the decoding score of the output embeddings or the decoding scores of the activations across layers of models.If set to output_only=True, it will plot the decoding scores of the output embeddings, otherwise it will plot the decoding scores of the activations across layers.

        Parameters:
        ----------
        results_dict : Dict[str, Dict[int, Tuple[np.float64, list, list]]]
            Dictionary containing the decoding results for each model and layer.
        title : str, optional
            The title of the plot. Default is "Decoding by layer".
        label : int, optional
            The label to plot. This is relevant only if the dataset label is not specified.
        figsize : tuple, optional
            The size of the figure to plot. Default is (15, 5).
        palette : str, optional
            The color palette to use for the plot. Default is "hls".
        plot_error : bool, optional
            Whether to plot the error score. Default is False, meaning the R^2 score will be plotted. This is relevant only if the dataset label is not specified.
        ax : Optional[matplotlib.axes.Axes], optional
            The axes to plot on. If None, a new figure and axes will be created. Default is None.

        Returns:
        -------
        matplotlib.axes.Axes
            The axes containing the plot. If ax is provided, it will return the same ax with the plot, otherwise it will create a new figure and return the ax.
        """

        if self.dataset_label is None:
            if label is None:
                raise ValueError(
                    "If dataset_label is not specified, label must be provided to plot the decoding scores for specified label."
                )
   
        if self.output_only:
            return plot_decoding(
                results_dict, palette, self.dataset_label, label, plot_error, ax
            )
        else:
            return plot_layer_decoding(
                results_dict, title, self.dataset_label, label, plot_error, figsize
            )
