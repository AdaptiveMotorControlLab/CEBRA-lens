from typing import Dict, Optional, Tuple, Type

import cebra
import matplotlib
import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
import torch as pt
import torch.nn as nn

from cebra_lens import utils_plot, utils_wrapper

from ..activations import get_activations_model
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from .base import _BaseMetric

#NOTE(eloise): resampling, subsampling and supervised model architecture is still not supported
#              checked via '''supported_model_architectures()''' function.


def decoding(
    embedding_train: pt.tensor,
    embedding_test: pt.tensor,
    train_label: npt.NDArray,
    test_label: npt.NDArray,
) -> Tuple[np.float64, list, list]:
    """Function to decode the embeddings using KNNDecoder from CEBRA. 
    
    The decoding scores are returned in the form of average $R^2$ score 
    across all labels, $R^2$ scores per label and error per label.

    Args:
        embedding_train : pt.tensor
            The part of the output embedding to use as training for the decoding.
        embedding_test : pt.tensor
            The part of the output embedding to use as testing for the decoding.
        train_label : npt.NDArray
            The true labels corresponding to the training data.
        test_label : npt.NDArray
            The true labels corresponding to the validation data.

    Returns:
        Tuple[np.float64, list, list]
            A tuple containing the overall test score (R^2), a list of median errors for each label, and a list of R^2 scores for each label.
    """
    if train_label.shape[1] > 1:
        num_labels = train_label.shape[1]
    else:
        num_labels = 1
        train_label = train_label.reshape(-1, 1)
        test_label = test_label.reshape(-1, 1)

    # for each label find another K
    predictions, labels_test_err, labels_test_score = [], [], []
    for i in range(num_labels):
        params = np.power(np.linspace(1, 10, 10, dtype=int), 2)
        errs = []
        for n in params:
            train_decoder = cebra.KNNDecoder(n_neighbors=n, metric="cosine")
            train_valid_idx = int(len(embedding_train) / 9 *
                                  8)  # NOTE(celia): for now 8/9 arbitrarily
            train_decoder.fit(embedding_train[:train_valid_idx],
                              train_label[:train_valid_idx, i])
            pred = train_decoder.predict(embedding_train[train_valid_idx:])
            err = train_label[train_valid_idx:, i] - pred
            errs.append(abs(err).sum())

        best_decoder = cebra.KNNDecoder(n_neighbors=params[np.argmin(errs)],
                                        metric="cosine")

        best_decoder.fit(embedding_train, train_label[:, i])
        label_pred = best_decoder.predict(embedding_test)

        predictions.append(label_pred)
        print("coucou", label_pred.shape, test_label[:, i].shape,
              embedding_test.shape)
        label_test_err = np.median(abs(label_pred - test_label[:, i]))
        labels_test_err.append(label_test_err)
        label_test_score = sklearn.metrics.r2_score(test_label[:, i],
                                                    label_pred)
        labels_test_score.append(label_test_score)

    predictions = np.stack(np.array(predictions), axis=1)

    # NOTE(eloise): Here we are only taking into account regression style labels
    # not classification
    test_score = sklearn.metrics.r2_score(test_label, predictions)

    # NOTE(eloise): For now, we always plot the test_score in R2 for overall labels
    return test_score, labels_test_err, labels_test_score


class Decoding(_BaseMetric):
    """
    Decoding class for decoding neural data by layer using a given CEBRA model.

    Attributes:
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
        session_id: int = 0,
        dataset_label: str = None,
        layer_type: Optional[Type[nn.Module]] = nn.Conv1d,
    ):

        self.train_label = train_label
        self.train_data = train_data
        self.test_label = test_label
        self.test_data = test_data
        self.session_id = session_id
        self.dataset_label = dataset_label
        self.layer_type = layer_type

    def output_information(self):
        print(
            "The decoding analysis initialized with the following parameters:")
        print(f"Session ID: {self.session_id}")
        print(f"Dataset label: {self.dataset_label}")
        print(f"Layer type: {self.layer_type}")
        # print(f"Output only: {self.output_only}")
        # if self.output_only:
        #    print(
        #        "The decoding analysis will only compute the decoding scores for the output embeddings of the model and plot them."
        #    )
        # else:
        #     print(
        #         "The decoding analysis will compute the decoding scores for the activations of the model and plot them across layers."
        #     )
        print(
            "If you want to change the parameters, please re-initialize the class with the new parameters ",
            "or if you want to change the output_only parameter, call the compute_metric function with the ",
            "output_only parameter set to True or False.")

    def _decode(
        self,
        embedding_train: npt.NDArray,
        train_label: npt.NDArray,
        embedding_test: npt.NDArray,
        test_label: npt.NDArray,
        dataset_label: str = None,
    ) -> npt.NDArray:
        """Decode a model by choosing the appropriate function base on the dataset.
        
        Note: 
            Currently compatible with multi-session and single-session data only.

        Args: 
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
            npt.NDArray
                Array containing the decoding results based on the given embeddings and labels. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset, or test_score, test_err, test_acc for Allen visual dataset.

        """
        if (embedding_train.shape[0]
                < embedding_train.shape[1]):  # should be samples X neurons
            embedding_train = embedding_train.T
        if (embedding_test.shape[0]
                < embedding_test.shape[1]):  # should be samples X neurons
            embedding_test = embedding_test.T

        if dataset_label == "visual":
            results = decoding_frames(
                embedding_train=embedding_train,
                train_label=train_label,
                embedding_test=embedding_test,
                test_label=test_label,
            )
        elif dataset_label == "HPC":
            results = decoding_pos_dir(
                embedding_train=embedding_train,
                embedding_test=embedding_test,
                train_label=train_label,
                test_label=test_label,
            )
        else:
            results = decoding(
                embedding_train=embedding_train,
                embedding_test=embedding_test,
                train_label=train_label,
                test_label=test_label,
            )
        return results

    def compute(
        self,
        model: cebra.integrations.sklearn.cebra.CEBRA,
        output_only: bool = True,
    ) -> npt.NDArray:
        """Decode neural data by layer using a given CEBRA model.

        Args:
            model : cebra.integrations.sklearn.cebra.CEBRA
                The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
            output_only: bool
                A bool which defines whether to calculation decoding scores for the activations layers of a model, or for the
                embeddings of the model. Default: True.

        Returns:
            npt.NDArray
                A numpy array containing the decoding results for each layer and the neural input baseline.
        """
        transform_kwargs = {}
        if output_only:
            num_layers = 0
            transform_kwargs.update({"session_id": self.session_id})

            train_embedding = utils_wrapper.transform(
                model=model,
                data=self.train_data,
                label=self.train_label,
                **transform_kwargs,
            )

            test_embedding = utils_wrapper.transform(
                model=model,
                data=self.test_data,
                label=self.test_label,
                **transform_kwargs,
            )
        else:
            activations_train = get_activations_model(
                model=model,
                data=self.train_data,
                labels=self.train_label,
                activations_keys_prefix=model.solver_.__class__.__name__
                if hasattr(model, 'solver_') else model.__class__.__name__,
                session_id=self.session_id,
                layer_type=self.layer_type,
            )

            activations_test = get_activations_model(
                model=model,
                data=self.test_data,
                labels=self.test_label,
                activations_keys_prefix=model.solver_.__class__.__name__
                if hasattr(model, 'solver_') else model.__class__.__name__,
                session_id=self.session_id,
                layer_type=self.layer_type,
            )
            num_layers = len(activations_train)
            keys = list(activations_train.keys())

        # Get the decoding labels
        if isinstance(model, cebra.solver.UnifiedSolver):
            train_decoding_labels = self.train_label[self.session_id]
            test_decoding_labels = self.test_label[self.session_id]
        else:
            train_decoding_labels = self.train_label
            test_decoding_labels = self.test_label

        results = {}
        for i in range(num_layers + 1):

            #NOTE(eloise): if output_only is True, then it will only do this iteration
            # of the for loop and for train_data it will take in the embeddings.
            #NOTE(celia): for now we skip the first layer if the model is a UnifiedSolver
            if i == 0 and not isinstance(model, cebra.solver.UnifiedSolver):
                if not output_only:
                    train_embedding = self.train_data
                    test_embedding = self.test_data

                results.update({
                    i:
                    self._decode(
                        train_embedding,
                        train_decoding_labels,
                        test_embedding,
                        test_decoding_labels,
                        self.dataset_label,
                    )
                })

            else:
                results.update({
                    i:
                    self._decode(
                        activations_train[keys[i - 1]],
                        train_decoding_labels,
                        activations_test[keys[i - 1]],
                        test_decoding_labels,
                        self.dataset_label,
                    )
                })

        return results

    @property
    def __name__(self):
        return "decode_by_layer"

    # TODO(celia): check that doesn't break anything to remove it
    # def set_output_only(self, output_only: bool) -> None:
    #     """
    #     Set the output_only parameter to True or False. If True, it will compute the decoding scores for the output embeddings of the model, otherwise it will compute the decoding scores for the activations of the model.

    #     Parameters:
    #     ----------
    #     output_only : bool
    #         If True, it will compute the decoding scores for the output embeddings of the model, otherwise it will compute the decoding scores for the activations of the model.
    #     """
    #     self.output_only = output_only

    def plot(
        self,
        results_dict: Dict[str, Dict[int, Tuple[np.float64, list, list]]],
        title: str = "Decoding by layer",
        label: int = None,
        figsize: tuple = (15, 5),
        palette: str = "hls",
        plot_error: bool = False,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.axes.Axes:
        """Plot the decoding score of the output embeddings or the decoding scores of the activations across layers of models.If set to output_only=True, it will plot the decoding scores of the output embeddings, otherwise it will plot the decoding scores of the activations across layers.

        Args:
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
            matplotlib.axes.Axes
                The axes containing the plot. If ax is provided, it will return the same ax with the plot, otherwise it will create a new figure and return the ax.
        """

        if self.dataset_label is None:
            if label is None:
                raise ValueError(
                    "If dataset_label is not specified, label must be provided to plot ",
                    "the decoding scores for specified label.",
                )

        if len(results_dict) == 1:
            return utils_plot.plot_decoding(results_dict, palette,
                                            self.dataset_label, label,
                                            plot_error, ax)
        else:
            return utils_plot.plot_layer_decoding(results_dict, title,
                                                  self.dataset_label, label,
                                                  plot_error, figsize)
