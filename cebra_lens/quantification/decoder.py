from typing import Dict, Optional, Tuple, Type

import cebra
import matplotlib
import numpy as np
import numpy.typing as npt
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, mean_absolute_error, r2_score
from sklearn.utils.multiclass import type_of_target
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
    """    
    Decode embeddings with per-label automated KNNDecoder hyperparameter search.
    Returns overall score (R² or accuracy), per-label error, and per-label score.

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
        
    label_types = [type_of_target(train_label[:, i]) for i in range(num_labels)]
    is_reg = [t == "continuous" for t in label_types]
    # for each label find another K
    param_grid = {"n_neighbors": np.arange(1, 11) ** 2}
    
    predictions, test_labels_err, test_labels_score = [], [], []
    for i in range(num_labels):
        y_train_i = train_label[:, i]
        y_test_i  = test_label[:, i]

        if not is_reg[i]:
            # force binary → integer so CEBRA picks classifier
            y_train_i = y_train_i.astype(np.int64)
            y_test_i  = y_test_i.astype(np.int64)

        # Choose scorer based on continuous vs. classification
        scorer = make_scorer(r2_score) if is_reg[i] else make_scorer(accuracy_score)

        gs = GridSearchCV(
            cebra.KNNDecoder(metric="cosine"),
            param_grid=param_grid,
            scoring=scorer,
            cv=2,
        )

        gs.fit(embedding_train, y_train_i)
        pred_label = gs.best_estimator_.predict(embedding_test)  

        predictions.append(pred_label)
        print("coucou", pred_label.shape, test_label[:, i].shape,
              embedding_test.shape)
        if is_reg[i]:
            test_labels_err.append(mean_absolute_error(y_test_i, pred_label))
            test_labels_score.append(r2_score(y_test_i, pred_label))
        else:
            acc = accuracy_score(y_test_i, pred_label)
            test_labels_err.append(1.0 - acc)
            test_labels_score.append(acc)  

    predictions = np.stack(np.array(predictions), axis=1)

    if all(is_reg):
        test_score = r2_score(test_label, predictions)
    elif not any(is_reg):
        test_score = accuracy_score(test_label.ravel(), predictions.ravel())
    else:
        test_score = None # for now, because not useful now

    # NOTE(eloise): For now, we always plot the test_score in R2 for overall labels
    return test_score, test_labels_err, test_labels_score


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
            model : cebra.intfegrations.sklearn.cebra.CEBRA
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
            if output_only:
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
                if i == 0 and not isinstance(model,
                                             cebra.solver.UnifiedSolver):
                    results.update({
                        i:
                        self._decode(
                            self.train_data,
                            train_decoding_labels,
                            self.test_data,
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
        label_is_binary: bool = False,
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
        return utils_plot.plot_layer_decoding(
            results_dict, title, self.dataset_label, label, plot_error, label_is_binary, figsize
        )
           
