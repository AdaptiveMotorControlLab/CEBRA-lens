"""All the functions relative to the Representation Dissimilarity Matrix (RDM) calculation"""

import numpy as np
from scipy.linalg import block_diag
from typing import List, Optional, Tuple, Union
from scipy.spatial.distance import correlation, pdist, squareform
from .misc import discrete_binning, continuous_binning
import torch
from .base import _BaseMetric
from ..matplotlib import *
import numpy.typing as npt


class RDM(_BaseMetric):
    """
    Compute the Representational Dissimilarity Matrix (RDM) for a given activation.

    Parameters:
    -----------
    data : torch.Tensor
        The data array of shape (num_samples, num_features).
    label : torch.Tensor
        The array of labels corresponding to the data.
    discrete : bool, optional
        Whether the labels are discrete or continuous. If None, it will be determined based on the dataset_label.
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
    metric : str, optional
        The distance metric to use for computing the RDMs. 'correlation' or 'euclidean' are supported.
    bool_oracle : bool, optional
        Whether to compute and compare with the Oracle RDM. Default is True.
    label_ind : int, optional
        The index of the label to use for the RDM calculation if there are multiple labels. If None, it will raise an error if the dataset is not HPC or visual.
    """

    def __init__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        discrete: bool = None,
        dataset_label: str = None,
        metric: str = "correlation",
        bool_oracle: bool = True,
        label_ind: int = None,
    ):
        super().__init__()
        self.data = data
        self.label = label
        self.label_ind = label_ind
        self.dataset_label = dataset_label
        # check that label is 1D if dataset_label is not HPC/visual, and the label_ind is not provided
        if (
            isinstance(self.label, np.ndarray)
            and self.label.ndim != 1
            and self.dataset_label not in ["HPC", "visual"]
        ):
            # if the dataset contains multiple labels check that if it is not HPC dataset the label_ind was given
            if self.label_ind != None:
                self.label = label[:, label_ind]

            else:
                raise KeyError(
                    "If dataset not HPC or visual and there are multiple possible labels, parameter label_ind must be provided to indicate which label will be used for the RDM calculation"
                )

        self.metric = metric
        self.bool_oracle = bool_oracle
        self.discrete = discrete
        self.idxs, self.num_bins = self._define_indices()

        self.output_information()

    def output_information(self):
        """
        Outputs information about the RDM class initialization parameters.
        """
        print("RDM class initialized with the following parameters:")
        if self.bool_oracle:
            print(
                "The chosen analyis will plot the correlation of the RDMs with the Oracle RDM."
            )
        else:
            print("The chosen analysis will plot the RDMs, no Oracle RDM comparison.")
        if self.dataset_label is None:
            print(
                f"The dataset label is not specified, the RDMs will be computed based on the label index {self.label_ind},\n this label has been noted DISCRETE = {self.discrete}."
            )
        else:
            print(f"The dataset label is specified as: {self.dataset_label}")
        print(
            "If this is not the desired behavior, please check the parameters passed to the RDM class."
        )

    def _define_indices(self) -> Tuple[npt.NDArray, Optional[int]]:
        """
        Defines the indices for the bins and repetitions based on the specified distance label.

        Returns:
        --------
        Tuple[npt.NDArray, Optional[int]]
            A tuple containing:
            - idxs: A 2D numpy array of shape (num_bins, num_samples) representing the indices of samples in each bin.
            - num_bins: The number of bins if applicable, otherwise None.
        """
        num_bins = None
        if self.dataset_label is not None:
            if self.dataset_label not in ["visual", "HPC"]:
                raise ValueError(
                    f"Dataset label {self.dataset_label} is not supported. Please use 'visual' or 'HPC' or None for general binning."
                )
            else:
                idxs, num_bins = continuous_binning(
                    data=self.data,
                    label=self.label,
                    dataset_label=self.dataset_label,
                    sample_mode="sub_sample",
                )
        else:

            if self.discrete is None:
                raise ValueError(
                    "The 'discrete' parameter must be specified.This parameter specifies whether the given label is discrete or continuous."
                )

            if self.discrete:
                # just detect the unique values and find the indices of the bins (each bin is a unique value)
                # dataset_label is None and discrete is True
                idxs = discrete_binning(
                    label=self.label,
                )
            else:
                # dataset_label is HPC or visual/ discrete is False (dataset_label is None)
                idxs, num_bins = continuous_binning(
                    data=self.data,
                    label=self.label,
                    dataset_label=self.dataset_label,
                    sample_mode="sub_sample",
                )

        return idxs, num_bins

    def _create_oracle_rdm(self):
        """
        Creates the Oracle RDM for the specified dataset.

        Returns:
        --------
        npt.NDArray
            The Oracle RDM as a squareform distance matrix.
        """

        if self.dataset_label == "visual":
            # Create Oracle RDM.
            one_class = np.ones((200, 200))
            all_classes = [one_class for _ in range(30)]
            block_rdm_sqform = 1 - block_diag(*all_classes)
            oracle_rdm = squareform(block_rdm_sqform)

        elif self.dataset_label == "HPC":
            one_class = np.ones((200, 200))
            all_classes = [one_class for _ in range(16)]
            block_rdm_sqform = 1 - block_diag(*all_classes)
            oracle_rdm = squareform(block_rdm_sqform)

        else:
            num_sample = self.idxs.shape[1]
            num_bins = self.idxs.shape[0]
            one_class = np.ones((num_sample, num_sample))
            all_classes = [one_class for _ in range(num_bins)]
            block_rdm_sqform = 1 - block_diag(*all_classes)
            oracle_rdm = squareform(block_rdm_sqform)

        return oracle_rdm

    def _compute_per_layer(
        self, layer_activation: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        """
        Computes the RDM for a given layer's activation.

        Parameters:
        -----------
        layer_activation : npt.NDArray
            A 2D numpy array representing the activation of neurons in a layer. The shape should be (num_neurons, num_samples) or (num_samples, num_neurons).

        Returns:
        --------
        Tuple[npt.NDArray, float]
            A tuple of the computed RDM as a squareform distance matrix and the similarity score between the computed RDM and the Oracle RDM, if applicable.
        """
        # to ensure the right shape: numSamples X numNeurons
        if layer_activation.shape[0] < layer_activation.shape[1]:
            layer_activation = layer_activation.T

        rdm = pdist(layer_activation[self.idxs.flatten(), :], metric=self.metric)
        if self.bool_oracle:
            oracle_rdm = self._create_oracle_rdm()
            comparison = 1 - correlation(oracle_rdm, rdm)
        else:
            comparison = None

        return squareform(rdm), comparison

    def compute(
        self,
        activations: List[Union[float, npt.NDArray]],
    ) -> List[Tuple[npt.NDArray, float]]:
        """
        Computes the RDMs (Representational Dissimilarity Matrices) for each layer's activations.

        Parameters:
        -----------
        activations : List[Union[float, npt.NDArray]]
            List of 2D numpy arrays representing the activation of neurons per layer.

        Returns:
        --------
        List[Tuple[npt.NDArray, float]]:
            A list of tuples, where each tuple contains the computed RDM and the correlation score with the Oracle RDM (if applicable) for each layer of a model.
        """
        if isinstance(
            activations, (np.ndarray, torch.Tensor)
        ):  # if only one activation is passed instead of a list of arrays
            activations = [activations]

        return super().iterate_over_layers(activations, self._compute_per_layer)

    @property
    def __name__(self):
        return "rdm"

    def set_num_bins(self, num_bins):
        """
        Sets the number of bins for the RDM computation.

        Parameters:
        ----------
        num_bins : int
            The number of bins to be used in the RDM computation.
        """
        self.num_bins = num_bins

    def plot(
        self,
        rdms: Dict[str, List[npt.NDArray]],
        titles: List[Tuple[npt.NDArray, float]] = None,
        metric: str = "Correlation",
        cmap: str = "viridis",
        figsize: tuple = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plots the RDM analysis results. If `bool_oracle` is True, it plots the correlation of the RDMs with the Oracle RDM, else it plots the RDMs in the rdms dictionary.

        Parameters:
        ----------
        rdms : Dict[str, List[npt.NDArray]]
            Dictionary where the key is the model category label (str), and the value is a list of npt.NDArray containing for all the models under that label the calculated RDMs.
        titles : List[Tuple[npt.NDArray, float]], optional
            List of tuples containing the RDM and the correlation score with the Oracle RDM for each layer of a model. Default is None.
        metric : str, optional
            The metric to be used for the plot. Default is "Correlation".
        cmap : str, optional
            The colormap to be used for the plot. Default is "viridis".
        figsize : tuple, optional
            The size of the figure for the plot. Default is None, which uses the default size.
        ax : Optional[matplotlib.axes.Axes], optional
            The axes on which to plot the RDMs. If None, a new figure and axes will be created. Default is None.
        Returns:
        -------
        matplotlib.figure.Figure
            The figure containing the plotted RDMs.
        """
        if self.bool_oracle:
            return plot_rdm_correlation(rdms)
        else:
            return plot_rdm_all(
                rdms=rdms,
                labels=self.label,
                num_bins=self.num_bins,
                discrete=self.discrete,
                titles=titles,
                metric=metric,
                dataset_label=self.dataset_label,
                cmap=cmap,
                figsize=figsize,
                ax=ax,
            )
