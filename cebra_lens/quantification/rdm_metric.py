"""All the functions relative to the Representation Dissimilarity Matrix (RDM) calculation"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import numpy as np
import numpy.typing as npt
import torch
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, pdist, squareform

from cebra_lens import utils_plot

from .base import _BaseMetric
from .misc import continuous_binning, discrete_binning


class RDM(_BaseMetric):
    """Compute the Representational Dissimilarity Matrix (RDM) for a given activation.

    Attributes:
        data : torch.Tensor
            The data array of shape (num_samples, num_features).
        label : torch.Tensor
            The array of labels corresponding to the data.
        is_discrete_labels : bool, optional
            Whether the labels are discrete or continuous. By default, it is False, meaning the labels are continuous.
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
        metric : str, optional
            The distance metric to use for computing the RDMs. 'correlation' or 'euclidean' are supported.
    """

    def __init__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        is_discrete_labels: bool = False,
        dataset_label: str = None,
        metric: str = "correlation",
    ):
        super().__init__()
        self.data = data
        self.label = label
        self.dataset_label = dataset_label
        # check that label is 1D if dataset_label is not HPC/visual
        if isinstance(self.label, np.ndarray) and self.dataset_label not in [
                "HPC",
                "visual",
        ]:
            if len(label.shape) > 1:
                raise ValueError("The label must be a 1D array.")
            self.label = label

        self.metric = metric
        self.discrete = is_discrete_labels
        self.idxs, self.num_bins = self._define_indices()

    def output_information(self):
        """Outputs information about the RDM class initialization parameters."""
        print("RDM class initialized with the following parameters:")
        # if self.bool_oracle:
        #     print(
        #         "The chosen analysis will plot the correlation of the RDMs with the Oracle RDM."
        #     )
        # else:
        #     print(
        #         "The chosen analysis will plot the RDMs, no Oracle RDM comparison."
        #     )
        if self.dataset_label is None:
            print(
                f"The dataset label is not specified,\n this label has been noted DISCRETE = {self.discrete}."
            )
        else:
            print(f"The dataset label is specified as: {self.dataset_label}")
        print(
            "If this is not the desired behavior, please check the parameters passed to the RDM class."
        )

    def _define_indices(self) -> Tuple[npt.NDArray, Optional[int]]:
        """Defines the indices for the bins and repetitions based on the specified distance label.

        Returns:
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
                idxs = discrete_binning(labels=self.label, )
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
        """Creates the Oracle RDM for the specified dataset.

        Returns:
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
            self,
            layer_activation: npt.NDArray,
            bool_oracle: bool = False) -> Tuple[npt.NDArray, float]:
        """Computes the RDM for a given layer's activation.

        Args:
            layer_activation : npt.NDArray
                A 2D numpy array representing the activation of neurons in a layer. The shape should be (num_neurons, num_samples) or (num_samples, num_neurons).

        Returns:
            Tuple[npt.NDArray, float]
                A tuple of the computed RDM as a squareform distance matrix and the similarity score between the computed RDM and the Oracle RDM, if applicable.
        """
        # to ensure the right shape: numSamples X numNeurons
        if layer_activation.shape[0] < layer_activation.shape[1]:
            layer_activation = layer_activation.T

        rdm = pdist(layer_activation[self.idxs.flatten(), :],
                    metric=self.metric)
        if bool_oracle:
            oracle_rdm = self._create_oracle_rdm()
            comparison = 1 - correlation(oracle_rdm, rdm)
        else:
            comparison = None

        return squareform(rdm), comparison

    def compute(
        self,
        activations: List[Union[float, npt.NDArray]],
        bool_oracle: bool = False,
        labels: Optional[List[npt.NDArray]] = None,
    ) -> List[Tuple[npt.NDArray, float]]:
        """Computes the RDMs (Representational Dissimilarity Matrices) for each layer's activations.

        Args:
            activations : List[Union[float, npt.NDArray]]
                List of 2D numpy arrays representing the activation of neurons per layer.

        Returns:
            List[Tuple[npt.NDArray, float]]:
                A list of tuples, where each tuple contains the computed RDM and the correlation score with the Oracle RDM (if applicable) for each layer of a model.
        """
        if isinstance(
                activations, (np.ndarray, torch.Tensor)
        ):  # if only one activation is passed instead of a list of arrays
            activations = [activations]

        if labels is not None:
            results: List[Tuple[npt.NDArray, float]] = []
            # stash originals so we can restore afterwards
            orig_label    = self.label
            orig_idxs     = self.idxs
            orig_num_bins = self.num_bins

            for A, L in zip(activations, labels):
                # 1) swap in the layer's own label
                self.label = L
                self.idxs, self.num_bins = self._define_indices()

                # 2) now compute exactly as before
                mat, corr = self._compute_per_layer(A, bool_oracle)
                results.append((mat, corr))

            # 3) restore the object back to its original state
            self.label, self.idxs, self.num_bins = (
                orig_label, orig_idxs, orig_num_bins
            )
            return results

        return super().iterate_over_layers(activations,
                                           self._compute_per_layer,
                                           bool_oracle=bool_oracle)

    @property
    def __name__(self):
        return "rdm"

    def plot(
        self,
        rdms: Dict[str, List[npt.NDArray]],
        bool_oracle: bool = False,
        titles: List[str] = None,
        cmap: str = "viridis",
        figsize: tuple = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.figure.Figure:
        """Plots the RDM analysis results. If `bool_oracle` is True, it plots the correlation of the RDMs with the Oracle RDM, else it plots the RDMs in the rdms dictionary.

        Args:
            rdms : Dict[str, List[npt.NDArray]]
                Dictionary where the key is the model category label (str), and the value is a list of npt.NDArray containing for all the models under that label the calculated RDMs.
            titles : List[str], optional
                List of title for the RDM plots, if it is different from ordered layers.
            cmap : str, optional
                The colormap to be used for the plot. Default is "viridis".
            figsize : tuple, optional
                The size of the figure for the plot. Default is None, which uses the default size.
            ax : Optional[matplotlib.axes.Axes], optional
                The axes on which to plot the RDMs. If None, a new figure and axes will be created. Default is None.
        
        Returns:
            matplotlib.figure.Figure
                The figure containing the plotted RDMs.
        """
        if bool_oracle:
            return utils_plot.plot_rdm_correlation(rdms)
        else:
            return utils_plot.plot_rdm_all(
                rdms=rdms,
                labels=self.label,
                num_bins=self.num_bins,
                discrete=self.discrete,
                titles=titles,
                metric=self.metric,
                dataset_label=self.dataset_label,
                cmap=cmap,
                figsize=figsize,
                ax=ax,
            )
