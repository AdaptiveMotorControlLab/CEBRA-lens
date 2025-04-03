"""All the functions relative to the Representation Dissimilarity Matrix (RDM) calculation"""

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, pdist, squareform
from tqdm import tqdm
from .misc import discrete_binning
import torch
from .base import _BaseMetric
import pickle
from pathlib import Path
from ..matplotlib import *


class RDM(_BaseMetric):
    def __init__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        dataset_label: str = None,
        metric: str = "correlation",
        bool_oracle: bool = True,
    ):
        super().__init__()
        self.data = data
        self.label = label
        self.dataset_label = dataset_label
        self.metric = metric
        self.bool_oracle = bool_oracle

        self.idxs = discrete_binning(
            data=self.data, label=self.label, dataset_label=self.dataset_label
        )

    def _create_oracle_rdm(self):
        """
        Creates the Oracle RDM for the specified dataset.

        Parameters:
        -----------
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'.

        Returns:
        --------
        np.ndarray
            The Oracle RDM as a squareform distance matrix.
        """

        if self.dataset_label == "visual":
            # Create Oracle RDM.
            one_class = np.ones((200, 200))
            all_classes = [one_class for _ in range(30)]
            block_rdm_sqform = 1 - block_diag(*all_classes)
            oracle_rdm = squareform(block_rdm_sqform)
        elif self.dataset_label == "HPC":
            # TODO: complete this
            raise NotImplementedError(
                f"Oracle RDM not defined for {self.dataset_label}. Please use 'visual'"
            )

        else:
            raise NotImplementedError(
                f"Oracle RDM not defined for {self.dataset_label}. Please use 'visual'"
            )

        return oracle_rdm

    def _compare_RDM(self, rdm_1: np.ndarray, rdm_2: np.ndarray) -> float:
        """
        Compares two RDMs using the specified metric.

        Parameters:
        -----------
        rdm_1 : np.ndarray
            The first RDM to compare.
        rdm_2 : np.ndarray
            The second RDM to compare.
        metric : str, optional
            The distance metric to use for comparison. Default is 'correlation'.

        Returns:
        --------
        float
            The similarity score between the two RDMs, based on the specified metric.
        """

        if self.metric == "correlation":
            comparison = 1 - correlation(rdm_1, rdm_2)
        else:
            raise NotImplementedError(
                f"The metric {self.metric} is not yet implemented. Please use 'correlation'."
            )

        return comparison

    def _compute_per_layer(self, layer_activation):
        # to ensure the right shape: numSamples X numNeurons
        if layer_activation.shape[0] < layer_activation.shape[1]:
            layer_activation = layer_activation.T

        rdm = pdist(layer_activation[self.idxs.flatten(), :], metric=self.metric)
        if self.bool_oracle:
            oracle_rdm = self._create_oracle_rdm()
            correlation = self._compare_RDM(rdm_1=oracle_rdm, rdm_2=rdm)
        else:
            correlation = None

        return squareform(rdm), correlation

    def compute(self, activations):
        """
        Computes the RDMs (Representational Dissimilarity Matrices) for each layer's activations.

        Parameters:
        -----------
        data : torch.Tensor
            The data array of shape (num_samples, num_features).
        label :torch.Tensor
            The array of labels corresponding to the data.
        activations : list
            A list of activations, each being an array of shape (num_neurons, num_samples).
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
        metric : str, optional
            The distance metric to use for computing the RDMs. Default is 'correlation'.
        bool_oracle : bool, optional
            Whether to compute and compare with the Oracle RDM. Default is True.

        Returns:
        --------
        list
            A list of tuples, where each tuple contains the RDM for the layer and the correlation with the Oracle RDM (if computed).
        """
        if isinstance(
            self.activations, (np.ndarray, torch.Tensor)
        ):  # if only one activation is passed instead of a list of arrays
            self.activations = [self.activations]

        return super().iterate_over_layers(activations, self._compute_per_layer)

    def plot(
        self,
        rdms: list,
        titles: list,
        metric: str = "Normalized Euclidean distance",
        dataset_label: str = "visual",
        cmap: str = "viridis",
        figsize: tuple = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        return plot_rdm(
            rdms,
            titles,
            metric,
            dataset_label,
            cmap,
            figsize,
            ax,
        )

    @property
    def __name__(self):
        return "rdm"
