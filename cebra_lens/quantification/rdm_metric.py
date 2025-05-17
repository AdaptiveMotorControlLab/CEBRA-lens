"""All the functions relative to the Representation Dissimilarity Matrix (RDM) calculation"""

import numpy as np
from scipy.linalg import block_diag
from typing import List, Optional, Tuple, Union
from scipy.spatial.distance import correlation, pdist, squareform
from .misc import discrete_binning
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
    dataset_label : str, optional
        The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
    metric : str, optional
        The distance metric to use for computing the RDMs. Default is 'correlation'.
    bool_oracle : bool, optional
        Whether to compute and compare with the Oracle RDM. Default is True.
    """

    def __init__(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        dataset_label: str = None,
        metric: str = "correlation",
        bool_oracle: bool = True,
        num_samples: int = None,
        num_bins: int = None,
        label_ind: int = None,
    ):
        super().__init__()
        self.data = data
        self.label = label
        self.label_ind = label_ind
        # check that label is 1D if dataset_label is not HPC/visual, and the label_ind is not provided
        if isinstance(self.label, np.ndarray) and self.label.ndim != 1:
            # if the dataset contains multiple labels check that if it is not HPC dataset the label_ind was given
            if self.dataset_label != "HPC" and self.label_ind != None:
                self.label = self.label[:, self.label_ind]
            else:
                raise KeyError(
                    "If dataset not HPC or visual and there are multiple possible labels, parameter label_ind must be provided to indicate which label will be used for the RDM calculation"
                )

        self.dataset_label = dataset_label
        self.metric = metric
        self.bool_oracle = bool_oracle
        self.num_samples = num_samples
        self.num_bins = num_bins

        self.idxs = discrete_binning(
            data=self.data, label=self.label, dataset_label=self.dataset_label
        )

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

    def _compare_RDM(self, rdm_1: npt.NDArray, rdm_2: npt.NDArray) -> float:
        """
        Compares two RDMs using the specified metric.

        Parameters:
        -----------
        rdm_1 : npt.NDArray
            The first RDM to compare.
        rdm_2 : npt.NDArray
            The second RDM to compare.

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
            correlation = self._compare_RDM(rdm_1=oracle_rdm, rdm_2=rdm)
        else:
            correlation = None

        return squareform(rdm), correlation

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

        if self.dataset_label != "visual" and self.dataset_label != "HPC":
            self.idxs = discrete_binning(
                self.data,
                self.label,
                self.dataset_label,
                num_bins=self.num_bins,
                max_num_samples=self.num_samples,
            )

        return super().iterate_over_layers(activations, self._compute_per_layer)

    @property
    def __name__(self):
        return "rdm"

    def set_num_bins(self, num_bins):
        self.num_bins = num_bins

    def set_num_samples(self, num_samples):
        self.num_samples = num_samples

    def plot(
        self,
        rdms: List[npt.NDArray],
        titles: List[Tuple[npt.NDArray, float]],
        metric: str = "Normalized Euclidean distance",
        cmap: str = "viridis",
        figsize: tuple = None,
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        return plot_rdm(
            rdms,
            titles,
            metric,
            self.dataset_label,
            cmap,
            figsize,
            ax,
        )
