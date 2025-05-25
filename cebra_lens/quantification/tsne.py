"""Functions to transform data e.g. tSNE, other functions can be added"""

from sklearn.manifold import TSNE
import numpy as np
from .base import _BaseMetric
from ..matplotlib import *
from typing import List, Optional, Union
import numpy.typing as npt


class Tsne(_BaseMetric):
    """
    Compute t-SNE (t-Distributed Stochastic Neighbor Embedding) on layer activation data.

    Parameters:
    -----------
    num_samples : int
        The number of samples to use for t-SNE transformation. Default is 200.
    """

    def __init__(
        self,
        num_samples: int = 200,
    ):
        super().__init__()
        self.num_samples = num_samples
        self._check_num_samples()

    def _compute_per_layer(self, layer_activation: npt.NDArray) -> npt.NDArray:
        """
        Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the given layer activation data.

        Parameters:
        -----------
        layer_activation : npt.NDArray
            A 2D numpy array representing the activation of neurons in a layer. The shape should be (num_neurons, num_samples) or (num_samples, num_neurons).

        Returns:
        --------
        tsne_embedding : npt.NDArray
            The 2D embedding produced by t-SNE.
        """
        if layer_activation.shape[0] > layer_activation.shape[1]:
            layer_activation = layer_activation.T

        tsne = TSNE(n_components=3)
        tsne_embedding = tsne.fit_transform(layer_activation[:, : self.num_samples].T)
        return tsne_embedding

    def compute(
        self, activations: List[Union[float, npt.NDArray]]
    ) -> List[Union[float, npt.NDArray]]:
        """
        Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the given activations data per layer.
        This function performs dimensionality reduction on each layer activation data to generate a 2D embeddings using t-SNE.

        Parameters:
        -----------
        activations : List[Union[float, npt.NDArray]]
            List of 2D numpy array representing the activation of neurons per layer.

        Returns:
        --------
        List[Union[float, npt.NDArray]]
            The 2D embedding produced by t-SNE for each layer of a model.
        """
        return super().iterate_over_layers(activations, self._compute_per_layer)

    def _check_num_samples(self):
        """Checks if the number of samples is less than 200. If so, it sets the number of samples to 200 and prints a warning message."""
        if self.num_samples < 200:
            print(
                f"Warning: Minimum number of samples is 200 to ensure good functioning. Provided: {self.num_samples}. Processing with 200..."
            )
            self.num_samples = 200

    @property
    def __name__(self):
        return "tsne"

    def plot(
        self,
        embeddings: Union[Dict[str, List[npt.NDArray]],List[npt.NDArray]],
        labels: npt.NDArray,
        sample_plot: int = 200,
        dataset_label: str = "HPC",
        group_name: str = "t-SNE",
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        """
        Plots the t-SNE embeddings for the first 3 dimensions.
        """
        return plot_embeddings(
            embeddings,
            labels,
            group_name,
            dataset_label,
            sample_plot,
            ax,
        )
