"""Functions to transform data e.g. tSNE, other functions can be added"""

from typing import List, Optional, Union

import matplotlib
import numpy as np
import numpy.typing as npt
from sklearn.manifold import TSNE

from ..utils_plot import *
from .base import _BaseMetric


class Tsne(_BaseMetric):
    """Compute t-SNE (t-Distributed Stochastic Neighbor Embedding) on layer activation data.

    Attributes:
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
        """Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the given layer activation data.

        Args:
            layer_activation : npt.NDArray
                A 2D numpy array representing the activation of neurons in a layer. The shape should be (num_neurons, num_samples) or (num_samples, num_neurons).

        Returns:
            tsne_embedding : npt.NDArray
                The 2D embedding produced by t-SNE.
        """
        if layer_activation.shape[0] > layer_activation.shape[1]:
            layer_activation = layer_activation.T

        tsne = TSNE(n_components=3)
        tsne_embedding = tsne.fit_transform(
            layer_activation[:, :self.num_samples].T)
        return tsne_embedding

    def compute(
        self, activations: List[Union[float, npt.NDArray]]
    ) -> List[Union[float, npt.NDArray]]:
        """Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the given activations data per layer.
        
        This function performs dimensionality reduction on each layer activation data to generate a 2D embeddings using t-SNE.

        Args:
            activations : List[Union[float, npt.NDArray]]
                List of 2D numpy array representing the activation of neurons per layer.

        Returns:
            List[Union[float, npt.NDArray]]
                The 2D embedding produced by t-SNE for each layer of a model.
        """
        return super().iterate_over_layers(activations,
                                           self._compute_per_layer)

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
        embeddings: Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]],
        labels: npt.NDArray,
        sample_plot: int = 200,
        dataset_label: str = None,
        group_name: str = "t-SNE",
        ax: Optional[matplotlib.axes.Axes] = None,
        **kwargs: dict,
    ) -> matplotlib.figure.Figure:
        """Plots the t-SNE embeddings for the first 3 dimensions.

        Args:
            embeddings : Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]]
                The t-SNE embeddings to plot. Can be a dictionary with group names or a list of embeddings.
            labels : npt.NDArray
                The labels corresponding to the embeddings, used for coloring the points in the plot.
            sample_plot : int, optional
                The number of samples to plot. Default is 200.
            dataset_label : str, optional
                The label of the dataset, used for determining the number of bins in the plot. Default is "HPC".
            group_name : str, optional
                The name of the group for which the embeddings are plotted. Default is "t-SNE".
            ax : Optional[matplotlib.axes.Axes], optional
                The axes on which to plot the embeddings. If None, a new figure and axes will be created. Default is None.

        Returns:
            matplotlib.figure.Figure
                The figure containing the t-SNE plot.
        """
        return plot_embeddings(
            data=embeddings,
            labels=labels,
            group_name=group_name,
            dataset_label=dataset_label,
            sample_plot=sample_plot,
            ax=ax,
            **kwargs,
        )
