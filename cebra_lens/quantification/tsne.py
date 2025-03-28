"""Functions to transform data e.g. tSNE, other functions can be added"""

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from .base import _BaseMetric
from ..matplotlib import *
import pickle
from pathlib import Path


class Tsne(_BaseMetric):
    def __init__(
        self,
        num_samples: int = 200,
        # activation: np.ndarray
    ):
        super().__init__(self)
        self.num_samples = num_samples
        self._check_num_samples()

    def _compute_per_layer(self, layer_activation):
        if layer_activation.shape[0] > layer_activation.shape[1]:
            layer_activation = layer_activation.T

        tsne = TSNE(n_components=3)
        tsne_embedding = tsne.fit_transform(layer_activation[:, : self.num_samples].T)
        return tsne_embedding

    def compute(self, activations) -> np.ndarray:
        """
        Applies t-SNE (t-Distributed Stochastic Neighbor Embedding) to the given layer activation data.
        This function performs dimensionality reduction on the layer activation data to generate a 2D embedding using t-SNE.

        Parameters:
        -----------
        layer_activation : np.ndarray
            A 2D numpy array representing the activation of neurons in a layer. The shape should be
            (num_neurons, num_samples) or (num_samples, num_neurons).
        num_samples : int
            The number of samples to use for t-SNE transformation.

        Returns:
        --------
        tsne_embedding : np.ndarray
            The 2D embedding produced by t-SNE.
        """
        return super().iterate_over_layers(activations, self._compute_per_layer)

    def _check_num_samples(self):
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
        embeddings_1: list,
        embeddings_2: list,
        labels: np.ndarray,
        sample_plot: int = 200,
        comparison_labels: tuple = ("tSNE", ["Untrained", "Trained"]),
        dataset_label: str = "HPC",
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        return compare_embeddings_layers(
            embeddings_1,
            embeddings_2,
            labels,
            sample_plot,
            comparison_labels,
            dataset_label,
            ax,
        )
