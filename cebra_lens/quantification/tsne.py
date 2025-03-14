"""Functions to transform data e.g. tSNE, other functions can be added"""

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import pickle
from .base import _BaseMetric, _MultiMetric

class TSne(_BaseMetric):
    def __init__(self, activation: np.ndarray):
        self.activation = activation

    def compute(self) -> np.ndarray:
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
        tsne_embeddings = []
        for layer_activation in self.activation:
            # Check that it's num_neurons X num_samples: Assumption that we always have num_neurons < num_samples
            if layer_activation.shape[0] > layer_activation.shape[1]:
                layer_activation = layer_activation.T

            tsne = TSNE(n_components=3)
            tsne_embedding = tsne.fit_transform(layer_activation[:, :self.num_samples].T)
            tsne_embeddings.append(tsne_embedding)

        return tsne_embeddings
    
class MultiTsne(_MultiMetric):

    def __init__(self, activations_dict, num_samples=200):
        self.activations_dict = activations_dict
        self.num_samples = num_samples
        if self.num_samples < 200:
            print(
                f"Warning: Minimum number of samples is 200 to ensure good functioning. Provided: {self.num_samples}. Processing with 200..."
            )
            self.num_samples = 200
        self.base = TSne
        self.data = super().transform(self.activations_dict,self.base)

    def compute(self, *args, **kwargs) -> dict:
        """
        Runs t-SNE on the provided activations dictionary and saves the results to a pickle file.
        This function performs t-SNE on the activations data.

        Parameters:
        -----------
        activations_dict : dict
            A dictionary containing the activations. More information on the format can be found in CEBRA_Lens.activations.
        num_samples : int
            The number of samples to use for t-SNE transformation.

        Returns:
        --------
        tsne_embeddings : dict
            A dictionary containing the t-SNE embeddings, structured exactly the same as the input `activations_dict`.
        """
        return super().compute(data_dict=self.data,*args, **kwargs)
