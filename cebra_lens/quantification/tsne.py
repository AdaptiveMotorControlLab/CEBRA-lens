"""Functions to transform data e.g. tSNE, other functions can be added"""

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
from .base import _BaseMetric


class Tsne(_BaseMetric):
    def __init__(self, num_samples:int = 200,
                 #activation: np.ndarray
                 ):
        super().__init__(self)
        self.num_samples = num_samples
        self._check_num_samples()

    def compute(self, activation) -> np.ndarray:
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
        for layer_activation in activation:
            # Check that it's num_neurons X num_samples: Assumption that we always have num_neurons < num_samples
            if layer_activation.shape[0] > layer_activation.shape[1]:
                layer_activation = layer_activation.T

            tsne = TSNE(n_components=3)
            tsne_embedding = tsne.fit_transform(layer_activation[:, :self.num_samples].T)
            tsne_embeddings.append(tsne_embedding)

        return tsne_embeddings
    
    # def save(self):
    #     return super().save()
    
    # def load(self):
    #     return super().load()
    
    # def plot(self):
    #     return super().plot()
    
    # @property
    # get_
    # set_for parameters which are extra in the compute

    def _check_num_samples(self):
        if self.num_samples < 200:
            print(
                f"Warning: Minimum number of samples is 200 to ensure good functioning. Provided: {self.num_samples}. Processing with 200..."
            )
            self.num_samples = 200

