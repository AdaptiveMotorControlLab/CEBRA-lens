"""
All relevant functions for Centered Kernel Alignment (CKA) analysis.

CKA computation was taken from https://github.com/amathislab/DeepDraw

"""

from tqdm import tqdm
import numpy as np
from .base import _BaseMetric
from ..matplotlib import *
from typing import Optional, List, Dict
import numpy.typing as npt


class CKA(_BaseMetric):
    """
    Compute the Centered Kernel Alignment (CKA) between two sets of model types.

    Parameters:
    -----------
    comparison : Tuple[str,str]
        A tuple containing two strings representing the models and training type to be compared.
        For example, ('single_UT', 'single_TR').
    """

    def __init__(self, comparison: Tuple[str, str]):

        if not isinstance(comparison, tuple):
            raise ValueError(
                f"A comparison must be a tuple. Comparison is of type: {type(comparison)}."
            )
        self.comparisonX = comparison[0]
        self.comparisonY = comparison[1]
        self.cka_matrix = None

    def center_gram(self, gram: npt.NDArray, unbiased: bool = False) -> npt.NDArray:
        """Center a symmetric Gram matrix.

        This is equvialent to centering the (possibly infinite-dimensional) features
        induced by the kernel before computing the Gram matrix.

        Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
            estimate of HSIC. Note that this estimator may be negative.

        Returns:
        A symmetric matrix with centered columns and rows.
        """
        if not np.allclose(gram, gram.T, rtol=1e-03, atol=0.004):
            raise ValueError("Input must be a symmetric matrix.")
        gram = gram.copy()

        if unbiased:
            # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
            # L. (2014). Partial distance correlation with methods for dissimilarities.
            # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
            # stable than the alternative from Song et al. (2007).
            n = gram.shape[0]
            np.fill_diagonal(gram, 0)
            means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
            means -= np.sum(means) / (2 * (n - 1))
            gram -= means[:, None]
            gram -= means[None, :]
            np.fill_diagonal(gram, 0)
        else:
            means = np.mean(gram, 0, dtype=np.float64)
            means -= np.mean(means) / 2
            gram -= means[:, None]
            gram -= means[None, :]

        return gram

    def cka(
        self, gram_x: npt.NDArray, gram_y: npt.NDArray, debiased: bool = False
    ) -> np.float64:
        """Compute CKA.

        Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

        Returns:
        The value of CKA between X and Y.
        """
        gram_x = self.center_gram(gram_x, unbiased=debiased)
        gram_y = self.center_gram(gram_y, unbiased=debiased)

        # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
        # n*(n-3) (unbiased variant), but this cancels for CKA.
        scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

        normalization_x = np.linalg.norm(gram_x)
        normalization_y = np.linalg.norm(gram_y)
        return scaled_hsic / (normalization_x * normalization_y)

    def gram_linear(self, x: npt.NDArray) -> npt.NDArray:
        """Compute Gram (kernel) matrix for a linear kernel.

        Args:
        x: A num_examples x num_features matrix of features.

        Returns:
        A num_examples x num_examples Gram matrix of examples.
        """

        return x.dot(x.T)

    def _compute_cka(
        self, embeddings_1: List[npt.NDArray], embeddings_2: List[npt.NDArray]
    ) -> npt.NDArray:
        """
        Compute the Centered Kernel Alignment (CKA) between two sets of embeddings for each layer.
        This function calculates the CKA score between corresponding layers of two sets of embeddings,
        assuming both sets have the same number of layers and that each corresponding layer has the same shape.

        Parameters:
        -----------
        embeddings_1 : List[npt.NDArray]
            A list of embeddings for the first set. Each element represents the embeddings for a specific layer.
        embeddings_2 : List[npt.NDArray]
            A list of embeddings for the second set. Each element represents the embeddings for a specific layer.

        Returns:
        --------
        cka_matrix : npt.NDArray
            A one-row array containing the CKA values for each layer.
        """

        if len(embeddings_1) != len(embeddings_2):
            raise ValueError(
                "The number of layers in embeddings_1 and embeddings_2 must be the same."
            )
        for i in range(len(embeddings_1)):
            if embeddings_1[i].shape != embeddings_2[i].shape:
                raise ValueError(
                    f"The shape of layer {i} in embeddings_1 and embeddings_2 must be the same."
                )

        cka_matrix = np.zeros((1, len(embeddings_1)))
        for i in range(len(embeddings_1)):
            cka_matrix[0, i] = self.cka(
                self.gram_linear(embeddings_1[i].T),
                self.gram_linear(embeddings_2[i].T),
            )
        return cka_matrix

    def _compute_per_layer(
        self,
        embeddings_1: List[npt.NDArray],
        embeddings_2: List[npt.NDArray],
        flag=False,
    ) -> npt.NDArray:
        """
        Compute the Centered Kernel Alignment (CKA) between two sets of embeddings.

        Parameters:
        -----------
        embeddings_1 : List[npt.NDArray]
            A list of embeddings for the first set. Each element represents the embeddings for a specific layer.
        embeddings_2 : List[npt.NDArray]
            A list of embeddings for the second set. Each element represents the embeddings for a specific layer.
        flag : bool
            If True, compute CKA for each layer of the first set against the second set.

        Returns:
        --------
        cka_matrix : npt.NDArray
            A matrix containing the CKA values for each layer.
        """
        cka_matrix = np.zeros((len(embeddings_1), len(embeddings_1[0])))
        for j in tqdm(range(len(embeddings_1))):
            if flag:
                cka_matrix[j, :] = self._compute_cka(embeddings_1[j], embeddings_2[j])
            else:
                cka_matrix[j, :] = self._compute_cka(embeddings_1[j], embeddings_2)
        return cka_matrix

    def compute(self, activations: Dict[str, npt.NDArray]) -> npt.NDArray:
        """
        Compute multi-layer Centered Kernel Alignment (CKA) between different sets of activations.
        This function calculates the CKA score between activations from different models and layers,
        comparing them based on the specified models and layers.

        Parameters:
        -----------
        activations : Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are 2d lists with the corresponding activations per layer.

        Returns:
        --------
        cka_matrix : npt.NDArray
            A CKA matrix with rows representing instances of the model and columns representing the layers.
        """

        activations_1 = activations[self.comparisonX]
        activations_2 = activations[self.comparisonY]
        if len(activations_1) != len(activations_2):
            if len(activations_1) > len(activations_2):
                embeddings_1 = activations_1
                embeddings_2 = activations_2[0]

            elif len(activations_1) < len(activations_2):
                embeddings_1 = activations_2
                embeddings_2 = activations_1[0]

            self.cka_matrix = self._compute_per_layer(embeddings_1, embeddings_2)

        # example when compare intra model single_TR v single_TR, only compare to the first instantiation
        elif self.comparisonX == self.comparisonY:
            embeddings_1 = activations_1
            embeddings_2 = activations_2[0]
            self.cka_matrix = self._compute_per_layer(embeddings_1, embeddings_2)

        else:
            embeddings_1 = activations_1
            embeddings_2 = activations_2
            self.cka_matrix = self._compute_per_layer(embeddings_1, embeddings_2, True)

        return self.cka_matrix

    @property
    def __name__(self) -> str:
        return "cka"

    def plot(
        self,
        cka_matrices: Dict[str, npt.NDArray],
        annot: bool,
        show_cbar: bool = True,
        cbar_label: str = "CKA score",
        color_map: str = "magma",
        figsize: tuple = (15, 5),
        ax: Optional[matplotlib.axes.Axes] = None,
    ):
        return plot_cka_heatmaps(
            cka_matrices,
            annot,
            show_cbar,
            cbar_label,
            color_map,
            figsize,
            ax,
        )
