"""All relevant functions for Centered Kernel Alignment (CKA) analysis."""

from tqdm import tqdm
import numpy as np
from .helper import *

class ComparisonCKA:
    def __init__(self,comparison):

        if not isinstance(comparison, tuple):
            raise ValueError(
                f"A comparison must be a tuple. Comparison is of type: {type(comparison)}."
            )
        self.comparisonX = comparison[0]
        self.comparisonY = comparison[1]
    
    def _compute(self,embeddings_1: list, embeddings_2: list) -> np.ndarray:
        """
        Compute the Centered Kernel Alignment (CKA) between two sets of embeddings for each layer.
        This function calculates the CKA score between corresponding layers of two sets of embeddings,
        assuming both sets have the same number of layers and that each corresponding layer has the same shape.

        Parameters:
        -----------
        embeddings_1 : list
            A list of embeddings for the first set. Each element represents the embeddings for a specific layer.
        embeddings_2 : list
            A list of embeddings for the second set. Each element represents the embeddings for a specific layer.

        Returns:
        --------
        cka_matrix : np.ndarray
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
            cka_matrix[0, i] = cka(
                gram_linear(embeddings_1[i].T),
                gram_linear(embeddings_2[i].T),
            )
        return cka_matrix

    def _compute_single(self, embeddings_1, embeddings_2, flag=False):
        self.cka_matrix = np.zeros((len(embeddings_1), len(embeddings_1[0])))
        for j in tqdm(range(len(embeddings_1))):
            if flag:
                self.cka_matrix[j, :] = self._compute(embeddings_1[j], embeddings_2[j])
            else:
                self.cka_matrix[j, :] = self._compute(embeddings_1[j], embeddings_2)

    def compute(self, activations_dict):
        """
        Compute multi-layer Centered Kernel Alignment (CKA) between different sets of activations.
        This function calculates the CKA score between activations from different models and layers,
        comparing them based on the specified models and layers.

        Parameters:
        -----------
        activations_dict : dict
            A dictionary where keys are strings in the format 'model_identifie' and values are 2d lists with the corresponding activations.
        comparison : tuple
            A tuple containing two strings representing the models and layers to be compared.
            For example, ('single_UT', 'single_TR').

        Returns:
        --------
        cka_matrix : np.ndarray
            A CKA matrix with rows representing instances of the model and columns representing the layers.
        """

        activations_1 = activations_dict[self.comparisonX]
        activations_2 = activations_dict[self.comparisonY]

        # if not the same length, compare embeddings to the first instance. else compare pairwise.
        if len(activations_1) != len(activations_2):
            if len(activations_1) > len(activations_2):
                embeddings_1 = activations_1
                embeddings_2 = activations_2[0]

            elif len(activations_1) < len(activations_2):
                embeddings_1 = activations_2
                embeddings_2 = activations_1[0]

            self._compute_single(embeddings_1, embeddings_2)

        # example when compare intra model single_TR v single_TR, only compare to the first instantiation
        elif self.comparisonX == self.comparisonY:
            embeddings_1 = activations_1
            embeddings_2 = activations_2[0]
            self._compute_single(embeddings_1, embeddings_2)

        else:
            embeddings_1 = activations_1
            embeddings_2 = activations_2
            self._compute_single(embeddings_1, embeddings_2, flag=True)

        return self.cka_matrix
