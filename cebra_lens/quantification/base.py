import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import correlation, pdist, squareform
from tqdm import tqdm
from .misc import discrete_binning
import torch


class _MultiMetric:

    def compute(self, activations_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for model_label, activations in activations_dict.items():
            result_dict[model_label] = []
            for activation in tqdm(activations, desc=f"Processing {model_label}"):
                result_dict[model_label].append(activation.compute(*args, **kwargs))
        return result_dict


class _BaseMetric:
    """
    Base class for metric computations.
    """

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class MultiRDM(_MultiMetric):
    def __init__(self, activations_dict: dict):
        self.activations_dict = activations_dict
        self.data = self._transform()

    def _transform(self):
        result_dict = {}
        for model_label, activations in self.activations_dict.items():
            result_dict[model_label] = []
            for activation in tqdm(activations, desc=f"Processing {model_label}"):
                result_dict[model_label].append(RDM(activation))
        return result_dict

    def compute(self, *args, **kwargs):
        return super().compute(activations_dict=self.data, *args, **kwargs)


class RDM(_BaseMetric):
    def __init__(
        self,
        activations: list,
    ):
        self.activations = activations

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

    def compute(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        dataset_label: str = "visual",
        metric: str = "correlation",
        bool_oracle: bool = "True",
    ):

        if isinstance(
            self.activations, (np.ndarray, torch.Tensor)
        ):  # if only one activation is passed instead of a list of arrays
            self.activations = [self.activations]

        idxs = discrete_binning(data=data, label=label, dataset_label=dataset_label)

        layer_rdm = []

        for layer in self.activations:
            # to ensure the right shape: numSamples X numNeurons
            if layer.shape[0] < layer.shape[1]:
                layer = layer.T

            rdm = pdist(layer[idxs.flatten(), :], metric=metric)
            if bool_oracle:
                oracle_rdm = self._create_oracle_rdm()
                correlation = self._compare_RDM(rdm_1=oracle_rdm, rdm_2=rdm)
            else:
                correlation = None

            layer_rdm.append((squareform(rdm), correlation))
        return layer_rdm
