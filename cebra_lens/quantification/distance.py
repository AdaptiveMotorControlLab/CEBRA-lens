"file containing all the functions relative to distance computing"

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple, Union, Dict
from .misc import discrete_binning, repetition_binning, continuous_binning
from .base import _BaseMetric
from ..matplotlib import *
import numpy.typing as npt
from ..utils import extract_label


class DistanceMetric:
    """
    Base class for distance metrics.
    This class provides methods to compute distances between embeddings and centroids.
    """

    def compute_centroid(
        self, embedding: npt.NDArray, indices: List[np.int64]
    ) -> np.float64:
        """
        Computes the centroid of a single embedding (e.g. single layer) for specified bin indices.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.
        indices : List[np.int64]
            A list of indices specifying the bin data to compute the centroid.

        Returns:
        --------
        np.float64
            The computed centroid value.
        """
        bin_data = embedding[:, indices.flatten()]  # Get data for the current bin
        return np.mean(bin_data, axis=1)  # Compute centroid

    def scale_embedding(
        self, embedding: npt.NDArray, metric: str = "cosine"
    ) -> npt.NDArray:
        """
        Scales the embedding data based on the specified metric.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.
        metric : str
            The distance metric to use for scaling the embedding, either "cosine" or "euclidean".

        Returns:
        --------
        npt.NDArray
            The scaled embedding data.
        """

        if metric == "euclidean":
            scaler = StandardScaler()
            return scaler.fit_transform(
                embedding.T
            ).T  # Standardize across each dimension
        elif metric == "cosine":
            return embedding
        else:
            raise NotImplementedError(
                f"The scaling for metric {metric} is not yet implemented. Please use 'cosine' or 'euclidean'."
            )

    def compute_centroids(
        self, embedding: npt.NDArray, indices: List[np.float64], metric: str = "cosine"
    ) -> List[np.float64]:
        """
        Computes the centroid of a single embedding (e.g. single layer) for all the bins.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.
        indices : List[np.float64]
            A list of indices specifying the bins to compute the centroids.
        metric : str, optional
            The distance metric to use for scaling the embedding (default is "cosine").

        Returns:
        --------
        list : List[np.float64]
            A list of computed centroid values.
        """

        centroids = []
        for bin_idx in range(indices.shape[0]):
            embedding_scaled = self.scale_embedding(embedding, metric)
            bin_indices = indices[bin_idx, :]
            centroids.append(self.compute_centroid(embedding_scaled, bin_indices))
        return centroids


class Intrabin(DistanceMetric):
    """
    Class to compute intra-bin distances for a given embedding data and indices.

    Parameters:
    -----------
    indices : List[np.int64]
        A list of indices specifying the bins.
    metric : str, optional
        The distance metric to use for computing distances (default is "cosine").
    """

    def __init__(self, indices: List[np.int64], metric: Optional[str] = "cosine"):
        self.indices = indices
        self.metric = metric

    def _compute_distance(self, embedding: npt.NDArray) -> np.float64:
        """
        Computes the mean intra-bin distance for the given embedding data and indices.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.

        Returns:
        --------
        np.float64
            The mean intra-bin distance.
        """

        distances = []
        for bin_idx in range(self.indices.shape[0]):
            embedding_scaled = self.scale_embedding(embedding, self.metric)
            bin_indices = self.indices[bin_idx, :]
            bin_data = embedding_scaled[:, bin_indices.flatten()].T

            intra_distances = pdist(
                bin_data, metric=self.metric
            )  # Pairwise distances within the bin -> distances is list of x1x2,x1x3,x1x4...
            mean_intra_distance = np.mean(
                intra_distances
            )  # Mean of the pairwise distances
            distances.append(mean_intra_distance)

        return np.mean(distances)

    def plot(
        self,
        distance_dict: Dict[str, npt.NDArray],
        title: str = "Intra-bin distance",
        figsize: tuple = (15, 5),
    )-> matplotlib.figure.Figure:
        """
        Plots the intra-bin distances.
        
        Parameters:
        -----------
        distance_dict : Dict[str, npt.NDArray]
            A dictionary containing the distances for each layer.
        title : str, optional
            The title of the plot (default is "Intra-bin distance").
        figsize : tuple, optional
            The size of the figure for the plot (default is (15, 5)).

        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        return super().plot(distance_dict, title)


class Interrep(DistanceMetric):
    """
    Class to compute inter-repetition distances for a given embedding data, indices, and repetition indices.

    Parameters:
    -----------
    indices : List[np.int64]
        A list of indices specifying the bins.
    repetition_indices : List[np.int64]
        A list of lists specifying the repetition indices.
    metric : str, optional
        The distance metric to use for computing distances (default is "cosine").

    """

    def __init__(
        self,
        indices: List[np.int64],
        repetition_indices: List[np.int64],
        metric: Optional[str] = "cosine",
    ):
        self.indices = indices
        self.repetition_indices = repetition_indices
        self.metric = metric

    def _compute_distance(self, embedding: npt.NDArray) -> np.float64:
        """
        Computes the mean distance between different repetitions for the given embedding data, indices, and repetition indices.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.

        Returns:
        --------
        np.float64
            The mean distance between different repetitions.
        """

        distances = []
        for bin_idx in range(self.indices.shape[0]):
            repetition_centroids = []

            for i in range(len(self.repetition_indices[0])):

                rep_indices = self.repetition_indices[bin_idx][
                    i
                ]  # Get indices for the current repetition
                embedding_scaled = self.scale_embedding(embedding, self.metric)
                repetition_centroids.append(
                    self.compute_centroid(embedding_scaled, rep_indices)
                )

            # Compute pairwise distances between centroids using cosine distance
            bin_distances = cdist(
                repetition_centroids, repetition_centroids, metric=self.metric
            )

            # Extract non-diagonal elements to get distances between different repetitions
            non_diagonal_distances = bin_distances[
                ~np.eye(bin_distances.shape[0], dtype=bool)
            ]
            mean_distance = np.mean(non_diagonal_distances)
            distances.append(mean_distance)

        return np.mean(distances)

    def plot(
        self,
        distance_dict: Dict[str, npt.NDArray],
        title: str = "Inter-repetition distance",
        figsize: tuple = (15, 5),
    )-> matplotlib.figure.Figure:
        """
        Plots the inter-repetition distances.
        
        Parameters:
        -----------
        distance_dict : Dict[str, npt.NDArray]
            A dictionary containing the distances for each layer.
        title : str, optional
            The title of the plot (default is "Inter-repetition distance").
        figsize : tuple, optional
            The size of the figure for the plot (default is (15, 5)).
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        return super().plot(distance_dict, title)


class Interbin(DistanceMetric):
    """
    Class to compute inter-bin distances for a given embedding data and indices.

    Parameters:
        indices : List[np.int64]
            A list of indices specifying the bins.
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").
    """

    def __init__(self, indices: List[np.int64], metric: Optional[str] = "cosine"):
        self.indices = indices
        self.metric = metric

    # Function to compute centroids and inter-bin distances for a given embedding
    def _compute_distance(self, embedding: npt.NDArray) -> np.float64:
        """
        Computes the mean inter-bin distance for the given embedding data (e.g. single layer) and indices.

        Parameters:
        -----------
        embedding : npt.NDArray
            The embedding data array of shape Neurons X Samples.

        Returns:
        --------
        np.float64
            The mean inter-bin distance across the embedding (e.g. across one layer).
        """

        centroids = self.compute_centroids(
            embedding=embedding, indices=self.indices, metric=self.metric
        )

        # Compute pairwise distances between centroids using metric
        distances = cdist(centroids, centroids, metric=self.metric)

        # Compute the mean inter-bin distance for each layer, excluding self-distances
        non_diagonal_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
        mean_distance = np.mean(non_diagonal_distances)

        return mean_distance

    def plot(
        self,
        distance_dict: Dict[str, npt.NDArray],
        title: str = "Inter-bin distance",
        figsize: tuple = (15, 5),
    )-> matplotlib.figure.Figure:
        """
        Plots the inter-bin distances.
        
        Parameters:
        -----------
        distance_dict : Dict[str, npt.NDArray]
            A dictionary containing the distances for each layer.
        title : str, optional
            The title of the plot (default is "Inter-bin distance").
        figsize : tuple, optional
            The size of the figure for the plot (default is (15, 5)).
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        return super().plot(distance_dict, title)


class Distance(_BaseMetric):
    """
    A Base class to compute distances between embeddings and centroids.

    Parameters:
    -----------
        data : torch.Tensor
            The data array of shape (num_samples, num_features).
        label : torch.Tensor
            The array of labels corresponding to the data.
        label_ind : int, optional
            The index of the label to extract from the array (default is 0). This is relevant when dataset_label is None.
        discrete : bool, optional
            Specifies whether the given label is discrete or continuous. This is relevant when dataset_label is None.
        dataset_label : str, optional
            The dataset type, either 'visual' or 'HPC'. Default is 'visual'.
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").
        distance_label : str, optional
            The type of distance to compute (default is "interbin").
    """

    def __init__(
        self,
        data,
        label,
        label_ind: int = 0,
        discrete: bool = None,
        dataset_label: str = None,
        metric: str = "cosine",
        distance_label: str = "interbin",
    ):

        super().__init__()
        self.data = data
        self.label = label
        self.dataset_label = dataset_label
        if self.dataset_label is None:
            if label_ind is None:
                raise ValueError(
                    "If dataset_label is None, label_ind must be provided to indicate which label will be used for the distance calculation."
                )
            if discrete is None:
                raise ValueError(
                    "If dataset_label is None, discrete must be specified to indicate whether the label is discrete or continuous."
                )
            self.label = extract_label(label, label_ind)
        self.metric = metric
        self.distance_label = distance_label

        self.indices, self.repetition_indices = self._define_indices(discrete)

    def _define_indices(
        self, discrete: bool = None
    ) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
        """
        Defines the how the labels are binned and the indices for each bin.

        Parameters:
        -----------
        discrete : bool, optional
            Specifies whether the given label is discrete or continuous. This is relevant when dataset_label is None.

        Returns:
        --------
        Tuple[npt.NDArray, Optional[npt.NDArray]]
            A tuple containing the indices for each bin and the repetition indices if applicable.
        """

        if self.dataset_label is not None:
            if self.dataset_label not in ["visual", "HPC"]:
                raise ValueError(
                    f"Dataset label {self.dataset_label} is not supported. Please use 'visual' or 'HPC' or None for general binning."
                )
            else:
                idxs = continuous_binning(
                    data=self.data,
                    label=self.label,
                    dataset_label=self.dataset_label,
                    sample_mode="all",
                )[0]
        else:

            if discrete is None:
                raise ValueError(
                    "The 'discrete' parameter must be specified.This parameter specifies whether the given label is discrete or continuous."
                )

            if discrete:
                # just detect the unique values and find the indices of the bins (each bin is a unique value)
                # dataset_label is None and discrete is True
                idxs = discrete_binning(
                    label=self.label,
                )
            else:
                # dataset_label is HPC or visual/ discrete is False (dataset_label is None)
                idxs = continuous_binning(
                    data=self.data,
                    label=self.label,
                    dataset_label=self.dataset_label,
                    sample_mode="all",
                )[0]

        if self.distance_label == "interrep":
            # only relevant for visual dataset
            repetition_indices = repetition_binning(
                indices=idxs, data=self.data, dataset_label=self.dataset_label
            )
        else:
            repetition_indices = None

        return idxs, repetition_indices

    def compute(
        self, activations: List[Union[np.float64, npt.NDArray]]
    ) -> List[np.float64]:
        """
        Computes specified type of distance for multiple layers of embedding data.

        Parameters:
        -----------
        activations : List[Union[np.float64, npt.NDArray]]
            List of 2D numpy array representing the activation of neurons per layer.

        Returns:
        --------
        List[np.float64]
            A list of computed distances for each layer.
        """
        if self.distance_label == "interbin":
            distance = Interbin(self.indices, self.metric)
        elif self.distance_label == "intrabin":
            distance = Intrabin(self.indices, self.metric)
        elif self.distance_label == "interrep":
            distance = Interrep(self.indices, self.repetition_indices, self.metric)
        else:
            raise NotImplementedError(
                f"Distance {self.distance_label} not yet implemented. Please use 'interbin','interrep' or 'intrabin'."
            )

        return super().iterate_over_layers(activations, distance._compute_distance)

    @property
    def __name__(self):
        return self.distance_label

    def plot(
        self,
        distance_dict: Dict[str, npt.NDArray],
        title: str = None,
        figsize: tuple = (15, 5),
    )-> matplotlib.figure.Figure:
        """
        Plots the computed distances.
        
        Parameters:
        -----------
        distance_dict : Dict[str, npt.NDArray]
            A dictionary containing the distances for each layer.
        title : str, optional
            The title of the plot (default is None, which will use the distance type).
        figsize : tuple, optional
            The size of the figure for the plot (default is (15, 5)).
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object containing the plot.
        """
        y_axis = f"{self.metric} distance"
        return plot_distance(distance_dict, title, figsize, y_axis)
