"file containing all the functions relative to distance computing"

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from .misc import discrete_binning, repetition_binning
from .base import _BaseMetric


class DistanceMetric:

    def compute_centroid(self, embedding: np.ndarray, indices: list) -> float:
        """
        Computes the centroid of a single embedding (e.g. single layer) for specified bin indices.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        indices : list
            A list of indices specifying the bin data to compute the centroid.

        Returns:
        --------
        float
            The computed centroid value.
        """
        bin_data = embedding[:, indices.flatten()]  # Get data for the current bin
        return np.mean(bin_data, axis=1)  # Compute centroid

    def scale_embedding(
        self, embedding: np.ndarray, metric: str = "cosine"
    ) -> np.ndarray:
        """
        Scales the embedding data based on the specified metric.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        metric : str
            The distance metric to use for scaling the embedding, either "cosine" or "euclidean".

        Returns:
        --------
        np.ndarray
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
        self, embedding: np.ndarray, indices: list, metric: str = "cosine"
    ) -> list:
        """
        Computes the centroid of a single embedding (e.g. single layer) for all the bins.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        indices : list
            A list of indices specifying the bins to compute the centroids.
        metric : str, optional
            The distance metric to use for scaling the embedding (default is "cosine").

        Returns:
        --------
        list
            A list of computed centroid values.
        """

        centroids = []
        for bin_idx in range(indices.shape[0]):
            embedding_scaled = self.scale_embedding(embedding, metric)
            bin_indices = indices[bin_idx, :]
            centroids.append(self.compute_centroid(embedding_scaled, bin_indices))
        return centroids


class Intrabin(DistanceMetric):
    def __init__(self, indices, metric):
        self.indices = indices
        self.metric = metric

    def _compute_distance(self, embedding: np.ndarray) -> float:
        """
        Computes the mean intra-bin distance for the given embedding data and indices.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        indices : list
            A list of indices specifying the bins.
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").

        Returns:
        --------
        float
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


class Interrep(DistanceMetric):
    def __init__(self, indices, repetition_indices, metric):
        self.indices = indices
        self.repetition_indices = repetition_indices
        self.metric = metric

    def _compute_distance(self, embedding: np.ndarray) -> float:
        """
        Computes the mean distance between different repetitions for the given embedding data, indices, and repetition indices.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        indices : list
            A list of indices specifying the bins.
        repetition_indices : list
            A list of lists specifying the repetition indices.
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").

        Returns:
        --------
        float
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


class Interbin(DistanceMetric):
    def __init__(self, indices, repetition_indices, metric):
        self.indices = indices
        self.metric = metric

    # Function to compute centroids and inter-bin distances for a given embedding
    def _compute_distance(self, embedding: np.ndarray) -> float:
        """
        Computes the mean inter-bin distance for the given embedding data (e.g. single layer) and indices.

        Parameters:
        -----------
        embedding : np.ndarray
            The embedding data array of shape Neurons X Samples.
        indices : list
            A list of indices specifying the bins.
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").

        Returns:
        --------
        float
            The mean inter-bin distance across the embedding (e.g. across one layer).
        """

        centroids = self.compute_centroids(
            embedding=embedding, indices=self.indices, metric=self.metric
        )

        # Compute pairwise distances between centroids using cosine distance
        distances = cdist(centroids, centroids, metric=self.metric)

        # Compute the mean inter-bin distance for each layer, excluding self-distances
        non_diagonal_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
        mean_distance = np.mean(non_diagonal_distances)

        return mean_distance


class Distance(_BaseMetric):
    def __init__(self,data, label, dataset_label,
        metric: str = "cosine",
        distance_label: str = "interbin"):

        super().__init__(self)
        self.data = data
        self.label = label
        self.dataset_label = dataset_label
        self.metric = metric
        self.distance_label = distance_label

        self.indices, self.repetition_indices = self._define_indices()
    
    def _define_indices(self):

        idxs = discrete_binning(
            data=self.data,
            label=self.label,
            dataset_label=self.dataset_label,
            sample_mode="all",
        )
        if self.distance_label == "interrep":

            repetition_indices = repetition_binning(
                indices=idxs, data=self.data, dataset_label=self.dataset_label
            )
        else:
            repetition_indices = None
        
        return idxs, repetition_indices
        

    def compute(
        self, activations
    ) -> list:
        """
        Computes specified type of distance for multiple layers of embedding data.

        Parameters:
        -----------
        embeddings : list
            A list of embedding data arrays for different layers.
        indices : list
            A list of indices specifying the bins.
        repetition_indices : list, optional
            A list of lists specifying the repetition indices (default is None).
        metric : str, optional
            The distance metric to use for computing distances (default is "cosine").
        distance_label : str, optional
            The type of distance to compute ("interbin", "intrabin", or "interrep") (default is "interbin").

        Returns:
        --------
        list
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
    
