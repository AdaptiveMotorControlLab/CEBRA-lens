import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.preprocessing import StandardScaler


def _compute_centroid(embedding: np.ndarray, indices: list) -> float:
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


def compute_centroids(
    embedding: np.ndarray, indices: list, metric: str = "cosine"
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
        embedding_scaled = scale_embedding(embedding, metric)
        bin_indices = indices[bin_idx, :]
        centroids.append(_compute_centroid(embedding_scaled, bin_indices))
    return centroids


def scale_embedding(embedding: np.ndarray, metric: str = "cosine") -> np.ndarray:
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
        return scaler.fit_transform(embedding.T).T  # Standardize across each dimension
    elif metric == "cosine":
        return embedding
    else:
        raise NotImplementedError(
            f"The scaling for metric {metric} is not yet implemented. Please use 'cosine' or 'euclidean'."
        )


# Function to compute centroids and inter-bin distances for a given embedding
def compute_interbin_distance(
    embedding: np.ndarray, indices: list, metric: str = "cosine"
) -> float:
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

    centroids = compute_centroids(embedding=embedding, indices=indices, metric=metric)

    # Compute pairwise distances between centroids using cosine distance
    distances = cdist(centroids, centroids, metric=metric)

    # Compute the mean inter-bin distance for each layer, excluding self-distances
    non_diagonal_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
    mean_distance = np.mean(non_diagonal_distances)

    return mean_distance


def compute_intrabin_distance(
    embedding: np.ndarray, indices: list, metric: str = "cosine"
) -> float:
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
    for bin_idx in range(indices.shape[0]):
        embedding_scaled = scale_embedding(embedding, metric)
        bin_indices = indices[bin_idx, :]
        bin_data = embedding_scaled[:, bin_indices.flatten()].T

        intra_distances = pdist(
            bin_data, metric=metric
        )  # Pairwise distances within the bin -> distances is list of x1x2,x1x3,x1x4...
        mean_intra_distance = np.mean(intra_distances)  # Mean of the pairwise distances
        distances.append(mean_intra_distance)

    return np.mean(distances)


def compute_interrep_distances(
    embedding: np.ndarray,
    indices: list,
    repetition_indices: list,
    metric: str = "cosine",
) -> float:
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
    for bin_idx in range(indices.shape[0]):
        repetition_centroids = []

        for i in range(len(repetition_indices[0])):

            rep_indices = repetition_indices[bin_idx][
                i
            ]  # Get indices for the current repetition
            embedding_scaled = scale_embedding(embedding, metric)
            repetition_centroids.append(
                _compute_centroid(embedding_scaled, rep_indices)
            )

        # Compute pairwise distances between centroids using cosine distance
        bin_distances = cdist(repetition_centroids, repetition_centroids, metric=metric)

        # Extract non-diagonal elements to get distances between different repetitions
        non_diagonal_distances = bin_distances[
            ~np.eye(bin_distances.shape[0], dtype=bool)
        ]
        mean_distance = np.mean(non_diagonal_distances)
        distances.append(mean_distance)

    return np.mean(distances)


def compute_distance(
    embedding: np.ndarray,
    indices: list,
    repetition_indices: list = None,
    metric: str = "cosine",
    distance_label: str = "interbin",
) -> float:
    """
    Computes a specified type of distance for the given embedding data and indices.

    Parameters:
    -----------
    embedding : np.ndarray
        The embedding data array.
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
    float
        The computed distance based on the specified label.
    """

    if distance_label == "interbin":
        distance = compute_interbin_distance(
            embedding=embedding, indices=indices, metric=metric
        )
    elif distance_label == "intrabin":
        distance = compute_intrabin_distance(
            embedding=embedding, indices=indices, metric=metric
        )
    elif distance_label == "interrep":
        distance = compute_interrep_distances(
            embedding=embedding,
            indices=indices,
            repetition_indices=repetition_indices,
            metric=metric,
        )
    else:
        raise NotImplementedError(
            f"Distance {distance_label} not yet implemented. Please use 'interbin','interrep' or 'intrabin'."
        )
    return distance


def compute_distance_layers(
    embeddings: list,
    indices: list,
    repetition_indices: list = None,
    metric: str = "cosine",
    distance_label: str = "interbin",
):
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

    layer_distances = []
    for embedding in embeddings:
        layer_distances.append(
            compute_distance(
                embedding=embedding,
                indices=indices,
                repetition_indices=repetition_indices,
                metric=metric,
                distance_label=distance_label,
            )
        )

    return layer_distances
