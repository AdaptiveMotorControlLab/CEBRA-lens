"""Functions to transform data e.g. tSNE, other functions can be added"""

from sklearn.manifold import TSNE
from tqdm import tqdm
import numpy as np
import pickle


def apply_tsne(layer_activation: np.ndarray, num_samples: int) -> np.ndarray:
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

    # Check that it's num_neurons X num_samples: Assumption that we always have num_neurons < num_samples
    if layer_activation.shape[0] > layer_activation.shape[1]:
        layer_activation = layer_activation.T

    tsne = TSNE(n_components=3)
    tsne_embedding = tsne.fit_transform(layer_activation[:, :num_samples].T)
    return tsne_embedding


def run_tsne_and_save(
    activations_dict: dict, filepath: str, num_samples: int = 200
) -> dict:
    """
    Runs t-SNE on the provided activations dictionary and saves the results to a pickle file.
    This function performs t-SNE on the activations data and stores the resulting embeddings in a pickle file for later use.

    Parameters:
    -----------
    activations_dict : dict
        A dictionary containing the activations. More information on the format can be found in CEBRA_Lens.activations.
    filepath : str
        The path where the t-SNE embeddings will be saved as a pickle file.
    num_samples : int
        The number of samples to use for t-SNE transformation.

    Returns:
    --------
    tsne_embeddings : dict
        A dictionary containing the t-SNE embeddings, structured exactly the same as the input `activations_dict`.
    """

    if num_samples < 200:
        print(
            f"Warning: Minimum number of samples is 200 to ensure good functioning. Provided: {num_samples}. Processing with 200..."
        )
        num_samples = 200

    tsne_embeddings = {}
    for outer_key, outer_value in activations_dict.items():
        tsne_embeddings[outer_key] = {}
        for inner_key, outer_list in tqdm(
            outer_value.items(), desc=f"Processing {outer_key}"
        ):
            tsne_embeddings[outer_key][inner_key] = []
            for inner_list in tqdm(
                outer_list, desc=f"Processing {outer_key} {inner_key}"
            ):
                processed_inner_list = [
                    apply_tsne(arr, num_samples=num_samples) for arr in inner_list
                ]
                tsne_embeddings[outer_key][inner_key].append(processed_inner_list)

    with open(filepath, "wb") as f:
        pickle.dump(tsne_embeddings, f)

    print("DONE")
    return tsne_embeddings
