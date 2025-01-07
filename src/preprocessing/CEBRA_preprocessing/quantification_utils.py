
import numpy as np

import cebra
import cebra.datasets
import copy
import torch
import sklearn.metrics
from scipy.spatial.distance import cosine, correlation,cdist,pdist, squareform
from sklearn.preprocessing import StandardScaler

# Save this in a file called `cka_computation.py`

import numpy as np
from tqdm import tqdm


# Min-Max normalization function
def normalize_minmax(rdm):
    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)



# Function to compute centroids and inter-bin distances for a given embeddings list
def compute_inter_bin_distances(embeddings_list,idxs,num_bins,metric = 'cosine'):
    centroids_all = []
    for i in range(len(embeddings_list)):
        layer_centroids = []
        for bin_idx in range(num_bins):
            if metric == 'euclidean':
                    scaler = StandardScaler()
                    emb = scaler.fit_transform(embeddings_list[i].T).T # standardize across each dimension

            else: emb = embeddings_list[i]

            bin_indices = idxs[bin_idx, :]  # Indices for the current bin
            bin_data = emb[:, bin_indices.flatten()].T  # Get data for the current bin
            centroid = np.mean(bin_data, axis=0)  # Compute centroid
            layer_centroids.append(centroid)
        centroids_all.append(np.array(layer_centroids))

    # Step 2: Compute Inter-Bin Distances using Cosine Distance
    inter_bin_distances_all = []
    for layer_centroids in centroids_all:
        # Compute pairwise distances between centroids using cosine distance
        distances = cdist(layer_centroids, layer_centroids, metric=metric)
        inter_bin_distances_all.append(distances)

    # Compute the mean inter-bin distance for each layer, excluding self-distances
    mean_inter_bin_distances_all = []
    for distances in inter_bin_distances_all:
        # Extract non-diagonal elements
        non_diagonal_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
        mean_distance = np.mean(non_diagonal_distances)
        mean_inter_bin_distances_all.append(mean_distance)

    return mean_inter_bin_distances_all

def compute_intra_bin_distances(embeddings_list,idxs,num_bins,metric = 'cosine'):
    intra_bin_distances_all = []
    for i in range(len(embeddings_list)):
        
        # one list per layer
        layer_intra_distances = []

        # for each bin compute the mean distance
        for bin_idx in range(num_bins):
            if metric == 'euclidean':
                    scaler = StandardScaler()
                    emb = scaler.fit_transform(embeddings_list[i].T).T # standardize across each dimension

            else: emb = embeddings_list[i]

            bin_indices = idxs[bin_idx, :]  
            bin_data = emb[:, bin_indices.flatten()].T 
            distances = pdist(bin_data, metric=metric)  # Pairwise distances within the bin -> distances is list of x1x2,x1x3,x1x4...
            mean_intra_distance = np.mean(distances)  # Mean of the pairwise distances
            layer_intra_distances.append(mean_intra_distance)
        
        # a list of arrays per layers
        intra_bin_distances_all.append(np.array(layer_intra_distances))
    
    # Compute the mean intra-bin distance for each layer
    mean_intra_bin_distances_all = []
    std_intra_bin_distances_all = []

    for layer_intra_distances in intra_bin_distances_all:
        mean_intra_distance = np.mean(layer_intra_distances)
        mean_intra_bin_distances_all.append(mean_intra_distance)
        std_intra_bin_distances_all.append(np.std(layer_intra_distances))
    
    return mean_intra_bin_distances_all,std_intra_bin_distances_all


# Function to compute mean repetition distances for a given embeddings list
def compute_mean_repetition_distances(embeddings_list, idxs,num_classes,num_repetitions,metric = 'cosine'):
    mean_repetition_distances_all = []
    
    for i in range(len(embeddings_list)):
        if metric == 'euclidean':
                scaler = StandardScaler()
                emb = scaler.fit_transform(embeddings_list[i].T).T # standardize across each dimension

        else: emb = embeddings_list[i]
        
        layer_mean_distances = []
        
        for class_idx in range(num_classes):
            print(class_idx)
            repetition_centroids = []
            
            for rep in range(num_repetitions):
                print(rep*900,rep*900+30)
                rep_indices = idxs[class_idx][rep*30:(rep+1)*30]  # Get indices for the current repetition
                print(rep_indices)
                rep_data = emb[:, rep_indices].T  # Get data for the current repetition
                centroid = np.mean(rep_data, axis=0)  # Compute centroid
                repetition_centroids.append(centroid)
            
            # Compute pairwise distances between centroids using cosine distance
            distances = cdist(repetition_centroids, repetition_centroids, metric=metric)
            
            # Extract non-diagonal elements to get distances between different repetitions
            non_diagonal_distances = distances[~np.eye(distances.shape[0], dtype=bool)]
            mean_distance = np.mean(non_diagonal_distances)
            layer_mean_distances.append(mean_distance)
        
        mean_repetition_distances_all.append(np.mean(layer_mean_distances))
    
    return mean_repetition_distances_all


# from https://cebra.ai/docs/demo_notebooks/Demo_decoding.html#
def decoding_pos_dir(embedding_train, embedding_test, label_train, label_test):
   pos_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")
   dir_decoder = cebra.KNNDecoder(n_neighbors=36, metric="cosine")

   pos_decoder.fit(embedding_train, label_train[:,0])
   dir_decoder.fit(embedding_train, label_train[:,1])

   pos_pred = pos_decoder.predict(embedding_test)
   dir_pred = dir_decoder.predict(embedding_test)

   prediction = np.stack([pos_pred, dir_pred],axis = 1)

   test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)
   pos_test_err = np.median(abs(prediction[:,0] - label_test[:, 0]))
   pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])

   return test_score, pos_test_err, pos_test_score

def decoding_models(models_untrained, models_single, models_multi, X, y, discrete_labels_val, valid_datas, decoding_frames):
    """
    Processes untrained, single, and multi models to obtain their decoding results.
    
    Parameters:
    - models_untrained: List of untrained models.
    - models_single: List of single models.
    - models_multi: List of multi models.
    - X: Training data for the models.
    - y: Labels for the training data.
    - discrete_labels_val: Validation labels for testing.
    - valid_datas: Validation data for testing.
    - decoding_frames: Function used to decode the frames.
    
    Returns:
    - results_untrained: Decoding results for untrained models.
    - results_single: Decoding results for single models.
    - results_multi: Decoding results for multi models.
    """
    
    # Initialize results arrays
    results_untrained = np.zeros((len(models_untrained), 3))
    results_single = np.zeros((len(models_single), 3))
    results_multi = np.zeros((len(models_multi), 3))
    
    # Labels and data
    label_train = y
    label_test = discrete_labels_val[3]
    data_train = X
    data_test = valid_datas[3].neural
    
    # UNTRAINED models
    for i, model in enumerate(models_untrained):
        if i == 1:  # multi-session, need to add session_id
            train = model.transform(data_train, session_id=3)
            test = model.transform(data_test, session_id=3)
        else:
            train = model.transform(data_train)
            test = model.transform(data_test)
        
        results_untrained[i] = decoding_frames(embedding_train=train, label_train=label_train, embedding_test=test, label_test=label_test)
    
    # SINGLE models
    for i, model in enumerate(models_single):
        train = model.transform(X)
        test = model.transform(data_test)
        results_single[i] = decoding_frames(embedding_train=train, label_train=label_train, embedding_test=test, label_test=label_test)
    
    # MULTI models
    for i, model in enumerate(models_multi):
        train = model.transform(data_train, session_id=3)
        test = model.transform(data_test, session_id=3)
        results_multi[i] = decoding_frames(embedding_train=train, label_train=label_train, embedding_test=test, label_test=label_test)
    
    return results_untrained, results_single, results_multi


########################################################################################################################
########################################################################################################################
######################################## TAKEN FROM https://github.com/amathislab/DeepDraw ############################################
########################################################################################################################
########################################################################################################################


def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
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
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
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


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


########################################################################################################################
########################################################################################################################
######################################## TAKEN FROM utils_allen.py of Célia ############################################
########################################################################################################################
########################################################################################################################


def _quantize_acc(frame_diff, time_window=1):

    true = (abs(frame_diff) < (time_window * 30)).sum()
    return true / len(frame_diff) * 100


def create_sequences(embedding, labels, seq_len=10):
    sequences = []
    sequence_labels = []
    for i in range(len(embedding) - seq_len):
        seq = embedding[i : i + seq_len]
        # Label is the frame number following the sequence
        label = labels[i + seq_len]
        sequences.append(seq)
        sequence_labels.append(label)
    return np.array(sequences), np.array(sequence_labels)


def decoding_frames(
    embedding_train, embedding_test, label_train, label_test, time_window=1, seq_len=1
):
    """ 1-frame decoding.
    
    TODO(celia): Implement n-frames decoding. Started but not functional yet.
    """
    if seq_len > 1:
        embedding_train, label_train = create_sequences(
            embedding_train, label_train, seq_len
        )
        embedding_test, label_test = create_sequences(
            embedding_test, label_test, seq_len
        )

    params = np.power(np.linspace(1, 10, 10, dtype=int), 2)
    errs = []
    for n in params:
        train_decoder = cebra.KNNDecoder(n_neighbors=n, metric="cosine")
        train_valid_idx = int(len(embedding_train) / 9 * 8)
        if seq_len > 1:
            train_decoder.fit(
                embedding_train[:train_valid_idx].reshape(
                    -1, seq_len * embedding_train.shape[2]
                ),
                label_train[:train_valid_idx],
            )
            pred = train_decoder.predict(
                embedding_train[train_valid_idx:].reshape(
                    -1, seq_len * embedding_train.shape[2]
                )
            )
        else:
            train_decoder.fit(
                embedding_train[:train_valid_idx], label_train[:train_valid_idx]
            )
            pred = train_decoder.predict(embedding_train[train_valid_idx:])
        err = label_train[train_valid_idx:] - pred
        errs.append(abs(err).sum())

    test_decoder = cebra.KNNDecoder(
        n_neighbors=params[np.argmin(errs)], metric="cosine"
    )
    if seq_len > 1:
        test_decoder.fit(
            embedding_train.reshape(-1, seq_len * embedding_train.shape[2]), label_train
        )
        frame_pred = test_decoder.predict(
            embedding_test.reshape(-1, seq_len * embedding_test.shape[2])
        )
    else:
        test_decoder.fit(embedding_train, label_train)
        frame_pred = test_decoder.predict(embedding_test)

    frame_errors = frame_pred - label_test
    test_score = sklearn.metrics.r2_score(label_test, frame_pred)
    test_err = np.median(abs(frame_errors))
    test_acc = _quantize_acc(frame_errors, time_window=1)

    return test_score, test_err, test_acc
