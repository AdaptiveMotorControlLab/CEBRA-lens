"All relevant functions for CKA analysis."

import numpy as np

def compute_single_CKA_layers(embeddings_1: list, embeddings_2: list) -> np.ndarray:
    """
    Compute the Centered Kernel Alignment (CKA) between two sets of embeddings for each layer.
    This function assumes that the two sets of embeddings have the same number of layers and that 
    each corresponding layer in the two sets has the same shape.
    Args:
      embeddings_1 (list): A list of embeddings for the first set. Each element in the list 
                 represents the embeddings for a specific layer.
      embeddings_2 (list): A list of embeddings for the second set. Each element in the list 
                 represents the embeddings for a specific layer.
    Returns:
      cka_matrix (np.ndarray): A one row array containing the CKA values for each layer.
    """

    # same number of layers and same shape across layers
    if len(embeddings_1) != len(embeddings_2):
      raise ValueError("The number of layers in embeddings_1 and embeddings_2 must be the same.")
    for i in range(len(embeddings_1)):
      if embeddings_1[i].shape != embeddings_2[i].shape:
        raise ValueError(f"The shape of layer {i} in embeddings_1 and embeddings_2 must be the same.")
    
    cka_matrix = np.zeros(
        (1, len(embeddings_1))
    ) 
    for i in range(len(embeddings_1)):
        cka_matrix[0, i] = cka(
                    gram_linear(embeddings_1[i].T),
                    gram_linear(embeddings_2[i].T),
                )
    return cka_matrix

def compute_multi_CKA_layers(activations_dict: dict, comparisons: list) -> list:
    """
    Compute multi-layer Centered Kernel Alignment (CKA) between different sets of activations.
    Args:
      activations_dict (dict): A dictionary where keys are strings in the format
                            'model_identifier_layer' and values are activations.
      comparisons (list): A list of tuples where each tuple contains two strings representing the models and layers 
                to be compared. For example, [('single_UT', 'single_TR'), ('single_TR', 'multi_TR')].
    Returns:
      list: A list of CKA matrices for each comparison. Each matrix is stored in a dictionary with keys formatted as 
          '{comparison[0]}_v_{comparison[1]}'. Example of key: 'single_UT_v_single_TR'.
    """
    # ex of comparisons : [('single_UT','single_TR'),('single_TR','multi_TR')] # make that if the number of models instantiated is not the same then it compares all to the 1st model
    # --> if len(single) != len(UT) use UT[0], single[i]
    # --> if = then just do multi[i], single[i]


    for i,comparison in enumerate(comparisons):

      if not isinstance(comparison, tuple):
            raise ValueError(f"Each comparison must be a tuple. Comparison {i} is of type: {type(comparison)}.")


  
      parts_1 = comparison[0].split('_')
      parts_2 = comparison[1].split('_')
      prefixes = (parts_1[0],parts_2[0])
      suffixes = (parts_1[1],parts_2[1])

      activations_1 = activations_dict[prefixes[0]][suffixes[0]]
      activations_2 = activations_dict[prefixes[1]][suffixes[1]]

      cka_matrices = {}

      # if not the same length, compare embeddings to the first instance. else compare pairwise.
      if  len(activations_1) != len(activations_2):
        if len(activations_1) > len(activations_2):
          embeddings_1 = activations_1
          embeddings_2 = activations_2[0] 

        elif len(activations_1) < len(activations_2):
          embeddings_1 = activations_2
          embeddings_2 = activations_1[0] 

        cka_matrix = np.zeros((len(embeddings_1),len(embeddings_1[0])))
        for j in range(len(embeddings_1)):
          cka_matrix[j,:] = compute_single_CKA_layers(embeddings_1[j],embeddings_2)
        

      else:
        cka_matrix = np.zeros((len(embeddings_1),len(embeddings_1[0])))
        for j in range(len(embeddings_1)):
          cka_matrix[j,:] = compute_single_CKA_layers(embeddings_1[j],embeddings_2[j])
        
      cka_matrices[f"{comparison[0]}_v_{comparison[1]}"] = cka_matrix





    return cka_matrices
    


################################################################
###### Taken from https://github.com/amathislab/DeepDraw #######
################################################################

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

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)
