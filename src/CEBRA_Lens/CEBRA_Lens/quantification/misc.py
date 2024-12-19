"misc functions like normalization and possibly others"

import numpy as np


# Min-Max normalization function
def normalize_minmax(rdm: np.ndarray) -> np.ndarray:
    """
    Normalizes a given array using Min-Max normalization.

    Parameters:
    -----------
    rdm : np.ndarray
        A NumPy array to be normalized. This can be any numeric array, such as an RDM (Representational
        Dissimilarity Matrix), where values are normalized to the range [0, 1].

    Returns:
    --------
    np.ndarray
        A normalized NumPy array where the minimum value is scaled to 0 and the maximum value is scaled to 1.
    """

    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)
