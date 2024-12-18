"misc functions like normalization and possibly others"

import numpy as np


# Min-Max normalization function
def normalize_minmax(rdm):
    rdm_min = np.min(rdm)
    rdm_max = np.max(rdm)
    return (rdm - rdm_min) / (rdm_max - rdm_min)
