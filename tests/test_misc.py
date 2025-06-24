import numpy as np
import pytest
import torch

from cebra_lens.quantification.misc import (continuous_binning,
                                            discrete_binning,
                                            repetition_binning)


def test_discrete_binning_shape_and_values():
    labels = np.array([0, 0, 1, 1, 2, 2])
    idxs = discrete_binning(labels)
    assert isinstance(idxs, np.ndarray)
    assert idxs.shape[0] == 3  # three unique labels
    assert all(len(set(row)) == len(row)
               for row in idxs)  # no duplicates in each bin


def test_continuous_binning_general_continuous():
    data = torch.rand((10000, 10))
    label = torch.linspace(0, 1, 10000)
    idxs, bins = continuous_binning(data, label, dataset_label=None)
    assert isinstance(idxs, np.ndarray)
    assert isinstance(bins, int)
    assert idxs.shape[1] > 0


def test_repetition_binning_invalid_dataset():
    with pytest.raises(NotImplementedError):
        repetition_binning(np.zeros((3, 90), dtype=int),
                           np.random.rand(900, 10),
                           dataset_label="HPC")
