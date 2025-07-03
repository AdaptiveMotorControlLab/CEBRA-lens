from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import cebra_lens


@pytest.fixture
def dummy_data():
    return torch.tensor(np.random.rand(10000, 10), dtype=torch.float32)


@pytest.fixture
def dummy_labels():
    return torch.tensor(np.random.randint(0, 5, size=10000), dtype=torch.int64)


@patch("cebra_lens.quantification.rdm_metric.continuous_binning")
def test_define_indices_continuous(mock_binning, dummy_data, dummy_labels):
    mock_binning.return_value = (np.array([[0, 1], [2, 3]]), 2)
    rdm = cebra_lens.quantification.rdm_metric.RDM(data=dummy_data,
                                                   label=dummy_labels,
                                                   dataset_label="visual")
    idxs, bins = rdm._define_indices()
    assert isinstance(idxs, np.ndarray)
    assert bins == 2


@patch("cebra_lens.quantification.rdm_metric.discrete_binning")
def test_define_indices_discrete(mock_binning, dummy_data):
    labels = torch.tensor([0, 1, 0, 1, 2])
    mock_binning.return_value = np.array([[0, 2], [1, 3]])
    rdm = cebra_lens.quantification.rdm_metric.RDM(data=dummy_data,
                                                   label=labels,
                                                   is_discrete_labels=True)
    idxs, bins = rdm._define_indices()
    assert isinstance(idxs, np.ndarray)
    assert bins is None


def test_init_with_label_ind():
    labels = np.array([[1, 2], [3, 4]])
    data = torch.tensor(np.random.rand(2, 5), dtype=torch.float32)
    rdm = cebra_lens.quantification.rdm_metric.RDM(data=data,
                                                   label=labels[:, 0],
                                                   dataset_label=None,
                                                   is_discrete_labels=True)
    assert rdm.label.tolist() == [1, 3]

    with pytest.raises(ValueError):
        cebra_lens.quantification.rdm_metric.RDM(data=data,
                                                 label=labels,
                                                 dataset_label=None,
                                                 is_discrete_labels=True)


def test_create_oracle_rdm_custom():
    rdm = cebra_lens.quantification.rdm_metric.RDM(
        data=torch.rand((10000, 5)),
        label=torch.randint(0, 5, (10000, )),
        dataset_label=None,
        is_discrete_labels=True,
    )
    rdm.idxs = np.array([[0, 1], [2, 3]])
    oracle = rdm._create_oracle_rdm()
    assert isinstance(oracle, np.ndarray)


def test_compute_per_layer_and_bool_oracle():
    rdm = cebra_lens.quantification.rdm_metric.RDM(
        data=torch.rand((10000, 5)),
        label=torch.randint(0, 5, (10000, )),
        is_discrete_labels=True,
    )
    rdm.idxs = np.array([[i] for i in range(10000)])
    dummy_layer = np.random.rand(10000, 5)
    _, corr = rdm._compute_per_layer(dummy_layer, bool_oracle=True)
    assert isinstance(corr, float)


def test_compute_single_activation_tensor():
    data = torch.rand((10000, 5))
    label = torch.randint(0, 3, (10000, ))
    rdm = cebra_lens.quantification.rdm_metric.RDM(data=data,
                                                   label=label,
                                                   is_discrete_labels=False)
    rdm.idxs = np.array([[i] for i in range(10000)])
    act = torch.rand((10000, 5)).numpy()
    result = rdm.compute(act)
    assert isinstance(result, list)
    assert isinstance(result[0], tuple)
    assert result[0][0].ndim == 2  # RDM squareform


@patch("cebra_lens.utils_plot.plot_rdm_correlation")
@patch("cebra_lens.utils_plot.plot_rdm_all")
def test_rdm_plot(mock_all, mock_corr):
    dummy_rdm = cebra_lens.quantification.rdm_metric.RDM(
        data=torch.rand((10000, 5)),
        label=torch.randint(0, 5, (10000, )),
        is_discrete_labels=False,
    )
    dummy_rdm.plot({"group": [np.random.rand(5, 5)]}, bool_oracle=True)
    assert mock_corr.called

    dummy_rdm.num_bins = 2
    dummy_rdm.is_discrete_labels = True
    dummy_rdm.plot({"group": [np.random.rand(5, 5)]}, bool_oracle=False)
    assert mock_all.called

    dummy_rdm.plot({"group": [np.random.rand(5, 5)]}, titles=None)
    dummy_rdm.plot({})


def test_define_indices_raises_valueerror_for_invalid_dataset_label():
    data = torch.tensor(np.random.rand(10, 5), dtype=torch.float32)
    labels = torch.tensor(np.random.randint(0, 5, size=10), dtype=torch.int64)
    with pytest.raises(ValueError):
        cebra_lens.quantification.rdm_metric.RDM(
            data=data,
            label=labels,
            is_discrete_labels=True,
            dataset_label="invalid_label",
        )


def test_compute_per_layer_transposes_if_needed():
    rdm = cebra_lens.quantification.rdm_metric.RDM(
        data=torch.rand((10, 5)),
        label=torch.randint(0, 5, (10, )),
        is_discrete_labels=True,
    )
    rdm.idxs = np.array([[i] for i in range(10)])
    # Provide activation with shape (features, samples)
    dummy_layer = np.random.rand(5, 10)
    rdm._compute_per_layer(dummy_layer)
