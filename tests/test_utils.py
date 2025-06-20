from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cebra_lens import compute_metric, extract_label, model_loader, plot_metric
from cebra_lens.quantification.cka_metric import CKA
from cebra_lens.quantification.decoder import Decoding
from cebra_lens.quantification.rdm_metric import RDM


def test_extract_label_single_dim():
    labels = np.array([1, 2, 3, 2])
    result = extract_label(labels, 0)
    assert result.tolist() == [1, 2, 3, 2]


def test_extract_label_multi_dim():
    labels = np.array([[1, 0], [2, 1], [3, 2]])
    result = extract_label(labels, 1)
    assert result.tolist() == [0, 1, 2]


def test_extract_label_out_of_range():
    labels = np.array([[1, 0], [2, 1], [3, 2]])
    with pytest.raises(ValueError):
        extract_label(labels, 3)


def test_compute_metric_with_mock_metric_class():
    dummy_data = {"group1": [np.array([1, 2]), np.array([3, 4])]}

    class DummyMetric:

        def compute(self, sample):
            return sample.sum()

    result = compute_metric(dummy_data, DummyMetric())
    assert result["group1"].tolist() == [3, 7]


def test_compute_metric_with_decoding():
    dummy_data = {"group": [np.ones((5, )), np.ones((5, ))]}

    mock_metric = MagicMock(spec=Decoding)
    mock_metric.compute.side_effect = lambda x: x.sum()
    result = compute_metric(dummy_data, mock_metric, output_only=True)
    assert result["group"].tolist() == [5.0, 5.0]


def test_plot_metric_single_model(monkeypatch):

    class DummyRDM:

        def plot(self, data_dict, **kwargs):
            assert isinstance(data_dict, dict)
            return "plotted"

    result = plot_metric({"data": np.array([1, 2, 3])},
                         DummyRDM(),
                         group_name="test")
    assert result == "plotted"
