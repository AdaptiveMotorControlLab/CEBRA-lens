from unittest.mock import MagicMock, patch

import cebra
import numpy as np
import pytest
import torch

from cebra_lens import compute_metric, model_loader, plot_metric
from cebra_lens.quantification.base import _BaseMetric
from cebra_lens.quantification.cka_metric import CKA
from cebra_lens.quantification.decoder import Decoding
from cebra_lens.quantification.rdm_metric import RDM


def make_mock_cebra_model(input_dim=10, label_dim=1):
    model = cebra.CEBRA(max_iterations=1, device="cpu")
    model.fit(torch.rand((5, input_dim)), torch.rand((5, label_dim)))
    #model.pad_before_transform = False
    return model


def test_compute_metric_with_mock_metric_class():
    dummy_data = {"group1": [np.array([1, 2]), np.array([3, 4])]}

    class DummyMetric(_BaseMetric):

        def compute(self, sample):
            return sample.sum()

    result = compute_metric(dummy_data, DummyMetric())
    assert result["group1"].tolist() == [3, 7]


def test_compute_metric_with_decoding():
    dummy_model = {"group1": make_mock_cebra_model(100, 1)}

    dec_metric = Decoding(
        train_data=torch.rand((300, 100)),
        train_label=np.random.rand(300, 1),
        test_data=torch.rand((100, 100)),
        test_label=np.random.rand(100, 1),
        dataset_label=None,
    )
    result = compute_metric(model_data=dummy_model,
                            metric_class=dec_metric,
                            output_only=False)


def test_plot_metric_single_model(monkeypatch):

    class DummyRDM:

        def plot(self, data_dict, **kwargs):
            assert isinstance(data_dict, dict)
            return "plotted"

    result = plot_metric({"data": np.array([1, 2, 3])},
                         DummyRDM(),
                         group_name="test")
    assert result == "plotted"
