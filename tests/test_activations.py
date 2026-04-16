from collections import namedtuple
from unittest.mock import MagicMock

import cebra
import numpy as np
import pytest
import torch

from cebra_lens import activations
from cebra_lens.activations import (_cut_array, get_activations_model,
                                    get_cut_indices)


def test_cut_array_no_cut():
    arr = np.arange(10).reshape(2, 5)
    result = _cut_array(arr, (0, 0))
    np.testing.assert_array_equal(result, arr)


def test_cut_array_with_cut():
    arr = np.array([[1, 2, 3, 4, 5]])
    result = _cut_array(arr, (1, -1))
    np.testing.assert_array_equal(result, np.array([[2, 3, 4]]))


def test_get_cut_indices():
    Offset = namedtuple("Offset", ["left", "right"])

    # Mock the model's get_offset behavior
    model_mock = MagicMock()
    model_mock.get_offset.return_value = Offset(left=2, right=2)

    result = get_cut_indices(model_mock, torch.nn.Conv1d, [3, 3])
    assert isinstance(result, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in result)

    with pytest.raises(NotImplementedError,
                       match="Padding handling not implemented*"):
        get_cut_indices(model_mock, None, [3, 3])


def make_mock_cebra_model():
    model = cebra.CEBRA(max_iterations=1, device="cpu")
    model.fit(torch.rand((5, 10)), torch.rand((5, 1)))
    model.pad_before_transform = False
    return model


def test_get_activations_model_basic(monkeypatch):
    model = make_mock_cebra_model()
    data = torch.rand((5, 10))

    monkeypatch.setattr(
        activations,
        "_attach_hooks",
        lambda *a, **kw: ({
            "test_layer": np.ones((5, 3))
        }, [], [3]),
    )
    print(type(model))
    result = get_activations_model(model=model,
                                   data=data,
                                   layer_type=torch.nn.Conv1d)
    assert isinstance(result, dict)
    assert "test_layer" in result
    assert result["test_layer"].shape == (5, 3)
