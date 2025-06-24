from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import cebra_lens


@pytest.fixture
def dummy_comparisons():
    return [("A", "B")]


@pytest.fixture
def dummy_cka(dummy_comparisons):
    return cebra_lens.quantification.cka_metric.CKA(
        comparisons=dummy_comparisons)


@pytest.fixture
def dummy_activations():
    # Simulate Conv1D and Linear layer activations as 2D arrays (samples, features)
    batch_size = 10
    conv_channels = 4
    conv_length = 8
    linear_features = 5

    # Conv1D output: (batch_size, conv_channels, conv_length) -> flatten to (batch_size, conv_channels * conv_length)
    conv1d_activations_A = np.random.rand(batch_size, conv_channels,
                                          conv_length).reshape(batch_size, -1)
    linear_activations_A = np.random.rand(batch_size, linear_features)
    conv1d_activations_B = np.random.rand(batch_size, conv_channels,
                                          conv_length).reshape(batch_size, -1)
    linear_activations_B = np.random.rand(batch_size, linear_features)

    # Each group has a list of 2D arrays (one per layer)
    return {
        "A":
        np.array([
            [np.random.rand(5, 10),
             np.random.rand(5, 10)],
            [np.random.rand(5, 10),
             np.random.rand(5, 10)],
        ]),
        "B":
        np.array([
            [np.random.rand(5, 10),
             np.random.rand(5, 10)],
            [np.random.rand(5, 10),
             np.random.rand(5, 10)],
        ]),
    }


def test_center_gram_symmetry(dummy_cka):
    mat = np.eye(5)
    centered = dummy_cka.center_gram(mat)
    assert np.allclose(centered, centered.T)


def test_center_gram_unbiased(dummy_cka):
    mat = np.eye(5)
    centered = dummy_cka.center_gram(mat, unbiased=True)
    assert np.allclose(centered, centered.T)


def test_gram_linear(dummy_cka):
    x = np.random.rand(10, 5)
    gram = dummy_cka.gram_linear(x)
    assert gram.shape == (10, 10)
    assert np.allclose(gram, gram.T)


def test_cka_value(dummy_cka):
    x = np.random.rand(10, 5)
    y = np.random.rand(10, 5)
    gram_x = dummy_cka.gram_linear(x)
    gram_y = dummy_cka.gram_linear(y)
    val = dummy_cka.cka(gram_x, gram_y)
    assert isinstance(val, float) or isinstance(val, np.floating)


def test_compute_cka_shape(dummy_cka):
    emb1 = [np.random.rand(5, 10), np.random.rand(5, 10)]
    emb2 = [np.random.rand(5, 10), np.random.rand(5, 10)]
    result = dummy_cka._compute_cka(emb1, emb2)
    assert result.shape == (1, 2)


def test_compute_per_layer_shape(dummy_cka):
    emb1 = [[np.random.rand(5, 10), np.random.rand(5, 10)] for _ in range(3)]
    emb2 = [[np.random.rand(5, 10), np.random.rand(5, 10)] for _ in range(3)]
    result = dummy_cka._compute_per_layer(emb1, emb2, flag=True)
    assert result.shape == (3, 2)


def test_compute(dummy_cka, dummy_activations):
    result = dummy_cka.compute(dummy_activations, ("A", "B"))
    assert isinstance(result, np.ndarray)


def test_compute_intra_label(dummy_cka, dummy_activations):
    result = dummy_cka.compute(dummy_activations, ("A", "A"))
    assert isinstance(result, np.ndarray)


@patch("cebra_lens.utils_plot.plot_cka_heatmaps")
def test_plot_calls_heatmap(mock_plot, dummy_cka):
    cka_matrices = {"A": np.random.rand(2, 2)}
    dummy_cka.plot(cka_matrices, annot=True)
    assert mock_plot.called
