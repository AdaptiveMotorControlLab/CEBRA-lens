import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from cebra_lens.quantification.decoding import decoding, Decoding


@pytest.fixture
def embeddings_labels():
    emb_train = torch.randn((1000, 32))
    emb_test = torch.randn((200, 32))
    label_train = np.random.rand(1000, 2)
    label_test = np.random.rand(200, 2)
    return emb_train, emb_test, label_train, label_test


def test_decoding_function(embeddings_labels):
    emb_train, emb_test, label_train, label_test = embeddings_labels

    with patch("cebra_lens.quantification.decoding.cebra.KNNDecoder") as mock_knn:
        mock_model = MagicMock()
        # Return prediction with correct shape each time
        mock_model.predict.side_effect = lambda x: np.random.rand(len(x))
        mock_knn.return_value = mock_model

        score, medians, r2s = decoding(emb_train, emb_test, label_train, label_test)

        assert isinstance(score, float)
        assert len(medians) == label_train.shape[1]
        assert len(r2s) == label_train.shape[1]


def make_mock_cebra_model():
    model = MagicMock()
    model.solver_name_ = "single-session"
    model.transform.side_effect = lambda x, **kwargs: x.detach().numpy()
    return model


def test_decoding_class_output_only_true():
    model = make_mock_cebra_model()
    decoding_class = Decoding(
        train_data=torch.rand((300, 100)),
        train_label=np.random.rand(300, 1),
        test_data=torch.rand((100, 100)),
        test_label=np.random.rand(100, 1),
        dataset_label=None,
        output_only=True,
    )
    results = decoding_class.compute(model)
    assert isinstance(results, dict)
    assert 0 in results


@patch("cebra_lens.quantification.decoding.get_activations_model")
def test_decoding_class_output_only_false(mock_get_act):
    model = make_mock_cebra_model()
    mock_get_act.side_effect = lambda **kwargs: {
        "layer1": np.random.rand(1000, 1000),
        "layer2": np.random.rand(1000, 1000),
    }

    decoding_class = Decoding(
        train_data=torch.rand((1000, 1000)),
        train_label=np.random.rand(1000, 1),
        test_data=torch.rand((1000, 1000)),
        test_label=np.random.rand(1000, 1),
        dataset_label=None,
        output_only=False,
    )

    results = decoding_class.compute(model)
    assert isinstance(results, dict)
    assert len(results) == 3  # baseline + 2 layers


def test_set_output_only():
    decoding_instance = Decoding(
        train_data=torch.rand((10, 5)),
        train_label=np.random.rand(10, 1),
        test_data=torch.rand((10, 5)),
        test_label=np.random.rand(10, 1),
    )
    decoding_instance.set_output_only(False)
    assert decoding_instance.output_only is False


@patch("cebra_lens.quantification.decoding.plot_decoding")
@patch("cebra_lens.quantification.decoding.plot_layer_decoding")
def test_plot_logic(mock_layer_plot, mock_decoding_plot):
    dummy_result = {"modelA": {0: (0.9, [0.1], [0.8])}}
    dec = Decoding(
        train_data=torch.rand((10, 10)),
        train_label=np.random.rand(10, 1),
        test_data=torch.rand((10, 10)),
        test_label=np.random.rand(10, 1),
    )

    dec.output_only = True
    dec.plot(dummy_result, label=0)
    assert mock_decoding_plot.called

    dec.output_only = False
    dec.plot(dummy_result, label=0)
    assert mock_layer_plot.called
