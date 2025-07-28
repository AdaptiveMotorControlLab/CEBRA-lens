from unittest.mock import MagicMock, patch

import cebra
import numpy as np
import pytest
import torch

import cebra_lens

class _LayerMap(list):
    """A little helper so that `.tolist()` returns the list of layer‐maps."""
    def tolist(self):
        return self
    
class DummyKNNDecoder:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.random.rand(len(X))

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self



def make_mock_cebra_model(input_dim=10, label_dim=1):
    model = cebra.CEBRA(max_iterations=1, device="cpu")
    model.fit(torch.rand((5, input_dim)), torch.rand((5, label_dim)))
    return model


@pytest.fixture
def embeddings_labels():
    emb_train = torch.randn((1000, 32))
    emb_test = torch.randn((200, 32))
    train_label = np.random.rand(1000, 2)
    test_label = np.random.rand(200, 2)
    return emb_train, emb_test, train_label, test_label


def test_decoding_function(embeddings_labels):
    emb_train, emb_test, train_label, test_label = embeddings_labels

    with patch("cebra.KNNDecoder", return_value=DummyKNNDecoder()):
        # decoding() now returns a DecodeResult namedtuple:
        dr = cebra_lens.quantification.decoder.decoding(
            emb_train, emb_test, train_label, test_label)
        # unpack fields rather than expecting three separate return values
        score   = dr.overall_score
        medians = dr.per_label_error
        r2s     = dr.per_label_score
        
        assert isinstance(score, float)
        assert len(medians) == train_label.shape[1]
        assert len(r2s) == train_label.shape[1]


def test_decoding_class_output_only_true():
    model = make_mock_cebra_model(100, 1)
    decoding_class = cebra_lens.quantification.decoder.Decoding(
        train_data=torch.rand((300, 100)),
        train_label=np.random.rand(300, 1),
        test_data=torch.rand((100, 100)),
        test_label=np.random.rand(100, 1),
        dataset_label=None,
    )
    results = decoding_class.compute(model, output_only=True)
    assert isinstance(results, dict)
    assert 0 in results


@patch("cebra_lens.activations.get_activations_model")
def test_decoding_class_output_only_false(mock_get_act):
    model = make_mock_cebra_model(1000, 1)
    mock_get_act.side_effect = lambda **kwargs: {
        "layer1": np.random.rand(1000, 1000),
        "layer2": np.random.rand(1000, 1000),
    }

    decoding_class = cebra_lens.quantification.decoder.Decoding(
        train_data=torch.rand((1000, 1000)),
        train_label=np.random.rand(1000, 1),
        test_data=torch.rand((1000, 1000)),
        test_label=np.random.rand(1000, 1),
        dataset_label=None,
    )

    results = decoding_class.compute(model, output_only=False)
    assert isinstance(results, dict)
    assert len(results) == 2  # only 2 Conv1d layer in the mock model

    decoding_class.layer_type = None
    with pytest.raises(NotImplementedError,
                       match="Padding handling not implemented*"):
        decoding_class.compute(model, output_only=False)

    decoding_class.layer_type = torch.nn.Linear
    with pytest.raises(NotImplementedError,
                       match="Padding handling not implemented*"):
        decoding_class.compute(model, output_only=False)


@patch("cebra_lens.utils_plot.plot_decoding")
@patch("cebra_lens.utils_plot.plot_layer_decoding")
def test_decoder_plot(mock_layer_plot, mock_decoding_plot):
    dec = cebra_lens.quantification.decoder.Decoding(
        train_data=torch.rand((10, 10)),
        train_label=np.random.rand(10, 1),
        test_data=torch.rand((10, 10)),
        test_label=np.random.rand(10, 1),
    )
    # single‐model, single‐layer → .tolist() must return a length‑1 mapping
    dummy_result = {
        "modelA": _LayerMap([{0: (0.9, [0.1], [0.8])}])
    }
    dec.plot(dummy_result, label=0)
    assert mock_decoding_plot.called

    # now two “models” → plot_layer_decoding
    dummy_result = {
        "modelA": _LayerMap([{0: (0.9,  [0.1],  [0.8])}]),
        "modelB": _LayerMap([{0: (0.85, [0.15], [0.75])}]),
    }
    dec.plot(dummy_result, label=0)
    assert mock_layer_plot.called
