import pytest
from CEBRA_Lens.activations import get_layer_activations

class MockModel:
    def predict(self, inputs):
        # Mock predict function that returns fixed activations
        return {'layer1': [1, 2, 3], 'layer2': [4, 5, 6]}

@pytest.fixture
def model():
    return MockModel()

@pytest.fixture # allows you to reuse inputs instead of redefining in each test
def inputs():
    return [0.5, 0.8, 0.2]

def test_get_layer_activations(model, inputs):
    layer_name = 'layer1'
    activations = get_layer_activations(model, layer_name, inputs)
    expected_activations = [1, 2, 3]  # Expected output for 'layer1'
    assert activations == expected_activations
 

@pytest.mark.parametrize("embeddings,labels,n_jobs,match",
                         _initialize_invalid_embedding_ensembling_data()) # allows you to have multiple values for parameters. e.g. if 3 values, test will be done 3 times
def test_invalid_embedding_ensembling(embeddings, labels, n_jobs, match):
    with pytest.raises(ValueError, match=match):
        _ = cebra_data_helper.ensemble_embeddings(
            embeddings=embeddings,
            labels=labels,
            n_jobs=n_jobs,
        )