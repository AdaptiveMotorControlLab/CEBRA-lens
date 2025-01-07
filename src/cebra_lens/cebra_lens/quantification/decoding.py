import cebra
import torch
import numpy as np
from ..utils_allen import decoding_frames
from ..utils_hpc import decoding_pos_dir
from ..activations import process_activations, get_activations_one_model


def decode_model(
    model: cebra.integrations.sklearn.cebra.CEBRA,
    train_data: torch.Tensor,
    train_label: np.ndarray,
    test_data: torch.Tensor,
    test_label: np.ndarray,
    session_id: int = -1,
    dataset_label: str = "visual",
) -> np.ndarray:
    """
    Decodes a single model.

    Parameters:
    -----------
    model : cebra.integrations.sklearn.cebra.CEBRA
        The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
    train_data : torch.Tensor
        The training data used for model transformation.
    train_label : np.ndarray
        The true labels corresponding to the training data.
    test_data : torch.Tensor
        The validation data used for testing the model.
    test_label : np.ndarray
        The true labels corresponding to the validation data.
    session_id : int, optional
        The session ID for multi-session models. For single-session no need to input it.
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "visual").

    Returns:
    --------
    np.ndarray : Array containing the results. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset.
    """

    if model.solver_name_ == "multi-session":

        embedding_train = model.transform(train_data, session_id)
        embedding_test = model.transform(test_data, session_id)

    elif model.solver_name_ == "single-session":

        embedding_train = model.transform(train_data)
        embedding_test = model.transform(test_data)

    else:
        raise NotImplementedError(
            f"Solver {model.solver_name_} is not yet implemented."
        )

    results = _decoding_function_selection(
        embedding_train, train_label, embedding_test, test_label, dataset_label
    )
    return np.array(results)


def _decoding_function_selection(
    embedding_train: np.ndarray,
    label_train: np.ndarray,
    embedding_test: np.ndarray,
    label_test: np.ndarray,
    dataset_label: str = "visual",
):
    """
    Decodes a model by choosing the appropriate function.

    Parameters:
    -----------
    embedding_train : np.ndarray
        The part of the output embedding to use as training for the decoding.
    train_label : np.ndarray
        The true labels corresponding to the training data.
    embedding_test : np.ndarray
        The part of the output embedding to use as testing for the decoding.
    test_label : np.ndarray
        The true labels corresponding to the validation data.
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "visual").

    Returns:
    --------
    np.ndarray : Array containing the results. Has different structure depending on the dataset used: e.g. 1D array of structure test_score, pos_test_err, pos_test_score for HPC dataset.
    """
    if (
        embedding_train.shape[0] < embedding_train.shape[1]
    ):  # should be samples X neurons
        embedding_train = embedding_train.T
    if embedding_test.shape[0] < embedding_test.shape[1]:  # should be samples X neurons
        embedding_test = embedding_test.T

    if dataset_label == "visual":

        results = decoding_frames(
            embedding_train=embedding_train,
            label_train=label_train,
            embedding_test=embedding_test,
            label_test=label_test,
        )
    elif dataset_label == "HPC":
        results = decoding_pos_dir(
            embedding_train=embedding_train,
            label_train=label_train,
            embedding_test=embedding_test,
            label_test=label_test,
        )
    else:
        raise NotImplementedError(
            f"Decoding not implemented for {dataset_label}. Please use 'visual' or 'HPC'."
        )
    return results


def decode_models(
    models: dict,
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    test_data: torch.Tensor,
    test_label: torch.Tensor,
    session_id: int = -1,
    dataset_label: str = "visual",
) -> dict:
    """
    Decodes multiple models and stores their results in a dictionary.

    Parameters:
    -----------
    models : dict
        A dictionary where keys are model names and values are lists of model objects to be decoded.
    train_data : torch.Tensor
        The training data used for model transformation.
    train_label : torch.Tensor
        The true labels corresponding to the training data.
    test_data : torch.Tensor
        The test data used for model transformation.
    test_label : torch.Tensor
        The true labels corresponding to the test data.
    session_id : int, optional
        The session ID for multi-session models (default is -1 for single-session models).
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "visual").

    Returns:
    --------
    results_dict : dict
        A dictionary where the keys are the model names, and the values are the corresponding decoding results.
    """

    results_dict = {}

    for key, models_list in models.items():

        results = np.zeros((len(models_list), 3))
        for i, model in enumerate(models_list):
            results[i, :] = np.array(
                decode_model(
                    model=model,
                    train_data=train_data,
                    train_label=train_label,
                    test_data=test_data,
                    test_label=test_label,
                    session_id=session_id,
                    dataset_label=dataset_label,
                )
            )

        results_dict[key] = results

    return results_dict


def decode_by_layer_single(
    model: cebra.integrations.sklearn.cebra.CEBRA,
    bool_train: bool,
    train_data: torch.Tensor,
    train_label: np.ndarray,
    test_data: torch.Tensor,
    test_label: np.ndarray,
    session_id: int,
    dataset_label: str = "visual",
    layer_type: str = "conv",
):
    """
    Decode neural data by layer using a given CEBRA model.

    Parameters:
    ----------
    model : cebra.integrations.sklearn.cebra.CEBRA
        The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
    bool_train : bool
        Flag indicating whether the model is trained or not.
    train_data : torch.Tensor
        The training data used for model transformation.
    train_label : np.ndarray
        The true labels corresponding to the training data.
    test_data : torch.Tensor
        The validation data used for testing the model.
    test_label : np.ndarray
        The true labels corresponding to the validation data.
    session_id : int, optional
        The session ID for multi-session models. For single-session no need to input it.
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "visual").
    layer_type : str, optional
        The type of layer to extract activations from. Defaults to 'conv'.

    Returns:
    -------
    np.ndarray
        A numpy array containing the decoding results for each layer and the neural input baseline.
    """

    activations_train = get_activations_one_model(
        model=model,
        data=train_data,
        name=model.solver_name_,
        session_id=session_id,
        bool_train=bool_train,
        layer_type=layer_type,
    )

    activations_test = get_activations_one_model(
        model=model,
        data=test_data,
        name=model.solver_name_,
        session_id=session_id,
        bool_train=bool_train,
        layer_type=layer_type,
    )

    num_layers = len(activations_train)

    if dataset_label in ["HPC", "visual"]:
        results = np.zeros((num_layers + 1, 3))
    else:
        raise NotImplementedError(
            f"Decoding not implemented for {dataset_label}. Please use 'visual' or 'HPC'."
        )
    keys = list(activations_train.keys())
    for i in range(num_layers + 1):

        if i == 0:
            results[i, :] = _decoding_function_selection(
                train_data, train_label, test_data, test_label, dataset_label
            )  # neural input baseline
        else:
            results[i, :] = _decoding_function_selection(
                activations_train[keys[i - 1]],
                train_label,
                activations_test[keys[i - 1]],
                test_label,
                dataset_label,
            )  # layer decoding

    return results


def decode_by_layer_all(
    models_dict: dict,
    train_data: torch.Tensor,
    train_label: np.ndarray,
    test_data: torch.Tensor,
    test_label: np.ndarray,
    session_id: int = -1,
    dataset_label: str = "visual",
    layer_type: str = "conv",
):
    """
    Decode neural data by layer using a given CEBRA model.

    Parameters:
    ----------
    models_dict : dict
        Dictionnary containing the CEBRA Models loaded by lens.model.model_loader().
    train_data : torch.Tensor
        The training data used for model transformation.
    train_label : np.ndarray
        The true labels corresponding to the training data.
    test_data : torch.Tensor
        The validation data used for testing the model.
    test_label : np.ndarray
        The true labels corresponding to the validation data.
    session_id : int, optional
        The session ID for multi-session models. For single-session no need to input it.
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "visual").
    layer_type : str, optional
        The type of layer to extract activations from. Defaults to 'conv'.

    Returns:
    -------
    dict
        A result dictionnary containing the results for each solver and trained or untrained in the same format as the layer activations (cf  lens.activations.process_activations()).
    """

    results_dict = {}
    for key, models in models_dict.items():
        results_list = []

        for model in models:
            results_list.append(
                decode_by_layer_single(
                    model=model,
                    bool_train="UT" not in key,
                    train_data=train_data,
                    train_label=train_label,
                    test_data=test_data,
                    test_label=test_label,
                    session_id=session_id,
                    dataset_label=dataset_label,
                    layer_type=layer_type,
                )
            )

        results_dict[key] = results_list

    results_dict = process_activations(results_dict)

    for key, value in results_dict.items():
        for inner_key, inner_value in value.items():
            results_dict[key][inner_key] = inner_value[0][0]

    return results_dict
