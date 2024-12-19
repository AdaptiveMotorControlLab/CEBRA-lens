import cebra
import torch
import numpy as np
from ..utils_allen import decoding_frames


def decode_model(
    model: cebra.integrations.sklearn.cebra.CEBRA,
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    test_data: torch.Tensor,
    test_label: torch.Tensor,
    session_id: int = -1,
    dataset_label: str = "Visual",
) -> np.ndarray:
    """
    Decodes a single model.

    Parameters:
    -----------
    model : cebra.integrations.sklearn.cebra.CEBRA
        The CEBRA model that will be used to transform the data (either multi-session or single-session model for now).
    train_data : torch.Tensor
        The training data used for model transformation.
    train_label : torch.Tensor
        The true labels corresponding to the training data.
    test_data : torch.Tensor
        The validation data used for testing the model.
    test_label : torch.Tensor
        The true labels corresponding to the validation data.
    session_id : int, optional
        The session ID for multi-session models. For single-session no need to input it.
    dataset_label : str, optional
        The type of dataset being used for decoding (default is "Visual").

    Returns:
    --------
    results : tuple if comes from Visual. Can be different formats depending on the decoding. (#TODO one we do HPC maybe we can have a standard format)
        The decoding coming from the decoding function of the data type.
    """

    if model.solver_name_ == "multi-session":

        train = model.transform(train_data, session_id)
        test = model.transform(test_data, session_id)

    elif model.solver_name_ == "single-session":

        train = model.transform(train_data)
        test = model.transform(test_data)

    else:
        raise NotImplementedError(
            f"Solver {model.solver_name_} is not yet implemented."
        )

    if dataset_label == "Visual":

        results = decoding_frames(
            embedding_train=train,
            label_train=train_label,
            embedding_test=test,
            label_test=test_label,
        )
    else:
        raise NotImplementedError(
            f"Decoding not implemented for {dataset_label}. Please use 'Visual'"
        )

    return np.array(results)


def decode_models(
    models: dict,
    train_data: torch.Tensor,
    train_label: torch.Tensor,
    test_data: torch.Tensor,
    test_label: torch.Tensor,
    session_id: int = -1,
    dataset_label: str = "Visual",
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
        The type of dataset being used for decoding (default is "Visual").

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
                    X=train_data,
                    y=train_label,
                    valid_datas=test_data,
                    discrete_labels_val=test_label,
                    session_id=session_id,
                    dataset_label=dataset_label,
                )
            )

        results_dict[key] = results

    return results_dict
