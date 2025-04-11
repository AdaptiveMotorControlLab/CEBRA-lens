"""Functions to retrieve and handle layer activations"""

import cebra
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, List, Type


def _cut_array(
    array: npt.NDArray, cut_indices: Tuple[np.int64, np.int64]
) -> npt.NDArray:
    """
    Slices the input array based on the provided cut indices.
    This is used to remove the padding from activations in `get_activations_model`.
    Parameters:
    -----------
    array : numpy.ndarray
        The input array to be sliced.
    cut_indices : Tuple[np.int64, np.int64]
        A tuple containing two integers, start and end indices for slicing.

    Returns:
    --------
    npt.NDArray
        The sliced array. If both start and end indices are 0, the whole array is returned.
    """

    start = cut_indices[0]
    end = cut_indices[1]
    if start == 0 and end == 0:
        # If both start and end are 0, take the whole array
        sliced_array = array
    else:
        # Otherwise, slice the array
        sliced_array = array[:, start:end]
    return sliced_array    

def get_activations_model(
    model: cebra.integrations.sklearn.cebra.CEBRA,
    data: torch.Tensor,
    session_id: int = -1,
    name: str = "single",
    instance: int = 0,
    layer_type: Type[nn.Module] = None,
) -> Dict[str, npt.NDArray]:
    """
    Extracts activations from a single model layer.
    This function extracts activations from the specified layer of a model and stores them in a dictionary.

    Parameters:
    -----------
    model : cebra.model
        The cebra model from which to extract activations.
    data : torch.Tensor
        The input data to be passed through the model. Shape of samples X channels (neurons).
    session_id : int, optional
        The session identifier used for selecting the appropriate model in multi-session solvers.
        For single-session, no need to input it.
    name : str
        A base name for the activation keys (e.g., "single", "multi").
    instance : int
        The instance number for the model, used to differentiate between models from the same model category.
    layer_type : Type[nn.Module]
        The type of layer to extract activations from. Defaults to None, meaning extracts activations from all layers.

    Returns:
    --------
    activations : Dict[str, npt.NDArray]
        A dictionary containing the activations from the layers of the model. Where the keys are str 'model_label_instance_layer_num , and the values are the activations for that model instance and layer. E.g.  {'model1_layer_1': [0.1, 0.2], 'model1_layer_2': [0.3, 0.4]}
    Notes:
    --------
    If the model includes padding, the padding is removed from the activations for easier downstream use.
    """

    activations = {}
    if model.solver_name_ == "multi-session":
        model_ = model.model_[session_id]
        activations, handles = _attach_hooks(
            activations=activations,
            model=model_,
            name=name,
            instance=instance,
            layer_type=layer_type,
        )
        _ = model.transform(
            data, session_id=session_id
        )  # no need to store the output embedding. we already add a hook on it

    elif model.solver_name_ == "single-session":
        model_ = model.model_
        activations, handles = _attach_hooks(
            activations=activations,
            model=model_,
            name=name,
            instance=instance,
            layer_type=layer_type,
        )
        _ = model.transform(
            data
        )  # no need to store the output embedding. we already add a hook on it

    else:
        raise NotImplementedError(
            f"Solver {model.solver_name_} is not yet implemented."
        )

    # remove all handles to avoid activation's problems
    for handle in handles:
        handle.remove()

    #TODO(eloise): implement general padding
    if model.pad_before_transform:
        if layer_type == nn.Conv1d:
            if model.model_architecture in ["offset10-model", "offset10-model-mse","offset10-model-adapt"]:
                cut_indices = [(4, -4), (3, -3), (2, -2), (1, -1), (0, 0), (0, 0)]
            elif model.model_architecture in ["offset5-model"]:
                cut_indices = [(1, -2), (0, -1), (0, 0), (0, 0)]

            else:
                raise NotImplementedError(
                    f"Padding handling for {model.model_architecture} not implemented yet."
                )
        elif layer_type==None:
            raise NotImplementedError("Padding handling not implemented for 'all'.")
        else:
            raise NotImplementedError(
                f"Padding handling not implemented for {layer_type}."
            )

        for i, (key, value) in enumerate(activations.items()):
            activations[key] = _cut_array(value, cut_indices[i])

    return activations


def get_activations_models(
    models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
    data: torch.Tensor,
    session_id: int,
    activations: Dict[str, npt.NDArray] = {},
    layer_type: Type[nn.Module] = None,
) -> Dict[str, npt.NDArray]:
    """
    Extracts activations from multiple models and stores them in a dictionary.
    This function demonstrates how to use the `get_activations_model` function by extracting activations from multiple models
    and storing them in a dictionary.

    Parameters:
    -----------
    models : Dict[str, List[ cebra.integrations.sklearn.cebra.CEBRA]]
        A dictionary containing different sets of models.
    data : torch.Tensor
        The input data for which activations are to be extracted. Shape of samples X channels (neurons).
    session_id : int
        The session identifier used for selecting the appropriate model in multi-session solvers.
    activations : Dict[str, npt.NDArray]
        A dictionary to store the activations. If passed as an argument, the new keys will be concatenated to the existing dictionary.
    layer_type : Type[nn.Module]
        The type of layer from which to extract activations (e.g., nn.Conv1d).

    Returns:
    --------
    activations : Dict[str, npt.NDArray]
        A dictionary containing the activations from all the models passed as input. A dictionary where keys are strings in the format 'model_identifier_layer_num' and values are activations.
    """

    for model_name, models in models.items():
        for i, model in enumerate(models):
            activations.update(
                get_activations_model(
                    model=model,
                    data=data,
                    session_id=session_id,
                    name=model_name,
                    instance=i,
                    layer_type=layer_type,
                )
            )

    return activations


# Function to create a hook that stores the activations in the dictionary
def _get_activation(name: str, activations: Dict):
    def hook(model, input, output):
        activations[name] = output.detach().squeeze().numpy()

    return hook, activations


def _attach_hooks(
    activations: Dict[str, npt.NDArray],
    model: cebra.integrations.sklearn.cebra.CEBRA,
    name: str,
    instance: int,
    layer_type: Type[nn.Module] = None,
) -> Dict[str, npt.NDArray]:  # only attaches hooks on convolutional layers
    """
    Attaches forward hooks to the specified layers of a given model to capture activations.
    This function attaches hooks to the specified layers of the model to capture activations during the forward pass.

    Parameters:
    -----------
    activations : Dict[str, npt.NDArray]
        A dictionary to store the activations. Please refer to ``activations`` returned by ``get_activations_model``.
    model : cebra.integrations.sklearn.cebra.CEBRA
        The model to which hooks will be attached.
    name : str
        A base name for the activation keys (e.g., "single", "multi").
    instance : int
        The instance number for the model, used to differentiate between models from the same model category.
    layer_type : Type[nn.Module]
        The type of layer from which to extract activations (e.g., nn.Conv1d).

    Returns:
    --------
    activations : Dict[str, npt.NDArray]
        The updated dictionary containing the activations captured by the hooks. Please refer to ``activations`` returned by ``get_activations_model``.
    """

    num_layer = 1

    handles = []

    if layer_type:
        for i in range(len(model.net)):
            if isinstance(model.net[i], layer_type) or i == len(model.net) - 1:
                hook, activations = _get_activation(
                    f"{name}_{instance}_layer_{num_layer}", activations
                )

                handle = model.net[i].register_forward_hook(hook)
                handles.append(handle)
                num_layer += 1

            elif bool(
                model.net[i]._modules
            ):
                for submodule in model.net[i].modules():
                    if isinstance(submodule, layer_type):
                        hook, activations = _get_activation(
                            f"{name}_{instance}_layer_{num_layer}",
                            activations,
                        )
                        handle = submodule.register_forward_hook(hook)
                        handles.append(handle)
                        num_layer += 1

    else:
        #layer_type is None meaning we want to attach hooks to every layer regardless
        for i in range(len(model.net)):
            if bool(
                model.net[i]._modules
            ):
                for submodule in model.net[i].modules():
                    hook, activations = _get_activation(
                        f"{name}_{instance}_layer_{num_layer}",
                        activations,
                    )
                    handle = submodule.register_forward_hook(hook)
                    handles.append(handle)
                    num_layer += 1

            else:
                hook, activations = _get_activation(
                    f"{name}_{instance}_layer_{num_layer}", activations
                )

                handle = model.net[i].register_forward_hook(hook)
                handles.append(handle)
                num_layer += 1

    return activations, handles


def _aggregate_activations(
    activations: Dict[str, npt.NDArray],
) -> Dict[str, npt.NDArray]:
    """
    Aggregates activations by model identifier aka. instance.
    This function takes a dictionary of activations where the keys are strings containing model identifiers and layer information,
    and the values are the corresponding activations. It aggregates the activations by model identifier, ignoring the layer information.

    Parameters:
    -----------
    activations : Dict[str, npt.NDArray]
        A dictionary where keys are strings in the format 'model_identifier_layer_num' and values are activations.

    Returns:
    --------
    Dict[str, npt.NDArray]
        A dictionary where keys are model identifiers and values are lists of activations corresponding to those model identifiers.

    Example:
    --------
    >>> activations = {
    ...     'model1_layer_1': [0.1, 0.2],
    ...     'model1_layer_2': [0.3, 0.4],
    ...     'model2_layer_1': [0.5, 0.6]
    ... }
    >>> _aggregate_activations(activations)
    {
        'model1': [[0.1, 0.2], [0.3, 0.4]],
        'model2': [[0.5, 0.6]]
    }
    """

    aggregated_activations = {}

    for key, value in activations.items():
        model_identifier = key.rsplit("_layer", 1)[0]

        if model_identifier not in aggregated_activations:
            aggregated_activations[model_identifier] = []

        aggregated_activations[model_identifier].append(value)
    return aggregated_activations


def process_activations(activations: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]:
    """
    Processes the activations and formats them into a structured dictionary.

    Parameters:
    -----------
    activations : Dict[str, npt.NDArray]
        A dictionary where the keys are in the format 'MODEL_LABEL_INSTANCE_layer_LAYER'
        (e.g., 'single_UT_1_layer_2') and the values contain the activations for that model instance and that layer.

    Returns:
    --------
    activations_dict : Dict[str, npt.NDArray]
        A dictionary where the keys are the model category names, and the values are arrays of activation values for each instance:
        e.g.{'single_UT': [[instance1_activations], [instance2_activations], ...], 'single_TR': [[instance1_activations], [instance2_activations], ...]}'
    """

    # first aggregate all the layers of the activations into models
    aggregated_activations = _aggregate_activations(activations=activations)

    activations_dict = {}

    for key, value in aggregated_activations.items():
        prefix = "_".join(key.split("_")[:-1])
        if prefix not in activations_dict.keys():
            activations_dict[prefix] = []
        activations_dict[prefix].append(value)

    return activations_dict
