"""Functions to retrieve and handle layer activations"""

import cebra
import torch
import torch.nn as nn
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, List, Type, Optional
from .matplotlib import plot_activations
import matplotlib.pyplot as plt


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
        sliced_array = array[:, start : end if end != 0 else start :]
    return sliced_array


def get_cut_indices(
    model_: cebra.integrations.sklearn.cebra.CEBRA,
    layer_type: Type[nn.Module],
    conv_layer_info: Optional[List[int]] = [],
) -> List[Tuple[int, int]]:
    """
    Function which computes indices for removing padding from activation layers.

    Parameters:
    -----------
    model_ : cebra.integrations.sklearn.cebra.CEBRA
        CEBRA model instance.
    layer_type : Type[nn.Module]
        The type of layer whose activations were extracted.
    conv_layer_info : Optional[List[int]]
        A list of the kernel sizes for the convolutional layers extracted, needed for receptive field calculation.

    Returns:
    --------
    cut_indices: List[Tuple[int,int]]
        A list of tuples, a tuple for each activation layer, where the first element is the amount of elements which need to be removed from the beginning and the abs(last element) the amount needed to be remove from the end.
        Example:
        cut_indices = [(4,-4), (3,-3), (2,-2), (1,-1), (0,0), (0,0)]
    """
    cut_indices = []
    offset = model_.get_offset()
    if layer_type == nn.Conv1d:
        # fix it so it's using only model_
        reduction = len(offset) - 1
        for k in conv_layer_info:
            reduction = max(0, reduction - (k - 1))
            left = reduction // 2  # lower
            right = reduction - left
            if offset.left > offset.right:
                right = left
                left = reduction - right
            cut_indices.append((left, -right))
        # add for output layer
        cut_indices.append((0, 0))
    elif layer_type == None:
        raise NotImplementedError("Padding handling not implemented for 'all'.")
    else:
        # need to analyze the padding from the last output of Conv1 and apply the same cut
        raise NotImplementedError(f"Padding handling not implemented for {layer_type}.")
    return cut_indices


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
    model : cebra.integrations.sklearn.cebra.CEBRA
        The cebra model from which to extract activations.
    data : torch.Tensor
        The input data to be passed through the model. Shape of samples X channels (neurons).
    session_id : int, optional
        The session identifier used for selecting the appropriate model in multi-session solvers.
        For single-session, no need to input it.
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
    transform_kwargs = {}
    if model.solver_name_ in [
        "multi-session",
        "multi-session-aux",
        "multiobjective-solver",
    ]:

        model_ = model.model_[session_id]
        transform_kwargs.update({"session_id": session_id})

    elif model.solver_name_ in [
        "single-session",
        "single-session-aux",
        "single-session-hybrid",
        "single-session-full",
    ]:
        model_ = model.model_

    else:
        raise NotImplementedError(
            f"Solver {model.solver_name_} is not yet implemented."
        )

    activations, handles, conv_layer_info = _attach_hooks(
        activations=activations,
        model=model_,
        name=name,
        instance=instance,
        layer_type=layer_type,
    )
    _ = model.transform(data, **transform_kwargs)

    # remove all handles to avoid activation's problems
    for handle in handles:
        handle.remove()

    if model.pad_before_transform:
        # Padding logic: calculate the total reduction which happens based on the kernel size per layer, divide the reduction per layer into 2 parts
        cut_indices = get_cut_indices(model_, layer_type, conv_layer_info)
        for i, (key, value) in enumerate(activations.items()):
            activations[key] = _cut_array(value, cut_indices[i])

    return activations


def process_activations(
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

    handles, conv_layer_info = [], []

    if layer_type:
        for i in range(len(model.net)):
            # attach hook to the layer_type and to the output layer
            if isinstance(model.net[i], layer_type) or i == len(model.net) - 1:
                hook, activations = _get_activation(
                    f"{name}_{instance}_layer_{num_layer}", activations
                )
                if isinstance(model.net[i], layer_type):
                    conv_layer_info.append(model.net[i].kernel_size[0])
                handle = model.net[i].register_forward_hook(hook)
                handles.append(handle)
                num_layer += 1

            elif bool(model.net[i]._modules):
                for submodule in model.net[i].modules():
                    if isinstance(submodule, layer_type):
                        hook, activations = _get_activation(
                            f"{name}_{instance}_layer_{num_layer}",
                            activations,
                        )
                        conv_layer_info.append(submodule.kernel_size[0])
                        handle = submodule.register_forward_hook(hook)
                        handles.append(handle)
                        num_layer += 1

    else:
        # layer_type is None meaning we want to attach hooks to every layer regardless

        for i in range(len(model.net)):
            if bool(model.net[i]._modules):
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

    return activations, handles, conv_layer_info


def aggregate_activations(
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


def get_activations(
    models: Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]],
    data: torch.Tensor,
    session_id: int,
    activations: Optional[Dict[str, npt.NDArray]] = None,
    layer_type: Optional[Type[nn.Module]] = None,
) -> Dict[str, npt.NDArray]:
    """
    Extracts and organizes activations from models.

    Parameters:
    -----------
    models : Dict[str, List[CEBRA]]
        Dictionary of models categorized by label.

    data : torch.Tensor
        Input tensor for the models, shape (samples, features).

    session_id : int
        Session identifier used for selecting the appropriate model.

    activations : Dict[str, npt.NDArray], optional
        Optional dictionary to store activations.

    layer_type : Type[nn.Module], optional
        Optional layer type (e.g., nn.Conv1d) to extract specific activations.

    Returns:
    --------
    Dict[str, npt.NDArray]
        Dictionary with model label prefixes as keys and lists of activation arrays as values.
    """
    activations = activations or {}

    aggregated_activations = aggregate_activations(
        process_activations(models, data, session_id, activations, layer_type)
    )

    activations_dict = {}
    for key, value in aggregated_activations.items():
        prefix = "_".join(key.split("_")[:-1])
        activations_dict.setdefault(prefix, []).append(value)

    return activations_dict
