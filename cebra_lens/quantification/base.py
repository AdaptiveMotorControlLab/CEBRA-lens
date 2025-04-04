from tqdm import tqdm
import numpy as np
import pickle
import types
from typing import List, Union
import abc
from pathlib import Path


class _BaseMetric:
    """
    Base class for metrics computations.
    """

    @abc.abstractmethod
    def compute(self, activations: dict) -> dict:
        raise NotImplementedError

    def iterate_over_layers(
        self, activations: List[Union[float, np.ndarray]], metric_func: types.FunctionType
    ) -> List[Union[float, np.ndarray]]:
        """
        Iterate over each layer of activations and apply the metric function to compute the desired metric.

        Parameters:
        -----------
            activations : List[np.ndarray]
            List of 2D numpy array representing the activation of neurons per layer.

        Returns:
        --------
            layer_data : List[Union[float, np.ndarray]]
            The computed metric for each layer.
        """
        layer_data = []
        for layer_activation in activations:
            layer_data.append(metric_func(layer_activation))
        return layer_data

    def save(self, filepath: str, data: dict) -> None:
        filepath = Path(filepath)
        custom_filepath = filepath.with_stem(
            filepath.stem + f"_{self.__class__.__name__}"
        )
        with open(custom_filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> dict:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    @abc.abstractmethod
    def plot(self):
        raise NotImplementedError
