from tqdm import tqdm
import numpy as np
import pickle
import types
from typing import List, Literal, Optional, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path


class _BaseMetric(ABC):
    """
    Base class for metrics computations.
    """

    @abstractmethod
    def compute(self, activations: dict) -> dict:
        raise NotImplementedError

    def iterate_over_layers(
        self, activations: np.ndarray, metric_func: types.FunctionType
    ) -> List[Union[float, np.ndarray]]:
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

    @abstractmethod
    def plot(self):
        raise NotImplementedError
