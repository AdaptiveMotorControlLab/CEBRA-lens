from tqdm import tqdm
import numpy as np
import pickle
import types
from typing import List, Union, Dict
from abc import *
from pathlib import Path
import numpy.typing as npt


class _BaseMetric:
    """
    Base class for metrics computations.
    """

    @abstractmethod
    def compute(self,
                activations: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]:
        """
        Every metric which inherits ``_BaseMetric`` needs to implement a compute function.
        The compute function is specific to a metric, e.g. intra-bin distance, RDM, CKA,...

        Parameters:
        -----------
        activations : Dict[str, npt.NDArray]
            Dictionary where the key is the model category group (str),
            and the value is a npt.NDArray containing for all the models under that group the activations per layer.

        """
        raise NotImplementedError

    def iterate_over_layers(
        self,
        activations: List[Union[float, npt.NDArray]],
        metric_func: types.FunctionType,
    ) -> List[Union[np.float64, npt.NDArray]]:
        """
        Iterate over each layer of activations and apply the metric function to compute the desired metric.

        Parameters:
        -----------
        activations : List[npt.NDArray]
            List of 2D numpy array representing the activation of neurons per layer.
        metric_func : types.FunctionType
            Function that computes the metric for a single layer's activations.

        Returns:
        --------
        layer_data : List[Union[float, npt.NDArray]]
            The computed metric for each layer.
        """
        layer_data = []
        for layer_activation in activations:
            layer_data.append(metric_func(layer_activation))
        return layer_data

    def save(self, filepath: str, data: Dict[str, npt.NDArray]) -> None:
        """
        Save data in the location filepath.

        Parameters:
        -----------
        filepath : str
            Filepath to the location where the data wants to be stored.
        data : Dict[str, npt.NDArray]
            Dictionary where the key is the model category label (str),
            and the value is a npt.NDArray containing for all the models under that label the calculated data.
        """
        filepath = Path(filepath)
        custom_filepath = filepath.with_stem(filepath.stem +
                                             f"_{self.__class__.__name__}")
        with open(custom_filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> Dict[str, npt.NDArray]:
        """
        Load data from the filepath location

        Parameters:
        -----------
        filepath : str
            Filepath to the location where the data is stored.

        Returns:
        --------
        Dict[str, npt.NDArray]
            Dictionary where the key is the model category label (str),
            and the value is a npt.NDArray containing for all the models under that label the calculated data.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    @abstractmethod
    def plot(self):
        """
        Every metric which inherits ``_BaseMetric`` needs to implement a plot function.
        The plot function is specific to a metric, e.g. intra-bin distance, RDM, CKA,...
        """
        raise NotImplementedError
