from .decoding import DecodeModel
from tqdm import tqdm
import numpy as np
from typing import Dict
import numpy.typing as npt


class MultiModel:
    """
    Compute the same metric across multiple models.

    Parameters:
    ----------
    metric_class : object
        The metric class to be used for computation.
    """

    def __init__(self, metric_class: object):
        self.metric_class = metric_class
        self.results_dict = {}

    def compute(self, activations: Dict[str, npt.NDArray]) -> Dict[str, npt.NDArray]:
        """
        Computes the metric based on metric_class given for each model and stores the results in a dictionary.

        Parameters:
        -----------
        activations : Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are 2d lists with the corresponding activations per layer.

        Returns:
        --------
        Dict[str, npt.NDArray]
            A dictionary where keys are strings which represent the model label and values are 2d lists with the calculated metric per layer.

        """
        self.result_dict = {}
        for model_label, activations_list in activations.items():
            self.result_dict[model_label] = []
            for activations in tqdm(activations_list,
                                    desc=f"Processing {model_label}"):
                self.result_dict[model_label].append(
                    self.metric_class.compute(activations))
            if isinstance(self.metric_class, DecodeModel):
                self.result_dict[model_label] = np.array(
                    self.result_dict[model_label])
        return self.result_dict

    def plot(self, *args, **kwargs):
        """
        Plots the results of the metric computation.
        """
        self.metric_class.plot(*args, **kwargs)
