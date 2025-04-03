from .decoding import DecodeModel
from tqdm import tqdm
import numpy as np


class MultiModel:
    """
    A class to compute the same metric across multiple models.

    Args:
        metric_class (object): The metric class to be used for computation.
    """

    def __init__(self, metric_class: object):
        self.metric_class = metric_class
        self.results_dict = {}

    def compute(self, activations: dict) -> dict:
        """
        Computes the metric based on metric_class given for each model and stores the results in a dictionary.
        """
        self.result_dict = {}
        for model_label, activations_list in activations.items():
            self.result_dict[model_label] = []
            for activations in tqdm(activations_list, desc=f"Processing {model_label}"):
                self.result_dict[model_label].append(
                    self.metric_class.compute(activations)
                )
            if isinstance(self.metric_class, DecodeModel):
                self.result_dict[model_label] = np.array(self.result_dict[model_label])
        return self.result_dict
