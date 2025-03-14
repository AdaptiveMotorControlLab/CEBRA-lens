from tqdm import tqdm
import numpy as np


class _BaseMetric:
    """
    Base class for metric computations.
    """

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class _MultiMetric(_BaseMetric):

    def transform(self, data_dict, BaseClass):
        result_dict = {}
        for model_label, activations in data_dict.items():
            result_dict[model_label] = []
            for activation in tqdm(activations, desc=f"Processing {model_label}"):
                result_dict[model_label].append(BaseClass(activation))
        return result_dict

    def compute(self, data_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for label, values in data_dict.items():
            result_dict[label] = []
            for value in tqdm(values, desc=f"Processing {label}"):
                result_dict[label].append(value.compute(*args, **kwargs))
        return result_dict

    def decode(self, data_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for label, values in data_dict.items():
            result_dict[label] = []
            for value in tqdm(values, desc=f"Processing {label}"):
                result_dict[label].append(value.decode(*args, **kwargs))
            result_dict[label] = np.array([result_dict])
        return result_dict
