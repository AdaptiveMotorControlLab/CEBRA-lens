from tqdm import tqdm
import numpy as np
import pickle
import types
from typing import List, Literal, Optional, Tuple, Union
from tsne import Tsne
from CKA import CKA
from RDM import RDM

class _BaseMetric:
    """
    Base class for metric computations.
    """

    def __init__(self):
        pass

    def compute(self, activations):
        #figure something out yo
        pass
    
    # def load(self, filepath):
    #     with open(filepath, "rb") as f:
    #         data = pickle.load(f)
    #     return data
    
    # def save(self, filepath, data):
    #     with open(filepath, "wb") as f:
    #         pickle.dump(data, f)
    
    # def plot(self):
    #     raise NotImplementedError
        

class MultiLayer:
    def _unpack_dataset_arguments(
        self, metrics
    ) -> List[_BaseMetric]:
        if len(metrics) == 0:
            raise ValueError("Need to supply at least one metric.")
        else:
            allowed_classes = [CKA, Tsne]
            #add classes later
            if not set(metrics).issubset(set(allowed_classes)):
                raise ValueError("Value supplied are not in the allowed metrics list. ")
        return metrics
        
    def __init__(self, activations):
        self.activations = activations
        self.metrics = []

    def _get_arguments(self, metric, **kwargs):

        if metric == Tsne:
            keys = ['num_samples']
        elif metric == RDM:
            keys = ['data','label','dataset_label','metric','bool_oracle']
       
        if all(key in kwargs for key in keys):
            parameters = {k: v for k, v in kwargs.items() if k in keys}
        else:
            raise ValueError(f"Missing arguments to do {str(metric)} analysis.")
        return parameters


    def _compute_metric(self, metric, **kwargs):
        result_dict = {}
        for label, values in self.activations.items():
            result_dict[label] = []
            for value in tqdm(values, desc=f"Processing {label}"):
                value = metric(value,**kwargs)
                result_dict[label].append(value.compute())
        return result_dict
    
    def compute(self, *metrics, **kwargs):
        """Computes metrics for multiple activation layers."""
        #unpack the kwargs which are the parameters which are common for all
        self.metrics = self._unpack_dataset_arguments(metrics)
        calculated_metrics= {}
        for metric in self.metrics:
            parameter = self._get_arguments(metric, **kwargs)
            result_dict = self._compute_metric(metric,parameter)
            calculated_metrics[metric] = result_dict
        return calculated_metrics


    #deal with decode later -----
    def decode(self, data_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for label, values in data_dict.items():
            result_dict[label] = []
            for value in tqdm(values, desc=f"Processing {label}"):
                result_dict[label].append(value.decode(*args, **kwargs))
            result_dict[label] = np.array(result_dict[label])
        return result_dict
