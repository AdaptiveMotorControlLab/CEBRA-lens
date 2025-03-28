from tqdm import tqdm
import numpy as np
import pickle
import types
from typing import List, Literal, Optional, Tuple, Union


class _BaseMetric:
    """
    Base class for metric computations.
    """
    def compute(self):
        raise NotImplementedError
    
    def iterate_over_layers(activations, metric_func):
        layer_data = []
        for layer_activation in activations:
            layer_data.append(metric_func(layer_activation))
        return layer_data
    
    def load(self, filepath, data):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data
    
    def save(self, filepath, data):
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    def plot(self):
        raise NotImplementedError
        

class MultiModel:
        
    def __init__(self, metric_class):
        self.metric_class = metric_class
        self.results_dict = {}

    def compute(self, activations_dict, flag=False):
        self.result_dict = {}
        for model_label, activations_list in activations_dict.items():
            self.result_dict[model_label] = []
            if not flag:
                for activations in tqdm(activations_list, desc=f"Processing {model_label}"):
                    self.result_dict[model_label].append(self.metric_class.compute(activations))
            else:
                for model in tqdm(activations_list, desc=f"Processing {model_label}"):
                    self.result_dict[model_label].append(self.metric_class.decode(model))
                self.result_dict[model_label] = np.array(self.result_dict[model_label])
        return self.result_dict
    
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.results_dict = pickle.load(f)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.results_dict, f)
