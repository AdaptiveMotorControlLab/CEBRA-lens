from .decoding import DecodeModel
from tqdm import tqdm
import numpy as np

class MultiModel:

    def __init__(self, metric_class):
        self.metric_class = metric_class
        self.results_dict = {}

    def compute(self, activations_dict):

        self.result_dict = {}
        for model_label, activations_list in activations_dict.items():
            self.result_dict[model_label] = []
            for activations in tqdm(activations_list, desc=f"Processing {model_label}"):
                self.result_dict[model_label].append(
                    self.metric_class.compute(activations)
                )
            if isinstance(self.metric_class, DecodeModel):
                self.result_dict[model_label] = np.array(self.result_dict[model_label])
        return self.result_dict