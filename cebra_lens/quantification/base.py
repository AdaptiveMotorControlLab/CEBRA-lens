from tqdm import tqdm


class _BaseMetric:
    """
    Base class for metric computations.
    """

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class _MultiMetric(_BaseMetric):

    def compute(self, activations_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for model_label, activations in activations_dict.items():
            result_dict[model_label] = []
            for activation in tqdm(activations, desc=f"Processing {model_label}"):
                result_dict[model_label].append(activation.compute(*args, **kwargs))
        return result_dict
