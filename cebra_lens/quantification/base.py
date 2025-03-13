from tqdm import tqdm


class _BaseMetric:
    """
    Base class for metric computations.
    """

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")


class _MultiMetric(_BaseMetric):

    def compute(self, data_dict, *args, **kwargs):
        """Computes metrics for multiple activation layers."""
        result_dict = {}
        for label, values in data_dict.items():
            result_dict[label] = []
            for value in tqdm(values, desc=f"Processing {label}"):
                result_dict[label].append(value.compute(*args, **kwargs))
        return result_dict
