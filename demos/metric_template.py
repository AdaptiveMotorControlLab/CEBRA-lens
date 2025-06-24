import numpy as np
from cebra_lens.quantification.base import _BaseMetric
from cebra_lens.utils_plot import *
from typing import List, Optional, Union
import numpy.typing as npt
import matplotlib


class NewMetric(_BaseMetric):
    """
    Compute New Metric on data

    Parameters:
    -----------
    num_samples : int
        The number of samples to use for t-SNE transformation. Default is 200.
    """

    def __init__(
        self,
        arguments,
    ):
        super().__init__()
        self.arguments = arguments

    def _compute_per_layer(self, layer_data: npt.NDArray) -> npt.NDArray:
        """
        Applies NewMetric to the given layer data.

        Parameters:
        -----------
        layer_data: npt.NDArray


        Returns:
        --------
        newmetric_for_layer : npt.NDArray
        """
        # New Metric computation for layer

        return NewMetric_for_layer

    def compute(
        self, data: List[Union[float, npt.NDArray]]
    ) -> List[Union[float, npt.NDArray]]:
        """
        Applies NewMetric to data

        Parameters:
        -----------
        data : List[Union[float, npt.NDArray]]

        Returns:
        --------
        List[Union[float, npt.NDArray]]
            The 2D embedding produced by t-SNE for each layer of a model.
        """

        # Computation logic insert

        # If the computation is done by layer then
        return super().iterate_over_layers(data, self._compute_per_layer)

        # If the computation is done on the data directly
        return result

    @property
    def __name__(self):
        return "NewMetric"

    def plot(
        self,
        embeddings: Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]],
        arguments,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.figure.Figure:
        """
        Plots the NewMetric.

        Parameters:
        -----------
        embeddings : Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]]

        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the NewMetric plot.
        """

        # Need to define the plot_newMetric function in the utils_plot.py
        return plot_newMetric(
            embeddings,
            labels,
            arguments,
            ax,
        )
