"""Matplotlib interface to CEBRA-Lens."""

from abc import *
from typing import Optional, Tuple, List, Dict
import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy.typing as npt


class _BasePlot:
    """Base plotting class.

    Attributes:
    ----------
    axis : matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    figsize : Tuple[np.float64, np.float64]
        Figure width and height in inches.
    """

    def __init__(
        self,
        axis: Optional[matplotlib.axes.Axes],
        figsize: Tuple[np.float64, np.float64],
    ):
        if axis is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.ax = axis
            self.fig = self.ax.figure

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError()


class _GenericPlot(_BasePlot):
    """Generic plot class for plotting RDM, distance, and decoding results.

    Attributes:
    ----------
    axis : matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    figsize : Tuple[np.float64, np.float64]
        Figure width and height in inches.
    title : str
        Title of the plot.

    """

    def __init__(
        self, axis: Optional[matplotlib.axes.Axes], figsize: Tuple, title: str
    ):
        super().__init__(axis, figsize)
        self.title = title
        self.unique_keys = []
        self.colors = []

    def plot(self, plot_data: Dict[str, npt.NDArray]) -> None:
        """Create a plot where the x-axis corresponds to layer and the y-axis to the calculated metric per layer.

        Parameters:
        -----------
        plot_data: Dict[str, npt.NDArray]
            Dictionary containing the data to be plotted. Where the keys represent the model label, and the values are the metric values per layer for each model inside a model label category.
        """
        for idx, (key, data_list) in enumerate(plot_data.items()):
            color = self.colors[idx]
            layer_values = []

            for values in data_list:
                layer_values.append(values)
                sns.lineplot(
                    x=np.arange(1, len(values) + 1),
                    y=values,
                    linestyle="-",
                    marker="D",
                    color=color,
                    alpha=0.5,
                    ax=self.ax,  # Ensure correct axis usage
                )

            layer_values = np.array(layer_values)

            mean_values = (
                layer_values if layer_values.ndim == 1 else np.mean(layer_values, axis=0)
            )

            sns.lineplot(
                x=np.arange(1, len(mean_values) + 1),
                y=mean_values,
                linestyle="-",
                marker="D",
                color=color,
                alpha=1,
                label=f"Mean {key}",
                ax=self.ax,
            )

        self.ax.set_title(self.title, fontsize=15)
        sns.despine(ax=self.ax)


class RDMPlot(_GenericPlot):
    """Plot the correlation of Representational Dissimilarity Matrices (RDMs) with Oracle data.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArray]
        Dictionary containing the correlations to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title : str
        Title of the plot.
    figsize : Tuple[np.float64, np.float64]
        Figure width and height in inches.
    axis : matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    """

    def __init__(
        self,
        results_dict: Dict[str, npt.NDArray],
        title: str = "Correlation of RDM with Oracle data across layers",
        figsize: Tuple[np.float64, np.float64] = (15, 5),
        axis: Optional[matplotlib.axes.Axes] = None,
    ):
        super().__init__(axis, figsize, title)

        self.results_dict = results_dict
        self.plot_data = self._transform()
        self.unique_keys = list(self.results_dict.keys())  # Define unique keys here
        self.colors = sns.color_palette("husl", len(self.unique_keys))

    def _transform(self):
        """Transforms ``results_dict`` into a dictionary where the key stays the same, but the values are now corresponding to the correlation between RDM and Oracle data across layers for model label.

        Returns:
        --------
        Dict[str,List[List[np.float64]]]
            Dictionary where the keys correspond to the model labels, and the value to the correlation between RDM and Oracle data for each layer for each model inside a model label category.
        """
        data = {}
        for key, data_list in self.results_dict.items():
            layer_values = []
            for inner_list in data_list:
                values = [arr[1] for arr in inner_list]  # Extract second column
                layer_values.append(values)
            data[key] = layer_values
        return data

    def plot(self):
        """Plots correlation of RDM with Oracle data across layers"""
        return super().plot(self.plot_data)


class DistancePlot(_GenericPlot):
    """Plot the distances across layers for models in ``results_dict``.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArray]
        Dictionary containing the distances to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title: str
        Title of the plot.
    figsize: Tuple[np.float64, np.float64]
        Figure width and height in inches.
    axis: matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    """

    def __init__(
        self,
        results_dict: Dict[str, npt.NDArray],
        title: str = "Distance across layers",
        figsize: Tuple[np.float64, np.float64] = (15, 5),
        axis: Optional[matplotlib.axes.Axes] = None,
    ):
        super().__init__(axis, figsize, title)

        self.results_dict = results_dict
        self.plot_data = self._transform()
        self.unique_keys = list(self.results_dict.keys())  # Define unique keys here
        self.colors = sns.color_palette("husl", len(self.unique_keys))

    def _transform(self) -> Dict[str, List[List[np.float64]]]:
        """Transforms ``results_dict`` into a dictionary where the key stays the same, but the values are now corresponding to the distance metric across layers for model label.

        Returns:
        --------
        Dict[str,List[List[np.float64]]]
            Dictionary where the keys correspond to the model labels, and the value to the distance metric for each layer for each model inside a model label category.
        """
        data = {}
        for idx, (key, data_list) in enumerate(self.results_dict.items()):
            layer_values = []

            for i, inner_list in enumerate(data_list):

                values = [arr for arr in inner_list]  # Extract second column
                layer_values.append(values)
            data[key] = layer_values
        return data

    def plot(self):
        """Plots distance metric across layers"""
        return super().plot(self.plot_data)


class DecodingPlot(_GenericPlot):
    """Plot the decoding accuracy across layers for models in ``results_dict``.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArray]
        Dictionary containing the decoding accuracies to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title: str
        Title of the plot.
    figsize: Tuple[np.float64, np.float64]
        Figure width and height in inches.
    axis: matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    """

    def __init__(
        self,
        results_dict: Dict[str, npt.NDArray],
        title: str = "Decoding accuracy across layers",
        figsize: Tuple[np.float64, np.float64] = (15, 5),
        axis: Optional[matplotlib.axes.Axes] = None,
    ):
        super().__init__(axis, figsize, title)

        self.results_dict = results_dict
        self.plot_data = self._transform()
        self.unique_keys = list(self.results_dict.keys())  # Define unique keys here
        self.colors = sns.color_palette("husl", len(self.unique_keys))

    def _transform(self) -> Dict[str, List[List[np.float64]]]:
        """Transforms ``results_dict`` into a dictionary where the key stays the same, but the values are now corresponding to the decoding accuracies across layers for model label.

        Returns:
        --------
        Dict[str,List[List[np.float64]]]
            Dictionary where the keys correspond to the model labels, and the value to the decoding accuracies for each layer for each model inside a model label category.
        """
        data = {}
        for idx, (key, data_list) in enumerate(self.results_dict.items()):
            layer_values = []

            for i, inner_list in enumerate(data_list):

                values = [arr[2] for arr in inner_list]  # Extract second column
                layer_values.append(values)
            data[key] = layer_values
        return data

    def plot(self):
        """Plots decoding accuracy across layers"""
        return super().plot(self.plot_data)


def plot_rdm_correlation(
    rdm_dict: Dict[str, npt.NDArray],
    title: str = "RDM comparison to Oracle",
    figsize: Tuple[np.float64, np.float64] = (15, 5),
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots the correlation of Representational Dissimilarity Matrices (RDMs) with Oracle data.

    Parameters:
    -----------
    rdm_dict : Dict[str, npt.NDArray]
        Dictionary containing the RDMs to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title : str, optional
        The title for the plot (default is "RDM comparison to Oracle").
    figsize : Tuple, optional
        A Tuple representing the figure size (default is (15, 5)).
    ax : matplotlib.axes.Axes, optional

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return RDMPlot(results_dict=rdm_dict, title=title, figsize=figsize, axis=ax).plot(
        **kwargs
    )


def plot_distance(
    distance_dict: Dict[str, npt.NDArray],
    title: str = "Inter-repetition distance",
    figsize: Tuple[np.float64, np.float64] = (15, 5),
    **kwargs,
) -> plt.Figure:
    """
    Plots the distances across layer for models in ``distance_dict``.

    Parameters:
    -----------
    distance_dict : Dict[str, npt.NDArray]
        A dictionary containing the distances to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title : str, optional
        The title for the plot (default is "Inter-repetition distance").
    figsize : Tuple, optional
        A Tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return DistancePlot(
        results_dict=distance_dict,
        title=title,
        figsize=figsize,
    ).plot(**kwargs)


def plot_layer_decoding(
    results_dict: Dict[str, npt.NDArray],
    title: str = "Decoding by layer",
    figsize: Tuple[np.float64, np.float64] = (15, 5),
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding accuracy across layer for models in results_dict.

    Parameters:
    -----------
    results_dict : Dict[str, npt.NDArray]
        A dictionary containing the decoding results to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title : str, optional
        The title for the plot (default is "Decoding by layer").
    figsize : Tuple, optional
        A Tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return DecodingPlot(
        results_dict=results_dict,
        title=title,
        figsize=figsize,
    ).plot(**kwargs)


class ModelDecodingPlot(_BasePlot):
    """Plotting decoding accuracy across models.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArrays]
        A dictionary where the keys are model category labels or model file names and the values are 2D arrays containing decoding results.
    palette : str
        The color palette to use for the plot. Default is "hls".
    dataset_label : str
        The dataset type. Currently only "visual" is supported.
    axis : matplotlib.axes.Axes, optional
        The axis on which to plot. If None, a new axis will be created.
    """

    def __init__(
        self,
        results_dict: Dict[str, npt.NDArrays],
        palette: str,
        dataset_label: str,
        axis: Optional[matplotlib.axes.Axes],
    ):

        self.figsize = (
            len(results_dict) * 2,
            6,
        )  # Set figure size based on the number of models
        super().__init__(
            axis, self.figsize
        )  # Call parent constructor to initialize self.fig and self.ax
        self.results_dict = results_dict
        self.palette = sns.color_palette(
            palette, len(results_dict)
        )  # Define a color palette
        self.dataset_label = dataset_label  # Define dataset label

    def plot(self, **kwargs) -> None:
        """Plotting logic to plot the decoding accuracies across models where the x-axis are the model labels, and the y-axis are the decoding accuracy values in (%)."""
        x_positions = list(
            range(1, len(self.results_dict) + 1)
        )  # X positions for scatter points

        if self.dataset_label == "visual":  # Only handle 'visual' dataset label for now
            for i, (key, results) in enumerate(self.results_dict.items()):
                acc = results[
                    :, 2
                ]  # Extract accuracy data from the 3rd column (index 2)
                mean_error = np.mean(acc)  # Calculate the mean accuracy
                color = self.palette[i]  # Get the color from the palette
                self.ax.scatter(
                    np.ones_like(acc) * x_positions[i], acc, color=color, alpha=0.3
                )

                # Plot the mean accuracy
                self.ax.scatter(
                    x_positions[i],
                    mean_error,
                    color=color,
                    s=50,
                    label=f"Mean {key}",
                    zorder=5,  # Bring mean point to the top
                )

            self.ax.set_xlabel("Model")
            self.ax.set_ylabel("Accuracy (%)")
            self.ax.set_title("Comparison of Accuracy Across Models")
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(
                self.results_dict.keys()
            )  # Set model names as x-tick labels
            self.ax.legend()  # Show legend for model labels
            sns.despine(ax=self.ax)  # Remove top and right spines for aesthetic reasons
        else:
            raise NotImplementedError(
                f"Plotting of {self.dataset_label} is not handled yet. Only 'visual' is for now. "
            )


def plot_decoding(
    results_dict: Dict[str, npt.NDArrays],
    palette: str = "hls",
    dataset_label="visual",
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding accuracy across multiple models.

    Parameters:
    -----------
    results_dict : Dict[str, npt.NDArrays]
        A dictionary where the keys are model category labels or model file names and the values are 2d-arrays containing decoding results.
    palette: str, optional (default is "hls")
        The color palette to use for the plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure displaying the comparison of decoding accuracy across models.
    """
    return ModelDecodingPlot(
        results_dict=results_dict,
        axis=ax,
        palette=palette,
        dataset_label=dataset_label,
    ).plot(**kwargs)


class _EmbeddingComparisonPlot:
    """Plot the embedding visualization for comparison across layers.

    Attributes:
    ----------
    embeddings_1 : List[npt.NDArray]
        A list of embeddings for the first set of data.
    embeddings_2 : List[npt.NDArray]
        A list of embeddings for the second set of data.
    labels : npt.NDArray
        An array of labels corresponding to the data labels.
    sample_plot : int
        The number of samples to plot from the embeddings.
    comparison_labels : Tuple
        A Tuple containing the type of embedding and a list of two strings representing the labels for the two sets of embeddings.
    dataset_label : str
        A string representing the label for the data being plotted.
    axis : matplotlib.axes.Axes, optional
        The axis on which to plot the embeddings.
    """

    def __init__(
        self,
        embeddings_1: List[npt.NDArray],
        embeddings_2: List[npt.NDArray],
        labels: npt.NDArray,
        sample_plot: int,
        comparison_labels: Tuple,
        dataset_label: str,
        axis: Optional[matplotlib.axes.Axes],
    ):

        self.figsize = (15, 10)
        self.embeddings_1 = embeddings_1
        self.embeddings_2 = embeddings_2
        self.labels = labels
        self.sample_plot = sample_plot
        self.comparison_labels = comparison_labels
        self.dataset_label = dataset_label

        self.num_layers_1 = len(embeddings_1)
        self.num_layers_2 = len(embeddings_2)

        # Padding the shorter embedding to match the number of layers in the longer embedding
        if self.num_layers_1 > self.num_layers_2:
            embeddings_2 += [np.empty_like(embeddings_2[0])] * (
                self.num_layers_1 - self.num_layers_2
            )
        elif self.num_layers_2 > self.num_layers_1:
            embeddings_1 += [np.empty_like(embeddings_1[0])] * (
                self.num_layers_2 - self.num_layers_1
            )

        self.ax = self._define_ax(axis)
        self.axs_1 = self.ax[0, :]
        self.axs_2 = self.ax[1, :]

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Parameters:
        -----------
        axis: matplotlib.axes.Axes, optional
            A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
        -----------
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.fig, self.ax = plt.subplots(
                2,
                max(self.num_layers_1, self.num_layers_2),
                figsize=(15, 10),
                subplot_kw={"projection": "3d"},
            )

        else:
            self.ax = axis
        return self.ax

    def _plot_hippocampus(
        ax: matplotlib.axes.Axes,
        embedding: npt.NDArray,
        label: str,
        gray: bool = False,
        idx_order: Tuple[int, int, int] = (0, 1, 2),
    ) -> matplotlib.axes.Axes:
        """Plot the hippocampus embedding."""
        r_ind = label[:, 1] == 1
        l_ind = label[:, 2] == 1

        if not gray:
            r_cmap = "cool"
            l_cmap = "magma"
            r_c = label[r_ind, 0]
            l_c = label[l_ind, 0]
        else:
            r_cmap = None
            l_cmap = None
            r_c = "gray"
            l_c = "gray"

        idx1, idx2, idx3 = idx_order
        r = ax.scatter(
            embedding[r_ind, idx1],
            embedding[r_ind, idx2],
            embedding[r_ind, idx3],
            c=r_c,
            cmap=r_cmap,
            s=0.05,
            alpha=0.75,
        )
        l = ax.scatter(
            embedding[l_ind, idx1],
            embedding[l_ind, idx2],
            embedding[l_ind, idx3],
            c=l_c,
            cmap=l_cmap,
            s=0.05,
            alpha=0.75,
        )

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        return ax

    def _plot_allen(
        self,
        ax: matplotlib.axes.Axes,
        embedding: npt.NDArray,
        label: str,
        gray: bool = False,
        idx_order: Tuple[int, int, int] = (0, 1, 2),
    ):
        """Plot the Allen embedding."""
        c = label

        idx1, idx2, idx3 = idx_order
        ax.scatter(
            embedding[:, idx1],
            embedding[:, idx2],
            embedding[:, idx3],
            c=c,
            cmap="magma",
            s=0.05,
            alpha=0.75,
        )

        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("w")
        ax.yaxis.pane.set_edgecolor("w")
        ax.zaxis.pane.set_edgecolor("w")

        return ax

    def _plot_embedding_layers(
        self,
        axs: List[matplotlib.axes.Axes],
        embeddings: List[npt.NDArray],
        title_prefix: str,
    ):
        """
        Plots the embedding layers on the provided axes. Used in tSNE and in normal CEBRA.

        Parameters:
        -----------
        axs : List[matplotlib.axes.Axes]
            List of matplotlib axes objects where the embeddings will be plotted.
        embeddings : List[npt.NDArray]
            List of numpy arrays containing the embeddings for each layer. Each array is shape Samples X num Neurons.
        title_prefix : str
            Title of the plot (e.g., 'single' or 'multi').
        """
        num_layers = len(embeddings)

        labels_list = [self.labels[: self.sample_plot]] * num_layers
        titles = [f"{title_prefix} Layer {layer}" for layer in range(1, num_layers)]
        titles.append(f"{title_prefix} Output")

        for i, (label, ax) in enumerate(zip(labels_list, axs)):
            if (
                embeddings[i].shape[0] < embeddings[i].shape[1]
            ):  # should be num Samples X num Neurons
                embedding = embeddings[i].T
            else:
                embedding = embeddings[i]

            embedding = embedding[: self.sample_plot, :]
            if self.dataset_label == "HPC":
                ax = self._plot_hippocampus(ax, embedding, label)
            elif self.dataset_label == "visual":
                ax = self._plot_allen(ax, embedding, label)
            else:
                raise NotImplementedError(
                    f"label {self.dataset_label} not yet implemented. Use either visual or HPC"
                )

            ax.set_title(titles[i], y=1)
            ax.axis("off")

    def plot(self):
        """Plots embedding layers for models being compared"""
        self._plot_embedding_layers(
            self.axs_1, self.embeddings_1, self.comparison_labels[1][0]
        )
        self._plot_embedding_layers(
            self.axs_2, self.embeddings_2, self.comparison_labels[1][1]
        )
        self.fig.suptitle(
            f"{self.comparison_labels[0]} across layers({self.comparison_labels[1][0]} - {self.comparison_labels[1][1]})",
            fontsize=20,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()


def compare_embeddings_layers(
    embeddings_1: List[npt.NDArray],
    embeddings_2: List[npt.NDArray],
    labels: npt.NDArray,
    sample_plot: int = 200,
    comparison_labels: Tuple = ("tSNE", ["Untrained", "Trained"]),
    dataset_label: str = "HPC",
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Compare embeddings across layers for two datasets.

    Parameters:
    -----------
    embeddings_1 : List[npt.NDArray]
        List of embeddings for the first dataset.
    embeddings_2 : List[npt.NDArray]
        List of embeddings for the second dataset.
    labels : npt.NDArray
        Array of labels for the data points.
    sample_plot : int, optional
        Number of samples to plot (default is 200).
    comparison_labels : Tuple, optional
        Labels describing the embeddings (default is ("tSNE", ["Untrained", "Trained"]) ).
    dataset_label : str, optional
        Dataset identifier (default is "HPC").
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object (default is None).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the embedding comparison plots.
    """
    return _EmbeddingComparisonPlot(
        embeddings_1=embeddings_1,
        embeddings_2=embeddings_2,
        labels=labels,
        sample_plot=sample_plot,
        comparison_labels=comparison_labels,
        dataset_label=dataset_label,
        axis=ax,
    ).plot(**kwargs)


class _ActivationPlot:
    """Class for plotting activations of a neural network model.

    Attributes:
    ----------
    input_data : torch.Tensor
        The input data tensor to be plotted.
    embeddings : List[npt.NDArray]
        A list of npt.NDArrays representing the embeddings/activations of each layer. Each array is shape Samples X num Neurons.
    figsize : Tuple[np.float64, np.float64]
        The size of the figure (width, height).
    axis : matplotlib.axes.Axes, optional
        The axis on which to plot the activations. If None, a new axis will be created.
    sample_plot : int
        The number of samples to plot along the time axis (default is 100).
    cmap : str
        The colormap to use for the embeddings (default is "magma").
    title : str
        The title of the plot (default is "Trained activations").
    """

    def __init__(
        self,
        input_data: torch.Tensor,
        embeddings: List[npt.NDArray],
        figsize: Tuple[np.float64, np.float64],
        axis: Optional[matplotlib.axes.Axes],
        sample_plot: int = 100,
        cmap: str = "magma",
        title: str = "Trained activations",
    ):
        self.figsize = figsize
        self.input_data = input_data
        self.embeddings = embeddings
        self.sample_plot = sample_plot
        self.cmap = cmap
        self.num_layers = len(embeddings)
        self._define_ax(axis)
        self.fig.suptitle(title, fontsize=20)

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Args:
        axis: matplotlib.axes.Axes, optional
            A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.fig, self.axes = plt.subplots(
                self.num_layers + 1, 1, figsize=self.figsize
            )
        else:
            self.axes = [axis] + [
                axis.figure.add_subplot(self.num_layers + 1, 1, i + 2)
                for i in range(self.num_layers)
            ]

    def plot(self):
        """Handles plotting logic."""
        self.axes[0].imshow(self.input_data.T[:, 0 : self.sample_plot], aspect="auto")
        self.axes[0].set_title("Input Data")
        self.axes[0].set_ylabel("Channel #")
        self.axes[0].set_xlabel("Time")
        self.axes[0].grid(False)

        # Plot the embeddings for each layer
        for i in range(self.num_layers):
            self.axes[i + 1].imshow(
                self.embeddings[i][:, 0 : self.sample_plot],
                cmap=self.cmap,
                aspect="auto",
            )
            if i == self.num_layers - 1:
                layer_title = "Output Layer"
            else:
                layer_title = f"Layer {i + 1}"
            self.axes[i + 1].set_title(layer_title)
            self.axes[i + 1].set_ylabel("Unit #")
            self.axes[i + 1].set_xlabel("Time")
            self.axes[i + 1].grid(False)

        # Adjust layout for better spacing
        plt.tight_layout()


def plot_activations(
    input_data: torch.Tensor,
    embeddings: List[npt.NDArray],
    sample_plot: int = 100,
    cmap: str = "magma",
    title: str = "Trained activations",
    figsize: Tuple = (10, 20),
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots the activations of a neural network model.

    Parameters:
    -----------
    input_data : torch.Tensor
        The input data tensor to be plotted.
    embeddings : List[npt.NDArray]
        A list of npt.NDArrays representing the embeddings/activations of each layer. Each array is shape Samples X num Neurons.
    sample_plot : int, optional
        The number of samples to plot along the time axis (default is 100).
    cmap : str, optional
        The colormap to use for the embeddings (default is "magma").
    title : str, optional
        The title of the plot (default is "Trained activations").
    figsize : Tuple, optional
        The size of the figure (default is (10, 20)).
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the activations. If None, a new axis will be created.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plots.
    """
    return _ActivationPlot(
        input_data=input_data,
        embeddings=embeddings,
        sample_plot=sample_plot,
        cmap=cmap,
        title=title,
        figsize=figsize,
        axis=ax,
    ).plot(**kwargs)


class _HeatMapsPlot:
    """Class for plotting CKA heatmaps.

    Attributes:
    ----------
    cka_matrices : Dict[str, npt.NDArrays]
        A dictionary where the keys are the comparison names and the values are the CKA matrices.
    annot : bool
        If True, shows the values in the heatmap cells.
    axis : matplotlib.axes.Axes
        The axis on which to plot the heatmaps. If None, a new axis will be created.
    show_cbar : bool
        If True, shows the color bar.
    cbar_label : str
        Label for the color bar.
    color_map : str
        The color map to use for the heatmaps.
    figsize : Tuple[np.float64, np.float64]
        The size of the figure (width, height).
    """

    def __init__(
        self,
        cka_matrices: Dict[str, npt.NDArrays],
        annot: bool,
        axis: Optional[matplotlib.axes.Axes],
        show_cbar: bool = True,
        cbar_label: str = "CKA score",
        color_map: str = "magma",
        figsize: Tuple = (15, 5),
    ):
        self.cka_matrices = cka_matrices
        self.annot = annot
        self.show_cbar = show_cbar
        self.cbar_label = cbar_label
        self.color_map = color_map
        self.figsize = figsize
        self.num_matrices = len(cka_matrices)

        self.num_comparisons = len(cka_matrices)
        self.axs = self._define_ax(axis)
        self.cbar_ax = self.fig.add_axes([0.13, -0.04, 0.3, 0.03])
        self.heatmap_kwargs = {
            "cbar": show_cbar,
            "cbar_ax": self.cbar_ax,
            "vmin": 0,
            "vmax": 1,
            "cmap": color_map,
            "cbar_kws": {"label": cbar_label, "orientation": "horizontal"},
        }
        if self.num_comparisons == 1:
            self.axs = [self.axs]  # handle the 1 comparison case

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Parameters:
        -----------
        axis: matplotlib.axes.Axes, optional
            A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
        -----------
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.fig, self.axs = plt.subplots(
                1, self.num_comparisons, figsize=self.figsize
            )

        else:
            self.axs = axis
        return self.axs

    def plot(self):
        """Handles plotting logic."""
        for i, (key, value) in enumerate(self.cka_matrices.items()):

            sns.heatmap(value, ax=self.axs[i], annot=self.annot, **self.heatmap_kwargs)

            num_layers = value.shape[1]
            num_models = value.shape[0]

            self.axs[i].set_title(key.replace("_", " "))
            self.axs[i].set_xlabel("Layer")
            if i == 0:
                self.axs[i].set_ylabel("Model Instantiation", fontsize=12)
                self.axs[i].set_yticks(np.arange(num_models) + 0.5)
                self.axs[i].set_yticklabels([m for m in range(1, num_models + 1)])
            else:
                self.axs[i].set_ylabel("")
                self.axs[i].set_yticks([])

            self.axs[i].set_xticks(np.arange(num_layers) + 0.5)
            self.axs[i].set_xticklabels([f"L{l}" for l in range(1, num_layers + 1)])

        # Adjust layout
        plt.subplots_adjust(wspace=0.1, right=0.9)
        self.fig.suptitle("Similarity between model representations (CKA)", fontsize=16)


def plot_cka_heatmaps(
    cka_matrices: Dict[str, npt.NDArrays],
    annot: bool,
    show_cbar: bool = True,
    cbar_label: str = "CKA score",
    color_map: str = "magma",
    figsize: Tuple = (15, 5),
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    This function generates heatmaps for various CKA matrices to visualize the similarity between different sets of embeddings.

    Parameters:
    -----------
    cka_matrices : Dict[str, npt.NDArrays]
        Dictionary of CKA matrices where the keys are the comparison names and the values the matrices.
    show_cbar : bool
        If True, shows the color bar.
    cbar_label : str
        Label for the color bar.
    color_map : str or matplotlib.colors.Colormap
        The palette to use for the heatmap.
    figsize : Tuple
        Size of the figure for the subplots (width, height).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the CKA heatmaps.
    """
    return _HeatMapsPlot(
        cka_matrices=cka_matrices,
        annot=annot,
        show_cbar=show_cbar,
        cbar_label=cbar_label,
        color_map=color_map,
        figsize=figsize,
        axis=ax,
    ).plot()


class _RDMPlots:
    """Class for plotting Representational Dissimilarity Matrices (RDMs).

    Attributes:
    ----------
    rdms : List[Tuple[npt.NDArray,np.float64]]
        A list of RDMs to be plotted. Each RDM should be a 2D array-like structure.
    titles : List[str]
        A list of titles for each RDM plot. The length of this list should match the length of `rdms`.
    metric : str
        The distance metric which was used for computing the RDMs. Default is 'correlation'.
    dataset_label : str
        The type of dataset being used for decoding (default is "visual").
    cmap : str
        The color map to use for the plotting (default is "viridis").
    figsize : Tuple[np.float64, np.float64]
        The size of the figure (width, height).
    """

    def __init__(
        self,
        rdms: List[Tuple[npt.NDArray, np.float64]],
        titles: List[str],
        axis: Optional[matplotlib.axes.Axes],
        metric: str = "Normalized Euclidean distance",
        dataset_label: str = "visual",
        cmap: str = "viridis",
        figsize: Tuple[np.float64, np.float64] = None,
    ):

        self.rdms = rdms
        self.titles = titles
        self.metric = metric
        self.dataset_label = dataset_label
        self.cmap = cmap
        self.figsize = figsize
        self.ax = self._define_ax(axis)

        if len(self.rdms) == 1:
            self.ax = [self.ax]

        if self.figsize == None:
            self.y_size = max(6, 3 * len(rdms))
            self.x_size = max(6, 5 * len(rdms))
        elif type(self.figsize) == Tuple:
            self.x_size = self.figsize[0]
            self.y_size = self.figsize[1]

        self.fig.set_size_inches(self.x_size, self.y_size)

        if len(self.rdms) != len(self.titles):
            raise ValueError(
                "The two lists (rdms and titles) must have the same length."
            )

        # Generate tick labels specific to the dataset
        if dataset_label == "visual":
            self.tick_labels = [str(i) for i in range(0, 930, 30)]

        elif self.dataset_label == "HPC":
            # TODO
            raise NotImplementedError
        else:
            raise NotImplementedError(
                f"RDM Plotting for dataset {self.dataset_label} not yet implemented. Please use 'visual' or 'HPC'."
            )

    def _define_ax(self, axis: Optional[matplotlib.axes.Axes]) -> matplotlib.axes.Axes:
        """Define the ax on which to generate the plot.

        Parameters:
        -----------
        axis : matplotlib.axes.Axes, optional
            A required ``matplotlib.axes.Axes``. If None, then add an axis to the current figure.

        Returns:
        -----------
            A ``matplotlib.axes.Axes`` on which to generate the plot.
        """
        if axis is None:
            self.fig, self.ax = plt.subplots(1, len(self.rdms))

        else:
            self.ax = axis
        return self.ax

    def plot(self):
        """Handles plotting logic."""
        for i, rdm in enumerate(self.rdms):

            cax = self.ax[i].imshow(rdm, cmap=self.cmap, aspect="auto")
            self.ax[i].set_title(self.titles[i])
            num_ticks = len(self.tick_labels)
            self.ax[i].set_xticks(np.linspace(0, rdm.shape[1] - 1, num_ticks))
            self.ax[i].set_yticks(np.linspace(0, rdm.shape[0] - 1, num_ticks))
            self.ax[i].set_xticklabels(self.tick_labels, rotation=90, ha="right")
            self.ax[i].set_yticklabels(self.tick_labels)

        plt.suptitle("Representational Dissimilarity Matrix (RDM)")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        self.fig.colorbar(
            cax, ax=self.ax, orientation="horizontal", fraction=0.05, label=self.metric
        )


def plot_rdm(
    rdms: list,
    titles: List[str],
    metric: Optional[str] = "Normalized Euclidean distance",
    dataset_label: Optional[str] = "visual",
    cmap: Optional[str] = "viridis",
    figsize: Optional[Tuple[np.float64, np.float64]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Plots Representational Dissimilarity Matrices (RDMs) with given titles and metric.

    Parameters:
    -----------
    rdms : List[Tuple[npt.NDArray,np.float64]]
        A list of RDMs to be plotted. Each RDM should be a 2D array-like structure.
    titles : List[str]
        A list of titles for each RDM plot. The length of this list should match the length of `rdms`.
    metric : str, optional
        The metric used for the RDM, which will be displayed as the colorbar label. Default is "Normalized Euclidean distance".
    dataset_label : str, optional
        The label of the dataset, which determines the tick labels. Default is "visual".
        Currently supported values are "visual" and "HPC".
    cmap : str, optional
        The color map to use for the plotting.
    figsize : Tuple[np.float64,np.float64], optional
        The size of the figure (width, height). Default is None.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the RDMs. If None, a new axis will be created.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plotted RDMs.
    """
    return _RDMPlots(
        rdms=rdms,
        titles=titles,
        metric=metric,
        dataset_label=dataset_label,
        cmap=cmap,
        figsize=figsize,
        axis=ax,
    ).plot()
