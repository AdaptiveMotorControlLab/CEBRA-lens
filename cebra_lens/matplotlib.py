"""Matplotlib interface to CEBRA-Lens."""

from abc import *
from typing import Optional, Tuple, List, Dict, Union
import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy.typing as npt
import random


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
                layer_values
                if layer_values.ndim == 1
                else np.mean(layer_values, axis=0)
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


class RDMPlotOracle(_GenericPlot):
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
                # getting the oracle data
                values = [arr[1] for arr in inner_list]
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

                values = [arr for arr in inner_list]
                layer_values.append(values)
            data[key] = layer_values
        return data

    def plot(self):
        """Plots distance metric across layers"""
        return super().plot(self.plot_data)


class DecodingPlot(_GenericPlot):
    """Plot the decoding scores across layers for models in ``results_dict``.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArray]
        Dictionary containing the decoding scores to be plotted. Please refer to the ``plot_data`` argument in the ``plot`` function from the inherited class.
    title: str
        Title of the plot.
    figsize: Tuple[np.float64, np.float64]
        Figure width and height in inches.
    axis: matplotlib.axes.Axes, optional
        Optional axis to create the plot on.
    """

    def __init__(
        self,
        results_dict: Dict[str, Dict[int, Tuple[np.float64, list, list]]],
        dataset_label: str = None,
        title: str = None,
        label: int = None,
        plot_error: bool = False,
        figsize: Tuple[np.float64, np.float64] = (15, 5),
        axis: Optional[matplotlib.axes.Axes] = None,
    ):

        if title is not None:
            if dataset_label == "visual":
                title = "Decoding accuracies across layers (%)"
            elif dataset_label == "HPC":
                title = "Decoding position errors across layers (cm)"
            else:
                title = "Decoding average R^2 scores across layers"
                if plot_error:
                    title = "Decoding error scores across layers"

        super().__init__(axis, figsize, title)
        self.label = label
        self.plot_error = plot_error
        self.dataset_label = dataset_label
        self.results_dict = results_dict
        self.plot_data = self._transform()
        self.unique_keys = list(self.results_dict.keys())  # Define unique keys here
        self.colors = sns.color_palette("husl", len(self.unique_keys))

    def _transform(self) -> Dict[str, List[List[np.float64]]]:
        """Transforms ``results_dict`` into a dictionary where the key stays the same, but the values are now corresponding to the decoding scores across layers for model label.

        Returns:
        --------
        Dict[str,List[List[np.float64]]]
            Dictionary where the keys correspond to the model labels, and the value to the decoding scores for each layer for each model inside a model label category.
        """
        data = {}
        for idx, (group_name, models) in enumerate(self.results_dict.items()):
            layer_values = []

            for i, model in enumerate(models):

                if self.dataset_label == "visual":
                    ind = 2
                elif self.dataset_label == "HPC":
                    ind = 2
                else:
                    ind = self.label
                layer_scores = []
                for layer, scores in model.items():
                    if self.dataset_label is None:
                        if self.plot_error:
                            layer_scores.append(scores[1][ind])
                        else:
                            layer_scores.append(scores[2][ind])
                    else:
                        layer_scores.append(scores[ind])
                layer_values.append(layer_scores)
            data[group_name] = layer_values
        return data

    def plot(self):
        """Plots decoding accuracy across layers"""
        return super().plot(self.plot_data)


def plot_rdm_correlation(
    rdm_dict: Dict[str, List[npt.NDArray]],
    title: str = "RDM comparison to Oracle",
    figsize: Tuple[np.float64, np.float64] = (15, 5),
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots the correlation of Representational Dissimilarity Matrices (RDMs) with Oracle data.

    Parameters:
    -----------
    rdm_dict : Dict[str, List[npt.NDArray]]
        Dictionary containing the RDMs to be plotted.
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

    return RDMPlotOracle(
        results_dict=rdm_dict, title=title, figsize=figsize, axis=ax
    ).plot(**kwargs)


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
    dataset_label: str = None,
    label: int = None,
    plot_error: bool = False,
    figsize: Tuple[np.float64, np.float64] = (15, 5),
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding score across layer for models in results_dict.

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
        The generated figure containing the decoding scored per layer per model.
    """

    return DecodingPlot(
        results_dict=results_dict,
        title=title,
        dataset_label=dataset_label,
        label=label,
        plot_error=plot_error,
        figsize=figsize,
    ).plot(**kwargs)


class ModelDecodingPlot(_BasePlot):
    """Plotting decoding scores across models.

    Attributes:
    ----------
    results_dict : Dict[str, npt.NDArray]
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
        results_dict: Dict[str, npt.NDArray],
        palette: str,
        dataset_label: str,
        axis: Optional[matplotlib.axes.Axes],
        label: int = None,
        plot_error: bool = False,
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
        self.plot_error = plot_error
        self.label = label
        if self.dataset_label is None and self.label:
            raise ValueError("Please define the label score you want to plot.")

    def plot(self, **kwargs) -> None:
        """Plotting logic to plot the decoding scores across models where the x-axis are the model labels, and the y-axis are the decoding scores values."""
        x_positions = list(
            range(1, len(self.results_dict) + 1)
        )  # X positions for scatter points

        for i, (key, results) in enumerate(self.results_dict.items()):
            if self.dataset_label == "visual":
                # for visual dataset get accuracy
                score = [dict_el[0][2] for dict_el in results]
                self.plot_label = "Accuracy"
                measure = "(%)"
            elif self.dataset_label == "HPC":
                # for HPC dataset get position error
                score = [dict_el[0][1] for dict_el in results]
                self.plot_label = "Position Error"
                measure = "(cm)"
            else:
                if self.plot_error:
                    # betwen error and R^2 score, you want to plot the error
                    score = [dict_el[0][1][i] for dict_el in results]
                    self.plot_label = "Error score"
                # choice of label to plot, self.metric
                else:
                    score = [dict_el[0][2][i] for dict_el in results]
                    self.plot_label = "R^2 score"
                    measure = ""

            mean_error = np.mean(score)
            color = self.palette[i]
            self.ax.scatter(
                np.ones_like(score) * x_positions[i], score, color=color, alpha=0.3
            )
            self.ax.scatter(
                x_positions[i],
                mean_error,
                color=color,
                s=50,
                label=f"Mean {key}",
                zorder=5,  # Bring mean point to the top
            )
        self.ax.set_xlabel("Model")
        self.ax.set_ylabel(f"{self.plot_label} {measure}")
        self.ax.set_title(f"Comparison of {self.plot_label} Across Models")
        self.ax.set_xticks(x_positions)
        self.ax.set_xticklabels(
            self.results_dict.keys()
        )  # Set model names as x-tick labels
        self.ax.legend()  # Show legend for model labels
        sns.despine(ax=self.ax)  # Remove top and right spines for aesthetic reasons


def plot_decoding(
    results_dict: Dict[str, npt.NDArray],
    palette: str = "hls",
    dataset_label: str = None,
    label: int = None,
    metric: int = 0,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding scores across multiple models for a label.

    Parameters:
    -----------
    results_dict : Dict[str, npt.NDArray]
        A dictionary where the keys are model category labels or model file names and the values are 2d-arrays containing decoding results.
    palette: str, optional (default is "hls")
        The color palette to use for the plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure displaying the comparison of decoding scores across models.
    """
    return ModelDecodingPlot(
        results_dict=results_dict,
        axis=ax,
        palette=palette,
        dataset_label=dataset_label,
        metric=metric,
        label=label,
    ).plot(**kwargs)


class _EmbeddingPlot:
    """Plot the embedding visualization across layers.

    Attributes:
    ----------
    embeddings : List[npt.NDArray]
        A list of embeddings. If it contains only one list inside then it is to plot embeddings, but if it contains two sets of data inside then it is for embedding comparison across layers
    labels : npt.NDArray
        An array of labels corresponding to the data labels.
    sample_plot : int
        The number of samples to plot from the embeddings.
    comparison_groups : Tuple
        A Tuple containing the type of embedding and a list of two strings representing the labels for the two sets of embeddings.
    dataset_label : str
        A string representing the label for the data being plotted.
    axis : matplotlib.axes.Axes, optional
        The axis on which to plot the embeddings.
    """

    def __init__(
        self,
        embeddings: List[npt.NDArray],
        labels: npt.NDArray,
        dataset_label: str,
        axis: Optional[matplotlib.axes.Axes],
        sample_plot: int = None,
        comparison_groups: Tuple = None,
    ):
        self.figsize = (15, 10)
        self.embeddings_list = embeddings
        self.labels = labels
        self.dataset_label = dataset_label
        self.axs = self._define_ax(axis)
        if len(embeddings) == 1:
            self.embeddings = embeddings[0]
            if sample_plot is None:
                self.sample_plot = self.embeddings[0].shape[1]
            else:
                self.sample_plot = sample_plot

        else:
            self.embeddings_1 = embeddings[0]
            self.embeddings_2 = embeddings[1]
            if sample_plot is None:
                self.sample_plot = self.embeddings_1[0].shape[1]
            else:
                self.sample_plot = sample_plot

            self.comparison_groups = comparison_groups

            self.axs_1 = self.ax[0, :]
            self.axs_2 = self.ax[1, :]

    def _multi_padding_check(self, embeddings_1, embeddings_2):

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
            if len(self.embeddings_list) == 2:
                self._multi_padding_check(
                    self.embeddings_list[0], self.embeddings_list[1]
                )
                self.fig, self.ax = plt.subplots(
                    2,
                    max(self.num_layers_1, self.num_layers_2),
                    figsize=(15, 10),
                    subplot_kw={"projection": "3d"},
                )
            else:
                self.fig, self.ax = plt.subplots(
                    1,
                    len(self.embeddings_list[0]),
                    figsize=(15, 10),
                    subplot_kw={"projection": "3d"},
                )
        else:
            self.ax = axis
        return self.ax

    def _plot_dataset(
        self,
        ax: matplotlib.axes.Axes,
        embedding: npt.NDArray,
        label: str,
        gray: bool = False,
        idx_order: Tuple[int, int, int] = (0, 1, 2),
    ):
        """
        Plot the dataset embedding, for generic dataset.
        Will plot all labels.
        """
        idx1, idx2, idx3 = idx_order
        available_palettes = list(sns.palettes.SEABORN_PALETTES.keys())
        label = np.atleast_2d(label)
        if label.shape[0] == 1 and label.shape[1] != 1:
            label = label.T

        for num_labels in range(len(label)):
            l_ind = label[:, num_labels]
            l_c = label[l_ind, 0]
            l_cmap = available_palettes[num_labels]

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

    def _plot_hippocampus(
        self,
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

    def plot_embedding_layers(
        self,
        axs: List[matplotlib.axes.Axes],
        embeddings: List[npt.NDArray],
        group_name: str,
    ):
        """
        Plots the embedding layers on the provided axes. Used in tSNE and in normal CEBRA.

        Parameters:
        -----------
        axs : List[matplotlib.axes.Axes]
            List of matplotlib axes objects where the embeddings will be plotted.
        embeddings : List[npt.NDArray]
            List of numpy arrays containing the embeddings for each layer. Each array is shape Samples X num Neurons.
        group_name : str
            Title of the plot (e.g., 'single' or 'multi').
        """
        num_layers = len(embeddings)
        self.fig.suptitle(
            f"{group_name}",
            fontsize=20,
        )
        labels_list = [self.labels[: self.sample_plot]] * num_layers
        titles = [f"Layer {layer}" for layer in range(1, num_layers)]
        titles.append("Output layer")

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
                ax = self._plot_dataset(ax, embedding, label)

            ax.set_title(titles[i], y=1)
            ax.axis("off")
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.tight_layout()

    def plot_embedding(self, group_name):
        return self.plot_embedding_layers(self.axs, self.embeddings, group_name)

    def plot_compare(self):
        """Plots embedding layers for models being compared"""
        self.plot_embedding_layers(
            self.axs_1, self.embeddings_1, self.comparison_groups[1][0]
        )
        self.plot_embedding_layers(
            self.axs_2, self.embeddings_2, self.comparison_groups[1][1]
        )
        self.fig.suptitle(
            f"CEBRA across layers comparison",
            fontsize=20,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()


def compare_embeddings_layers(
    embeddings_1: List[npt.NDArray],
    embeddings_2: List[npt.NDArray],
    labels: npt.NDArray,
    comparison_groups: Tuple = ("tSNE", ["Untrained", "Trained"]),
    dataset_label: str = "HPC",
    sample_plot: int = None,
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
    comparison_groups : Tuple, optional
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
    return _EmbeddingPlot(
        embeddings=[embeddings_1, embeddings_2],
        labels=labels,
        comparison_groups=comparison_groups,
        dataset_label=dataset_label,
        sample_plot=sample_plot,
        axis=ax,
    ).plot_compare(**kwargs)


def plot_embeddings(
    data: Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]],
    labels: npt.NDArray,
    group_name: str = None,
    dataset_label: str = "HPC",
    sample_plot: int = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    **kwargs,
) -> plt.Figure:

    data_dict = data
    if not isinstance(data, Dict):
        if group_name is None:
            raise ValueError(
                "If data is not a dictionary, group_name must be provided."
            )
        data_dict = {group_name: [data]}

    for group_name, models in data_dict.items():
        for i, embs in enumerate(models):
            fig = _EmbeddingPlot(
                embeddings=[embs],
                labels=labels,
                dataset_label=dataset_label,
                sample_plot=sample_plot,
                axis=ax,
            ).plot_embedding(group_name=f"{group_name} instance {i}", **kwargs)


class _ActivationPlot:
    """Class for plotting activations of a neural network model.

    Attributes:
    ----------
    input_data : torch.Tensor
        The input data tensor to be plotted.
    embeddings : List[npt.NDArray]
        A list of npt.NDArray representing the embeddings/activations of each layer. Each array is shape Samples X num Neurons.
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
        self.title = title
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
    data: Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]],
    sample_plot: int = 100,
    cmap: str = "magma",
    plot_title: str = "Trained activations per layer",
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
    data : Union[Dict[str, List[npt.NDArray]], List[npt.NDArray]]
        A list of npt.NDArray representing the embeddings/activations of each layer. Each array is shape Samples X num Neurons or a dictionary where the keys are group names and the values are lists of embeddings.
    sample_plot : int, optional
        The number of samples to plot along the time axis (default is 100).
    cmap : str, optional
        The colormap to use for the embeddings (default is "magma").
    plot_title : str, optional
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

    data_dict = data
    if not isinstance(data, Dict):
        data_dict = {plot_title: [data]}

    for group_name, models in data_dict.items():
        for i, embs in enumerate(models):
            fig = _ActivationPlot(
                input_data=input_data,
                embeddings=embs,
                sample_plot=sample_plot,
                cmap=cmap,
                title=f"{group_name} instance {i} activations across layers",
                figsize=figsize,
                axis=ax,
            ).plot(**kwargs)


class _HeatMapsPlot:
    """Class for plotting CKA heatmaps.

    Attributes:
    ----------
    cka_matrices : Dict[str, npt.NDArray]
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
        cka_matrices: Dict[str, npt.NDArray],
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
    cka_matrices: Dict[str, npt.NDArray],
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
    cka_matrices : Dict[str, npt.NDArray]
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
        rdms: List[npt.NDArray],
        axis: Optional[matplotlib.axes.Axes],
        titles: List[str] = None,
        metric: str = "Correlation",
        dataset_label: str = None,
        cmap: str = "viridis",
        figsize: Tuple[np.float64, np.float64] = None,
    ):

        self.rdms = rdms
        if titles is None:
            titles = [f"Layer {i+1}" for i in range(len(rdms))]
            titles[-1] = "Output Layer"
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
        elif isinstance(self.figsize, tuple):
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
            self.tick_positions = (
                np.arange(0, 34, 2) / 10
            )  # Ticks at 0, 0.2, 0.4,..., 1.6
            self.tick_labels = [
                "0.0",
                "0.2",
                "0.4",
                "0.6",
                "0.8",
                "1.0",
                "1.2",
                "1.4",
                "0.0",
                "0.2",
                "0.4",
                "0.6",
                "0.8",
                "1.0",
                "1.2",
                "1.4",
                "1.6",
            ]

        else:

            # TODO(eloise): think about this
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
            self.ax[i].set_title(self.titles[i], fontsize=14)

            if self.dataset_label == "HPC":
                # Set the x and y ticks
                self.ax[i].set_xticks(
                    self.tick_positions * len(rdm) // 1.6 / 2
                )  # Scale ticks to the range of data
                self.ax[i].set_yticks(
                    self.tick_positions * len(rdm) // 1.6 / 2
                )  # Same for y-axis

                # Set the tick labels to show 0, 0.2, ..., 1.6
                self.ax[i].set_xticklabels(self.tick_labels)
                self.ax[i].set_yticklabels(self.tick_labels)
                if i == 0:
                    self.ax[i].text(
                        -0.15,
                        0.6,
                        "Direction 1",
                        color="black",
                        fontsize=14,
                        transform=self.ax[i].transAxes,
                        rotation=90,
                    )
                    self.ax[i].text(
                        -0.15,
                        0.1,
                        "Direction 2",
                        color="black",
                        fontsize=14,
                        transform=self.ax[i].transAxes,
                        rotation=90,
                    )
                self.ax[i].text(
                    0.1,
                    -0.15,
                    "Direction 1",
                    color="black",
                    fontsize=14,
                    transform=self.ax[i].transAxes,
                )
                self.ax[i].text(
                    0.6,
                    -0.15,
                    "Direction 2",
                    color="black",
                    fontsize=14,
                    transform=self.ax[i].transAxes,
                )
            else:
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
    rdms: Dict[str, List[npt.NDArray]],
    titles: Optional[List[str]] = None,
    metric: Optional[str] = "Correlation",
    dataset_label: Optional[str] = "visual",
    cmap: Optional[str] = "viridis",
    figsize: Optional[Tuple[np.float64, np.float64]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    Plots Representational Dissimilarity Matrices (RDMs) with given titles and metric.

    Parameters:
    -----------
    rdms : Dict[str, List[npt.NDArray]]
        A dictionary containing for key a group name, a for values a list of models and for each layer their respective rdms and correlation to Oracle rdm matrix.
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


def plot_rdm_all(
    rdms: Dict[str, npt.NDArray],
    titles: Optional[List[str]] = None,
    metric: Optional[str] = "Correlation",
    dataset_label: Optional[str] = "visual",
    cmap: Optional[str] = "viridis",
    figsize: Optional[Tuple[np.float64, np.float64]] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
) -> plt.Figure:
    """
    The input to plot_rdm needs to be a list of rdms nothing more
    """
    for key, data_list in rdms.items():
        for inner_list in data_list:
            # getting the rdm matrices for each layer per model in group
            values = [arr[0] for arr in inner_list]
            plot_rdm(
                rdms=values,
                metric=metric,
                titles=titles,
                dataset_label=dataset_label,
                cmap=cmap,
                figsize=figsize,
                ax=ax,
            )
