"""Matplotlib interface to CEBRA-Lens."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch


class _BasePlot:
    def __init__(
        self,
        results_dict: dict,
        title: str = "Plotting dict",
        figsize: tuple = (15, 5),
        plotting_type: str = "rdm",
    ):
        self.dictionary = results_dict
        self.title = title
        self.figsize = figsize
        self.plotting_type = plotting_type
        self.unique_keys = list(self.dictionary.keys())
        self.colors = sns.color_palette("husl", len(self.unique_keys))

        self.fig, self.axs = plt.subplots(1, 1, figsize=self.figsize)

    def _plot(self):
        for idx, (key, data_list) in enumerate(self.dictionary.items()):
            color = self.colors[idx]  # Assign a unique color per key
            layer_values = []

            for i, inner_list in enumerate(data_list):
                if self.plotting_type == "rdm":
                    values = [arr[1] for arr in inner_list]  # Extract second column
                elif self.plotting_type == "distance":
                    values = [arr for arr in inner_list]  # Use raw values
                elif self.plotting_type == "decoding":
                    values = [arr[2] for arr in inner_list]  # Extract third column
                else:
                    raise NotImplementedError(
                        f"Plotting not yet implemented for {self.plotting_type}. Please use 'rdm', 'distance', or 'decoding'."
                    )

                layer_values.append(values)

                # Plot individual layers
                sns.lineplot(
                    x=np.arange(1, len(values) + 1),
                    y=values,
                    linestyle="-",
                    marker="D",
                    color=color,
                    alpha=0.5,
                )

            layer_values = np.array(layer_values)

            # Compute mean if multiple layers exist
            mean_values = (
                layer_values if layer_values.ndim == 1 else np.mean(layer_values, axis=0)
            )

            # Plot mean line
            sns.lineplot(
                x=np.arange(1, len(mean_values) + 1),
                y=mean_values,
                linestyle="-",
                marker="D",
                color=color,
                alpha=1,
                label=f"Mean {key}",
            )

            # Customize x-ticks for decoding plots
            if self.plotting_type == "decoding":
                plt.xticks(
                    np.arange(1, len(mean_values) + 1),
                    ["Neural input"] + [str(i) for i in range(1, len(mean_values))],
                )
            plt.title(self.title, fontsize=15)
            sns.despine()
        return self.fig
    
    def plot(self) -> plt.Figure:
        """Generates and returns the plot."""
        return self._plot()
    
def plot_rdm_correlation(
    rdm_dict: dict, title: str = "RDM comparison to Oracle", figsize: tuple = (15, 5),**kwargs
) -> plt.Figure:
    """
    Plots the correlation of Representational Dissimilarity Matrices (RDMs) with Oracle data.

    Parameters:
    -----------
    rdm_dict : dict
        A dictionary containing the RDMs to be plotted. Obtained by using lens.quantification.RDM.compute_multi_RDM_layers, where values should
        be dictionaries containing RDMs for different layers.
    title : str, optional
        The title for the plot (default is "RDM comparison to Oracle").
    figsize : tuple, optional
        A tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return _BasePlot(rdm_dict=rdm_dict, title=title, figsize=figsize, plotting_type="rdm").plot(**kwargs)

def plot_distance(
    distance_dict: dict,
    title: str = "Inter-repetition distance",
    figsize: tuple = (15, 5),
    **kwargs,
) -> plt.Figure:
    """
    Plots the distances across layer for models in results_dict.

    Parameters:
    -----------
    distance_dict : dict
        A dictionary containing the distances to be plotted. Obtained by using lens.quantification.distance.compute_distance_layers, where values should
        be dictionaries containing distances for different layers.  The format is the same as the activations_dict.
    title : str, optional
        The title for the plot (default is "Inter-repetition distance").
    figsize : tuple, optional
        A tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return _BasePlot(distance_dict=distance_dict, title=title, figsize=figsize, plotting_type="distance").plot(**kwargs)

def plot_layer_decoding(
    results_dict: dict, title: str = "Decoding by layer", figsize: tuple = (15, 5), **kwargs
) -> plt.Figure:
    """
    Plots the decoding accuracy across layer for models in results_dict.

    Parameters:
    -----------
    results_dict : dict
        A dictionary containing the decoding results to be plotted. Obtained by using lens.quantification.decoding.decode_layer_models, where values should
        be lists containing decoding 2d-arrays for different layers.
    title : str, optional
        The title for the plot (default is "Decoding by layer").
    figsize : tuple, optional
        A tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return _BasePlot(results_dict=results_dict, title=title, figsize=figsize, plotting_type="decoding").plot(**kwargs)

class _EmbeddingLayersPlot:
    def __init__(self, embeddings_1: list, embeddings_2: list, labels: np.ndarray, sample_plot: int, comparison_labels: tuple, dataset_label: str):
        """
        Initializes the EmbeddingLayersPlot class.

        Args:
            embeddings_1 (list): A list of embeddings for the first set of data.
            embeddings_2 (list): A list of embeddings for the second set of data.
            labels (np.ndarray): An array of labels corresponding to the data labels.
            sample_plot (int): The number of samples to plot from the embeddings.
            comparison_labels (tuple): A tuple containing the type of embedding and a list of two strings representing the labels for the two sets of embeddings.
            dataset_label (str, optional): A string representing the label for the data being plotted.
        """
        self.embeddings_1 = embeddings_1
        self.embeddings_2 = embeddings_2
        self.labels = labels
        self.sample_plot = sample_plot
        self.comparison_labels = comparison_labels
        self.dataset_label = dataset_label

        num_layers_1 = len(embeddings_1)
        num_layers_2 = len(embeddings_2)

        # Padding the shorter embedding to match the number of layers in the longer embedding
        if num_layers_1 > num_layers_2:
            embeddings_2 += [np.empty_like(embeddings_2[0])] * (num_layers_1 - num_layers_2)
        elif num_layers_2 > num_layers_1:
            embeddings_1 += [np.empty_like(embeddings_1[0])] * (num_layers_2 - num_layers_1)

        self.fig, self.axs = plt.subplots(
            2,
            max(num_layers_1, num_layers_2),
            figsize=(15, 10),
            subplot_kw={"projection": "3d"},
        )

        self.axs_1 = self.axs[0, :]
        self.axs_2 = self.axs[1, :]

    def _plot_hippocampus(ax, embedding, label, gray=False, idx_order=(0, 1, 2)):
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

    def _plot_allen(self, ax, embedding, label, gray=False, idx_order=(0, 1, 2)):
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
            axs,
            embeddings: list,
            title_prefix: str,
        ):
        """
        Plots the embedding layers on the provided axes. Used in tSNE and in normal CEBRA.

        Parameters:
        -----------
        axs : list
            List of matplotlib axes objects where the embeddings will be plotted.
        embeddings : list
            List of numpy arrays containing the embeddings for each layer. Each array is shape Samples X num Neurons.
        labels : np.ndarray
            Array of labels corresponding to the embeddings (e.g., frame number).
        title_prefix : str
            Title of the plot (e.g., 'single' or 'multi').
        sample_plot : int
            Number of samples to plot from the embeddings (default is 200).
        dataset_label : str
            Label indicating data source. Can be "HPC" or "visual".
        """
        num_layers = len(embeddings)

        labels_list = [self.labels[:self.sample_plot]] * num_layers
        titles = [f"{title_prefix} Layer {layer}" for layer in range(1, num_layers)]
        titles.append(f"{title_prefix} Output")

        for i, (label, ax) in enumerate(zip(labels_list, axs)):
            if (
                embeddings[i].shape[0] < embeddings[i].shape[1]
            ):  # should be num Samples X num Neurons
                embedding = embeddings[i].T
            else:
                embedding = embeddings[i]

            embedding = embedding[:self.sample_plot, :]
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
    
    def _plot(self):
        """Handles plotting logic."""
        self._plot_embedding_layers(self.axs_1, self.embeddings_1, self.comparison_labels[1][0])
        self._plot_embedding_layers(self.axs_2, self.embeddings_2, self.comparison_labels[1][1])
        self.fig.suptitle(
        f"{self.comparison_labels[0]} across layers({self.comparison_labels[1][0]} - {self.comparison_labels[1][1]})",
        fontsize=20,
        )
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
    
    def plot(self) -> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig

    

class _DecodingPlot:
    """Plot the decoding accuracy across multiple models."""
    def __init__(self, results_dict: dict, palette: str, dataset_label: str):
        """
        Initializes the DecodingPlot class.

        Args:
            results_dict (dict): A dictionary where the keys are model category labels or model file names 
                and the values are 2D arrays containing decoding results.
            palette (str, optional): The color palette to use for the plot. Default is "hls".
            dataset_label (str, optional): The dataset type. Currently only "visual" is supported.
        """
        self.results_dict = results_dict
        self.palette = sns.color_palette(palette, len(results_dict))
        self.dataset_label = dataset_label

        self.fig, self.ax = plt.subplots(figsize=(len(results_dict) * 2, 6))

    def _plot(self):
        """Handles plotting logic"""
        x_positions = list(range(1, len(self.results_dict) + 1))

        if self.dataset_label == "visual":
            for i, (key, results) in enumerate(self.results_dict.items()):
                acc = results[:, 2]  # accuracy
                mean_error = np.mean(acc)
                color = self.palette[i]
                self.ax.scatter(np.ones_like(acc) * x_positions[i], acc, color=color, alpha=0.3)

                # Plot the means
                self.ax.scatter(
                    x_positions[i],
                    mean_error,
                    color=color,
                    s=50,
                    label=f"Mean {key}",
                    zorder=5,
                )

            self.ax.set_xlabel("Model")
            self.ax.set_ylabel("Accuracy (%)")
            self.ax.set_title("Comparison of Accuracy Across Models")
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(self.results_dict.keys())
            self.ax.legend()
            sns.despine()
        else:
            raise NotImplementedError(
                f"Plotting of {self.dataset_label} is not handled yet. Only 'visual' is for now. "
            )
    def plot(self)-> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig


def plot_decoding(
    results_dict: dict,
    palette: str = "hls",
    dataset_label="visual",
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding accuracy across multiple models.

    Parameters:
    -----------
    results_dict : dict
        A dictionary where the keys are model category labels or model file names and the values are 2d-arrays containing decoding results gathered by lens.quantification.decoding.decode_models.
    palette: str, optional (default is "hls")
        The color palette to use for the plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure displaying the comparison of decoding accuracy across models.
    """
    return _DecodingPlot(
        results_dict=results_dict,
        palette = palette,
        dataset_label=dataset_label,
    ).plot(**kwargs)


def compare_embeddings_layers(
    embeddings_1: list,
    embeddings_2: list,
    labels: np.ndarray,
    sample_plot=200,
    comparison_labels: tuple = ("tSNE", ["Untrained", "Trained"]),
    dataset_label="HPC",
    **kwargs,
) -> plt.Figure:
    """
    Compare embeddings across layers for two sets of embeddings.
    Embedding are activations.process_activations(activation) outputs.
    This function takes two sets of embeddings and compares them layer by layer. It plots the embeddings in a 3D space
    for visual comparison. Used with CEBRA embeddings and tSNE embeddings.
    Parameters:
    -----------
    embeddings_1 : list
        A list of embeddings for the first set of data (e.g. activation_dict[model_name][0]).
    embeddings_2 : list
        A list of embeddings for the second set of data.
    labels : np.ndarray
        An array of labels corresponding to the data labels (e.g. frame number).
    sample_plot : int
        The number of samples to plot from the embeddings (default is 200).
    comparison_labels : tuple
        A tuple containing the type of embedding and a list of two strings representing the labels for the two sets of embeddings. Example: ('tSNE',["Untrained", "Trained"]).
    dataset_label : str, optional
        A string representing the label for the data being plotted (default is "HPC").
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plots of the t-SNE embeddings.
    """
    return _EmbeddingLayersPlot(
        embeddings_1=embeddings_1,
        embeddings_2=embeddings_2,
        labels=labels,
        sample_plot=sample_plot,
        comparison_labels=comparison_labels,
        dataset_label=dataset_label,
    ).plot(**kwargs)


class _ActivationPlot:
    def __init__(self, input_data: torch.Tensor,
    embeddings: list,
    sample_plot: int = 100,
    cmap: str = "magma",
    title: str = "Trained activations",
    figsize: tuple = (10, 20),
    ):
        self.intput_data = input_data
        self.embeddings = embeddings
        self.sample_plot = sample_plot
        self.cmap = cmap
        self.title = title
        self.figsize = figsize
        self.num_layers = len(embeddings)

        # Set up the figure
        self.fig, self.axes = plt.subplots(self.num_layers + 1, 1, figsize=figsize)
        self.fig.suptitle(title, fontsize=20)
    
    def _plot(self):
        self.axes[0].imshow(self.input_data.T[:, 0:self.sample_plot], aspect="auto")
        self.axes[0].set_title("Input Data")
        self.axes[0].set_ylabel("Channel #")
        self.axes[0].set_xlabel("Time")
        self.axes[0].grid(False)

        # Plot the embeddings for each layer
        for i in range(self.num_layers):
            self.axes[i + 1].imshow(self.embeddings[i][:, 0:self.sample_plot], cmap=self.cmap, aspect="auto")
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

    def plot(self) -> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig
    
def plot_activations(
    input_data: torch.Tensor,
    embeddings: list,
    sample_plot: int = 100,
    cmap: str = "magma",
    title: str = "Trained activations",
    figsize: tuple = (10, 20),
    **kwargs,
) -> plt.Figure:
    """
    Plots the activations of a neural network model.
    Parameters:
    -----------
    input_data : torch.Tensor
        The input data tensor to be plotted.
    embeddings : list
        A list of np.ndarrays representing the embeddings/activations of each layer. Each array is shape Samples X num Neurons.
    sample_plot : int, optional
        The number of samples to plot along the time axis (default is 100).
    cmap : str, optional
        The colormap to use for the embeddings (default is "magma").
    title : str, optional
        The title of the plot (default is "Trained activations").
    figsize : tuple, optional
        The size of the figure (default is (10, 20)).
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
    ).plot(**kwargs)

class _HeatMapsPlot:
    def __init__(self, cka_matrices: dict,
    annot: bool,
    show_cbar: bool = True,
    cbar_label: str = "CKA score",
    color_map: str = "magma",
    figsize: tuple = (15, 5),
    ):
        self.cka_matrices = cka_matrices
        self.annot = annot
        self.show_cbar = show_cbar
        self.cbar_label = cbar_label
        self.color_map = color_map
        self.figsize = figsize
        self.num_matrices = len(cka_matrices)

        self.num_comparisons = len(cka_matrices)
        # Create a figure and set of subplots
        self.fig, self.axs = plt.subplots(1, self.num_comparisons, figsize=figsize)

        # Define the heat map parameters
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
            axs = [axs]  # handle the 1 comparison case
    
    def _plot(self):
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
    
    def plot(self) -> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig

def plot_cka_heatmaps(
    cka_matrices: dict,
    annot: bool,
    show_cbar: bool = True,
    cbar_label: str = "CKA score",
    color_map: str = "magma",
    figsize: tuple = (15, 5),
    ) -> plt.Figure:
        """
    This function generates heatmaps for various CKA matrices to visualize the similarity between different sets of embeddings.

    Parameters:
    -----------
    cka_matrices : dict
        Dictionary of CKA matrices where the keys are the comparison names and the values the matrices.
    show_cbar : bool
        If True, shows the color bar.
    cbar_label : str
        Label for the color bar.
    color_map : str or matplotlib.colors.Colormap
        The palette to use for the heatmap.
    figsize : tuple
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
        ).plot()

class _RDMPlots:
    
    def __init__(
        self,
        rdms: list,
        titles: list,
        metric: str = "Normalized Euclidean distance",
        dataset_label: str = "visual",
        cmap: str = "viridis",
        figsize: tuple = None,
        ):
        self.rdms = rdms
        self.titles = titles
        self.metric = metric
        self.dataset_label = dataset_label
        self.cmap = cmap
        self.figsize = figsize

        if len(self.rdms) != len(self.titles):
            raise ValueError("The two lists (rdms and titles) must have the same length.")

        self.fig, self.ax = plt.subplots(1, len(self.rdms))
        if len(self.rdms) == 1:
            self.ax = [self.ax]

        if self.figsize == None:
            self.y_size = max(6, 3 * len(rdms))
            self.x_size = max(6, 5 * len(rdms))
        elif type(self.figsize) == tuple:
            self.x_size = self.figsize[0]
            self.y_size = self.figsize[1]

        self.fig.set_size_inches(self.x_size, self.y_size)

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
    def _plot(self):
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

        self.fig.colorbar(cax, ax=self.ax, orientation="horizontal", fraction=0.05, label=self.metric)

    def plot(self) -> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig


def plot_rdm(
    rdms: list,
    titles: list,
    metric: str = "Normalized Euclidean distance",
    dataset_label: str = "visual",
    cmap: str = "viridis",
    figsize: tuple = None,
) -> plt.Figure:
    """
    Plots Representational Dissimilarity Matrices (RDMs) with given titles and metric.
    Parameters:
    -----------
    rdms : list
        A list of RDMs to be plotted. Each RDM should be a 2D array-like structure.
    titles : list
        A list of titles for each RDM plot. The length of this list should match the length of `rdms`.
    metric : str, optional
        The metric used for the RDM, which will be displayed as the colorbar label. Default is "Normalized Euclidean distance".
    dataset_label : str, optional
        The label of the dataset, which determines the tick labels. Default is "visual".
        Currently supported values are "visual" and "HPC".
    cmap : str, optional
        The color map to use for the plotting.
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
    ).plot()
        