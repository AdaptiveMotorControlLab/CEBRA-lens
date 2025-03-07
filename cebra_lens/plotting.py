"""Functions that handle the plotting (CKA, RDM, layer representation...)"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch


def plot_simple_activations(
    input_data: torch.Tensor,
    embeddings: list,
    sample_plot: int = 100,
    cmap: str = "magma",
    title: str = "Trained activations",
    figsize: tuple = (10, 20),
):
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

    num_layers = len(embeddings)

    # Set up the figure
    fig, axes = plt.subplots(num_layers + 1, 1, figsize=figsize)
    fig.suptitle(title, fontsize=20)

    # Plot the input data
    axes[0].imshow(input_data.T[:, 0:sample_plot], aspect="auto")
    axes[0].set_title("Input Data")
    axes[0].set_ylabel("Channel #")
    axes[0].set_xlabel("Time")
    axes[0].grid(False)

    # Plot the embeddings for each layer
    for i in range(num_layers):
        axes[i + 1].imshow(embeddings[i][:, 0:sample_plot], cmap=cmap, aspect="auto")
        if i == num_layers - 1:
            layer_title = "Output Layer"
        else:
            layer_title = f"Layer {i + 1}"
        axes[i + 1].set_title(layer_title)
        axes[i + 1].set_ylabel("Unit #")
        axes[i + 1].set_xlabel("Time")
        axes[i + 1].grid(False)

    # Adjust layout for better spacing
    plt.tight_layout()
    return fig


def plot_embedding_layers(
    axs,
    embeddings: list,
    labels: np.ndarray,
    title_prefix: str,
    sample_plot: int = 200,
    dataset_label: str = "visual",
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

    labels_list = [labels[:sample_plot]] * num_layers
    titles = [f"{title_prefix} Layer {layer}" for layer in range(1, num_layers)]
    titles.append(f"{title_prefix} Output")

    for i, (label, ax) in enumerate(zip(labels_list, axs)):
        if (
            embeddings[i].shape[0] < embeddings[i].shape[1]
        ):  # should be num Samples X num Neurons
            embedding = embeddings[i].T
        else:
            embedding = embeddings[i]

        embedding = embedding[:sample_plot, :]
        if dataset_label == "HPC":
            ax = plot_hippocampus(ax, embedding, label)
        elif dataset_label == "visual":
            ax = plot_allen(ax, embedding, label)
        else:
            raise NotImplementedError(
                f"label {dataset_label} not yet implemented. Use either visual or HPC"
            )

        ax.set_title(titles[i], y=1)
        ax.axis("off")


def compare_embeddings_layers(
    embeddings_1: list,
    embeddings_2: list,
    labels: np.ndarray,
    sample_plot=200,
    comparison_labels: tuple = ("tSNE", ["Untrained", "Trained"]),
    dataset_label="HPC",
) -> plt.Figure:
    """
    Compare embeddings across layers for two sets of embeddings.
    This function takes two sets of embeddings and compares them layer by layer. It plots the embeddings in a 3D space
    for visual comparison. Used with CEBRA embeddings and tSNE embeddings.
    Parameters:
    -----------
    embeddings_1 : list
        A list of embeddings for the first set of data (e.g., untrained model).
    embeddings_2 : list
        A list of embeddings for the second set of data (e.g., trained model).
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

    num_layers_1 = len(embeddings_1)
    num_layers_2 = len(embeddings_2)

    # Padding the shorter embedding to match the number of layers in the longer embedding
    if num_layers_1 > num_layers_2:
        embeddings_2 += [np.empty_like(embeddings_2[0])] * (num_layers_1 - num_layers_2)
    elif num_layers_2 > num_layers_1:
        embeddings_1 += [np.empty_like(embeddings_1[0])] * (num_layers_2 - num_layers_1)

    fig, axs = plt.subplots(
        2,
        max(num_layers_1, num_layers_2),
        figsize=(15, 10),
        subplot_kw={"projection": "3d"},
    )

    axs_1 = axs[0, :]
    axs_2 = axs[1, :]

    plot_embedding_layers(
        axs=axs_1,
        embeddings=embeddings_1,
        labels=labels,
        title_prefix=comparison_labels[1][0],
        sample_plot=sample_plot,
        dataset_label=dataset_label,
    )
    plot_embedding_layers(
        axs=axs_2,
        embeddings=embeddings_2,
        labels=labels,
        title_prefix=comparison_labels[1][1],
        sample_plot=sample_plot,
        dataset_label=dataset_label,
    )

    fig.suptitle(
        f"{comparison_labels[0]} across layers({comparison_labels[1][0]} - {comparison_labels[1][1]})",
        fontsize=20,
    )
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    return fig


# could potentially be removed and just use the function compare_embeddings_layers
def plot_embeddings_singlevmulti(
    embeddings_single: list,
    embeddings_multi: list,
    embeddings_untrained_single: np.array,
    embeddings_untrained_multi: np.array,
    y: np.ndarray,
) -> plt.Figure:
    """
    Plot the 3D embeddings for both single and multi session, comparing untrained and trained models.
    This function plots the 3D embeddings to visually compare the trained and untrained models in both single-session and multi-session scenarios.

    Parameters:
    -----------
    embeddings_single : list
        List of trained embeddings (single-session).
    embeddings_multi : list
        List of trained embeddings (multi-session).
    embeddings_untrained_single : np.ndarray
        Only one instance of the untrained embedding (single-session).
    embeddings_untrained_multi : np.ndarray
        Only one instance of the untrained embedding (multi-session).
    y : np.ndarray
        Labels of the embeddings.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the 3D plots of the embeddings.
    """

    # Assert that the inputs are of correct type
    assert isinstance(embeddings_single, list), "embeddings_single should be a list"
    assert isinstance(embeddings_multi, list), "embeddings_multi should be a list"
    assert isinstance(
        embeddings_untrained_single, np.ndarray
    ), "embeddings_untrained_single should be a numpy array"
    assert isinstance(
        embeddings_untrained_multi, np.ndarray
    ), "embeddings_untrained_multi should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a np.ndarray"

    num_models = len(embeddings_single)
    if num_models > 5:  # truncate above 5 for plotting clarity
        num_models = 5
        embeddings_single = embeddings_single[:5]
        embeddings_multi = embeddings_multi[:5]

    fig, axs = plt.subplots(
        2, num_models + 1, figsize=(15, 10), subplot_kw={"projection": "3d"}
    )

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Separate the axes into untrained and trained lists
    axs_single = axs[: num_models + 1]
    axs_multi = axs[num_models + 1 :]

    i = 0
    for ax, ax_multi in zip(axs_single, axs_multi):
        if i == 0:
            embeddings_single_plot = embeddings_untrained_single
            embeddings_multi_plot = embeddings_untrained_multi
            title = "Untrained"
        else:
            embeddings_single_plot = embeddings_single[i - 1]
            embeddings_multi_plot = embeddings_multi[i - 1]
            title = i
        ax = plot_allen(
            ax, embeddings_single_plot, y
        )  # Assuming `plot_allen` is defined elsewhere
        ax.set_title(f"Single-{title}", y=1, pad=-20)
        ax.axis("off")
        ax_multi = plot_allen(ax_multi, embeddings_multi_plot, y)
        ax_multi.set_title(f"Multi-{title}", y=1, pad=-20)
        ax_multi.axis("off")
        i += 1

    plt.subplots_adjust(wspace=0, hspace=0)
    return fig


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

    num_comparisons = len(cka_matrices)
    # Create a figure and set of subplots
    fig, axs = plt.subplots(1, num_comparisons, figsize=figsize)

    # Define the heat map parameters
    cbar_ax = fig.add_axes([0.13, -0.04, 0.3, 0.03])  # Position for the color bar

    heatmap_kwargs = {
        "cbar": show_cbar,
        "cbar_ax": cbar_ax,
        "vmin": 0,
        "vmax": 1,
        "cmap": color_map,
        "cbar_kws": {"label": cbar_label, "orientation": "horizontal"},
    }
    if num_comparisons == 1:
        axs = [axs]  # handle the 1 comparison case

    # Plot each heat map
    for i, (key, value) in enumerate(cka_matrices.items()):

        sns.heatmap(value, ax=axs[i], annot=annot, **heatmap_kwargs)

        num_layers = value.shape[1]
        num_models = value.shape[0]

        axs[i].set_title(key.replace("_", " "))
        axs[i].set_xlabel("Layer")
        if i == 0:
            axs[i].set_ylabel("Model Instantiation", fontsize=12)
            axs[i].set_yticks(np.arange(num_models) + 0.5)
            axs[i].set_yticklabels([m for m in range(1, num_models + 1)])
        else:
            axs[i].set_ylabel("")
            axs[i].set_yticks([])

        axs[i].set_xticks(np.arange(num_layers) + 0.5)
        axs[i].set_xticklabels([f"L{l}" for l in range(1, num_layers + 1)])

    # Adjust layout
    plt.subplots_adjust(wspace=0.1, right=0.9)
    fig.suptitle("Similarity between model representations (CKA)", fontsize=16)
    return fig


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

    if len(rdms) != len(titles):
        raise ValueError("The two lists (rdms and titles) must have the same length.")

    fig, ax = plt.subplots(1, len(rdms))
    if len(rdms) == 1:
        ax = [ax]

    if figsize == None:
        y_size = max(6, 3 * len(rdms))
        x_size = max(6, 5 * len(rdms))
    elif type(figsize) == tuple:
        x_size = figsize[0]
        y_size = figsize[1]

    fig.set_size_inches(x_size, y_size)

    # Generate tick labels specific to the dataset
    if dataset_label == "visual":
        tick_labels = [str(i) for i in range(0, 930, 30)]

    elif dataset_label == "HPC":
        # TODO
        raise NotImplementedError
    else:
        raise NotImplementedError(
            f"RDM Plotting for dataset {dataset_label} not yet implemented. Please use 'visual' or 'HPC'."
        )

    for i, rdm in enumerate(rdms):

        cax = ax[i].imshow(rdm, cmap=cmap, aspect="auto")
        ax[i].set_title(titles[i])
        num_ticks = len(tick_labels)
        ax[i].set_xticks(np.linspace(0, rdm.shape[1] - 1, num_ticks))
        ax[i].set_yticks(np.linspace(0, rdm.shape[0] - 1, num_ticks))
        ax[i].set_xticklabels(tick_labels, rotation=90, ha="right")
        ax[i].set_yticklabels(tick_labels)

    plt.suptitle("Representational Dissimilarity Matrix (RDM)")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    fig.colorbar(cax, ax=ax, orientation="horizontal", fraction=0.05, label=metric)

    return fig


def plot(
    dictionary: dict,
    title: str = "Plotting dict",
    figsize: tuple = (15, 5),
    plotting_type: str = "rdm",
) -> list[plt.Figure]:
    """
    Goes through a dictionary and creates a plot where the x-axis represents the model layers, while on the y-axis is the plotting_type value. It is used to plot RDMs, distances, or decoding values. Depending on the number of keys in the dictionary, it will create a separate plot for each key, a key being a model category label.

    Parameters:
    -----------
    dictionary : dict
        A dictionary containing the values to be plotted, where keys are the model category labels or if the model labels where not given the model name,
        and values are lists of arrays containing data - data can be RDM, distance metric or decoding values.
    title : str, optional
        The base title for each plot.
    figsize : tuple, optional
        A tuple representing the figure size (default is (15, 5)).
    plotting_type : str, optional
        The type of plot: 'rdm', 'distance', or 'decoding'.

    Returns:
    --------
    figures : list of matplotlib.figure.Figure
        A list of generated figures, one per dictionary key.
    """

    # Get a color palette with a unique color per key
    unique_keys = list(dictionary.keys())
    colors = sns.color_palette("husl", len(unique_keys))

    fig, axs = plt.subplots(1, 1, figsize=figsize)

    for idx, (key, data_list) in enumerate(dictionary.items()):
        color = colors[idx]  # Assign a unique color per key
        layer_values = []

        for i, inner_list in enumerate(data_list):
            if plotting_type == "rdm":
                values = [arr[1] for arr in inner_list]  # Extract second column
            elif plotting_type == "distance":
                values = [arr for arr in inner_list]  # Use raw values
            elif plotting_type == "decoding":
                values = [arr[2] for arr in inner_list]  # Extract third column
            else:
                raise NotImplementedError(
                    f"Plotting not yet implemented for {plotting_type}. Please use 'rdm', 'distance', or 'decoding'."
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
        if plotting_type == "decoding":
            plt.xticks(
                np.arange(1, len(mean_values) + 1),
                ["Neural input"] + [str(i) for i in range(1, len(mean_values))],
            )
        plt.title(title, fontsize=15)
        sns.despine()

    return fig


def plot_rdm_correlation(
    rdm_dict: dict, title: str = "RDM comparison to Oracle", figsize: tuple = (15, 5)
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

    return plot(rdm_dict, title=title, figsize=figsize, plotting_type="rdm")


def plot_distance(
    distance_dict: dict,
    title: str = "Inter-repetition distance",
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Plots the distances across layer.

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

    return plot(
        distance_dict, title=title, figsize=figsize, plotting_type="distance"
    )


def plot_layer_decoding(
    results_dict: dict, title: str = "Decoding by layer", figsize: tuple = (15, 5)
) -> plt.Figure:
    """
    Plots the correlation of Representational Dissimilarity Matrices (RDMs) with Oracle data.

    Parameters:
    -----------
    results_dict : dict
        A dictionary containing the decoding results to be plotted. Obtained by using lens.quantification.decoding.decode_by_layer_all, where values should
        be dictionaries containing decoding arrays for different layers.
    title : str, optional
        The title for the plot (default is "Decoding by layer").
    figsize : tuple, optional
        A tuple representing the figure size (default is (15, 5)).

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure containing the RDM comparison plot.
    """

    return plot(
        results_dict, title=title, figsize=figsize, plotting_type="decoding"
    )


def plot_decoding(
    results_dict: dict,
    palette: str = "hls",
    dataset_label="visual",
) -> plt.Figure:
    """
    Plots the decoding accuracy across multiple models.

    Parameters:
    -----------
    results_dict : dict
        A dictionary where the keys are model names and the values are arrays containing decoding results gathered by lens.quantification.decoding.decode_models. The names must contain either TR or UT: e.g. multi1_TR, single3_UT
    palette: str, optional (default is "hls")
        The color palette to use for the plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure displaying the comparison of decoding accuracy across models.
    """

    palette = sns.color_palette(palette, len(results_dict))

    fig, ax = plt.subplots(figsize=(len(results_dict) * 2, 6))

    # X positions for each model type
    x_positions = list(range(1, len(results_dict) + 1))

    if dataset_label == "visual":
        for i, (key, results) in enumerate(results_dict.items()):
            acc = results[:, 2]  # accuracy
            mean_error = np.mean(acc)
            color = palette[i]
            ax.scatter(np.ones_like(acc) * x_positions[i], acc, color=color, alpha=0.3)

            # Plot the means
            ax.scatter(
                x_positions[i],
                mean_error,
                color=color,
                s=50,
                label=f"Mean {key}",
                zorder=5,
            )

        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Comparison of Accuracy Across Models")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(results_dict.keys())
        ax.legend()
        sns.despine()
    else:
        raise NotImplementedError(
            f"Plotting of {dataset_label} is not handled yet. Only 'visual' is for now. "
        )

    return fig


def plot_hippocampus(ax, embedding, label, gray=False, idx_order=(0, 1, 2)):
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


def plot_allen(ax, embedding, label, gray=False, idx_order=(0, 1, 2)):
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
