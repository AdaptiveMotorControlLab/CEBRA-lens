import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


#######################################################################
############################### CLEANED ###############################
#######################################################################

def plot_embedding_layers(axs, embeddings: list, labels: np.ndarray, title_prefix: str, sample_plot: int = 200, data_label="HPC"):
    """
    Plots the embedding layers on the provided axes. Used in tSNE and in normal CEBRA.
    
    Parameters:
    -----------
    axs : list
        List of matplotlib axes objects where the embeddings will be plotted.
    embeddings : list
        List of numpy arrays containing the embeddings for each layer.
    labels : np.ndarray
        Array of labels corresponding to the embeddings (e.g., frame number).
    title_prefix : str
        Title of the plot (e.g., 'single' or 'multi').
    sample_plot : int
        Number of samples to plot from the embeddings (default is 200).
    data_label : str
        Label indicating data source. Can be "HPC" or "Visual".
    """
    
    num_layers = len(embeddings)

    labels_list = [labels[:sample_plot]] * num_layers
    titles = [f"{title_prefix} Layer {layer}" for layer in range(1, num_layers)]
    titles.append(f"{title_prefix} Output")

    for i, (label, ax) in enumerate(zip(labels_list, axs)):
        embedding = embeddings[i][:sample_plot, :]
        if data_label == "HPC":
            ax = plot_hippocampus(ax, embedding, label)
        elif data_label == "Visual":
            ax = plot_allen(ax, embedding, label)
        else:
            raise NotImplementedError(f"label {data_label} not yet implemented. Use either Visual or HPC")

        ax.set_title(titles[i], y=1)
        ax.axis("off")

def compare_embeddings_layers(embeddings_1: list, embeddings_2: list, labels: np.ndarray, sample_plot=200, comparison_labels: tuple =('tSNE',["Untrained","Trained"]), data_label="HPC"):
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
    data_label : str, optional
        A string representing the label for the data being plotted (default is "HPC").
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plots of the t-SNE embeddings.
    """
    
    num_layers_1 = len(embeddings_1)
    num_layers_2 = len(embeddings_2)

    #Padding the shorter embedding to match the number of layers in the longer embedding
    if num_layers_1 > num_layers_2:
        embeddings_2 += [np.empty_like(embeddings_2[0])] * (num_layers_1 - num_layers_2)
    elif num_layers_2 > num_layers_1:
        embeddings_1 += [np.empty_like(embeddings_1[0])] * (num_layers_2 - num_layers_1)

    fig, axs = plt.subplots(2, max(num_layers_1, num_layers_2), figsize=(15, 10), subplot_kw={"projection": "3d"})

    axs_1 = axs[0, :]
    axs_2 = axs[1, :]

    
    plot_embedding_layers(axs = axs_1, embeddings_tsne = embeddings_1, labels = labels, title_prefix = comparison_labels[1][0], sample_plot = sample_plot,data_label = data_label)
    plot_embedding_layers(axs = axs_2, embeddings_tsne = embeddings_2, labels = labels, title_prefix = comparison_labels[1][1], sample_plot = sample_plot,data_label = data_label)

    fig.suptitle(f"{comparison_labels[0]} across layers({comparison_labels[1][0] - comparison_labels[1][1]})", fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    return fig


#######################################################################
############################ TO BE CLEANED ############################
#######################################################################
def plot_cka_heatmaps(
    cka_matrices,
    annot,
    titles,
    cbar_label="CKA score",
    color_map="magma",
    figsize=(15, 5),
):
    """
    Plot CKA heatmaps for different matrix comparisons.

    Parameters:
    - cka_matrices: List of matrices to plot (should be a list of 3 matrices)
    - titles: List of titles corresponding to each heatmap
    - cbar_label: Label for the color bar
    - color_map: The color map to use for the heatmap
    - figsize: Size of the figure for the subplots
    """
    # Create a figure and set of subplots
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # Define the heat map parameters
    cbar_ax = fig.add_axes([0.13, -0.04, 0.3, 0.03])  # Position for the color bar

    heatmap_kwargs = {
        "cbar": True,
        "cbar_ax": cbar_ax,
        "vmin": 0,
        "vmax": 1,
        "cmap": color_map,
        "cbar_kws": {"label": cbar_label, "orientation": "horizontal"},
    }

    num_layers = cka_matrices[0].shape[1]
    num_models = cka_matrices[0].shape[0]

    # Plot each heat map
    for i, matrix in enumerate(cka_matrices):
        sns.heatmap(matrix, ax=axs[i], annot=annot, **heatmap_kwargs)
        axs[i].set_title(titles[i])

    # Set labels and titles
    for ax in axs:
        ax.set_xlabel("Layer")
        if ax == axs[0]:
            ax.set_ylabel("Model Instantiation", fontsize=12)
            ax.set_yticks(np.arange(num_models) + 0.5)
            ax.set_yticklabels([m for m in range(1, num_models + 1)])
        else:
            ax.set_ylabel("")
            ax.set_yticks([])

        ax.set_xticks(np.arange(num_layers) + 0.5)
        ax.set_xticklabels([f"L{l}" for l in range(1, num_layers + 1)])

    # Adjust layout
    plt.subplots_adjust(wspace=0.1, right=0.9)
    fig.suptitle("Similarity between model representations (CKA)", fontsize=16)
    plt.show()


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


def plot_activations(
    input_data,
    embeddings_untrained,
    embeddings_trained,
    sample_plot=100,
    solver="single",
    comparison="untrained",
):
    num_layers = len(embeddings_trained)

    # Set up the figure and gridspec layout
    fig = plt.figure(figsize=(10, 15))
    fig.suptitle(f"Comparison between layers ({solver})", fontsize=20)
    gs = gridspec.GridSpec(num_layers + 1, 2)  # Create a grid with 5 rows and 2 columns

    # Create a subplot that spans both columns for the top plot
    ax_top = fig.add_subplot(gs[0, :])  # Top row, spanning both columns
    ax_top.imshow(input_data.T[:, 0:sample_plot])
    ax_top.set_title("Input Data")
    # ax_top.axis('off')
    ax_top.set_ylabel("Channel #")
    ax_top.set_xlabel("Time")
    ax_top.grid(False)

    # Now set up the existing subplots for the comparisons
    titles = []
    if comparison == "untrained":
        for layer in range(1, num_layers):
            titles.append(f"Layer {layer} Untrained")
            titles.append(f"Layer {layer} Trained")
        titles.append(f"Output Untrained")
        titles.append(f"Output Trained")
    else:
        for layer in range(1, num_layers):
            titles.append(f"Layer {layer} Single")
            titles.append(f"Layer {layer} Multi")

        titles.append(f"Output Single")
        titles.append(f"Output Multi")

    # Create each subplot
    axes = []
    for i in range(num_layers):
        for j in range(2):
            ax = fig.add_subplot(gs[i + 1, j])  # Shift down by 1 row for each layer
            axes.append(ax)

    # Interleave untrained and trained embeddings
    ax_images = [None] * (2 * num_layers)
    ax_images[::2] = embeddings_untrained  # Place untrained embeddings at even indices
    ax_images[1::2] = embeddings_trained  # Place trained embeddings at odd indices

    for ax, img, title in zip(axes, ax_images, titles):
        ax.imshow(img[:, 0:sample_plot], cmap="magma")
        ax.set_title(title)
        ax.set_ylabel("Unit #")
        ax.set_xlabel("Time")
        ax.grid(False)  # Hide gridlines
        # ax.axis('off')  # Hide axis for the subplots

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_cebra_embeddings_layers(
    embeddings_untrained,
    embeddings_trained,
    labels_list,
    solver,
    comparison="untrained",
    dataset="HPC",
):

    num_layers = len(embeddings_trained)

    fig, axs = plt.subplots(
        2, num_layers, figsize=(15, 10), subplot_kw={"projection": "3d"}
    )
    fig.suptitle(f"Embeddings across layers ({solver})", fontsize=20)

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Separate the axes into untrained and trained lists
    axs_untrained = axs[:num_layers]
    axs_trained = axs[num_layers:]

    # Now set up the existing subplots for the comparisons
    titles = []
    if comparison == "untrained":
        for layer in range(1, num_layers):
            titles.append((f"Layer {layer} Untrained", f"Layer {layer} Trained"))
        titles.append(("Output untrained", "Output Trained"))
    else:
        for layer in range(1, num_layers):
            titles.append((f"Layer {layer} Single", f"Layer {layer} Multi"))
        titles.append(("Output Single", "Output Multi"))

    i = 0
    for label, ax, ax_trained in zip(labels_list, axs_untrained, axs_trained):
        emb_untrained = embeddings_untrained[i].T
        emb_trained = embeddings_trained[i].T
        if dataset == "HPC":
            ax = plot_hippocampus(ax, emb_untrained, label)
        else:
            ax = plot_allen(ax, emb_untrained, label)

        ax.set_title(titles[i][0], y=1, pad=-20)
        ax.axis("off")
        ax_trained = plot_allen(ax_trained, emb_trained, label)
        ax_trained.set_title(titles[i][1], y=1, pad=-20)
        ax_trained.axis("off")
        i = i + 1

    # plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout()  # pad=0.5
    plt.show()


def plot_rdm(rdms, titles, metric="Normalized Euclidean distance"):
    # Create the plot
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 7)

    # Generate tick labels
    tick_labels = [str(i) for i in range(0, 930, 30)]

    for i, rdm in enumerate(rdms):
        # Display the RDM using imshow
        cax = ax[i].imshow(rdm, cmap="viridis", aspect="auto")

        # Set title and show the plot
        ax[i].set_title(titles[i])

        # Set ticks and tick labels
        num_ticks = len(tick_labels)
        ax[i].set_xticks(np.linspace(0, rdm.shape[1] - 1, num_ticks))
        ax[i].set_yticks(np.linspace(0, rdm.shape[0] - 1, num_ticks))
        ax[i].set_xticklabels(tick_labels, rotation=90, ha="right")
        ax[i].set_yticklabels(tick_labels)

    plt.suptitle("Representational Dissimilarity Matrix (RDM)")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Add colorbar with a label
    fig.colorbar(cax, ax=ax, orientation="horizontal", fraction=0.05, label=metric)

    plt.show()


def plot_accuracy_comparison(results_untrained, results_single, results_multi):
    """
    Plot the accuracy comparison across untrained, single-session, and multi-session models.

    Parameters:
    - results_untrained: Results for untrained models (array).
    - results_single: Results for single-session models (array).
    - results_multi: Results for multi-session models (array).
    """
    # Define pastel colors
    colors = sns.color_palette("hls", 8)
    pastel_purple = colors[6]
    pastel_blue = colors[4]
    grey = sns.color_palette("Greys")[5]

    # Extract the position error (accuracy) from each model's results
    acc1 = results_untrained[:, 2]
    acc2 = results_single[:, 2]
    acc3 = results_multi[:, 2]

    # Compute the mean of the accuracy for each model
    mean_error1 = np.mean(acc1)
    mean_error2 = np.mean(acc2)
    mean_error3 = np.mean(acc3)

    # Create the figure and axis for plotting
    fig, ax = plt.subplots(figsize=(5, 3))

    # X positions for each model
    x_positions = [1, 2, 3]

    # Plot the accuracy errors (scatter points)
    ax.scatter(np.ones_like(acc1) * x_positions[0], acc1, color=grey, alpha=0.3)
    ax.scatter(np.ones_like(acc2) * x_positions[1], acc2, color=pastel_blue, alpha=0.3)
    ax.scatter(
        np.ones_like(acc3) * x_positions[2], acc3, color=pastel_purple, alpha=0.3
    )

    # Plot the means (highlighted as larger points)
    ax.scatter(
        x_positions[0], mean_error1, color=grey, s=50, label="Mean untrained", zorder=5
    )
    ax.scatter(
        x_positions[1],
        mean_error2,
        color=pastel_blue,
        s=50,
        label="Mean single-session",
        zorder=5,
    )
    ax.scatter(
        x_positions[2],
        mean_error3,
        color=pastel_purple,
        s=50,
        label="Mean multi-session",
        zorder=5,
    )

    # Set labels, title, and other plot settings
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Comparison of Accuracy Across Models")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Untrained", "Single-session", "Multi-session"])
    ax.legend()
    sns.despine()

    # Show the plot
    plt.show()

def plot_embeddings_singlevmulti(
    embeddings_single: list,
    embeddings_multi: list,
    embeddings_untrained_single: np.array,
    embeddings_untrained_multi: np.array,
    y: np.ndarray,
):
    """
    Plot the 3D embeddings for both single and multi session, comparing untrained and trained models.

    Parameters:
    - embeddings_single: List of trained embeddings (single-session).
    - embeddings_multi: List of trained embeddings (multi-session).
    - embeddings_untrained_single: Only one instance of the untrained embedding (single-session).
    - embeddings_untrained_multi: Only one instance of the untrained embedding (multi-session).
    - y: labels of the embeddings.
    """
    # Assert that the inputs are of correct type
    assert isinstance(embeddings_single, list), "embeddings_single should be a list"
    assert isinstance(embeddings_multi, list), "embeddings_multi should be a list"
    assert isinstance(embeddings_untrained_single, np.ndarray), "embeddings_untrained_single should be a numpy array"
    assert isinstance(embeddings_untrained_multi, np.ndarray), "embeddings_untrained_multi should be a numpy array"
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
    #plt.show()
    return fig
