import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns


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


def plot_tsne_embeddings_layers(
    embeddings_tsne_untrained,
    embeddings_tsne_trained,
    labels,
    sample_plot=100,
    solver="single-session",
    comparison="untrained",
    data="HPC",
):

    num_layers = len(embeddings_tsne_trained)

    fig, axs = plt.subplots(
        2, num_layers, figsize=(15, 10), subplot_kw={"projection": "3d"}
    )

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Separate the axes into untrained and trained lists
    axs_untrained = axs[:num_layers]
    axs_trained = axs[num_layers:]

    # Prepare data for embedding, labels, and titles
    """labels_list = [label_extended[1:-1], label_extended[2:-2], label_extended[4:-4],
                  label_extended[5:-5], label_extended[5:-5]]"""  # COMPUTE ALL THE EMBEDINGS AFTER

    labels_list = [labels[:sample_plot]] * num_layers

    titles = []
    if comparison == "untrained":
        for layer in range(1, num_layers):
            titles.append((f"Layer {layer} Untrained", f"Layer {layer} Trained"))
        titles.append(("Output untrained", "Output Trained"))
    else:
        for layer in range(1, num_layers):
            titles.append((f"Layer {layer} Trained", f"Layer {layer} Trained"))
        titles.append(("Output Single", "Output Multi"))

    i = 0
    for label, ax, ax_trained in zip(labels_list, axs_untrained, axs_trained):
        embedding = embeddings_tsne_untrained[i][:sample_plot, :]
        embedding_trained = embeddings_tsne_trained[i][:sample_plot, :]
        if data == "HPC":
            ax = plot_hippocampus(ax, embedding, label)
        else:
            ax = plot_allen(ax, embedding, label)

        ax.set_title(titles[i][0], y=1)
        ax.axis("off")
        if data == "HPC":
            ax_trained = plot_hippocampus(ax_trained, embedding_trained, label)
        else:
            ax_trained = plot_allen(ax_trained, embedding_trained, label)

        ax_trained.set_title(titles[i][1], y=1)
        ax_trained.axis("off")
        i = i + 1

    fig.suptitle(f"t-SNE across layers ({solver})", fontsize=20)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.show()


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
    embeddings_single,
    embeddings_multi,
    embeddings_untrained_single,
    embeddings_untrained_multi,
    y,
):
    """
    Plot the 3D embeddings for both single and multi layers, comparing untrained and trained models.

    Parameters:
    - embeddings_single: List of trained embeddings (single layer).
    - embeddings_multi: List of trained embeddings (multi-layer).
    - embeddings_untrained_single: Untrained embeddings (single layer).
    - embeddings_untrained_multi: Untrained embeddings (multi-layer).
    - y: Data to be plotted alongside embeddings.
    """
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
    plt.show()
