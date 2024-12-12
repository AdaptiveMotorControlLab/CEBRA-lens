import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    r_ind = label[:,1] == 1
    l_ind = label[:,2] == 1

    if not gray:
        r_cmap = 'cool'
        l_cmap = 'magma'
        r_c = label[r_ind, 0]
        l_c = label[l_ind, 0]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'

    idx1, idx2, idx3 = idx_order
    r=ax.scatter(embedding [r_ind,idx1],
               embedding [r_ind,idx2],
               embedding [r_ind,idx3],
               c=r_c,
               cmap=r_cmap, s=0.05, alpha=0.75)
    l=ax.scatter(embedding [l_ind,idx1],
               embedding [l_ind,idx2],
               embedding [l_ind,idx3],
               c=l_c,
               cmap=l_cmap, s=0.05, alpha=0.75)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    return ax


def plot_allen(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    c = label

    idx1, idx2, idx3 = idx_order
    ax.scatter(embedding [:,idx1],
               embedding [:,idx2],
               embedding [:,idx3],
               c=c,
               cmap='magma', s=0.05, alpha=0.75)

    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    return ax



def plot_tsne_embeddings_layers(embeddings_tsne_untrained, embeddings_tsne_trained, labels, sample_plot = 100, solver = 'single-session', comparison = 'untrained',data = 'HPC'):


    num_layers = len(embeddings_tsne_trained)

    fig, axs = plt.subplots(2, num_layers, figsize=(15, 10), subplot_kw={'projection': '3d'})

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Separate the axes into untrained and trained lists
    axs_untrained = axs[:num_layers]
    axs_trained = axs[num_layers:]


    # Prepare data for embedding, labels, and titles
    '''labels_list = [label_extended[1:-1], label_extended[2:-2], label_extended[4:-4],
                  label_extended[5:-5], label_extended[5:-5]]''' # COMPUTE ALL THE EMBEDINGS AFTER

    labels_list = [labels[:sample_plot]] * num_layers

    titles = []
    if comparison == 'untrained':
      for layer in range(1, num_layers):
          titles.append((f'Layer {layer} Untrained', f'Layer {layer} Trained'))
      titles.append(('Output untrained','Output Trained'))
    else:
      for layer in range(1, num_layers):
          titles.append((f'Layer {layer} Trained', f'Layer {layer} Trained'))
      titles.append(('Output Single','Output Multi'))

    i = 0
    for label, ax, ax_trained in zip( labels_list, axs_untrained, axs_trained):
        embedding = embeddings_tsne_untrained[i][:sample_plot,:]
        embedding_trained = embeddings_tsne_trained[i][:sample_plot,:]
        if data == 'HPC':
          ax = plot_hippocampus(ax, embedding, label)
        else:
          ax = plot_allen(ax, embedding, label)

        ax.set_title(titles[i][0], y=1)
        ax.axis('off')
        if data =='HPC':
          ax_trained = plot_hippocampus(ax_trained, embedding_trained, label)
        else:
          ax_trained = plot_allen(ax_trained, embedding_trained, label)

        ax_trained.set_title(titles[i][1], y=1)
        ax_trained.axis('off')
        i = i + 1

    fig.suptitle(f't-SNE across layers ({solver})', fontsize= 20)

    plt.subplots_adjust(wspace=0,
                        hspace=0)
    plt.tight_layout()

    plt.show()


def plot_activations(input_data,embeddings_untrained,embeddings_trained,sample_plot = 100, solver = 'single', comparison = 'untrained'):
    num_layers = len(embeddings_trained)

    # Set up the figure and gridspec layout
    fig = plt.figure(figsize=(10, 15))
    fig.suptitle(f'Comparison between layers ({solver})', fontsize=20)
    gs = gridspec.GridSpec(num_layers+1, 2)  # Create a grid with 5 rows and 2 columns

    # Create a subplot that spans both columns for the top plot
    ax_top = fig.add_subplot(gs[0, :])  # Top row, spanning both columns
    ax_top.imshow(input_data.T[:, 0:sample_plot])
    ax_top.set_title('Input Data')
    #ax_top.axis('off')
    ax_top.set_ylabel('Channel #')
    ax_top.set_xlabel('Time')
    ax_top.grid(False)

    # Now set up the existing subplots for the comparisons
    titles = []
    if comparison == 'untrained':
      for layer in range(1, num_layers):
          titles.append(f'Layer {layer} Untrained')
          titles.append(f'Layer {layer} Trained')
      titles.append(f'Output Untrained')
      titles.append(f'Output Trained')
    else:
      for layer in range(1, num_layers):
          titles.append(f'Layer {layer} Single')
          titles.append(f'Layer {layer} Multi')

      titles.append(f'Output Single')
      titles.append(f'Output Multi')

    # Create each subplot
    axes = []
    for i in range(num_layers):
        for j in range(2):
            ax = fig.add_subplot(gs[i + 1, j])  # Shift down by 1 row for each layer
            axes.append(ax)

    # Interleave untrained and trained embeddings
    ax_images = [None] * (2 * num_layers)
    ax_images[::2] =  embeddings_untrained# Place untrained embeddings at even indices
    ax_images[1::2] =  embeddings_trained # Place trained embeddings at odd indices

    for ax, img, title in zip(axes, ax_images, titles):
        ax.imshow(img[:, 0:sample_plot], cmap = 'magma')
        ax.set_title(title)
        ax.set_ylabel('Unit #')
        ax.set_xlabel('Time')
        ax.grid(False)  # Hide gridlines
        #ax.axis('off')  # Hide axis for the subplots

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_cebra_embeddings_layers(embeddings_untrained,embeddings_trained,labels_list, solver, comparison = 'untrained',dataset = 'HPC'):

    num_layers = len(embeddings_trained)

    fig, axs = plt.subplots(2, num_layers, figsize=(15, 10), subplot_kw={'projection': '3d'})
    fig.suptitle(f'Embeddings across layers ({solver})', fontsize=20)

    # Flatten the array of axes for easier indexing
    axs = axs.flatten()

    # Separate the axes into untrained and trained lists
    axs_untrained = axs[:num_layers]
    axs_trained = axs[num_layers:]

    # Now set up the existing subplots for the comparisons
    titles = []
    if comparison == 'untrained':
      for layer in range(1, num_layers):
          titles.append((f'Layer {layer} Untrained', f'Layer {layer} Trained'))
      titles.append(('Output untrained','Output Trained'))
    else:
      for layer in range(1, num_layers):
          titles.append((f'Layer {layer} Single', f'Layer {layer} Multi'))
      titles.append(('Output Single','Output Multi'))

    i = 0
    for label, ax, ax_trained in zip(labels_list, axs_untrained, axs_trained):
        emb_untrained = embeddings_untrained[i].T
        emb_trained = embeddings_trained[i].T
        if dataset == 'HPC':
           ax = plot_hippocampus(ax, emb_untrained, label)
        else:
          ax = plot_allen(ax, emb_untrained, label)

        ax.set_title(titles[i][0], y=1, pad=-20)
        ax.axis('off')
        ax_trained = plot_allen(ax_trained, emb_trained, label)
        ax_trained.set_title(titles[i][1], y=1, pad=-20)
        ax_trained.axis('off')
        i = i + 1


    #plt.subplots_adjust(wspace=0,hspace=0)
    plt.tight_layout() #pad=0.5
    plt.show()


def plot_rdm(rdms, titles, metric='Normalized Euclidean distance'):
    # Create the plot
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(10, 7)

    # Generate tick labels
    tick_labels = [str(i) for i in range(0, 930, 30)]

    for i, rdm in enumerate(rdms):
        # Display the RDM using imshow
        cax = ax[i].imshow(rdm, cmap='viridis', aspect='auto')

        # Set title and show the plot
        ax[i].set_title(titles[i])

        # Set ticks and tick labels
        num_ticks = len(tick_labels)
        ax[i].set_xticks(np.linspace(0, rdm.shape[1] - 1, num_ticks))
        ax[i].set_yticks(np.linspace(0, rdm.shape[0] - 1, num_ticks))
        ax[i].set_xticklabels(tick_labels, rotation=90, ha='right')
        ax[i].set_yticklabels(tick_labels)

    plt.suptitle('Representational Dissimilarity Matrix (RDM)')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    # Add colorbar with a label
    fig.colorbar(cax, ax=ax, orientation='horizontal', fraction=0.05, label=metric)

    plt.show()

