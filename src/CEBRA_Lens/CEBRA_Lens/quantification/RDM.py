import numpy as np
from random import sample

def compute_single_RDM_layers(train_data,train_label,):
    # BINNING
    num_bins = 30
    num_samples = 200 if len(train_data) / 30 >= 200 else int(len(train_data) / 30)
    step_distance = 30
    idxs = np.zeros((num_bins, num_samples))

    j = 0
    for i in range(num_bins):

        full_idxs = np.where(
            (train_label[:] >= j * step_distance)
            & (train_label[:] < (j + 1) * step_distance)
        )[0]
        idxs[i, :] = sample(list(full_idxs), num_samples)
        j = j + 1

    idxs = idxs.astype(int)


    # Neural
    neural_data = train_data[list(idxs.flatten()), :]

    # Activations
    activations_UT = activations_dict["act_UT"]
    activations_multi = activations_dict["act_multi"]
    activations_single = activations_dict["act_single"]

    num_layers = (
        len(activations_UT) - 2
    ) // 2  # This assumes that there will always be only 1 instance of single UT and 1 multi UT.
    output_embeddings_idxs = [i for i in range(1, num_trained_models + 1)][::-1]

    # Output Embedding of first multi-session model instance
    embeddings_trained_multi_all = activations_multi[:num_layers] + [
        activations_multi[-output_embeddings_idxs[0]]
    ]

    embeddings_trained_multi = activations_multi[-output_embeddings_idxs[0]][
        :, idxs.flatten()
    ].T  # .T to keep consistency between neural data and this

    neural_data_rdm = squareform(pdist(neural_data, metric="euclidean"))
    embedding_rdm = squareform(pdist(embeddings_trained_multi, metric="euclidean"))
    # just to show that with correlation it doesn't work with neural input
    neural_data_rdm_corr = squareform(pdist(neural_data, metric="correlation"))
    embedding_rdm_corr = squareform(
        pdist(embeddings_trained_multi, metric="correlation")
    )
    # Normalize the RDMs using Min-Max normalization
    rdm1_normalized = normalize_minmax(neural_data_rdm)
    rdm2_normalized = normalize_minmax(embedding_rdm)
