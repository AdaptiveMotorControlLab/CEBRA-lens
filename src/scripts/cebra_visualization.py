# run using python -m GithubFolder.src.scripts.cebra_visualization

import pickle
import argparse
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt


def main(
    activations_filepath="data/activations/offset10.pkl",
    session_id=3,
):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading activations...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    _, _, discrete_labels_train, _ = lens.utils_allen.get_single_session_datasets()
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    fig1 = lens.plotting.compare_embeddings_layers(
        activations_dict["single"]["UT"][0],
        activations_dict["single"]["TR"][0],
        labels=train_label,
        data_label="Visual",
        sample_plot=activations_dict["single"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Untrained Single", "Trained Single"]),
    )
    fig2 = lens.plotting.compare_embeddings_layers(
        activations_dict["multi"]["UT"][0],
        activations_dict["multi"]["TR"][0],
        labels=train_label,
        data_label="Visual",
        sample_plot=activations_dict["multi"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Untrained Multi", "Trained Multi"]),
    )
    print(len(activations_dict["multi"]["TR"][0][0]))
    print(activations_dict["multi"]["TR"][0][0].shape)
    fig3 = lens.plotting.compare_embeddings_layers(
        activations_dict["single"]["TR"][0],
        activations_dict["multi"]["TR"][0],
        labels=train_label,
        data_label="Visual",
        sample_plot=activations_dict["multi"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Single", "Multi"]),
    )

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--activations_filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="Path to the activations file.",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="Session ID to use for the analysis.",
    )

    args = parser.parse_args()

    main(
        args.activations_filepath,
        args.session_id,
    )
