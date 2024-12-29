import pickle
import argparse
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt
import os
import logging


def setup_logging():

    # Get directory and filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_filename = os.path.splitext(os.path.basename(__file__))[0]

    logs_dir = os.path.join(script_dir, "logs")

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_path = os.path.join(logs_dir, f"{script_filename}.log")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main(
    activations_filepath="data/activations/offset10.pkl",
    session_id=3,
    dataset_label="visual",
):

    logging.info("Script started with arguments:")
    for arg, value in locals().items():
        logging.info(f"{arg}: {value}")

    _, _, discrete_labels_train, _ = lens.utils_allen.get_single_session_datasets()
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    fig1 = lens.plotting.compare_embeddings_layers(
        activations_dict["single"]["UT"][0],
        activations_dict["single"]["TR"][0],
        labels=train_label,
        dataset_label=dataset_label,
        sample_plot=activations_dict["single"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Untrained Single", "Trained Single"]),
    )
    fig2 = lens.plotting.compare_embeddings_layers(
        activations_dict["multi"]["UT"][0],
        activations_dict["multi"]["TR"][0],
        labels=train_label,
        dataset_label=dataset_label,
        sample_plot=activations_dict["multi"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Untrained Multi", "Trained Multi"]),
    )
    fig3 = lens.plotting.compare_embeddings_layers(
        activations_dict["single"]["TR"][0],
        activations_dict["multi"]["TR"][0],
        labels=train_label,
        dataset_label=dataset_label,
        sample_plot=activations_dict["multi"]["TR"][0][0].shape[1],
        comparison_labels=("CEBRA embeddings", ["Single", "Multi"]),
    )

    plt.show()


if __name__ == "__main__":

    setup_logging()

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

    parser.add_argument(
        "--dataset_label",
        type=str,
        default="visual",
    )

    args = parser.parse_args()

    main(
        args.activations_filepath,
        args.session_id,
    )
