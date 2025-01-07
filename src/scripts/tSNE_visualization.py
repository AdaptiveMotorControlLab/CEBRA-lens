import pickle
import argparse
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt
import logging
import os


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
    tsne_filepath="data/tSNE/offset10.pkl",
    session_id=3,
    bool_comput=False,
    num_samples=200,
):
    logging.info("Script started with arguments:")
    for arg, value in locals().items():
        logging.info(f"{arg}: {value}")

    _, _, discrete_labels_train, _ = lens.utils_allen.get_single_session_datasets()
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    if bool_comput:

        tSNE_dict = lens.transform.run_tsne_and_save(
            activations_dict, tsne_filepath, num_samples
        )

    else:

        with open(tsne_filepath, "rb") as f:
            tSNE_dict = pickle.load(f)

    fig1 = lens.plotting.compare_embeddings_layers(
        tSNE_dict["single"]["UT"][0],
        tSNE_dict["single"]["TR"][0],
        labels=train_label,
        dataset_label="visual",
        sample_plot=200,
    )
    fig2 = lens.plotting.compare_embeddings_layers(
        tSNE_dict["multi"]["UT"][0],
        tSNE_dict["multi"]["TR"][0],
        labels=train_label,
        dataset_label="visual",
        sample_plot=200,
    )
    fig3 = lens.plotting.compare_embeddings_layers(
        tSNE_dict["single"]["TR"][0],
        tSNE_dict["multi"]["TR"][0],
        labels=train_label,
        dataset_label="visual",
        sample_plot=200,
        comparison_labels=("tSNE", ["Single", "Multi"]),
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
        "--tsne_filepath",
        type=str,
        default=None,
        help="Path to the tSNE embeddings file.",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="Session ID to use for the analysis.",
    )

    parser.add_argument(
        "--bool_comput",
        type=int,
        default=0,
        help="If True, will recompute and overwrite the tSNE embeddings (0 or 1).",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples to use for tSNE computation.",
    )

    args = parser.parse_args()
    if args.tsne_filepath == None:
        filename = args.activations_filepath.split("/")[-1].split(".")[0]
        args.tsne_filepath = f"data/tsne/{filename}.pkl"

    main(
        args.activations_filepath,
        args.tsne_filepath,
        args.session_id,
        args.bool_comput,
        args.num_samples,
    )
