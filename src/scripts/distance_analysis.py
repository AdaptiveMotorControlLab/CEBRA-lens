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
    activations_filepath,
    bool_comput,
    distance_filepath,
    session_id,
    dataset_label="visual",
    metric="cosine",
):

    logging.info("Script started with arguments:")
    for arg, value in locals().items():
        logging.info(f"{arg}: {value}")

    # LOAD DATA
    train_datas, _, discrete_labels_train, _ = (
        lens.utils_allen.get_single_session_datasets()
    )

    train_data = train_datas[session_id].neural
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    if bool_comput:

        interbin_distances_dict = (
            lens.quantification.distance.compute_multi_distance_layers(
                data=train_data,
                label=train_label,
                activations_dict=activations_dict,
                dataset_label=dataset_label,
                metric=metric,
                distance_label="interbin",
            )
        )
        intrabin_distances_dict = (
            lens.quantification.distance.compute_multi_distance_layers(
                data=train_data,
                label=train_label,
                activations_dict=activations_dict,
                dataset_label=dataset_label,
                metric=metric,
                distance_label="intrabin",
            )
        )
        interrep_distances_dict = (
            lens.quantification.distance.compute_multi_distance_layers(
                data=train_data,
                label=train_label,
                activations_dict=activations_dict,
                dataset_label=dataset_label,
                metric=metric,
                distance_label="interrep",
            )
        )

        distances = {
            "inter-bin": interbin_distances_dict,
            "intra-bin": intrabin_distances_dict,
            "inter-rep": interrep_distances_dict,
        }
        with open(distance_filepath, "wb") as f:
            pickle.dump(distances, f)

    else:

        with open(distance_filepath, "rb") as f:
            distances = pickle.load(f)

    figs = []
    for key, value in distances.items():
        title = f"Distance: {key}"

        figs.append(lens.plotting.plot_distance(distance_dict=value, title=title))

    plt.show()


if __name__ == "__main__":

    setup_logging()

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--activations_filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="Activation's filepath",
    )

    parser.add_argument(
        "--bool_comput",
        type=int,
        default=0,
        help="If True, will recompute and overwrite the distances (0 or 1)",
    )

    parser.add_argument(
        "--distance_filepath",
        type=str,
        default=None,
        help="Saving filepath of the distances dictionnary",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--dataset_label",
        type=str,
        default="visual",
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        help="metric to compute the distance: euclidean or cosine",
    )
    args = parser.parse_args()

    if args.distance_filepath is None:

        filename = args.activations_filepath.split("/")[-1].split(".")[0]
        args.distance_filepath = f"data/distances/{filename}.pkl"

    main(
        args.activations_filepath,
        args.bool_comput,
        args.distance_filepath,
        args.session_id,
        args.dataset_label,
    )
