# run using python -m GithubFolder.src.scripts.distance_analysis --activations_filepath data/activations/offset10.pkl --bool_comput 1 --distance_filepath --session_id 3 --dataset_label Visual
import pickle
from tqdm import tqdm
import argparse
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt


def main(
    activations_filepath,
    bool_comput,
    distance_filepath,
    session_id,
    dataset_label,
):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading Data and activations...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    # LOAD DATA
    train_datas, _, discrete_labels_train, _ = (
        lens.utils_allen.get_single_session_datasets()
    )

    train_data = train_datas[session_id].neural
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    #####################################
    ####### DISTANCE  CALCULATION #######
    #####################################

    if bool_comput:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Calculating Distances...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # BINNING
        idxs = lens.quantification.misc.discrete_binning(
            train_data=train_data,
            train_label=train_label,
            dataset_label=dataset_label,
            sample_mode="all",
        )
        repetition_indices = lens.quantification.misc.repetition_binning(
            indices=idxs, train_data=train_data, dataset_label=dataset_label
        )

        # meme forme que activations dict.
        interbin_distances_dict = {}
        intrabin_distances_dict = {}
        interrep_distances_dict = {}

        for outer_key, outer_value in activations_dict.items():  # "single" or "multi"
            interbin_distances_dict[outer_key] = {}
            intrabin_distances_dict[outer_key] = {}
            interrep_distances_dict[outer_key] = {}

            for inner_key, outer_list in tqdm(
                outer_value.items(), desc=f"Processing {outer_key}"
            ):  # "UT" or "TR"
                interbin_distances_dict[outer_key][inner_key] = []
                intrabin_distances_dict[outer_key][inner_key] = []
                interrep_distances_dict[outer_key][inner_key] = []

                for inner_list in tqdm(
                    outer_list, desc=f"Processing {outer_key} {inner_key}"
                ):  # for each model instance

                    interbin_distances_dict[outer_key][inner_key].append(
                        lens.quantification.distance.compute_distance_layers(
                            embeddings=inner_list,
                            indices=idxs,
                            metric="cosine",
                            distance_label="interbin",
                        )
                    )
                    intrabin_distances_dict[outer_key][inner_key].append(
                        lens.quantification.distance.compute_distance_layers(
                            embeddings=inner_list,
                            indices=idxs,
                            metric="cosine",
                            distance_label="intrabin",
                        )
                    )
                    interrep_distances_dict[outer_key][inner_key].append(
                        lens.quantification.distance.compute_distance_layers(
                            embeddings=inner_list,
                            indices=idxs,
                            repetition_indices=repetition_indices,
                            metric="cosine",
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
        default="Visual",
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    args = parser.parse_args()

    if args.distance_filepath is None:

        filename = args.activations_filepath.split("/")[-1].split(".")[0]
        args.distance_filepath = f"data/distances/{filename}.pkl"

    print("INPUT: ", args)
    main(
        args.activations_filepath,
        args.bool_comput,
        args.distance_filepath,
        args.session_id,
        args.dataset_label,
    )
