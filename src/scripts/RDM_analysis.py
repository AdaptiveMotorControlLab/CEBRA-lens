# run using python -m GithubFolder.src.scripts.RDM_analysis --bool_comput 0 --saving_filepath data/RDM/offset10testing.pkl

import pickle
import matplotlib.pyplot as plt
import argparse
import os
from GithubFolder.src.cebra_lens import cebra_lens as lens


def main(
    filepath="data/activations/offset10.pkl",
    bool_comput=0,
    saving_filepath="data/RDM/offset10.pkl",
    session_id=3,
    bool_example: bool = True,
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

    with open(filepath, "rb") as f:
        activations_dict = pickle.load(f)

    if bool_example:

        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Calculating example RDM...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        # example of single instance usage with plotting (neural vs multi here)
        neural_rdm = lens.quantification.RDM.compute_single_RDM_layers(
            train_data=train_data,
            train_label=train_label,
            activations=[train_data],
            metric="euclidean",
            bool_oracle=False,
        )

        multi_rdm = lens.quantification.RDM.compute_single_RDM_layers(
            train_data=train_data,
            train_label=train_label,
            activations=activations_dict["multi"]["TR"][0],
            metric="euclidean",
            bool_oracle=False,
        )
        # Normalize the RDMs using Min-Max normalization
        rdm1_normalized = lens.quantification.misc.normalize_minmax(neural_rdm[0][0])
        rdm2_normalized = lens.quantification.misc.normalize_minmax(multi_rdm[-1][0])

        fig1 = lens.plotting.plot_rdm(
            [rdm1_normalized, rdm2_normalized],
            ["Neural input", "Output Layer"],
            metric="Normalized Euclidean distance",
        )
        multi_rdm_corr = lens.quantification.RDM.compute_single_RDM_layers(
            train_data=train_data,
            train_label=train_label,
            activations=activations_dict["multi"]["TR"][0],
            metric="correlation",
            bool_oracle=False,
        )
        fig2 = lens.plotting.plot_rdm(
            [multi_rdm_corr[0][0], multi_rdm_corr[-1][0]],
            ["Layer 1", "Output Layer"],
            metric="Correlation",
        )

        plt.show()

    if bool_comput:

        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(" Calculating RDM matrices and comparing to Oracle RDM...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        rdm_dict = lens.quantification.RDM.compute_multi_RDM_layers(
            train_data=train_data,
            train_label=train_label,
            activations_dict=activations_dict,
            dataset_label="visual",
            metric="correlation",
            bool_oracle=True,
        )

        with open(saving_filepath, "wb") as f:
            pickle.dump(rdm_dict, f)

    else:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Loading RDM matrices...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        with open(saving_filepath, "rb") as f:
            rdm_dict = pickle.load(f)

    fig = lens.plotting.plot_rdm_correlation(rdm_dict=rdm_dict)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="name of the activations (assuming they are under data/activations)",
    )

    parser.add_argument(
        "--bool_comput",
        type=int,
        default=0,
        help="If True, will recompute and overwrite the cka matrices (0 or 1)",
    )

    parser.add_argument(
        "--saving_filepath",
        type=str,
        default=None,
        help="name of the file where to save the RDM matrices (it will be under data/RDM/saving_filename)",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--bool_example",
        type=int,
        default=1,
        help="Shows an example usage.",
    )
    args = parser.parse_args()

    if args.saving_filepath is None:
        filename = args.filepath.split("/")[-1]
        args.saving_filepath = os.path.join("data/CKA/", filename)

    print("INPUT: ", args)
    main(
        args.filepath,
        args.bool_comput,
        args.saving_filepath,
        args.session_id,
        args.bool_example,
    )
