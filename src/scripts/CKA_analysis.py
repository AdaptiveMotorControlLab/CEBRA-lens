# run using python -m GithubFolder.src.scripts.CKA_analysis --filename offset10-mse --bool_comput True
import os
from tqdm import tqdm
import argparse
import pickle
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt


def main(
    activations_filepath="data/activations/offset10.pkl",
    bool_comput=0,
    saving_filepath="data/CKA/offset10.pkl",
):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    ######################
    ####### LOADING ######
    ######################

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading activations...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)

    if bool_comput:
        comparisons = [
            ("single_UT", "single_TR"),
            ("multi_UT", "multi_TR"),
            ("single_TR", "multi_TR"),
            ("single_TR", "single_TR"),
            ("multi_TR", "multi_TR"),
        ]

        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Calculating CKA matrices...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        cka_matrices = {}
        for comparison in tqdm(comparisons):
            cka_matrix = lens.quantification.compute_multi_CKA_layers(
                activations_dict=activations_dict, comparison=comparison
            )
            cka_matrices[f"{comparison[0]}_v_{comparison[1]}"] = cka_matrix

        with open(saving_filepath, "wb") as f:
            pickle.dump(cka_matrices, f)
            print(f"Succesfully saved the matrices here: {saving_filepath} ")

    else:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Loading CKA matrices...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        with open(saving_filepath, "rb") as f:
            cka_matrices = pickle.load(f)

    fig = lens.plotting.plot_cka_heatmaps(
        cka_matrices=cka_matrices,
        annot=False,
    )

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--activations_filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="filepath of the activation's dictionnary",
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
        help="filepath where to save the CKA dictionnary",
    )

    args = parser.parse_args()

    if args.saving_filepath is None:
        filename = args.activations_filepath.split("/")[-1]
        args.saving_filepath = os.path.join("data/CKA/", filename)
    print(args.saving_filepath)
    main(
        args.activations_filepath,
        args.bool_comput,
        args.saving_filepath,
    )
