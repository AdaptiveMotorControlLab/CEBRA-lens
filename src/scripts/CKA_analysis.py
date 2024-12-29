import os
from tqdm import tqdm
import argparse
import pickle
from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt
import logging

def setup_logging():

    # Get directory and filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_filename = os.path.splitext(os.path.basename(__file__))[0]

    logs_dir = os.path.join(script_dir, 'logs')

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    log_file_path = os.path.join(logs_dir, f'{script_filename}.log')

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main(
    activations_filepath="data/activations/offset10.pkl",
    bool_comput=0,
    saving_filepath="data/CKA/offset10.pkl",
):
    logging.info("Script started with arguments:")
    for arg, value in locals().items():
        logging.info(f"{arg}: {value}")

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

        cka_matrices = {}
        for comparison in tqdm(comparisons):
            cka_matrix = lens.quantification.compute_multi_CKA_layers(
                activations_dict=activations_dict, comparison=comparison
            )
            cka_matrices[f"{comparison[0]}_v_{comparison[1]}"] = cka_matrix

        with open(saving_filepath, "wb") as f:
            pickle.dump(cka_matrices, f)

    else:

        with open(saving_filepath, "rb") as f:
            cka_matrices = pickle.load(f)

    fig = lens.plotting.plot_cka_heatmaps(
        cka_matrices=cka_matrices,
        annot=False,
    )

    plt.show()


if __name__ == "__main__":

    setup_logging()


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
    main(
        args.activations_filepath,
        args.bool_comput,
        args.saving_filepath,
    )
