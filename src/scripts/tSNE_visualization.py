# run using python -m GithubFolder.src.scripts.tSNE_visualization

import pickle
import argparse
from GithubFolder.src.CEBRA_Lens import CEBRA_Lens as Lens
from GithubFolder.src.preprocessing.CEBRA_preprocessing.plotting_utils import compare_embeddings_layers
from GithubFolder.src.preprocessing.CEBRA_preprocessing.data_utils import get_single_session_datasets
import matplotlib.pyplot as plt

def main(activations_filepath = 'data/activations/offset10.pkl',tsne_filepath = 'data/tSNE/offset10.pkl', session_id = 3, bool_comput = False, num_samples = 200):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading activations...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    _, _, discrete_labels_train, _ = (
        get_single_session_datasets()
    )
    train_label = discrete_labels_train[session_id]

    with open(activations_filepath, "rb") as f:
        activations_dict = pickle.load(f)
    
    if bool_comput:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Computing tSNE embeddings...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        tSNE_dict = Lens.transform.run_tsne_and_save(activations_dict,tsne_filepath,num_samples)
    
    else:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Loading tSNE embeddings...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        with open(tsne_filepath, "rb") as f:
            tSNE_dict = pickle.load(f)

    print('yeee')
    fig1 = compare_embeddings_layers(tSNE_dict["single"]["UT"][0],tSNE_dict["single"]["TR"][0],labels=train_label, data_label="Visual",sample_plot=200)
    fig2 = compare_embeddings_layers(tSNE_dict["multi"]["UT"][0],tSNE_dict["multi"]["TR"][0],labels=train_label, data_label="Visual",sample_plot=200)
    fig3 = compare_embeddings_layers(tSNE_dict["single"]["TR"][0],tSNE_dict["multi"]["TR"][0],labels=train_label, data_label="Visual",sample_plot=200, comparison_labels= ('tSNE',["Single", "Multi"]))

    plt.show()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--activations_filepath",
        type=str,
        default='data/activations/offset10.pkl',
        help="Path to the activations file.",
    )

    parser.add_argument(
        "--tsne_filepath",
        type=str,
        default='data/tSNE/offset10.pkl',
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

    main(
        args.activations_filepath,
        args.tsne_filepath,
        args.session_id,
        args.bool_comput,
        args.num_samples,
    )
