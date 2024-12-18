# run using (e.g): python -m GithubFolder.src.scripts.layer_activation_retrieval --layer_type conv --session_id 3 --filename offset10alllayers
# attention: need to be one step above the GithubFolder to have data and finalmodels

import os
import pickle
import torch.nn as nn
import argparse
import cebra

from GithubFolder.src.cebra_lens import cebra_lens as lens
from GithubFolder.src.preprocessing.CEBRA_preprocessing.data_utils import (
    get_single_session_datasets,
    model_loader,
)
from GithubFolder.src.preprocessing.CEBRA_preprocessing.plotting_utils import (
    plot_embeddings_singlevmulti,
)



def main(model_name, layer_type, session_id, filename, bool_plot_embeddings):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading Data and models...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    # LOAD DATA
    train_datas, valid_datas, discrete_labels_train, discrete_labels_val = (
        get_single_session_datasets()
    )

    train_data = train_datas[session_id].neural
    train_label = discrete_labels_train[session_id]

    # LOAD MODELS
    models = model_loader(model_name=model_name)

    if bool_plot_embeddings:
        print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Calculating output embeddings...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

        X = train_data
        y = train_label
        embeddings_single = []
        embeddings_multi = []

        # Go to 5 max for plotting clarity (works even if there are less than 5 models)
        for model in models["multi_TR"][:5]:
            embeddings_multi.append(model.transform(X, session_id=session_id))
        for model in models["single_TR"][:5]:
            embeddings_single.append(model.transform(X))

        # Align the single session embeddings to the first rat
        alignment = cebra.data.helper.OrthogonalProcrustesAlignment()

        for i in range(len(embeddings_single)):
            embeddings_single[i] = alignment.fit_transform(
                embeddings_single[0], embeddings_single[i], y, y
            )

        for i in range(len(embeddings_multi)):
            embeddings_multi[i] = alignment.fit_transform(
                embeddings_multi[0], embeddings_multi[i], y, y
            )

        embeddings_untrained_single = models["single_UT"][0].transform(
            X
        )  # only take the first untrained model for plotting
        embeddings_untrained_multi = models["multi_UT"][0].transform(
            X, session_id=session_id
        )  # only take the first untrained model for plotting

        fig = plot_embeddings_singlevmulti(
            embeddings_single,
            embeddings_multi,
            embeddings_untrained_single,
            embeddings_untrained_multi,
            y,
        )
        fig.show()
        
    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Retrieving layer activations...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    activations2 = {}
    activations2 = lens.activations.get_activations_multi_model(
        models=models,
        data=train_data,
        session_id=session_id,
        activations=activations2,
        layer_type="conv",
    )

    activations_dict = lens.activations.process_activations(activations2)


    with open(f"{filename}.pkl", "wb") as f:
        pickle.dump(activations_dict, f)
        print(
            "\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        )
        print("Layer activations saved!")
        print(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="offset10",
        help="name of the folder where the models (assuming they are under FinalModels/VISION)",
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="conv",
        help="Type of layer to process ('all' or 'conv')",
    )
    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--filename", type=str, default="offset10", help="filename of the activations"
    )
    parser.add_argument(
        "--bool_plot_embeddings",
        type=int,
        default=1,
        help="Plots the output embeddings of the models (0 or 1)",
    )

    args = parser.parse_args()
    print(args)
    main(
        args.model_name,
        args.layer_type,
        args.session_id,
        args.filename,
        args.bool_plot_embeddings,
    )
