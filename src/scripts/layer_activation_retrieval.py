import pickle
import argparse
import cebra
from GithubFolder.src.cebra_lens import cebra_lens as lens
import os
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
    model_name, session_id, activations_filepath, bool_plot_embeddings, layer_type
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

    # LOAD MODELS
    models = lens.model.model_loader(model_name=model_name)

    if bool_plot_embeddings:

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

        fig = lens.plotting.plot_embeddings_singlevmulti(
            embeddings_single,
            embeddings_multi,
            embeddings_untrained_single,
            embeddings_untrained_multi,
            y,
        )
        fig.show()

    activations = {}
    activations = lens.activations.get_activations_multi_model(
        models=models,
        data=train_data,
        session_id=session_id,
        activations=activations,
        layer_type=layer_type,
    )

    activations_dict = lens.activations.process_activations(activations)

    with open(activations_filepath, "wb") as f:
        pickle.dump(activations_dict, f)

if __name__ == "__main__":

    setup_logging()
    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="offset10",
        help="name of the folder where the models (assuming they are under FinalModels/VISION)",
    )

    parser.add_argument(
        "--session_id",
        type=int,
        default=3,
        help="session id for the analysis, used to retrieve the correct data and multi-session model",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="data/activations/offset10.pkl",
        help="filename of the activations",
    )
    parser.add_argument(
        "--bool_plot_embeddings",
        type=int,
        default=0,
        help="Plots the output embeddings of the models (0 or 1)",
    )
    parser.add_argument(
        "--layer_type",
        type=str,
        default="conv",
        help="Type of layer: e.g. 'conv', 'all'",
    )

    args = parser.parse_args()
    main(
        args.model_name,
        args.session_id,
        args.activations_filepath,
        args.bool_plot_embeddings,
        args.layer_type,
    )
