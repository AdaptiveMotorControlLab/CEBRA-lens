# run using % python -m GithubFolder.src.scripts.model_decoding --model_name offset10 --session_id 3

from GithubFolder.src.cebra_lens import cebra_lens as lens
import matplotlib.pyplot as plt
import argparse


def main(model_name, session_id, bool_plot_loss):

    print("\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("BEGINNING OF SCRIPT")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Loading Data and models...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    # LOAD DATA
    train_datas, valid_datas, discrete_labels_train, discrete_labels_val = (
        lens.utils_allen.get_single_session_datasets()
    )

    train_data = train_datas[session_id].neural
    test_data = valid_datas[session_id].neural
    train_label = discrete_labels_train[session_id]
    test_label = discrete_labels_val[session_id]

    # LOAD MODELS
    models = lens.utils_allen.model_loader(model_name=model_name)

    if bool_plot_loss:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Plotting lossess...")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))

        # Plot for single models
        for i in range(len(models["single_TR"])):
            axs[0].plot(models["single_TR"][i].state_dict_["loss"], c="blue", alpha=0.6)
        axs[0].set_xlabel("Steps", fontsize=15)
        axs[0].set_ylabel("Loss", fontsize=15)
        axs[0].set_title("Single-session", fontsize=20)

        # Plot for multi models
        for i in range(len(models["multi_TR"])):
            axs[1].plot(
                models["multi_TR"][i].state_dict_["loss"], c="orange", alpha=0.6
            )
        axs[1].set_xlabel("Steps", fontsize=15)
        axs[1].set_ylabel("Loss", fontsize=15)
        axs[1].set_title("Multi-session", fontsize=20)

        fig.suptitle("Losses", fontsize=30)
        plt.show()

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Decoding models...")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n")

    results_dict = lens.quantification.decoding.decode_models(
        models=models,
        train_data=train_data,
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
        session_id=3,
    )

    fig = lens.plotting.plot_decoding(results_dict=results_dict, palette_tr="cool")
    plt.show()


if __name__ == "__main__":

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
        "--bool_plot_loss", type=int, default=1, help="Plots losses of the models"
    )

    args = parser.parse_args()
    print("INPUT PARAMETERS: ", args)
    main(args.model_name, args.session_id, args.bool_plot_loss)
