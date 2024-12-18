import os
import numpy as np
import torch
import cebra
import cebra.datasets
import copy


def model_loader(model_name):
    """
    Load and categorize models based on their training status and session type.
    Args:
        model_name (str): The name of the model to load. This should correspond to a folder
                          within "FinalModels/VISION/" containing the model files.
    Returns:
        dict: A dictionary containing the loaded models categorized into four keys:
              - "single_UT": List of single-session untrained models.
              - "multi_UT": List of multi-session untrained models.
              - "single_TR": List of single-session trained models.
              - "multi_TR": List of multi-session trained models.
    """

    # LOAD MODELS
    models_folder_path = f"FinalModels/VISION/{model_name}"
    files_list = os.listdir(models_folder_path)

    models_list = []
    for file in files_list:  # load only the torch models for cpu usage
        if file.endswith("torch.pt"):
            models_list.append(file)

    models_list
    print("Number of models: ", len(models_list))
    print(models_list)

    models_single_UT = []  # will be all the singles untrained
    models_multi_UT = []  # will be all the singles untrained
    models_single_TR = []  # will be all the singles trained
    models_multi_TR = []  # will be all the multi trained

    for model in models_list:

        loaded_model = cebra.CEBRA.load(
            os.path.join(models_folder_path, model),
            backend="torch",
            map_location=torch.device("cpu"),
        ).to("cpu")
        if "_UT" in model:
            if loaded_model.solver_name_ == "multi-session":
                models_multi_UT.append(loaded_model)
            elif loaded_model.solver_name_ == "single-session":
                models_single_UT.append(loaded_model)
            else:  # e.g. Unified
                raise NotImplementedError(
                    "Only single session and multi session are implemented"
                )

        # TODO: This should be changed to elif "TR" in model.  This comes from the name of the file when you save the model
        # Models were trained without this label but it might be clearer in the future
        else:
            if loaded_model.solver_name_ == "multi-session":
                models_multi_TR.append(loaded_model)
            elif loaded_model.solver_name_ == "single-session":
                models_single_TR.append(loaded_model)
            else:  # e.g. Unified
                raise NotImplementedError(
                    "Only single session and multi session are implemented"
                )

    models = {
        "single_UT": models_single_UT,
        "multi_UT": models_multi_UT,
        "single_TR": models_single_TR,
        "multi_TR": models_multi_TR,
    }

    # check the models
    print("# of Single Untrained models: ", len(models["single_UT"]))
    print("# of Single Trained models: ", len(models["single_TR"]))
    print("# of Multi Untrained models: ", len(models["multi_UT"]))
    print("# of Multi Trained models: ", len(models["multi_TR"]))

    return models


########################################################################################################################
########################################################################################################################
######################################## TAKEN FROM utils_allen.py of Célia ############################################
########################################################################################################################
########################################################################################################################


def get_single_session_datasets(
    test_session=9,
    corrupted=False,
    pseudomice=False,
    mice=4,
    shot_noise: float = None,
    gaussian_noise: float = None,
):
    """
    Args:
        test_session: The session ID to consider as the test session. NOTE(celia): this will need
            to be changed if we want to test smaller training set number of repeats.
        corrupted: If True, loads the corrupted dataset, see `datasets/allen/single_session_ca.py` in CEBRA
            codebase.
        pseudomice: If True, uses pseudomice rather than full, with default number of neurons per
            mouse.
        mice: Number of mice to use (max is 4, for now). NOTE(celia): this could be increased by
            pre-processing more mice.
        shot_noise: Level of shot noise (Poisson noise) to apply on the dataset. Default is None,
            and that means that no noise is applied.
        gaussian_noise: Value of the standard deviaiton of the Gaussian noise to add on the data.
            Default is None, and that means that no noise is applied.

    Returns:
        The train and valid datasets, and the train and valid frame IDs.
    """
    train_datas, valid_datas = [], []
    if corrupted:
        for i in range(mice):
            train_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-corrupt-{i}-repeat-{test_session}-train"
                )
            )
            valid_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-corrupt-{i}-repeat-{test_session}-test"
                )
            )
    else:
        for i in range(4):
            train_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-{i}-repeat-{test_session}-train"
                )
            )
            valid_datas.append(
                cebra.datasets.init(
                    f"allen-movie1-ca-single-session-decoding-{i}-repeat-{test_session}-test"
                )
            )
    if pseudomice:
        for i in range(len(train_datas)):
            train_datas[i].neural = torch.from_numpy(
                obtain_pseudomice(
                    [train_datas[i].neural for i in range(len(train_datas))]
                )
            )
            valid_datas[i].neural = torch.from_numpy(
                obtain_pseudomice(
                    [valid_datas[i].neural for i in range(len(train_datas))]
                )
            )

    # Add noise to the 4th mouse only
    if shot_noise is not None:
        # train_datas[0].neural = _add_shot_noise(train_datas[0].neural, scale_factor=shot_noise)
        valid_datas[3].neural = _add_shot_noise(
            valid_datas[3].neural, scale_factor=shot_noise
        )
    elif gaussian_noise is not None:
        # train_datas[0].neural = _add_gaussian_noise(train_datas[0].neural, sigma=gaussian_noise)
        valid_datas[3].neural = _add_gaussian_noise(
            valid_datas[3].neural, sigma=gaussian_noise
        )

    # discrete_labels = [np.tile(np.arange(900), 10) for i in range(len(mice))]
    discrete_labels_train = [np.tile(np.arange(900), 9) for i in range(mice)]
    discrete_labels_val = [np.tile(np.arange(900), 1) for i in range(mice)]

    return train_datas, valid_datas, discrete_labels_train, discrete_labels_val


def obtain_pseudomice(mice, num_neurons_per_mouse=80):
    """
    Creates a pseudomouse by selecting a random subset of neurons from each mouse's neural data.

    Parameters:
    mice (list): List of neural data for different mice
    num_neurons_per_mouse (int): the number of neurons to select from each mouse

    Returns:
    Mouse: a pseudomouse object with the concatenated neural data from the selected neurons
    """
    neuron_ids = []
    pseudomice_matrix = None
    for i, session in enumerate(mice):
        session_length = session.shape[1]
        selected_neurons = np.random.choice(
            session_length, replace=False, size=num_neurons_per_mouse
        )
        neuron_ids.append(selected_neurons)
        if pseudomice_matrix is None:
            pseudomice_matrix = session[:, selected_neurons]
        else:
            pseudomice_matrix = np.concatenate(
                (pseudomice_matrix, session[:, selected_neurons]), axis=1
            )

    pseudomouse = copy.deepcopy(mice[0])
    pseudomouse = pseudomice_matrix
    return pseudomouse


def _add_gaussian_noise(neural_data, sigma: float = 2):
    gaussian_noise = torch.normal(mean=0.0, std=sigma, size=neural_data.size())
    return neural_data + gaussian_noise


def _add_shot_noise(neural_data, scale_factor: float = 1.0):
    # Neural data * scale_factor = Poisson lambda
    return torch.poisson(neural_data * scale_factor) / scale_factor
