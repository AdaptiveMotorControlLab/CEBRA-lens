"Model handling. For now only loading is used."
import os
import cebra
import torch


def model_loader(
    model_path: str = "FinalModels/VISION", model_name: str = "offset10"
) -> dict:
    """
    Load and categorize models based on their training status and session type.
    Parameters:
    -----------
    model_path : str
        The path of the models: e.g. FinalModels/VISION.
    model_name : str
        The name of the model to load. e.g. 'offset10'.
    Returns:
        dict: A dictionary containing the loaded models categorized into four keys:
              - "single_UT": List of single-session untrained models.
              - "multi_UT": List of multi-session untrained models.
              - "single_TR": List of single-session trained models.
              - "multi_TR": List of multi-session trained models.
    """

    # LOAD MODELS
    models_folder_path = os.path.join(model_path, model_name)
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
    # models_unified_UT = []
    # models_unified_TR = []

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
                    f"Solver {loaded_model.solver_name_} not yet implemented. Only single session and multi session are implemented."
                )

        # TODO: This should be changed to elif "TR" in model.  This comes from the name of the file when you save the model
        # Models were trained without this label but future training should include _TR to the modelname.

        # elif "_TR" in model:

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
