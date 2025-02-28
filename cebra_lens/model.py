"Model handling. For now only loading is used."

import pathlib
import cebra
import torch


def model_loader(model_dir: str, labels: dict = {}) -> dict:
    """
    Load and categorize models based on their training status and session type.
    Parameters:
    -----------
    model_dir : str
        The path of the models: e.g. FinalModels/VISION

    #maybe tell the user to add a dictionary with the labels of the models he wants, kwargs can be the labels for models, example :
    labels{
    'model1': 'single_UT',
    'model2': 'single_UT',
    'model3': 'multi_UT',
    'model4': 'multi_UT',
    'model5': 'single_TR',
    'model6': 'single_TR'
    }
    Returns:
        dict: A dictionary containing the loaded models (label, model) where label is taken from the input dictionary given by user?
    """

    # LOAD MODELS

    models_folder_path = pathlib.Path(model_dir)
    if not pathlib.Path.exists(models_folder_path):
        raise FileNotFoundError(f"Folder {models_folder_path} not found.")
    models = {}
    for file in pathlib.Path.iterdir(models_folder_path):
        if str(file).endswith(".pt") or str(file).endswith(".pth"):
            print(f"Model {file.stem} loading...")
            model_path = models_folder_path / file
            loaded_model = cebra.CEBRA.load(
                model_path,
                backend="torch",
                map_location=torch.device("cpu"),
            ).to("cpu")
            key = labels.get(file.stem, False)
            if not key or not models.get(key, False):
                models[file.stem] = [loaded_model]
            else:
                models[key].append(loaded_model)
            # what is solver_name and how is it chosen from the model file?

            # for now this just assigns label = model file name
            # print(f"Solver_name =  {loaded_model.solver_name_}")
            print(f"Model {file.stem} loaded succesfully.")

    return models
