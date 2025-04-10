"Model handling. For now only loading is used."

import pathlib
import cebra
import torch
from typing import Dict, List

def model_loader(model_dir: str, labels: Dict = {}) -> Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]:
    """
    Load and categorize models based on their training status and session type.
    Parameters:
    -----------
    model_dir : str
        The path of the models: e.g. FinalModels/VISION

    labels : Dict, optional
        A dictionary containing the labels for the models. The keys should be the model file names and the values should be the model category labels. Default is an empty dictionary:
    e.g. {'model1': 'single_UT',
        'model2': 'single_UT',
        'model3': 'multi_UT',
        'model4': 'multi_UT',
        'model5': 'single_TR',
        'model6': 'single_TR'}
    Returns:
         Dict[str, List[cebra.integrations.sklearn.cebra.CEBRA]]: A dictionary containing the loaded models (label, model) where label is taken from the input dictionary given by user or if not given, the model file name.
    """

    # LOAD MODELS

    models_folder_path = pathlib.Path(model_dir)
    if not pathlib.Path.exists(models_folder_path):
        raise FileNotFoundError(f"Folder {models_folder_path} not found.")
    models = {}
    for file in pathlib.Path.iterdir(models_folder_path):
        if str(file).endswith((".pt", ".pth")):
            model_path = models_folder_path / file
            loaded_model = cebra.CEBRA.load(
                model_path,
                backend="torch",
                map_location=torch.device("cpu"),
            ).to("cpu")
            key = labels.get(file.stem, None)
            if key is None:
                models[file.stem] = [loaded_model]
            else:
                if key not in models:
                    models[key] = [loaded_model]
                else:
                    models[key].append(loaded_model)
            print(f"Model {file.stem} loaded succesfully.")

    return models
