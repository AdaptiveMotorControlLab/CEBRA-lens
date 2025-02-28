"Model handling. For now only loading is used."
import pathlib
import cebra
import torch


def model_loader(
    model_dir: str
) -> dict:
    """
    Load and categorize models based on their training status and session type.
    Parameters:
    -----------
    model_dir : str
        The path of the models: e.g. FinalModels/VISION

    #maybe tell the user to add a dictionary with the labels of the models he wants
    Returns:
        dict: A dictionary containing the loaded models (label, model) where label is taken from the input dictionary given by user? 
    """

    # LOAD MODELS
    models = {}
    models_folder_path = pathlib.Path(model_dir)
    if not pathlib.Path.exists(models_folder_path):
        raise FileNotFoundError(f"Folder {models_folder_path} not found.")
    
    for file in pathlib.Path.iterdir(models_folder_path):
        if str(file).endswith(".pt") or str(file).endswith(".pth"):
                model_path = models_folder_path / file
                loaded_model = cebra.CEBRA.load(model_path,
                backend="torch",
                map_location=torch.device("cpu"),
                )
                print("Loaded model")
                loaded_model.to("cpu")

                #what is solver_name and how is it chosen from the model file?

                #for now this just assigns label = model file name
                print(f"Solver_name =  {loaded_model.solver_name_}")
                models[file.stem] = loaded_model
                print(f"Model {file.stem} loaded succesfully.")

    return models
