import itertools
import math
import os
from typing import Any, Dict, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)

        return output


class Backbone:
    def __init__(self, model_type: str = "vits14"):
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_type}")

    def _to_tensor(self, x: Union[List[np.ndarray], np.ndarray]) -> torch.Tensor:

        x = np.array(x)
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x / 255.0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)

        return x

    def extract_embeddings(self, x: Union[torch.Tensor, np.ndarray]):
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)

        if hasattr(self.model, "patch_size"):
            self.model.register_forward_pre_hook(
                lambda _, x: CenterPadding(self.model.patch_size)(x[0])
            )

        self.model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        x = x.to(device)
        print("DEVICE", device)

        with torch.no_grad():
            embedding_size = self.model(x[0:1, :, :, :]).shape[
                1
            ]  # Get embedding size from the model
            num_embeddings = x.shape[0]  # Total number of items in x

            # Preallocate embeddings tensor
            embeddings = torch.empty((num_embeddings, embedding_size), device=x.device)
            print("Total number of Batches: ", num_embeddings // 10)

            # Process in batches
            for i in tqdm(range(0, num_embeddings, 10), desc="Processing batches"):
                # Define the end index for the current batch
                end_idx = min(
                    i + 10, num_embeddings
                )  # Handle the last batch if it's smaller than 10
                # Get the current batch
                embeddings_temp = self.model(x[i:end_idx, :, :, :])
                # Store the results in the preallocated tensor
                embeddings[i:end_idx] = embeddings_temp
        return embeddings

    @staticmethod
    def save_embeddings(embeddings: torch.Tensor, key: Dict[str, Any]) -> str:
        print(f"Saving embeddings for {key['stimulus']}")
        save_path = os.path.join(
            os.environ["DATA_PATH"],
            key["stimulus"],
            "dinov2",
            f'{key["backbone_name"]}.pt',
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)

        return save_path


# was in thirdparty utils of anastasiia
'''
# TO DO: Make this work
def extract_image_paths(stimulus: str) -> List[str]:
    """Load images from the database and return their absolute paths.
    
    Args: 
        stimulus: Name of the video (stimulus) whose absolute path to extract.
        
    Return: 
        List of absolute paths to the `stimulus` data.
    """

    #restriction = schema.ImageTable & {"stimulus": stimulus}
    #image_relative_paths = restriction.fetch("image_path")


    try:
        data_dir = os.environ["DATA_PATH"]
    except KeyError:
        raise ValueError("DATA_PATH environment variable is not set")
    image_absolute_paths = [
        os.path.join(data_dir, path) for path in image_relative_paths
    ]

    return image_absolute_paths
'''


def extract_stimuli_paths(stimuli_names):
    stimuli_paths = {}

    try:
        data_dir = os.environ["DATA_PATH"]
    except KeyError:
        raise ValueError("DATA_PATH environment variable is not set")

    # Loop over each stimulus name to find its path
    for stimulus in stimuli_names:
        stimulus_path = os.path.join(data_dir, stimulus)

        # Check if the path exists and is a directory
        if os.path.exists(stimulus_path) and os.path.isdir(stimulus_path):

            stimuli_paths[stimulus] = stimulus_path

        else:
            print(
                f"Warning: Path for stimulus '{stimulus}' does not exist or is not a directory."
            )

    return stimuli_paths


def save_features(
    image_features, stimulus, backbone_name, feature_name: str = "features"
):
    """Save the embeddings in `image_features` to the corresponding `stimulus`.

    Args:
        image_features: Embeddings to save as a `.pt` file.
        stimulus: The stimulus it corresponds to
        backbone_name: Name of the backbone model used.
        feature_name: Str describing the type of embedding that is saved.

    """

    try:
        data_dir = os.environ["DATA_PATH"]
    except KeyError:
        raise ValueError("DATA_PATH environment variable is not set")

    save_path = os.path.join(
        os.environ["DATA_PATH"], stimulus, feature_name, f"{backbone_name}.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(image_features, save_path)
    print(f"Saved features for {stimulus} at {save_path}")
