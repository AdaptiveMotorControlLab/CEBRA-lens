import os
from typing import List
import torch


def extract_image_paths(stimulus: str) -> List[str]:
    """Load images from the database and return their absolute paths.

    Args:
        stimulus: Name of the video (stimulus) whose absolute path to extract.

    Return:
        List of absolute paths to the `stimulus` data.
    """

    # restriction = schema.ImageTable & {"stimulus": stimulus}
    # image_relative_paths = restriction.fetch("image_path")

    try:
        data_dir = os.environ["DATA_PATH"]
    except KeyError:
        raise ValueError("DATA_PATH environment variable is not set")
    image_absolute_paths = [
        os.path.join(data_dir, path) for path in image_relative_paths
    ]

    return image_absolute_paths


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
