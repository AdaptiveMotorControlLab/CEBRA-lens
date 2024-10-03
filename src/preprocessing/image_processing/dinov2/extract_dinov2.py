import sys
import cv2
import torch
from torchvision import transforms


from utils import Backbone, save_features,  extract_stimuli_paths

#NOTE(celia): to add the segmentation mask, see /src/video_processing/dinov2/segmentation.py.

import glob
import os

if __name__ == "__main__":

    import argparse
    #import parser

    parser = argparse.ArgumentParser(description="Extract features from images")
    #NOTE(celia): depending on the backbone size wanted: 
    # -> small: "vits14"
    # -> base: "vitb14"
    # -> large: "vitl14"
    # -> giant: "vitg14"
    parser.add_argument(
        "--backbone_name", type=str, default=f"vitg14", help="backbone name"
    )
    parser.add_argument("--stimuli", nargs="+", help="stimuli to use", default="all")
    parser.add_argument(
        "--skip", type=str, default=["dots", "spontaneous", "noise", "natural_movie_1"]
    )
    args = parser.parse_args()
    locals().update(args.__dict__)

    #if stimuli == "all":
    #    stimuli = set(schema.ImageTable.fetch("stimulus"))
    #else:
    #    stimuli = set(stimuli)
  
    stimuli = set(stimuli)

    if skip is None:
        skip = []

    skip = set(skip)
    stimuli = stimuli - skip
    image_paths = extract_stimuli_paths(stimuli)
    model = Backbone(backbone_name)
    
    for stimulus in stimuli:
        trans = transforms.Compose([transforms.ToTensor()])

        
        # Define the path to the folder that contains the images
        #image_folder = '/content/drive/MyDrive/CEBRA/Allen/snake'
        image_folder = image_paths[stimulus]

        # Get all the image paths in the folder. Assuming images have a .jpg extension.
        # If your images have a different extension, adjust the pattern accordingly.
        image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')))  # Adjust the pattern if necessary

        images = torch.stack(
            [trans(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)) for image_path in image_paths]
        )
        
        '''
        # Modified version with print statement:
        images = []
        for image_path in image_paths:
            print(f"Processing image: {image_path}")  # Print the current image path
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = trans(img_rgb)
                images.append(img_tensor)
            else:
                print(f"Failed to read image: {image_path}")
        '''
        # Stack the list of images into a single tensor
        #images = torch.stack(images)
        
        print(type(images))
        print(images.shape)
        with torch.no_grad():
            image_features = model.extract_embeddings(images)
            #image_features = model.encode_image(images)
            print("extracted embeddings")
        save_features(image_features, stimulus, backbone_name, feature_name="Dinov2_embeddings")
        print("saved")