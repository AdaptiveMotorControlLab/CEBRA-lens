# Riccardo's space - Exploring nonlinear encoders for robust vision decoding

This repository contains the code for Riccardo's semester's project (FALL 2024) on robustness in vision encoders. 

## 💫  Initial pitch

*TO DO: UPDATE PITCH*

Measuring the robustness of a model is critical in vision decoding. To measure robustness, there are two approaches we will take in the project: 

- **What representations is the model learning?** We will look at the NN units themselves after the model is trained. Using “neuroscientist methods” such as CKA, PCA/tSNE, GLMs (See Sandbrink et al 2023). 
- **How well does the model generalize to unseen data?** We will set up a new task. On already processed data: if we train on video 1 with 9 repeats and hold out 1, or train on 1 and hold out 9, how is decoding affected? And then, how does this transfer to other settings, i.e. other movies in the Allen dataset and brain observatory movies.

## Repository organization

The Github repository contains all the code used for the project without the Models and the Data. The data can be found [here](https://figshare.com/s/60adb075234c2cc51fa3?file=36869049) and an example usage of the data can be found in this [demo](https://cebra.ai/docs/demo_notebooks/Demo_Allen.html).

The `Notebooks` folder contains all the relevant notebooks:
- `ModelGeneratorVISUAL.ipynb`and `ModelGeneratorHPC.ipynb` were used to generate the models used in the analysis notebooks
- `AnalaysisHPC.ipynb` was used for the analysis on the hippocampus dataset. The analysis in this notebook are the following:
	- Layer analysis:
		- Single-unit activation plotting across layers
		- CEBRA embeddings across layers
		- t-SNE embeddings across layers
		- CKA
		- RDM
	- fitting of a GLM on each neuron using the 3 labels as features to measure feature importance across layers
- `AnalaysisVISUAL.ipynb`and `AnalaysisVISUALtest.ipynb` were used for the analysis on the train and test set respectively of the Allen visual dataset. The main analysis was carried on the training  set to focus on the model training and its learning. Test analysis was carried to see if any difference would arise on unseen data. The analysis carried here are the following:
	- Single-unit activation plotting across layers
	- CEBRA embeddings across layers
	- t-SNE embeddings across layers
	- CKA
	- RDM  
	- Distance across layers (euclidean and cosine):
		- Intraclass
		- Interclass
		- Across repetitions
- `decodingacrosslayerVISUAL.ipynb` computes the decoding across layer on the Allen visual dataset
- `noise_testVISUAL.ipynb`computes the decoding across varying levels of noise on the data. Here the models are trained on the clean data but during decoding the train and test data are injected with Poisson or Gaussian noise.  A variant where the models are trained on noisy data will have to be made for comparison

*TO DO: change the folder arrangement in src --> remove preprocessing and clean the names. also clean the paths in the notebooks*
The `src`folder contains all the preprocessing and utils code. It contains two main directories `image_processing` and `CEBRA_preprocessing`. 

- `image_processing` contains the functions used during the first weeks of the project to extract the images from the Openscope dataset (`_extract_images.py`) and, from these images, extract the DINOv2 features (`extract_dinov2.py`). Please note that the folder `image_processing` does not contain any code that was used in the final notebooks and submission.
- `CEBRA_preprocessing` contains three utils used in the notebooks. They are separated based on their main functions:
	- `data_utils.py` contains all the relevant functions for data handling (loading, splitting, separation)
	- `plotting_utils.py` contains all the functions to plot (CEBRA embeddings, t-SNE embeddings, layer activations, RDM matrices)
	- `quantification.py` contains all the functions for metrics computation (distances, decoding, CKA, RDM)

