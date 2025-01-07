# Riccardo's space - Exploring nonlinear encoders for robust vision decoding

This repository contains the code for Riccardo's semester's project (FALL 2024) on robustness in vision encoders. 


[Initial pitch](initial_pitch.pdf) | [Final report](final_report.pdf) | [CEBRA Lens](https://github.com/AdaptiveMotorControlLab/riccardo_workspace/)

## :dizzy:  Initial pitch

Measuring the robustness of a model is critical in vision decoding. We can thus wonder **what representations is the model learning?** By looking at the NN units themselves after the model is trained, using “neuroscientist methods” such as CKA, PCA/tSNE (See Sandbrink et al 2023), we can get a glimpse of what the models learn. A more detailed description can be found in this [initial pitch](initial_pitch.pdf). The final report resuming all the analysis and detailing the methods is [here](final_report.pdf).

## :mag: cebra_lens package
The final aim of this Github folder is to mimic a python package that allows the user to analyze the layers of the CEBRA models. This package is called cebra_lens and allows to replicate the analysis pursued during the project with commented and detailed functions that can take multiple types of models with different architectures. Work has to be done to adapt some code for different models, for now it works for offset5, 10 and 10-mse. This is due to small differences in padding that could not be solved in a flexible way. Most functions work with hippocampus and Allen visual dataset, however, the focus being on the visual allen dataset, most effort has been put into making that part work.


<img src="zebra.png" alt="zebra" width="200" height="194">


## :books: Repository organization
This Github repository contains scripts, notebooks and package code without the Models and the Data. The data can be found [here](https://figshare.com/s/60adb075234c2cc51fa3?file=36869049) and an example usage of the data can be found in this [demo](https://cebra.ai/docs/demo_notebooks/Demo_Allen.html). 

The `Notebooks` folder contains two notebooks:
- `ModelGeneratorVISUAL.ipynb` which was used to generate the models for the demo analysis notebook. It is meant to be run on Colab and was thus not scripted.
- `Demo-Notebook-Allen.ipynb` contains the demo that goes over the whole package. It is more descriptive than the scripts and more flexible. This is the reference for understanding how the functions work.

The `src` folder contains two main folders:

1. The `scripts` folder contains the scripted versions of the analysis with the exception of the decoding by layer:
- tSNE 
- CKA 
- RDM
- Layer activation retrieval
- Model decoding
- Distance calculation

2. The `cebra_lens folder contains the package build to replicate the analysis on the same or different models.

![Abstract figure](abstractfig.png)

Note: The notebooks that reproduce exactly the results of the report have been deleted because cleaning them would have been too time consuming and the results can be now reproduced using the package. Analysis such as GLMs were not pushed far enough to be kept in this final version of the repository and have thus been deleted. 
All these notebooks can still be found travelling back to the commit [f89dd1b](https://github.com/AdaptiveMotorControlLab/riccardo_workspace/tree/f89dd1b801144912348e414c53dc21e9b5c6c937).





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

=======

