# Riccardo's space - Exploring nonlinear encoders for robust vision decoding

This repository contains the code for Riccardo's semester's project (FALL 2024) on robustness in vision encoders. 

## 💫  Initial pitch

Measuring the robustness of a model is critical in vision decoding. We can thus wonder **what representations is the model learning?** By looking at the NN units themselves after the model is trained, using “neuroscientist methods” such as CKA, PCA/tSNE (See Sandbrink et al 2023), we can get a glimpse of what the models learn. A more detailed description can be found in this [PDF](docs/your-file.pdf).

The final aim of this Github folder is to mimic a python package that allows the user to analyze the layers of the CEBRA models.

## Repository organization

This Github repository contains scripts and package code without the Models and the Data. The data can be found [here](https://figshare.com/s/60adb075234c2cc51fa3?file=36869049) and an example usage of the data can be found in this [demo](https://cebra.ai/docs/demo_notebooks/Demo_Allen.html). 

The `Notebooks` folder contains two notebooks:
- `ModelGeneratorVISUAL.ipynb` which was used to generate the models for the demo analysis notebook. It is meant to be run on Colab and was thus not scripted.
- `Demo-Notebook-Allen.ipynb` contains the demo that goes over the whole package. It is more descriptive than the scripts and more flexible. This is the reference for understanding how the functions work.

The `scripts` folder contains the scripted versions of the analysis with the exception of the decoding by layer.

Note: The notebooks that reproduce exactly the results of the report have been deleted because cleaning them would have been too time consuming and the results can be now reproduced using the package. Analysis such as GLMs were not pushed far enough to be kept in this final version of the repository and have thus been deleted. 
All these notebooks can still be found travelling back to the commit XXXX.
