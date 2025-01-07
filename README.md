# Riccardo's space - Exploring nonlinear encoders for robust vision decoding

This repository contains the code for Riccardo's semester's project (FALL 2024) on robustness in vision encoders. 

[Initial pitch](initial_pitch.pdf) | [Final report](final_report.pdf) | [CEBRA Lens](https://github.com/AdaptiveMotorControlLab/riccardo_workspace/tree/riccardo-packaging/src/cebra_lens)

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



