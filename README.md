# CEBRA-Lens: a helper package for interpretable latent spaces
<img src="figures/zebra.png" alt="zebra" width="200" height="194">

What is CEBRA-Lens?

This python codebase allows for neural representation analysis of CEBRA models. It contains tools to help answer the question: **What representations is my model learning?**  We can get a glimpse of what the models learn by looking at the NN units themselves after the model is trained, using “neuroscientist methods” such as CKA, PCA/tSNE (See Sandbrink et al 2023). Percisely these "neuroscientist methods" are implemented in this codebase.

The current version of CEBRA-Lens supports specific analysis on the Allen visual dataset and hippocampus dataset, and for general analysis on other datasets.

## 🔍 Analysis

The implemented "neuroscientist methods" for neural representation analysis are
- Model decoding:
    - R^2 score
    - error score
- Model decoding per layer
- Retrive activations:
    - 2D plot
    - 3D plot
- tSNE
- Central Kernal Alignment (CKA)
- Representational Dissmilarity Matrix (RDM):
    - plot model layer RDM
    - plot correlation with Oracle RDM across layers
- Distance:
    - intra-bin
    - inter-bin
    - inter-repetition

<img src="figures/analysis.png" alt="analysis">

## 📚 Codebase folder structure

Below is the folder structure of the repository with the main folder and files. The `cebra_lens` folder contains all the code for the analysis with the metric class definitions in the `quantification` folder, the `demos` folder contains the usage jupyter notebooks and finally there is a `tests` folder which contains some pytest for the repo.

    CEBRA_lens/
    ├── README.md
    ├── cebra_lens/
    │   ├── quantification/
    │   │   ├── base.py
    │   │   ├── cka_metric.py
    │   │   ├── decoding.py
    │   │   ├── distance.py
    │   │   ├── misc.py
    │   │   ├── rdm_metric.py
    │   │   └── tsne.py
    │   ├── activations.py
    │   ├── matplotlib.py
    │   ├── utils_allen.py
    │   ├── utils_hpc.py
    │   └── utils.py
    │
    ├── demos/
    │   ├── UsageDemoVISUAL.ipynb
    │   └── UsageDemoGENERAL.ipynb
    └── tests/

## 📊Usage

The CEBRA-Lens package allows for analyzing the embeddings of CEBRA models, but also offers the functionality of comparing embeddings and behavior through layers between models. For this purpose the code logic is centered around "metric classes". Before every analysis you first must initalize the corresponding metric class with the necessary arguments, and then to compute the metric the overhead function `compute_metric(data, metric_class)` needs to be called, this is the same for plotting, `plot_metric(data, metric_class)`.
For example:

    interbin_class = lens.Distance(
        data=train_data,
        label=train_label,
        dataset_label=dataset_label,
        metric=metric,
        distance_label="interbin",
    )
    interbin_dict = lens.compute_metric(
        activations_dict,
        interbin_class
    )
    fig = lens.plot_metric(
        interbin_dict, 
        interbin_class, 
        title="Inter-bin distance"
    )

The full demonstration of the usage is in the form of 2 jupyter notebooks:
- UsageDemoVISUAL: analysis on the Allen visual dataset
- UsageDemoGENERAL: analysis on the Hippocampus dataset, but without specific dataset functions

These two notebooks showcase the different approach when analyzing a pre-defined dataset and a non-defined dataset.

## 🛠️ Download dataset

The utils.py file contains a over-arching `get_data` function which checks for a pre-defined dataset label and accordingly loads the data based on specific functions for the dataset. If you want to load data from a non-defined dataset, you need to first import the loading function inside the utils.py file as so:

    from .utils_new import get_datasets as get_datasets_new

then add a if clause for your new dataset:

    elif dataset_label == "new_dataset":
            return get_datasets_new(session_id=session_id)

This is briefly repeated in the usage demo notebooks.

## 🛠️ Environment set-up

Make sure that the environment in which you trained the CEBRA models in has the same torch version as the environmnet used for CEBRA-Lens.

    !pip install --pre 'cebra[datasets,demos]'


**Adaptation for use on CEBRA-Unified and xCEBRA models is needed.**

Have fun!