# Riccardo's space - Exploring nonlinear encoders for robust vision decoding

This repository contains the code for Riccardo's semester's project (FALL 2024) on robustness in vision encoders. 

### 💫  Initial pitch

Measuring the robustness of a model is critical in vision decoding. To measure robustness, there are two approaches we will take in the project: 

- **What representations is the model learning?** We will look at the NN units themselves after the model is trained. Using “neuroscientist methods” such as CKA, PCA/tSNE, GLMs (See Sandbrink et al 2023). 
- **How well does the model generalize to unseen data?** We will set up a new task. On already processed data: if we train on video 1 with 9 repeats and hold out 1, or train on 1 and hold out 9, how is decoding affected? And then, how does this transfer to other settings, i.e. other movies in the Allen dataset and brain observatory movies.
