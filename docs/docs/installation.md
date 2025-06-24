# 🛠️ Installation

🚨 Make sure that the environment in which you trained the CEBRA models in **has the same torch version** as the environment used for CEBRA-Lens.

Quick Install Guide:

```bash
conda create -n CEBRAlens python=3.12
conda activate CEBRAlens
conda install -c conda-forge pytables==3.8.0

# install PyTorch with your desired CUDA version (or for CPU only)- check their website: https://pytorch.org/get-started/locally/
# example: GPU version of pytorch for CUDA 11.3
conda install pytorch cudatoolkit=11.3 -c pytorch

# install CEBRA and CEBRA-lens
pip install --pre 'cebra[datasets,demos]
pip install -- cebra_lens
```