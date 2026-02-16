# LAPS - Accelerating MRI with Longitudinally-informed Latent Posterior Sampling

This repository contains code for longitudinal reconstruction of MR images using latent diffusion models. LAPS implements longitudinally-informed latent posterior sampling techniques for improved MRI reconstruction across time series data.

Paper: https://arxiv.org/abs/2407.00537

## Setup

### Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/SetsompopLab/LAPS.git
cd LAPS
```

2. Create and activate the mamba environment:
```bash
mamba env create -f environment.yml
mamba activate laps
```

3. If not already done, create a `wandb` account as described here: https://docs.wandb.ai/models/quickstart

4. Log into huggingface credentials with `hf auth login` to retrieve models.

## Usage

### Retrieving SLAM dataset 
The SLAM dataset is available at https://purl.stanford.edu/rq296rb2765. To retrieve the data, run the following code.
All data will be saved to `./data` relative to this project's root. 
To start, it is reccomended to set `load_full` to False to retrieve a minimally sized dataset.
```python
from laps.slam import (
    pull_slam_dataset_volumes,
    prepare_slam_train,
    prepare_slam_test,
)

# If true, loads 200+ train and 20 test volumes, all with kspace. This can take 2-3 days.
# If false, will only load 5 training examples without kspace, and 1 test example with kspace. 
load_full = False

# First, loads the volumetric data from online repository
pull_slam_dataset_volumes(minimal = not load_full)

# If desired to train, can prepare slice-by-slice data for training dataloader
prepare_slam_train(dicom=False)
# to include training with the dicom images (magnitude-weighted prior scans), also run:
prepare_slam_train(dicom=True)

# To prepare test slice-by-slice data, run:
prepare_slam_test()
```

### Inference
After retrieving SLAM data, inference can be done with this script:
`python src/laps/recon_test.py`

Various reconstruction methods can be compared by adding or removing to the `recons` dictionary in the `Config` class. See `laps.recon` module for the different reconstruction methods implemented and their respective parameters.

## Model Fine-tuning

We have shared our code for fine-tuning both MedVAE and Stable Diffusion for our LAPS model development.

### Fine Tuning Stable Diffusion for Complex MRI Image Generation
To finetune stable diffusion model, a medvae model is first needed. Download ours here with this python code:
```python
from laps.model.medvae_download import download_model_folder
download_model_folder()
```

Then you can run a training script like this example: 

```bash
# Make sure the environment is active
mamba activate laps

# Run the training script
bash scripts/train_sd.sh
```
If you want to set up `accelerator` for multi gpu training run `accelerate config` and follow the instruction.

### Fine-tuning MedVAE
An example of our medvae finetune is shown with this script:
```bash
# Make sure the environment is active
mamba activate laps

# Run the training script
bash scripts/train_medvae_x4.sh
```

If you want to set up `accelerator` for multi gpu training run `accelerate config` and follow the instruction.
