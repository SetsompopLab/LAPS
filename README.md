# PIPS - Prior Informed Latent Posterior Sampling

This repository contains code for longitudinal reconstruction of MR images using latent diffusion models. PIPS implements prior-informed latent posterior sampling techniques for improved MRI reconstruction across time series data.

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

### SD Model and Fine-tuning development

We have shared our code for fine-tuning both MedVAE and Stable Diffusion for our LAPS model development.

### Fine-tuning MedVAE
To finetune medvae, run:
```bash
# Make sure the environment is active
mamba activate laps

# Run the training script
bash scripts/train_medvae_x4.sh
```

If you want to set up `accelerator` for multi gpu training run `accelerate config` and follow the instruction.

### FT Stable Diffusion
To finetune stable diffusion model, run:

```bash
# Make sure the environment is active
mamba activate pips

# Run the training script
bash scripts/train_sd.sh
```
If you want to set up `accelerator` for multi gpu training run `accelerate config` and follow the instruction.


## Adding a dataset
Link the csv table to ./data folder using:
`ln -s /path/to/csv ./data/<dsname>.csv`

link the data:
`ln -s /path/to/data ./data/`

In `src/pips/dataloaders/loaders.py` add a loader type, and update `get_loader` with a loader for the new dataset.

In `src/pips/dataset.py` update PIPS_DATASETS with the new dataset.

For training the SD model, update the config in `src/pips/config/sd_training.py`.

For medvae, update the config: `/data/yurman/repos/pips/submodules/pips-medvae/configs/dataloader/slam.yaml`

## Components usage examples
Can be found under tests.
