# Project Mandrill Age Recognition

The goal of this project is to explore solutions to predict the age of young mandrills from pictures.
This project uses the mandrillus database that is freely available.

This repository allows to:

- Train a regression model
- Train a face identification model
- Convert a torch network to onnx
- Perform inference with a script or a streamlit app

# Installation

- Create a virtual environnment
  `conda create --name mar python=3.9`

- Install dependancies
  `pip install -r requirements.txt`

- Install the project library
  `pip install .`

# Data format

We expect the input format to be a csv with the following columns:

- id
- photo_name
- parent_folder
- shootdate
- dob
- face_view
- face_qual
- sex
- error_dob

In the mandrillus database we have a csv called `MFD_metadatas.csv` where we added the estimated error on the DoB (error_dob)

Then, from the rows we construct the path to the photos using the parent folder the id and the photo name.
The full path build using this formula: `{parent_folder}/{id}/{photo_name}`.

# Run experiments

All experiments were trained using a SLURM environment that takes advantage of a singularity container.
For all the following experiments, we omit the slurm reservation script and the creation of the singularity container.

**We assume that all parameters given are using 1x V100-32Go GPU and at least 30Go of RAM. Please adjust the parameters to your hardware capabilities**

The way we run an experiment training is by calling the `train.py` script and by specifying which experiment we want to run. We can additionnaly overwrite the parameters directly in the command line interface.

## Regression

You can train a regression model on the full age range with the best model (dinov2 large).

```sh
HYDRA_FULL_ERROR=1 python train.py experiment=regression\   # Use the regression parameters
        training.batch_size=32\                             # Adjust the batch size if necessary
        name="regression_baseline_dinov2_large"\            # The name of the experiment
        resume=False\                                       # Start from scratch
        train=True\                                         # We want to train
        training.learning_rate=0.0000005\                   # Don't have to change the learning rate
        training.epochs=100\                                # Train for 100 epochs
        similarity_head=\                                   # Do not use the similarity head
        backbone.dino_type=large\                           # We use the dinov2 large model
        losses@train_regression_loss=mse\                   # MSE loss
        training.use_augmentation=True\
        regression_head.lin_start=256\
        regression_head.n_lin=0\
        regression_head.sigmoid=False
```

To perform a cross validation, we can use a script as follow

```sh
split_index=$1
train_max_age=$2
val_max_age=$3
date=$4

HYDRA_FULL_ERROR=1 python train.py experiment=regression\
training.batch_size=96\
        name="regression_baseline_dinov2_large_0-${train_max_age}"\
        resume=False\
        train=True\
        test=True\
        training.learning_rate=0.0000005\
        training.epochs=30\
        training.float16=True\
        training.optimizer_weight_decay=0.001\
        similarity_head=\
        backbone.dino_type=large\
        losses@train_regression_loss=mse\
        training.augmentation=True\
        regression_head.lin_start=256\
        regression_head.n_lin=0\
        regression_head.sigmoid=False\
        date=$date\
        kfold_index=$split_index\
        dataset.train_max_age=$train_max_age\
        dataset.val_max_age=$val_max_age
```

Where the split_index define the index of the split (kfold=5 so index={0,1,2,3,4}).
This script is an optimized version that trains in mixed precision (fp16) and use the one cycle learning rate to train the model in only 30 epochs. The `train_max_age` and `val_max_age` correspond to the maximum age for respectively the training and validation. The age is in year here.

# Build the singularity image

You can easily build the singularity image (assuming you have singularity installed).
`sudo singularity build --nv mandrillage.sif mandrillage.def`

If this fails because you don't have enough disk space in the `/tmp` partition you can change the location of the tmp folder:
`sudo SINGULARITY_TMPDIR=<path_to_tmp> singularity build --nv mandrillage.sif mandrillage.def`

# How to export model (torch -> onnx)

On JZ you can make an interactive reservation where you will start a singularity container in shell mode.

Then if you already downloaded default model weights you can specify the path using the variable `TORCH_HOME` like so `TORCH_HOME=/mandrillagerecognition/data` from within the singularity container.

You can easily convert a model using the script located in `tools/export_dino.py`.
It has 3 simple parameters:

- `--model_path` which is the path where the checkpoint is
- `--export_path` the path where you want your onnx export
- `--dino_type` this is the dino size (small/medium/large) but usually is it large
