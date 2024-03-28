# Project Mandrill Age Recognition

# Installation

`conda create --name mar python=3.9`

`python train.py --config-path=config --config-name=regression`

# How to export model (torch -> onnx)

On JZ you can make an interactive reservation where you will start a singularity container in shell mode.

Then if you already downloaded default model weights you can specify the path using the variable `TORCH_HOME` like so `TORCH_HOME=/mandrillagerecognition/data` from within the singularity container.

You can easily convert a model using the script located in `tools/export_dino.py`.
It has 3 simple parameters:

- `--model_path` which is the path where the checkpoint is
- `--export_path` the path where you want your onnx export
- `--dino_type` this is the dino size (small/medium/large) but usually is it large
