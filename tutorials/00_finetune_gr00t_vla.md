# Finetune gr00t to control a SO-100 robot arm

In this tutorial, we are going to train a Vision-Language Model (VLA) to control a SO-100 robot arm.

## Prerequisites

- You have recorded a dataset of the SO-100 robot arm in LeRobot dataset format (using phospho or LeRobot).
- You have a NVIDIA GPU with at least XXX GB of memory.

If you don't have access to such a machine, you can rent a cloud instance with a GPU from any provider. We recommend getting an A-100 80GB.

## Installation and setup

First, if you haven't already, clone this repository:

```bash
git clone https://github.com/phospho-app/phosphobot.git
cd phosphobot
```

We are going to use the phospho fork of Isaac-GR00T repo.

```bash
git clone https://github.com/phospho-app/Isaac-GR00T.git
cd Isaac-GR00T
```

Then install the dependencies using conda:

```bash
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
cd ..
```

You should now have a working environment to finetune the model in gr00t.

## Finetune the model

In the phosphobot folder, still in the `gr00t` conda environment, run:

```bash
python scripts/gr00t/train.py --dataset-id YOUR-DATASET-ID
```

Adjust the batch size to use as much as possible of the GPU memory.
If you encounter an Out of Memory error, try to reduce the batch size.

Use tools like `nvidia-smi` or `nvitop` to monitor the GPU memory usage.

## Evaluate the finetuned model

TODO

## Test the finetuned model on your robot

Make sure the setup of your robot matches the dataset setup (number of cameras, order of cameras, etc...).

TODO

## Troubleshooting common issues

### Git LFS not installed

On your linux machine, run:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
