# phosphobot

A community-driven platform for robotics enthusiasts to share and explore creative projects built with the phospho starter pack.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the phospho starter pack. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting Started

1. **Get Your Dev Kit**: Purchase your Phospho starter pack at [robots.phospho.ai](https://robots.phospho.ai). Unbox it and set it up following the instructions in the box.

2. **Control your Robot**: Donwload the Meta Quest app, connect it to your robot, start teleoperating it.

3. **Record a Dataset**: Record a dataset using the app. Do the same gesture 30-50 times (depending on the task complexity) to create a dataset.

4. **Install the Package**:

```bash
pip install --upgrade phosphobot
```

5. **Train a Model**: Use [Le Robot](https://github.com/huggingface/lerobot) to train a policy on the dataset you just recorded.

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

Add the `configs/policy/act_so100_phosphobot.yaml`file from this repository to the `lerobot/configs/policy` directory in the `lerobot` repository.

Launch the training script with the following command from the `lerobot` repository (change the device to `cuda` if you have an NVIDIA GPU, `mps` if you use a MacBook Pro Sillicon, and `cpu` otherwise):

```bash
sudo python lerobot/scripts/train.py \
  --dataset.repo_id=<HF_USERNAME>/<DATASET_NAME> \
  --policy.type=<act or diffusion or tdmpc or vqbet> \
  --output_dir=outputs/train/phoshobot_test \
  --job_name=phosphobot_test \
  --device=cpu \
  --wandb.enable=true
```

6. **Use a model to control your robot**: Launch a server (see [here](inference/) how to serve a policy).

Launch the phosphobot server:

```bash
phosphobot run
```

Run this script to control your robot using the model:

```python
from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ACT

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Instantiate the model
model = ACT()

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
        allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
    ]

    # Get the robot state
    state = client.control.read_joints()

    inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

    # Go through the model
    actions = model(inputs)

    for action in actions:
        # Send the new joint postion to the robot
        client.control.write_joints(angles=action.tolist())
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
```

For the full detailed instructions and other model (Pi0, OpenVLA,...), refer to the [docs](https://docs.phospho.ai/basic-usage/inference).

## Join the Community

Connect with other developers and share your experience in our [Discord community](https://discord.gg/cbkggY6NSK)

## Community Projects

Explore projects created by our community members in the [code_examples](./code_examples) directory. Each project includes its own documentation and setup instructions.

## Support

- **Documentation**: Read the [documentation](https://docs.phospho.ai)
- **Community Support**: Join our [Discord server](https://discord.gg/cbkggY6NSK)
- **Issues**: Submit problems or suggestions through [GitHub Issues](https://github.com/phospho-app/phosphobot/issues)

## License

MIT License

---

Made with 💚 by the Phospho community
