# phosphobot

A community-driven platform for robotics enthusiasts to share and explore creative projects built with the phospho Junior Dev Kit.

<div align="center">

<a href="https://pypi.org/project/phosphobot/"><img src="https://img.shields.io/pypi/v/phosphobot?style=flat-square&label=pypi+phospho" alt="phosphobot Python package on PyPi"></a>
<a href="https://www.ycombinator.com/companies/phospho"><img src="https://img.shields.io/badge/Y%20Combinator-W24-orange?style=flat-square" alt="Y Combinator W24"></a>
<a href="https://discord.gg/cbkggY6NSK"><img src="https://img.shields.io/discord/1106594252043071509" alt="phospho discord"></a>

</div>

## Overview

This repository contains demo code and community projects developed using the phospho Junior Dev Kit. Whether you're a beginner or an experienced developer, you can explore existing projects or contribute your own creations.

## Getting Started

1. **Get Your Dev Kit**: Purchase your Phospho Junior Dev Kit at [robots.phospho.ai](https://robots.phospho.ai). Unbox it and set it up following the instructions in the box.

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

6. **Use a Model**:
   Install the phosphobot package with the support for action models:

```bash
pip install --upgrade "phosphobot[am]"
```

Start your phosphobot server

```
curl -fsSL https://raw.githubusercontent.com/phospho-app/phosphobot/main/install.sh | bash
phosphobot run
```

Control your robot with the model you trained:

```python
from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ActionModel

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

allcameras = AllCameras()

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json
# Adapt it for your setup
images = [
allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
]

# Get the current robot state
state = client.control.read_joints()

# Replace by your model
model = ActionModel.from_pretrained(
"LegrandFrederic/Orange-brick-in-black-box", device="mps"
)

inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}
action = model(inputs)

# Move the robot
client.control.write_joints(angles=action[0].tolist())
```

For the full detailed instructions, refer to the [guide available here](https://docs.phospho.ai/learn/ai-models).

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
