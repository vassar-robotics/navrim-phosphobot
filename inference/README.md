# Inference scripts

These scripts will help you do the inference of a trained model to control your robot.

We assume you have:

- a SO-100 controlled by phosphobot
- a trained model uploaded to Hugging Face

We recommend you clone this repo to access and run the inference scripts.
You will also need to clone the lerobot repo.

```bash
git clone https://github.com/phospho-app/phosphobot.git
git clone https://github.com/huggingface/lerobot.git
```

# Setup an inference server

## Inference on ACT models

If you use pip, just run

```bash
cd phosphobot/inference
pip install .
python ACT/inference.py --model_id=<YOUR_HF_DATASET_NAME>
```

This will load your model and create an /act endpoint that expects the robot's current position and images. If using a local model, you can pass the path of a local model instead of a HF model with the "--model_id=..." flag.

This inference script is intended to run on a machine with a GPU, separately from your phosphobot installation.

You can edit the URL and Port parameters in your phohsphobot Admin panel to connect your robot to the inference server. By default, these should be localhost and 8080.

## Inference on Pi0 models

Follow the instruction in the [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) repo to setup a inference server.

# Call your inference server from a python script

We provide clients for ACT servers and Pi0 servers.
You can implement the `ActionModel` class with you own logic [here](phosphobot/am/models.py).

At this point, go to your phosphobot dashboard > docs and launch the auto/start endpoint that will communicate with the inference server to automatically control the robot.

You can stop at any time by calling the auto/stop endpoint.

## Example script for ACT

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

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json
images = [
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=2, resize=(240, 320)),
]

# Get the robot state
state = client.control.read_joints()

# Instanciate our model
model = ACT()

inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}

# Go through the model
actions = model(inputs)

# Send the new joint postion to the robot
client.control.write_joints(angles=actions[0].tolist())
```

## Example script for Pi0

```python
from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import Pi0

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Need to wait for the cameras to initialize
time.sleep(1)

# Get the frames from the cameras
# We will use this model: PLB/pi0-so100-orangelegobrick-wristcam
# It requires 2 cameras (a context cam and a wrist cam)
images = [
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
]

# Get the robot state
state = client.control.read_joints()

# Instanciate our model
model = Pi0(server_url="YOUR_SERVER_URL")

inputs = {
    "state": np.array(state.angles_rad),
    "images": np.array(images),
    "prompt": "Pick up the orange brick",
}

# Go through the model
actions = model(inputs)

# Send the new joint postion to the robot
client.control.write_joints(angles=actions[0].tolist())
```

### Notes

If running the inference script on a different machine than what you use to control your so-100 robot, you will need to change the URL in the admin panel. Set it to the URL of the machine running the inference script.

You can also change the port of the inference server by passing "--port=..."

If using a local model, you can pass the path of a local model instead of a HF model with the "--model_id=..." flag.
