from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.ai import ActionModel

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json
images = [
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=1, resize=(240, 320)),
]

# Get the robot state
state = client.control.read_joints()

model = ActionModel.from_pretrained(
    "LegrandFrederic/Orange-brick-in-black-box", device="mps"
)

inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}
# Go trhough the model
action = model(inputs)

# Send the new joint postion to the robot
client.control.write_joints(angles=action[0].tolist())
