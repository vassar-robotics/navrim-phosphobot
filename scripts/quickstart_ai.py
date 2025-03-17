from phosphobot.camera import AllCameras
from phosphobot.api.client import PhosphoApi
from phosphobot.am import ACT

import time
import numpy as np

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

# Get a camera frame
allcameras = AllCameras()

print(allcameras.get_rgb_frames_for_all_cameras(resize=(240, 320)))

# Get the frames from the cameras
# We will use this model: LegrandFrederic/Orange-brick-in-black-box
# It requires 3 cameras as you can see in the config.json
# https://huggingface.co/LegrandFrederic/Orange-brick-in-black-box/blob/main/config.json
images = [
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
    allcameras.get_rgb_frame(camera_id=0, resize=(240, 320)),
]

print(images)

image = np.zeros((240, 320, 3))
images = [image] * 3

# Get the robot state
state = client.control.read_joints()

model = ACT()

# print(images)
print(state)

inputs = {"state": np.array(state.angles_rad), "images": np.array(images)}
# Go trhough the model
actions = model(inputs)

print(f"actions {actions}")

# Send the new joint postion to the robot
client.control.write_joints(angles=actions[0].tolist())
