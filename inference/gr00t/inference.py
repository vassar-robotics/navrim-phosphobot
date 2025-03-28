# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cv2",
#     "phosphobot",
#     "torch",
#     "zmq",
# ]
# ///
import time

import cv2
import numpy as np

from phosphobot.am import Gr00tN1
from phosphobot.api.client import PhosphoApi
from phosphobot.camera import AllCameras

host = "20.199.85.87"
port = 5555

# Change this by your task description
TASK_DESCRIPTION = (
    "Pick up the green lego brick from the table and put it in the black container."
)

# Connect to the phosphobot server
client = PhosphoApi(base_url="http://localhost:80")

allcameras = AllCameras()
time.sleep(1)  # Wait for the cameras to initialize

while True:
    images = [
        allcameras.get_rgb_frame(camera_id=0, resize=(320, 240)),
        allcameras.get_rgb_frame(camera_id=1, resize=(320, 240)),
    ]

    for i in range(0, len(images)):
        image = images[i]
        if image is None:
            print(f"Camera {i} is not available.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Add a batch dimension (from (240, 320, 3) to (1, 240, 320, 3))
        converted_array = np.expand_dims(image, axis=0)
        converted_array = converted_array.astype(np.uint8)
        images[i] = converted_array

    # Create the model
    model = Gr00tN1(server_url=host, server_port=port)

    state = np.array(client.control.read_joints().angles_rad)
    obs = {
        "video.cam_context": images[0],
        "video.cam_wrist": images[1],
        "state.single_arm": state[0:5].reshape(1, 5),
        "state.gripper": np.array([state[5]]).reshape(1, 1),
        "annotation.human.action.task_description": [TASK_DESCRIPTION],
    }

    action = model.sample_actions(obs)

    for i in range(0, action.shape[0]):
        client.control.write_joints(angles=action[i])
        # Wait to respect frequency control (30 Hz)
        time.sleep(1 / 30)
