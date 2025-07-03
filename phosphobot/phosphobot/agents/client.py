#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "phosphobot",
#     "numpy",
#     "httpx",
#     "json_numpy",
# ]
#
# ///

from phosphobot.camera import AllCameras
import httpx
from phosphobot.am import ACT
import time
import numpy as np
from loguru import logger

# Initialize hardware interfaces
PHOSPHOBOT_API_URL = "http://localhost:80"
allcameras = AllCameras()
time.sleep(1)  # Camera warmup

# Connect to ACT server
model = ACT()

while True:
    # Capture multi-camera frames (adjust camera IDs and size as needed)
    images = []
    num_cameras = 2  # Adjust based on your setup

    for i in range(num_cameras):
        frame = allcameras.get_rgb_frame(i, resize=(240, 320))
        if frame is not None:
            images.append(frame)

    # Get current robot state
    state_response = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

    # Prepare inputs in the correct format for ACT server
    inputs = {
        "observation.state": np.array(state_response["angles"]),
    }

    # Add each image with its own key
    for i, image in enumerate(images):
        inputs[f"observation.images.{i}"] = image

    # Generate actions
    actions = model(inputs)

    httpx.post(
        f"{PHOSPHOBOT_API_URL}/joints/write",
        json={
            "angles": actions.tolist(),
            "unit": "rad",  # Specify the unit (required field)
        },
    )
    time.sleep(1 / 30)
    logger.info(f"action: {actions}")
