"""
Setup the simulation environment for the robot
"""

import os
import subprocess
import time

import pybullet as p
from loguru import logger


def simulation_init():
    """
    Initialize the pybullet simulation environment based on the configuration.
    """
    from phosphobot.configs import config

    if config.SIM_MODE == "headless":
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        logger.debug("Headless mode enabled")

    elif config.SIM_MODE == "gui":
        # Spin up a new process for the simulation

        # Run a new python process
        # cd ./simulation/pybullet && uv run --python 3.8 main.py
        absolute_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "simulation", "pybullet"
            )
        )
        subprocess.Popen(["uv", "run", "--python", "3.8", "main.py"], cwd=absolute_path)
        # Wait for 1 second to allow the simulation to start
        time.sleep(1)
        p.connect(p.SHARED_MEMORY)
        logger.debug("GUI mode enabled")

    else:
        raise ValueError("Invalid simulation mode")


def simulation_stop():
    """
    Cleanup the simulation environment.
    """
    from phosphobot.configs import config

    if p.isConnected():
        p.disconnect()
        logger.info("Simulation disconnected")

    if config.SIM_MODE == "gui":
        # Kill the simulation process: any instance of python 3.8
        subprocess.run(["pkill", "-f", "python3.8"])
