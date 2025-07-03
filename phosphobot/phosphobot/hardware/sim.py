"""
Setup the simulation environment for the robot
"""

import os
import subprocess
import time

from loguru import logger


def simulation_init():
    """
    Initialize the simulation environment based on the configuration.
    Note: PyBullet simulation has been removed from this project.
    """
    logger.debug("Simulation init called (no-op - pybullet removed)")
    pass


def simulation_stop():
    """
    Cleanup the simulation environment.
    Note: PyBullet simulation has been removed from this project.
    """
    logger.debug("Simulation stop called (no-op - pybullet removed)")
    pass
