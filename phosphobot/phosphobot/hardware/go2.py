from typing import Optional

import numpy as np

from phosphobot.hardware.base import BaseMobileRobot
from phosphobot.models import RobotConfigStatus
from loguru import logger


class UnitreeGo2(BaseMobileRobot):
    name = "unitree-go2"

    def set_motors_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot
        """
        raise NotImplementedError

    def get_info_for_dataset(self):
        """
        Generate information about the robot useful for the dataset.
        Return a BaseRobotInfo object. (see models.dataset.BaseRobotInfo)
        Dict returned is info.json file at initialization
        """
        raise NotImplementedError

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.
        This method should return the observation of the robot.
        Will be used to build an observation in a Step of an episode.
        Returns:
            - state: np.array state of the robot (7D)
            - joints_position: np.array joints position of the robot
        """
        raise NotImplementedError

    def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method is called after the __init__ method.

        raise: Exception if the setup fails. For example, if the robot is not plugged in.
            This Exception will be caught by the __init__ method.
        """
        raise NotImplementedError

    def disconnect(self) -> None:
        """
        Close the connection to the robot.

        This method is called on __del__ to disconnect the robot.
        """
        raise NotImplementedError

    def move_robot_absolute(
        self,
        target_position: np.ndarray,  # cartesian np.array
        target_orientation_rad: np.ndarray | None,  # rad np.array
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation.
        This method should be implemented by the robot class.
        """
        raise NotImplementedError

    @classmethod
    def from_ip(cls, ip: str, **kwargs) -> Optional["UnitreeGo2"]:
        """
        Return the robot class from the port information.
        """
        raise NotImplementedError

    def status(self) -> RobotConfigStatus:
        return RobotConfigStatus(
            name=self.name,
            usb_port=getattr(self, "SERIAL_ID", None),
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.
        The initial position is a safe position for the robot, where it is moved before starting the calibration.
        This method should be implemented by the robot class.
        """
        raise NotImplementedError

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.
        The sleep position is a safe position for the robot, where it is moved before disabling the motors.
        This method should be implemented by the robot class.
        """
        raise NotImplementedError
