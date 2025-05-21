import json
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.utils import get_home_app_path

DEFAULT_FILE_ENCODING = "utf-8"


class BaseRobot(ABC):
    name: str

    @abstractmethod
    def set_motors_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> "BaseRobotInfo":  # type: ignore
        """
        Get information about the robot
        Dict returned is info.json file at initialization
        """
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method is called after the __init__ method.

        raise: Exception if the setup fails. For example, if the robot is not plugged in.
            This Exception will be caught by the __init__ method.
        """
        raise NotImplementedError("The robot setup method must be implemented.")

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the robot.

        This method is called on __del__ to disconnect the robot.
        """
        raise NotImplementedError("The robot setup method must be implemented.")

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["BaseRobot"]:
        """
        Return the robot class from the port information.
        """
        logger.error(
            f"For automatic detection of {cls.__name__}, the method from_port must be implemented. Skipping autodetection."
        )
        return None


class BaseRobotPIDGains(BaseModel):
    """
    PID gains for servo motors
    """

    p_gain: float
    i_gain: float
    d_gain: float


class BaseRobotConfig(BaseModel):
    """
    Calibration configuration for a robot
    """

    name: str
    servos_voltage: float
    servos_offsets: List[float] = Field(
        default_factory=lambda: [
            2048.0,
            2048.0,
            2048.0,
            2048.0,
            2048.0,
            2048.0,
        ]
    )
    # Default factory: default offsets for SO-100
    servos_calibration_position: List[float]
    servos_offsets_signs: List[float] = Field(
        default_factory=lambda: [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    pid_gains: List[BaseRobotPIDGains] = Field(default_factory=list)

    # Torque value to consider that an object is gripped
    gripping_threshold: int = 0
    non_gripping_threshold: int = 0  # noise

    @classmethod
    def from_json(cls, filepath: str) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a JSON file
        """
        try:
            with open(filepath, "r", encoding=DEFAULT_FILE_ENCODING) as f:
                data = json.load(f)

        except FileNotFoundError:
            return None

        # Fix issues with the JSON file
        servos_offsets = data.get("servos_offsets", [])
        if len(servos_offsets) == 0:
            data["servos_offsets"] = [2048.0] * 6

        servos_offsets_signs = data.get("servos_offsets_signs", [])
        if len(servos_offsets_signs) == 0:
            data["servos_offsets_signs"] = [-1.0] + [1.0] * 5

        try:
            return cls(**data)
        except Exception as e:
            logger.error(f"Error loading configuration from {filepath}: {e}")
            return None

    @classmethod
    def from_serial_id(
        cls, serial_id: str, name: str
    ) -> Union["BaseRobotConfig", None]:
        """
        Load a configuration from a serial ID and a name.
        """
        filename = f"{name}_{serial_id}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        return cls.from_json(filepath)

    def to_json(self, filename: str) -> None:
        """
        Save the configuration to a JSON file
        """
        with open(filename, "w", encoding=DEFAULT_FILE_ENCODING) as f:
            f.write(self.model_dump_json(indent=4))

    def save_local(self, serial_id: str) -> str:
        """
        Save the configuration to the local calibration folder

        Returns:
            The path to the saved file
        """
        filename = f"{self.name}_{serial_id}_config.json"
        filepath = str(get_home_app_path() / "calibration" / filename)
        logger.info(f"Saving configuration to {filepath}")
        self.to_json(filepath)
        return filepath
