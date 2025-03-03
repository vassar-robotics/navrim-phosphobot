"""
Store constants and configurations for the app in this file.
"""

from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel

from phosphobot.types import SimulationMode, VideoCodecs
from phosphobot.utils import get_home_app_path

YAML_CONFIG_PATH = str(get_home_app_path() / "config.yaml")


def rename_keys_for_config(config_dict: dict) -> dict:
    """
    Rename the keys to correspond to the fields in the Configuration class.
    We write the keys in uppercase and add DEFAULT_ in front of them.
    """
    return {
        f"DEFAULT_{key.upper()}" if not key.startswith("DEFAULT_") else key: value
        for key, value in config_dict.items()
    }


def remove_default_prefix(config_dict: dict) -> dict:
    """
    Remove the DEFAULT_ prefix from the keys.
    """
    return {
        key.replace("DEFAULT_", "").lower(): value for key, value in config_dict.items()
    }


class Configuration(BaseModel):
    # Server
    PORT: int = 80

    # Profiling (creates a profile.html file in the root directory)
    PROFILE: bool = False

    # Camera
    CAMERA_DISABLED: bool = False

    # Recording
    MAIN_CAMERA_ID: int | None = None  # defaults to min(detected cameras)

    # If Set to True, the main camera will be the realsense camera and we ignore MAIN_CAMERA_ID
    USE_REALSENSE_MAIN: bool = True

    # How simulation should be run
    SIM_MODE: SimulationMode = SimulationMode.headless
    # Only simulation: Only use the simulation
    ONLY_SIMULATION: bool = False

    SIMULATE_CAMERAS: bool = False

    # Serial ports of leader robots
    LEADER_ARM_DEVICE_NAMES: list[str] | None = None

    # These fields will be set after loading the user config
    DEFAULT_DATASET_NAME: str = "example_dataset"
    DEFAULT_FREQ: int = 10
    DEFAULT_EPISODE_FORMAT: Literal["lerobot_v2", "json"] = "lerobot_v2"
    DEFAULT_VIDEO_CODEC: VideoCodecs = "mp4v"
    DEFAULT_VIDEO_SIZE: tuple[int, int] = (320, 240)
    DEFAULT_REMOTE_SERVER_INFERENCE_URL: str = "http://localhost"
    DEFAULT_REMOTE_SERVER_INFERENCE_PORT: int = 8080
    DEFAULT_TASK_INSTRUCTION: str = "None"

    @classmethod
    def from_yaml(cls, config_path: str | Path | None = None) -> "Configuration":
        """
        Load configuration from a YAML file.
        Args:
            config_path (str): Path to the YAML configuration file.
        Returns:
            dict: Dictionary containing configuration values.
        """
        if config_path is None:
            config_path = YAML_CONFIG_PATH

        logger.info(f"Loading configuration from {config_path}")

        # Ensure the file exists. If not, create it.
        if not Path(config_path).exists():
            logger.info(
                f"Configuration file not found. Creating a new one at {config_path}"
            )

            with open(config_path, "w") as file:
                file.write("")
                return cls()

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if config is None or not isinstance(config, dict):
                config = {}
            config = rename_keys_for_config(config)

        logger.info(f"Loaded configuration: {config}")

        return cls(**config)

    def save_user_settings(self, user_settings: dict) -> None:
        """
        Save user settings to a YAML file.
        Args:
            user_settings (dict): Dictionary containing user settings.
        """

        try:
            # Read the user_settings as a config
            user_settings = rename_keys_for_config(user_settings)
            new_config = Configuration(**user_settings)
        except Exception as e:
            logger.error(f"Error saving user settings: {e}")
            return

        # If the user_settings are valid, save them to the YAML file
        with open(YAML_CONFIG_PATH, "w") as file:
            yaml.dump(user_settings, file)

        # Then, replace inplace the fields of the current instance with the new instance
        for field in new_config.model_fields.keys():
            setattr(self, field, getattr(new_config, field))


config = Configuration.from_yaml()
