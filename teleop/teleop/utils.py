import os
import re
import subprocess
import sys
import zipfile
import toml  # type: ignore
from pathlib import Path
from typing import List, Literal

import numpy as np
from phosphobot.models.dataset import Dataset
import requests
from fastapi import HTTPException
from huggingface_hub import HfApi, login
from loguru import logger
from phosphobot.configs import config
from phosphobot.utils import get_home_app_path, parse_hf_username_or_orgid

from teleop.models import ItemInfo
from dataclasses import dataclass


def is_running_on_pi() -> bool:
    """Validate that we're running on a Raspberry Pi"""
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()
        return "Raspberry Pi" in cpuinfo or "BCM" in cpuinfo
    except:  # noqa: E722
        return False


def is_running_on_linux() -> bool:
    """Validate that we're running on a Linux machine"""
    return os.name == "posix"


def step_simulation(steps=960):
    """
    When running the simulation in headless mode,
    Pybullet forces us to step the simulation manually.

    The sim runs at 240 Hz, so we need to step it 240 times per second.
    """
    import pybullet as p  # type: ignore

    # When running the simulation in headless mode,
    # Pybullet forces us to step the simulation manually
    for _ in range(steps):
        p.stepSimulation()


def euler_from_quaternion(quaternion: np.ndarray, degrees: bool) -> np.ndarray:
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    """
    # Skipping analyzing "scipy.spatial.transform": module is installed, but missing library stubs or py.typed marker
    from scipy.spatial.transform import Rotation as R  # type: ignore

    try:
        return R.from_quat(quaternion).as_euler("xyz", degrees=degrees)
    except ValueError as e:
        logger.error(
            f"Error converting quaternion to Euler angles. Returning zeros. {e}"
        )
        return np.zeros(3)


def get_quaternion_from_euler(euler_angles: np.ndarray, degrees: bool) -> np.ndarray:
    """
    Convert an Euler angle to a quaternion.
    """
    from scipy.spatial.transform import Rotation as R  # type: ignore

    return R.from_euler("xyz", angles=euler_angles, degrees=degrees).as_quat()


def print_numpy_array(
    arr,
    precision=2,
    max_line_width=100,
    suppress_small=True,
    show_shape=True,
    show_dtype=True,
):
    """
    Customizable pretty-printer for NumPy arrays.

    Parameters:
    -----------
    arr : numpy.ndarray
        The array to be printed
    precision : int, optional
        Number of decimal places to display (default: 2)
    max_line_width : int, optional
        Maximum width of each line (default: 100)
    suppress_small : bool, optional
        Suppress small floating point values (default: True)
    show_shape : bool, optional
        Display array shape information (default: True)
    show_dtype : bool, optional
        Display array data type (default: True)

    Examples:
    ---------
    >>> arr = np.random.rand(3, 4)
    >>> print_array(arr)
    """
    # Set print options
    np.set_printoptions(
        precision=precision, linewidth=max_line_width, suppress=suppress_small
    )

    # Print shape and dtype if requested
    if show_shape:
        print(f"Shape: {arr.shape}")

    if show_dtype:
        print(f"Dtype: {arr.dtype}")

    # Print the array
    print(arr)

    # Reset print options to default
    np.set_printoptions()


def cartesian_to_polar(x, y, z):
    """
    Convert cartesian coordinates to polar coordinates
    """

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    return r, theta, z


def polar_to_cartesian(r, theta, z):
    """
    Convert polar coordinates to cartesian coordinates
    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y, z


def get_base_path() -> Path:
    """
    Return the base path of the app.
    This is used to load bundled resources.

    teleop/    <-- base path
        teleop/

            main.py
            ...
        resources/
        ...

    """
    return Path(__file__).parent.parent


def get_resources_path() -> Path:
    """
    Return the path of the resources directory.
    This is used to load bundled resources.

    teleop/
        teleop/

            main.py
            ...
        resources/    <-- resources path
        ...

    """
    return get_base_path() / "resources"


def get_dashboard_path() -> Path:
    """
    Return the path of the dashboard directory.
    It's used to serve the dashboard's index.html file.
    """
    return get_base_path() / "dashboard"


def login_to_hf(revalidate: bool = True) -> bool:
    """
    Return True if we successfully logged in to Hugging Face.
    """

    if not revalidate and config.HF_TOKEN_VALID:
        logger.debug("Skipping revalidation of Hugging Face token.")
        return config.HF_TOKEN_VALID

    token_file = get_home_app_path() / "huggingface.token"

    # Chec the file exists
    if not token_file.exists():
        logger.debug("Hugging Face token file not found.")
        config.HF_TOKEN_VALID = False
        return False

    with open(token_file, "r") as file:
        hf_token = file.read().strip()

    if hf_token:
        try:
            login(hf_token)
            logger.debug("Successfully logged in to Hugging Face.")
            api = HfApi()
            user_info = api.whoami(token=hf_token)
            # Get the username or org where we have write access
            username_or_orgid = parse_hf_username_or_orgid(user_info)
            logger.debug(f"HF username or org ID: {username_or_orgid}")
            config.HF_TOKEN_VALID = True
            return True
        except Exception as e:
            logger.warning(f"Error logging in to Hugging Face: {str(e)}")
            config.HF_TOKEN_VALID = False
            return False
    else:
        logger.warning(f"Hugging Face Token file is empty: {token_file}. Won't login.")
        config.HF_TOKEN_VALID = False
        return False


def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)


def is_can_active(interface: str = "can0") -> bool:
    """
    Checks if a specified CAN interface exists and is in UP state.

    Args:
        interface: CAN interface name (e.g., 'can0', 'can1')

    Returns:
        bool: True if interface exists and is up, False otherwise

    Raises:
        OSError: If the platform is not supported
        subprocess.SubprocessError: If the system command fails unexpectedly
    """
    try:
        if sys.platform == "linux":
            # Linux implementation using ip command
            result = subprocess.run(
                ["ip", "link", "show", interface],
                capture_output=True,
                text=True,
                check=True,
            )
            return "state UP" in result.stdout

        elif sys.platform == "darwin":  # macOS
            # macOS implementation using ifconfig
            result = subprocess.run(
                ["ifconfig", interface],
                capture_output=True,
                text=True,
                check=True,
            )
            return "status: active" in result.stdout.lower()

        elif sys.platform == "win32":  # Windows
            # Windows implementation using netsh
            result = subprocess.run(
                ["netsh", "interface", "show", "interface", interface],
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.stdout is None:
                return False
            return "connected" in result.stdout.lower()

        else:
            raise OSError(f"Unsupported platform: {sys.platform}")

    except subprocess.CalledProcessError as e:
        # Interface doesn't exist or command failed
        if e.returncode == 1:
            return False
        raise subprocess.SubprocessError(
            f"Failed to check CAN interface status: {str(e)}"
        )
    except FileNotFoundError as e:
        raise OSError(f"Required system command not found: {str(e)}")


def is_can_plugged(interface: str = "can0") -> bool:
    """
    Checks if a specified CAN interface exists.
    """

    try:
        if sys.platform == "linux":
            # Linux implementation using ip command
            result = subprocess.run(
                ["ip", "link", "show", interface],
                capture_output=True,
                text=True,
                check=True,
            )
            return "does not exist" not in result.stdout.lower()
        elif sys.platform == "darwin":
            # macOS implementation using ifconfig
            result = subprocess.run(
                ["ifconfig", interface],
                capture_output=True,
                text=True,
                check=True,
            )
            return "does not exist" not in result.stdout.lower()
        # Adds windows support
        elif sys.platform == "win32":
            # Windows implementation using ipconfig
            result = subprocess.run(
                ["ipconfig", "/all"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout is None:
                return False
            return interface in result.stdout.lower()
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")
    except subprocess.CalledProcessError as e:
        # Interface doesn't exist or command failed
        if e.returncode == 1:
            return False
        logger.error(f"Failed to check CAN interface status: {str(e)}")
        return False
    except FileNotFoundError as e:
        logger.error(f"OSError: Required system command not found: {str(e)}")
        return False


def sanitize_path(path: str) -> str:
    # Normalize the path to avoid directory traversal
    safe_path = os.path.normpath(path)
    if safe_path.startswith(".."):
        raise HTTPException(
            status_code=400, detail=f"Invalid path: {path} (starts with ..)"
        )
    return safe_path


def list_directory_items(path: str, root_dir: str = "") -> List[ItemInfo]:
    full_path = os.path.join(root_dir, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {full_path}")

    # Remove DS_Store files if they exist
    Dataset.remove_ds_store_files(full_path)

    items = os.listdir(full_path)
    items_info = []
    username_or_org_id = None
    api = None
    if path.endswith("lerobot_v2") or path.endswith("lerobot_v2.1"):
        try:
            api = HfApi()
            user_info = api.whoami()
            username_or_org_id = parse_hf_username_or_orgid(user_info)
        # If we can't get the username or org ID, we can't delete the dataset
        except Exception as e:
            logger.debug(f"Error getting Hugging Face username or org ID: {str(e)}")
            pass

    for item in items:
        item_path = os.path.join(path, item)
        absolute_item_path = os.path.join(full_path, item_path)
        is_dir = os.path.isdir(os.path.join(root_dir, item_path))
        info = ItemInfo(
            name=item,
            path=item_path,
            absolute_path=absolute_item_path,
            is_dir=is_dir,
            browseUrl=f"/browse?path={item_path}",
            downloadUrl=f"/dataset/download?folder_path={item_path}"
            if is_dir
            else None,
        )
        if is_dir:
            if api is not None and username_or_org_id is not None:
                info.previewUrl = f"https://lerobot-visualize-dataset.hf.space/{username_or_org_id}/{info.name}"
                info.huggingfaceUrl = (
                    f"https://huggingface.co/datasets/{username_or_org_id}/{info.name}"
                )

            # Only add the delete button if the dataset's path ends with "json" or "lerobot_v2"
            if (
                path.endswith("json")
                or path.endswith("lerobot_v2")
                or path.endswith("lerobot_v2.1")
            ):
                info.canDeleteDataset = True
                info.deleteDatasetAction = "/dataset/delete"

        # Check if this is a dataset: it's a directory and the parent is lerobot_v2
        if is_dir and (path.endswith("lerobot_v2") or path.endswith("lerobot_v2.1")):
            info.is_dataset_dir = True

        items_info.append(info)
    return items_info


def fetch_latest_brew_version(fail_silently: bool = False) -> str:
    """
    Fetch the latest version of the brew package from the tap.
    """
    url = "https://raw.githubusercontent.com/phospho-app/homebrew-phosphobot/refs/heads/main/Formula/phosphobot.rb"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise HTTP errors

        content = response.text
        # Match 'version "x.y.z"' or similar patterns
        version_match = re.search(
            r'url\s+"https://[^/]+/[^/]+/[^/]+/releases/download/([^/]+)/', content
        )
        if version_match:
            return version_match.group(1)
        else:
            if fail_silently:
                return "unknown"
            else:
                raise ValueError("No version found in the tap.")
    except requests.exceptions.RequestException as e:
        if fail_silently:
            return "unknown"
        raise HTTPException(status_code=500, detail=f"Error fetching package info: {e}")


@dataclass
class Tokens:
    ENV: Literal["dev", "prod"] = "dev"
    SENTRY_DSN: str | None = None
    # This is used to track the app usage
    POSTHOG_API_KEY: str | None = "phc_EesFKS4CVoyc0URzJN0FOETpg7KipCBEpRvHEvv5mDF"
    POSTHOG_HOST: str | None = "https://us.i.posthog.com"
    SUPABASE_URL: str | None = None
    SUPABASE_KEY: str | None = None
    MODAL_API_URL: str | None = None


def get_tokens() -> Tokens:
    """
    Load the tokens.toml file
    """
    tokens_toml_path = get_resources_path() / "tokens.toml"
    if not tokens_toml_path.exists():
        return Tokens()
    # Load with toml
    tokens = toml.load(tokens_toml_path)
    return Tokens(**tokens)
