import torch
from torch import nn
import numpy as np
from torch import Tensor
from lerobot.common.policies.act.modeling_act import ACTPolicy
import logging

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
import json
from pathlib import Path
from typing import List
import cv2


### UTILS ###


def get_safe_torch_device(device_str: str, log: bool = True) -> torch.device:
    """Get a safe torch device, defaulting to CPU if requested device is not available."""
    if device_str == "cuda" and not torch.cuda.is_available():
        if log:
            logging.warning("CUDA requested but not available. Using CPU instead.")
        return torch.device("cpu")
    elif device_str == "mps" and not torch.backends.mps.is_available():
        if log:
            logging.warning("MPS requested but not available. Using CPU instead.")
        return torch.device("cpu")
    return torch.device(device_str)


def parse_input_features(file_path: Path) -> dict:
    """
    Parse JSON data from a file path and return the input_features dictionary.

    Args:
        file_path (str): Path to the JSON file

    Returns:
        dict: The input_features portion of the JSON data

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        # Open and read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Return the input_features dictionary
        return data.get("input_features", {})

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find JSON file at: {file_path}")


def get_pretrained_policy_path(pretrained_policy_name_or_path, revision=None):
    """Get the path to the pretrained policy, either from HF Hub or local."""
    try:
        pretrained_policy_path = Path(
            snapshot_download(pretrained_policy_name_or_path, revision=revision)
        )
    except (HFValidationError, RepositoryNotFoundError) as e:
        if isinstance(e, HFValidationError):
            error_message = "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
        else:
            error_message = "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
        logging.warning(f"{error_message} Treating it as a local directory.")
        pretrained_policy_path = Path(pretrained_policy_name_or_path)

    if not pretrained_policy_path.is_dir() or not pretrained_policy_path.exists():
        raise ValueError(
            "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
            "repo ID, nor is it an existing local directory."
        )
    return pretrained_policy_path


### END UTILS ###


class ActionModel:
    """
    A PyTorch model for generating robot actions from robot state, camera images, and text prompts.
    Inspired by the simplicity and flexibility of pytorch-pretrained-bert.
    """

    def __init__(self, model_id: str, revision: str | None = None, device: str = "cpu"):
        """
        Initialize the ActionModel.

        Args:
            model_name (str): Name of the pre-trained model (e.g., "PLB/pi0-so100-orangelegobrick-wristcam").
            revision: default None which will resolve to main
        """
        policy_path = get_pretrained_policy_path(model_id, revision=revision)
        self.device = get_safe_torch_device(device)

        self.policy = ACTPolicy.from_pretrained(policy_path, revision=revision)
        self.policy.to(self.device)

        self.input_features = parse_input_features(policy_path / "config.json")

    def to(self, device: str) -> None:
        self.device = get_safe_torch_device(device)
        self.policy.to(self.device)

    def select_action(self, inputs: dict) -> Tensor:
        """
        Select a single action.

        Args:
            inputs (dict): Dictionary with keys:
                - "state": Tensor or list of floats representing robot state.
                - "images": List of images (numpy arrays or tensors).
                - "prompt": String text prompt (optional for ACT).

        Returns:
            torch.Tensor: Sequence of actions (shape: [max_seq_length, n_actions]).
        """
        # TODO: Check we have the right inputs
        # input_features = parse_input_features(policy_path / "config.json")

        # Get feature names
        image_names = [
            feature
            for feature in self.input_features.keys()
            if "observation.images" in feature
        ]

        shape = self.input_features[image_names[0]]["shape"]
        target_size = (shape[2], shape[1])

        current_qpos = inputs["state"]
        images = inputs["images"]

        if len(images) == 0:
            raise ValueError("No images provided")

        if len(images[0].shape) == 3 and images[0].shape[2] == 3:
            try:
                if self.device is None:
                    raise ValueError(
                        "Device is not set. Please ensure the policy is loaded correctly."
                    )
                with torch.no_grad(), torch.autocast(device_type=self.device.type):
                    # Prepare state tensor
                    current_qpos = current_qpos.copy()
                    state_tensor = (
                        torch.from_numpy(current_qpos)
                        .view(1, len(current_qpos))
                        .float()
                        .to(self.device)
                    )

                    # Create batch dictionary
                    batch = {
                        "observation.state": state_tensor,
                    }

                    # Convert images to tensors (B, C, H, W), normalize
                    processed_images = []
                    for i, image in enumerate(images):
                        if image.shape[:2] != target_size:
                            image = cv2.resize(image, target_size)
                        tensor_image = (
                            torch.from_numpy(image)
                            .permute(2, 0, 1)
                            .unsqueeze(0)
                            .float()
                            .to(self.device)
                        )
                        tensor_image = tensor_image / 255.0
                        processed_images.append(tensor_image)
                        batch[image_names[i]] = tensor_image

                    # Get action using select_action
                    action = self.policy.select_action(batch)
                    return action.cpu().numpy()

            except Exception as e:
                logging.error(f"Error during inference: {str(e)}")
                raise
        else:
            raise ValueError("Invalid image format. Expected RGB image.")

        return action

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        """
        Load a pre-trained ActionModel.

        Args:
            model_name (str): Name of the pre-trained model.
            **kwargs: Additional arguments for initialization.

        Returns:
            ActionModel: Initialized model with pre-trained weights.
        """
        return cls(model_name, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Makes the model instance callable, delegating to the forward method.

        Args:
            *args: Variable positional arguments passed to forward.
            **kwargs: Variable keyword arguments passed to forward.

        Returns:
            The output of the forward method.
        """
        return self.select_action(*args, **kwargs)
