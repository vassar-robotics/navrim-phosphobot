import requests
import numpy as np
from typing import List
import json_numpy  # type: ignore


"""
SERVER URL or IP
SERVER port
TODO: api_key

Images
State
Instruction (optional)
"""


class ActionModel:
    """
    A PyTorch model for generating robot actions from robot state, camera images, and text prompts.
    Inspired by the simplicity and flexibility of pytorch-pretrained-bert.
    """

    def __init__(self, server_url: str = "http://localhost", server_port: int = 8080):
        """
        Initialize the ActionModel.

        Args:
            model_name (str): Name of the pre-trained model (e.g., "PLB/pi0-so100-orangelegobrick-wristcam").
            revision: default None which will resolve to main
        """
        self.server_url = server_url
        self.server_port = server_port

    def select_action(self, inputs: dict) -> np.ndarray:
        """
        Select a single action.

        Args:
            inputs (dict): Dictionary with keys:
                - "state": Tensor or list of floats representing robot state.
                - "images": List of images (numpy arrays or tensors).
                - "prompt": String text prompt (optional for ACT).

        Returns:
            np.ndarray: Sequence of actions (shape: [max_seq_length, n_actions]).
        """
        raise NotImplementedError("""You cannot directly call the ActionModel class. 
                                  You need to use an implementation ( ACT, PI0,...) or implement you own class.""")

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


class ACT(ActionModel):
    def __init__(self, server_url: str = "http://localhost", server_port: int = 8080):
        super().__init__(server_url, server_port)
        self.required_input_keys: List[str] = ["images", "state"]

    def select_action(self, inputs: dict) -> np.ndarray:
        # Buuild the payload
        payload = {
            "observation.state": inputs["state"],
        }
        for i in range(0, len(inputs["images"])):
            payload[f"observation.images.{i}"] = inputs["images"][i]

        # Double-encoded version (to send numpy arrays as JSON)
        encoded_payload = {"encoded": json_numpy.dumps(payload)}

        response = requests.post(
            f"{self.server_url}:{self.server_port}/act",
            json=encoded_payload,
            timeout=5.0,  # Add timeout to prevent hanging
        ).json()

        actions = json_numpy.loads(response)
        return actions
