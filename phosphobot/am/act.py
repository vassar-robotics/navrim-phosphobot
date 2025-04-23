from typing import List

import json_numpy  # type: ignore
import numpy as np
import requests

from phosphobot.am.base import ActionModel


class ACT(ActionModel):
    def __init__(
        self,
        server_url: str = "http://localhost",
        server_port: int = 8080,
        timeout: int = 10,
    ):
        super().__init__(server_url, server_port)
        self.required_input_keys: List[str] = ["images", "state"]
        self.timeout = timeout

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Build the payload
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
            timeout=self.timeout,
        ).json()

        action = json_numpy.loads(response)

        return np.array([action])
