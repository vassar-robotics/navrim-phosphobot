from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict, List, Literal, Tuple

from fastapi import HTTPException
from huggingface_hub import HfApi
from loguru import logger
import numpy as np
from phosphobot.camera import AllCameras
from phosphobot.models.dataset import BaseRobot
from pydantic import BaseModel, Field, model_validator
import zmq
import pickle

from phosphobot.am.base import ActionModel
from phosphobot.utils import get_hf_token


# Code from: https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/eval/service.py#L111


class TorchSerializer:
    # TODO: Rename as PickleSerializer

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        # torch.save(data, buffer)
        # use pickle instead of torch
        pickle.dump(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        # obj = torch.load(buffer, weights_only=False)
        # use pickle instead of torch
        obj = pickle.load(buffer)
        return obj


@dataclass
class EndpointHandler:
    handler: Callable
    requires_input: bool = True


class BaseInferenceServer:
    """
    An inference server that spin up a ZeroMQ socket and listen for incoming requests.
    Can add custom endpoints by calling `register_endpoint`.
    """

    def __init__(self, host: str = "*", port: int = 5555):
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{host}:{port}")
        self._endpoints: dict[str, EndpointHandler] = {}

        # Register the ping endpoint by default
        self.register_endpoint("ping", self._handle_ping, requires_input=False)
        self.register_endpoint("kill", self._kill_server, requires_input=False)

    def _kill_server(self):
        """
        Kill the server.
        """
        self.running = False

    def _handle_ping(self) -> dict:
        """
        Simple ping handler that returns a success message.
        """
        return {"status": "ok", "message": "Server is running"}

    def register_endpoint(
        self, name: str, handler: Callable, requires_input: bool = True
    ):
        """
        Register a new endpoint to the server.

        Args:
            name: The name of the endpoint.
            handler: The handler function that will be called when the endpoint is hit.
            requires_input: Whether the handler requires input data.
        """
        self._endpoints[name] = EndpointHandler(handler, requires_input)

    def run(self):
        addr = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        print(f"Server is ready and listening on {addr}")
        while self.running:
            try:
                message = self.socket.recv()
                request = TorchSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "get_action")

                if endpoint not in self._endpoints:
                    raise ValueError(f"Unknown endpoint: {endpoint}")

                handler = self._endpoints[endpoint]
                result = (
                    handler.handler(request.get("data", {}))
                    if handler.requires_input
                    else handler.handler()
                )
                self.socket.send(TorchSerializer.to_bytes(result))
            except Exception as e:
                print(f"Error in server: {e}")
                import traceback

                print(traceback.format_exc())
                self.socket.send(b"ERROR")


class BaseInferenceClient:
    def __init__(
        self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """
    Client for communicating with the RealRobotServer
    """

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the action from the server.
        The exact definition of the observations is defined
        by the policy, which contains the modalities configuration.
        """
        return self.call_endpoint("get_action", observations)


class Stats(BaseModel):
    max: list[float]
    min: list[float]
    mean: list[float]
    std: list[float]
    q01: list[float]
    q99: list[float]


class ComponentStatistics(BaseModel):
    active_components: Dict[str, Stats] = Field(
        default_factory=dict,
        description="Dictionary mapping component names to their valid Stats objects",
    )

    class Config:
        extra = "allow"

    @model_validator(mode="before")
    def collect_active_components(self):
        """
        Collect names and values of all fields containing valid Stats before validation.
        Ensures extra fields are included in active_components.
        """
        if not isinstance(self, dict):
            return self

        active = {}
        # Process defined fields and extra fields
        for field_name, value in self.items():
            if field_name == "active_components":
                continue
            # Check if value is a dict that matches Stats structure
            if isinstance(value, dict) and all(
                key in value for key in ["max", "min", "mean", "std", "q01", "q99"]
            ):
                try:
                    # Attempt to parse as Stats
                    stats = Stats(**value)
                    active[field_name] = stats
                except ValueError:
                    pass  # Skip invalid Stats structures
            elif isinstance(value, Stats):
                active[field_name] = value

        # Update active_components in the data
        self["active_components"] = active
        return self

    @property
    def component_names(self) -> list[str]:
        """
        Return a list of active component names for convenience.
        """
        return list(self.active_components.keys())

    @property
    def number_of_arms(self) -> int:
        """
        We assume each arm has 6 joints.
        """
        total_joints = sum(len(stats.max) for stats in self.active_components.values())
        return total_joints // 6

    def get_max_value(self) -> float:
        """
        Return the maximum value across all 'max' fields of Stats instances.
        """
        max_values = []
        for stats in self.active_components.values():
            max_values.extend(stats.max)
        return max(max_values) if max_values else float("-inf")


class StateStatistics(ComponentStatistics):
    pass


class ActionStatistics(ComponentStatistics):
    pass


class Statistics(BaseModel):
    state: StateStatistics
    action: ActionStatistics


class CameraConfig(BaseModel):
    resolution: Tuple[int, int] = Field(
        ...,
        examples=[[320, 240]],
        description="Camera resolution in (width, height) format",
    )
    channels: int = Field(
        ...,
        examples=[3],
        description="Number of color channels (3 for RGB)",
        ge=1,
        le=4,
    )
    fps: float = Field(..., examples=[30.0], description="Frames per second", gt=0)


class ModalitiesConfig(BaseModel):
    video: Dict[str, CameraConfig] = Field(
        ..., description="Dictionary of camera configurations keyed by camera name"
    )


class EmbodimentConfig(BaseModel):
    modalities: ModalitiesConfig
    statistics: Statistics
    embodiment_tag: str


class HuggingFaceModelConfig(BaseModel):
    """
    We use a model validator to extract the embodiment config from the model config.
    """

    embodiment: EmbodimentConfig | None = (
        None  # This will store the found embodiment config
    )
    embodiment_field_name: str | None = None  # This will store the original field name

    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def extract_embodiment_config(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that at least one field contains an EmbodimentConfig.
        Extract and set it to the 'embodiment' field for easier access.
        """
        if not isinstance(data, dict):
            return data

        embodiment_field = None

        # Look through all fields for one that matches the EmbodimentConfig structure
        for field_name, value in data.items():
            # Check if the value is a dict and has the required keys for an EmbodimentConfig
            if isinstance(value, dict) and all(
                key in value for key in ["modalities", "statistics", "embodiment_tag"]
            ):
                # We found an embodiment config
                embodiment_field = field_name
                # Store the original field name for reference if needed
                data["embodiment_field_name"] = field_name
                # Store the actual embodiment config in our standard field
                data["embodiment"] = value
                break

        # If no embodiment field was found, raise a validation error
        if embodiment_field is None:
            raise ValueError(
                "No valid embodiment configuration found in the model config"
            )

        return data


class ModelSpawnConfig(BaseModel):
    video_keys: list[str]
    state_keys: list[str]
    action_keys: list[str]
    embodiment_tag: str
    unit: Literal["degrees", "rad"]
    hf_model_config: HuggingFaceModelConfig


class Gr00tN1(ActionModel):
    def __init__(
        self,
        action_keys: list[str],
        server_url: str = "localhost",
        server_port: int = 5555,
        **kwargs,
    ):
        super().__init__(server_url, server_port)
        self.client = ExternalRobotInferenceClient(server_url, server_port)
        self.action_keys = action_keys

    def sample_actions(self, inputs: dict) -> np.ndarray:
        # Get the dict from the server
        response = self.client.get_action(inputs)
        action_parts = []
        for key in self.action_keys:
            new_action = response[key]

            if isinstance(new_action, np.ndarray):
                if new_action.ndim == 1 and len(new_action) == 16:
                    # Handle 1D array of shape (16,) by reshaping to (16, 1)
                    new_action = new_action.reshape(16, 1)
                elif new_action.ndim == 2 and new_action.shape[0] == 16:
                    # Already a 2D array with batch size 16, no reshaping needed
                    pass
                else:
                    raise ValueError(
                        f"Unexpected array shape for key {key}: {new_action.shape}"
                    )

                # Array case: shape is (16, action_size)
                batch_size, action_size = new_action.shape
                if batch_size != 16:
                    raise ValueError(
                        f"Expected batch size 16, got {batch_size} for key {key}"
                    )

                # If action_size is 1 or 6, assume the last column is the gripper
                if action_size in [1, 6]:
                    new_action[:, -1] = np.where(
                        new_action[:, -1] < 0.35, 0.0, new_action[:, -1]
                    )

                action_parts.append(new_action)
            else:
                raise ValueError(
                    f"Unexpected new_action format for key {key}: {type(new_action)}, "
                    f"shape/len: {getattr(new_action, 'shape', len(new_action))}"
                )

        # Concatenate along axis=1 to combine features, preserving batch size of 16
        if not action_parts:
            raise ValueError("No valid actions found to concatenate")

        concatenated_actions = np.concatenate(action_parts, axis=1)

        return concatenated_actions

    @classmethod
    def verify_huggingface_model(
        cls, model_id: str, all_cameras: AllCameras, robots: List[BaseRobot]
    ) -> ModelSpawnConfig:
        """
        Verify if the HuggingFace model is compatible with the current setup.
        """
        try:
            api = HfApi(token=get_hf_token())
            model_info = api.model_info(model_id)
            if model_info is None:
                raise Exception(f"Model {model_id} not found on Hugging Face Hub.")
            # Download file from the model repo
            config_path = api.hf_hub_download(
                repo_id=model_id,
                filename="experiment_cfg/metadata.json",
                force_download=True,
            )
            # Read the file
            with open(config_path, "r") as f:
                config_content = f.read()
            # Parse the file
            hf_model_config = HuggingFaceModelConfig.model_validate_json(config_content)
            if hf_model_config.embodiment is None:
                raise Exception(
                    f"Model {model_id} embodiment config experiment_cfg/metadata.json not found."
                )
            video_keys = [
                "video." + key
                for key in hf_model_config.embodiment.modalities.video.keys()
            ]
            state_keys = [
                "state." + key
                for key in hf_model_config.embodiment.statistics.state.component_names
            ]
            action_keys = [
                "action." + key
                for key in hf_model_config.embodiment.statistics.action.component_names
            ]
        except Exception as e:
            raise Exception(
                f"Error loading model {model_id} from Hugging Face Hub: {e}"
            )

        number_of_cameras = len(hf_model_config.embodiment.modalities.video.keys())
        number_of_robots = hf_model_config.embodiment.statistics.state.number_of_arms

        # Check if the number of cameras in the model config matches the number of cameras connected
        if len(all_cameras.video_cameras) < number_of_cameras:
            logger.warning(
                f"Model has {len(hf_model_config.embodiment.modalities.video)} cameras but {len(all_cameras.video_cameras)} cameras are plugged."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model has {len(hf_model_config.embodiment.modalities.video)} cameras but {len(all_cameras.video_cameras)} cameras are plugged.",
            )

        # Check if the number of robots in the model config matches the number of robots connected
        if number_of_robots != len(robots):
            logger.warning(
                f"Model has {number_of_robots} robots but {len(robots)} robots are connected."
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model has {number_of_robots} robots but {len(robots)} robots are connected.",
            )

        # Determine angle unit based on state statistics
        max_values = hf_model_config.embodiment.statistics.state.get_max_value()
        use_degrees = max_values > 3.2
        angle_unit: Literal["degrees", "rad"] = "degrees" if use_degrees else "rad"

        return ModelSpawnConfig(
            video_keys=video_keys,
            state_keys=state_keys,
            action_keys=action_keys,
            embodiment_tag=hf_model_config.embodiment.embodiment_tag,
            unit=angle_unit,
            hf_model_config=hf_model_config,
        )
