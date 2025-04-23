import random
import string
from typing import Optional
import numpy as np
from fastapi import HTTPException
from abc import abstractmethod, ABC
from pydantic import BaseModel, Field, field_validator


class ActionModel(ABC):
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

    @abstractmethod
    def sample_actions(self, inputs: dict) -> np.ndarray:
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
        return self.sample_actions(*args, **kwargs)


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation"""

    dataset: str = Field(
        ...,
        description="Dataset repository ID on Hugging Face, should be a public dataset",
    )
    model_name: str = Field(
        ...,
        description="Name of the trained model to upload to Hugging Face, should be in the format phospho-app/<model_name> or <model_name>",
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        # We add random characters to the model name to avoid collisions
        random_chars = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=10)
        )
        # We need to make sure that the model is called phospho-app/...
        # So we can upload it to the phospho Hugging Face repo
        size = model_name.split("/")
        if len(size) == 1:
            model_name = "phospho-app/" + model_name + "-" + random_chars
        elif len(size) == 2:
            if size[0] != "phospho-app":
                model_name = "phospho-app/" + size[1] + "-" + random_chars
        else:
            raise HTTPException(
                status_code=400,
                detail="Model name should be in the format phospho-app/<model_name> or <model_name>",
            )
        return model_name

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, dataset: str) -> str:
        import requests

        try:
            url = f"https://huggingface.co/api/datasets/{dataset}/tree/main"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise ValueError()
            return dataset
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {dataset} is not a valid, public Hugging Face dataset. Please check the URL and try again. Your dataset name should be in the format <username>/<dataset_name>",
            )


class HuggingFaceInfoModel(BaseModel):
    """
    Pydantic model used to check the number of episodes in a dataset
    This only checks the number of episodes in the dataset !
    """

    total_episodes: int

    class Config:
        extra = "allow"
