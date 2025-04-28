from pathlib import Path
import random
import string
import requests  # type: ignore
import numpy as np
from huggingface_hub import HfApi
from fastapi import HTTPException
from abc import abstractmethod, ABC
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


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


class TrainingParamsAct(BaseModel):
    batch_size: int | None = Field(
        default=100, description="Batch size for training, we run this on L40S", gt=0
    )
    steps: int | None = Field(
        default=8000,
        description="Number of training steps, default is 10000, increase it if you want to train longer",
        gt=0,
        le=100000,
    )


class TrainingParamsGr00T(BaseModel):
    train_test_split: float = Field(
        default=1.0,
        description="Train test split ratio, default is 1.0 (no split), should be between 0 and 1",
        gt=0,
        le=1,
    )
    batch_size: int = Field(
        default=64,
        description="Batch size for training, default is 64, decrease it if you get an out of memory error",
        gt=0,
        le=80,
    )
    epochs: int = Field(
        default=10,
        description="Number of epochs to train for, default is 10",
        gt=0,
        le=50,
    )
    learning_rate: float = Field(
        default=0.0001,
        description="Learning rate for training, default is 0.0001",
        gt=0,
        le=1,
    )


class TrainingRequest(BaseModel):
    """Pydantic model for training request validation"""

    model_type: Literal["ACT", "gr00t"] = Field(
        ...,
        description="Type of model to train, either 'ACT' or 'gr00t'",
    )
    dataset_name: str = Field(
        ...,
        description="Dataset repository ID on Hugging Face, should be a public dataset",
    )
    model_name: str = Field(
        ...,
        description="Name of the trained model to upload to Hugging Face, should be in the format phospho-app/<model_name> or <model_name>",
    )
    wandb_api_key: Optional[str] = Field(
        default=None,
        description="WandB API key for tracking training, you can find it at https://wandb.ai/authorize",
    )
    training_params: Optional[TrainingParamsAct | TrainingParamsGr00T] = Field(
        default=None,
        description="Training parameters for the model, if not provided, default parameters will be used",
    )

    @field_validator("model_name", mode="before")
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
            raise ValueError(
                "Model name should be in the format phospho-app/<model_name> or <model_name>",
            )
        return model_name

    @field_validator("dataset_name", mode="before")
    def validate_dataset(cls, dataset_name: str) -> str:
        try:
            url = f"https://huggingface.co/api/datasets/{dataset_name}/tree/main"
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                raise ValueError()
            return dataset_name
        except Exception:
            raise ValueError(
                f"Dataset {dataset_name} is not a valid, public Hugging Face dataset. Please check the URL and try again. Your dataset name should be in the format <username>/<dataset_name>",
            )

    @model_validator(mode="after")
    def validate_training_params(self) -> "TrainingRequest":
        correspondance = {
            "gr00t": TrainingParamsGr00T(),
            "ACT": TrainingParamsAct(),
        }

        if self.training_params is None:
            self.training_params = correspondance[self.model_type]  # type: ignore

        else:
            if not isinstance(
                self.training_params, correspondance[self.model_type].__class__
            ):
                raise ValueError(
                    f"Training parameters for {self.model_type} should be of type {correspondance[self.model_type].__class__.__name__}",
                )
        return self


class HuggingFaceTokenValidator:
    @staticmethod
    def has_write_access(hf_token: str, hf_model_name: str) -> bool:
        """Check if the HF token has write access by attempting to create a repo."""
        api = HfApi()
        try:
            api.create_repo(hf_model_name, private=False, exist_ok=True, token=hf_token)
            return True  # The token has write access
        except Exception as e:
            print(f"Write access check failed: {e}")
            return False  # The token does not have write access


def generate_readme(
    model_type: str,
    dataset_repo_id: str,
    folder_path: Path | None = None,
    wandb_run_url: Optional[str] = None,
    steps: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    error_traceback: Optional[str] = None,
    return_readme_as_bytes: bool = False,
):
    readme = f"""
---
tags:
- phosphobot
- {model_type}
task_categories:
- robotics                                               
---

# {model_type} Model - phospho Training Pipeline

"""
    if error_traceback:
        readme += f"""
## Error Traceback
We faced an issue while training your model.

```
{error_traceback}
```

"""
    else:
        readme += """
## This model was trained using **phospho**.

Training was successfull, try it out on your robot!

"""

    readme += f"""
## Training parameters:

- **Dataset**: [{dataset_repo_id}](https://huggingface.co/datasets/{dataset_repo_id})
- **Wandb run URL**: {wandb_run_url}
- **Epochs**: {epochs}
- **Batch size**: {batch_size}
- **Training steps**: {steps}

📖 **Get Started**: [docs.phospho.ai](https://docs.phospho.ai?utm_source=replicate_groot_training_pipeline)

🤖 **Get your robot**: [robots.phospho.ai](https://robots.phospho.ai?utm_source=replicate_groot_training_pipeline)
"""
    if return_readme_as_bytes:
        return readme.encode("utf-8")
    if folder_path is not None:
        readme_path = folder_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme)
        return readme_path
    else:
        raise ValueError(
            "folder path is None and return_readme_as_bytes is False. Please provide a valid folder path. If you want to return the readme as bytes, set return_readme_as_bytes to True."
        )
