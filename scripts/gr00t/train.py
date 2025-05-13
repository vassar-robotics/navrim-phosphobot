import tyro
import time
from dataclasses import dataclass
from pathlib import Path
import os
from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from phosphobot.am.base import (
    HuggingFaceTokenValidator,
    generate_readme,
    resize_dataset,
)
import json
import asyncio


@dataclass
class Config:
    """
    If no indexes_to_delete is provided, will attempt to repair the dataset.
    If indexes_to_delete is provided, will delete the specified indexes from the dataset and repair the rest.
    """

    dataset_id: str
    """the Hugging Face id of the dataset to train on"""

    data_dir: str = "data/"
    """the directory to save the dataset to"""

    output_dir: str = "outputs/"
    """the directory to save the model to"""

    output_hf_model_id: str | None = None
    """the Hugging Face id of the model to save the finetuned model to. If not provided, will not save the model to Hugging Face."""

    epochs: int = 10
    """the number of epochs to train for"""

    batch_size: int = 64
    """the batch size to train for. Higher is better. Lower if you run out of memory."""

    learning_rate: float = 0.0002
    """the learning rate to train for"""


def generate_modality_json(data_dir) -> tuple[int, int]:
    # Load the metadata file to get image keys
    with open(data_dir / "meta" / "info.json", "r") as f:
        metadata = json.load(f)
        image_keys = []
        for key in metadata["features"].keys():
            if "image" in key:
                image_keys.append(key)

    number_of_cameras = len(image_keys)
    number_of_robots: int = metadata["features"]["action"]["shape"][0] // 6
    print(f"Number of cameras: {number_of_cameras}")
    print(f"Number of robots: {number_of_robots}")

    # Create the action/state keys based on the number of robots
    # Each robot has 5 arm keys and 1 gripper key
    robot_keys = []
    for i in range(number_of_robots):
        robot_keys.append(f"arm_{i}")

    # Create the action/state keys
    robot_structure = {}
    index = 0
    for key in robot_keys:
        robot_structure[key] = {"start": index, "end": index + 6}
        index += 6

    # Populate the video section with the image keys
    video_structure: dict = {}
    camera_name = [f"image_cam_{i}" for i in range(number_of_cameras)]
    for i, image_key in enumerate(image_keys):
        video_structure[camera_name[i]] = {"original_key": image_key}  # type: ignore

    # Create the base modality structure
    modality_json = {
        "state": robot_structure,
        "action": robot_structure,
        "video": video_structure,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }

    print(f"Modality JSON: {modality_json}")

    # Write the modality.json file
    with open(data_dir / "meta" / "modality.json", "w") as f:
        json.dump(modality_json, f, indent=4)

    return number_of_robots, number_of_cameras


async def run_gr00t_training(
    data_dir,
    output_dir,
    batch_size,
    epochs,
    number_of_robots,
    number_of_cameras,
    learning_rate,
    wandb_enabled: bool,
    timeout_seconds: int,
):
    cmd = [
        "python",
        "Isaac-GR00T/scripts/gr00t_finetune.py",
        "--dataset-path",
        str(data_dir),
        # Only 1 GPU for now
        # Open an issue for multi-GPU support
        "--num-gpus",
        "1",
        "--output-dir",
        str(output_dir),
        "--batch-size",
        str(batch_size),
        "--num-epochs",
        str(epochs),
        "--save-steps",
        "10000",
        "--num-arms",
        str(number_of_robots),
        "--num-cams",
        str(number_of_cameras),
        "--learning_rate",
        str(learning_rate),
        "--report_to",
        "wandb" if wandb_enabled else "tensorboard",
        "--video_backend",
        "torchvision_av",
    ]

    logger.info(f"Starting training with command: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )

    output_lines = []

    async def read_output():
        assert process.stdout is not None
        async for line in process.stdout:
            stripped_line = line.decode().strip()
            print(stripped_line)
            output_lines.append(stripped_line)

    try:
        await asyncio.wait_for(read_output(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        logger.error(f"Training process timed out after {timeout_seconds} seconds.")
        raise TimeoutError(
            f"Training process exceeded timeout of {timeout_seconds} seconds. Please consider lowering the number of epochs or batch size."
        )

    await process.wait()

    if process.returncode != 0:
        error_output = "\n".join(output_lines[-10:])
        error_msg = f"Training process failed with exit code {process.returncode}:\n{error_output}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    return output_lines


def main(config: Config):
    logger.info(f"Starting training for dataset={config.dataset_id}")

    # Create output directory
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    selected_branch = "main"
    print(f"Using branch {selected_branch}")

    if config.output_hf_model_id is not None:
        # We check if the user has write access to the model-id
        if not HuggingFaceTokenValidator().has_write_access(
            hf_model_name=config.output_hf_model_id
        ):
            raise ValueError(
                f"The provided HF token does not have write access to {config.output_hf_model_id}"
            )

    # Download huggingface dataset with huggingface_hub
    logger.info(f"Downloading dataset {config.dataset_id} to {data_dir}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            dataset_path_as_str = snapshot_download(
                repo_id=config.dataset_id,
                repo_type="dataset",
                revision=selected_branch,
                local_dir=str(data_dir),
                # token=hf_token,
            )
            DATASET_PATH = Path(dataset_path_as_str)
            logger.info(f"Dataset {config.dataset_id} downloaded to {DATASET_PATH}")
            break  # Exit the loop if download is successful
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait for 1 second before retrying
            else:
                raise RuntimeError(
                    f"Failed to download dataset {config.dataset_id} after {max_retries} attempts, is Hugging Face down ? : {e}"
                )

    resized_successful, _ = resize_dataset(
        dataset_root_path=DATASET_PATH, resize_to=(224, 224)
    )
    if not resized_successful:
        raise RuntimeError(
            f"Resizing dataset {config.dataset_id} to 224x224 failed: {resized_successful}"
        )
    logger.info(f"Resized dataset {config.dataset_id} to 224x224")

    # Create the modality json file in meta folder
    logger.info("Generating modality.json file")
    number_of_robots, number_of_cameras = generate_modality_json(data_dir)

    # Find the total number of frames in the dataset in meta / info.json
    with open(data_dir / "meta" / "info.json", "r") as f:
        info = json.load(f)
        total_frames = info["total_frames"]

    steps = total_frames * config.epochs // config.batch_size + 1

    logger.info(f"Will train for {config.epochs} epochs, which is {steps} steps")

    asyncio.run(
        run_gr00t_training(
            data_dir=data_dir,
            output_dir=output_dir,
            batch_size=config.batch_size,
            epochs=config.epochs,
            number_of_robots=number_of_robots,
            number_of_cameras=number_of_cameras,
            learning_rate=config.learning_rate,
            wandb_enabled=True,
            timeout_seconds=3 * 60 * 60,
        )
    )
    logger.info("Training finished")
    # Upload to Hugging Face if token is provided
    if config.output_hf_model_id is not None:
        logger.info(f"Uploading model to Hugging Face as {config.output_hf_model_id}")

        # Upload using huggingface_hub
        api = HfApi()
        files_directory = output_dir

        api.create_repo(
            repo_id=config.output_hf_model_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )

        # Then upload main files to the main branch
        for item in files_directory.glob("*"):
            if item.is_file() and item.name != "README.md":
                logger.info(
                    f"Uploading {item} to {config.output_hf_model_id} as {item.name}"
                )
                api.upload_file(
                    repo_id=config.output_hf_model_id,
                    repo_type="model",
                    path_or_fileobj=str(item.resolve()),
                    path_in_repo=item.name,
                )

        # Upload experiment_cfg and its contents
        exp_cfg_dir = files_directory / "experiment_cfg"
        if exp_cfg_dir.exists() and exp_cfg_dir.is_dir():
            for item in exp_cfg_dir.glob("**/*"):
                if item.is_file():
                    # Keep the directory structure
                    rel_path = item.relative_to(files_directory)
                    logger.info(f"Uploading config file: {item} as {rel_path}")
                    api.upload_file(
                        repo_type="model",
                        path_or_fileobj=str(item.resolve()),
                        path_in_repo=str(rel_path),
                        repo_id=config.output_hf_model_id,
                    )

        # Upload README last
        readme = generate_readme(
            model_type="gr00t",
            dataset_repo_id=config.dataset_id,
            epochs=config.epochs,
            batch_size=config.batch_size,
            return_readme_as_bytes=True,
        )

        api.upload_file(
            repo_type="model",
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=config.output_hf_model_id,
        )

        # Get the model URL
        huggingface_model_url = f"https://huggingface.co/{config.output_hf_model_id}"
        logger.info(f"Model successfully uploaded to {huggingface_model_url}")

        logger.info("Model successfully uploaded to Hugging Face")
        logger.info(f"Find your trained model on Hugging Face: {huggingface_model_url}")

    else:
        logger.info("Skipping upload to Hugging Face")
        logger.info("Provide a model-id to enable automatic upload to Hugging Face")


if __name__ == "__main__":
    config = tyro.cli(Config)

    main(config)
