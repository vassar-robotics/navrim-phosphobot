import asyncio
import os
import sys
import time
from datetime import datetime
from typing import cast
import lerobot
import torch
import re
import shlex
import atexit

import av
import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import HfApi
from loguru import logger

from phosphobot.am.base import TrainingParamsGr00T, TrainingRequest
from phosphobot.models import (
    CustomTrainingRequest,
    InfoModel,
    StartTrainingResponse,
    CancelTrainingRequest,
    StatusResponse,
    SupabaseTrainingModel,
    TrainingConfig,
)
from phosphobot.api_supabase import user_is_logged_in
from phosphobot.utils import get_hf_token, get_home_app_path, get_tokens
from phosphobot.workaround.db import DatabaseManager

router = APIRouter(tags=["training"])


@router.post("/training/models/read", response_model=TrainingConfig)
async def get_models(
    session=Depends(user_is_logged_in),
) -> TrainingConfig:
    """Get the list of models to be trained"""
    # client = await get_client()
    # user_id = session.user.id

    # model_list = await (
    #     client.table("trainings")
    #     .select("*")
    #     .eq("user_id", user_id)
    #     .order("requested_at", desc=True)
    #     .limit(1000)
    #     .execute()
    # )

    with DatabaseManager.get_instance() as db:
        trainings_data = db.get_trainings_by_user("default_user")[:1000]

    class MockResponse:
        def __init__(self, data):
            self.data = data

    model_list = MockResponse(trainings_data)

    if not model_list.data or len(model_list.data) == 0:
        return TrainingConfig(
            models=[],
        )

    # Convert the list of models to a TrainingConfig object
    training_config = TrainingConfig(models=[SupabaseTrainingModel.model_validate(model) for model in model_list.data])
    return training_config


@router.post(
    "/training/start",
    response_model=StartTrainingResponse,
    summary="Start training a model",
    description=(
        "Start training an ACT or gr00t model on the specified dataset. This will upload a trained model to the "
        "Hugging Face Hub using the main branch of the specified dataset."
    ),
)
async def start_training(
    request: TrainingRequest,
    session=Depends(user_is_logged_in),
) -> StartTrainingResponse | HTTPException:
    """
    Trigger training for a gr00t or ACT model on the specified dataset.

    This will upload a trained model to the Hugging Face Hub using the main branch of the specified dataset.

    Before launching a training, please make sure that:
    - Your dataset is uploaded to Hugging Face
    - Your dataset is in the Le Robot format (>= v2.0)
    - Your dataset has at least 10 episodes
    - You are logged in to phosphobot

    Pro usage:
    - (You can add a wandb token in phosphobot to track your training)
    """
    logger.debug(f"Training request: {request}")
    tokens = get_tokens()
    if not tokens.MODAL_API_URL:
        raise HTTPException(
            status_code=400,
            detail="Modal API url not found. Please check your configuration.",
        )

    wandb_token_path = str(get_home_app_path()) + "/wandb.token"
    if os.path.exists(wandb_token_path):
        logger.debug("WandB token found. Will be used for the training.")
        # If present, we add the wandb api key to the training request
        with open(wandb_token_path, "r") as f:
            request.wandb_api_key = f.read().strip()

    # Check that the given dataset has enough episodes to train
    api = HfApi(token=get_hf_token())
    try:
        info_file_path = api.hf_hub_download(
            repo_id=request.dataset_name,
            repo_type="dataset",
            filename="meta/info.json",
            force_download=True,
        )
        meta_folder_path = os.path.dirname(info_file_path)
        validated_info_model = InfoModel.from_json(meta_folder_path=meta_folder_path)
        if validated_info_model.total_episodes < 10:
            raise HTTPException(
                status_code=400,
                detail="The dataset has less than 10 episodes. Please record more episodes before training.",
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.warning(f"Error accessing dataset info: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download and parse meta/info.json.\n{e}",
        )
    # The check is only done for gr00t models
    if request.model_type == "gr00t":
        # We cast the training params to the correct type
        training_params = cast(TrainingParamsGr00T, request.training_params)
        if training_params.validation_dataset_name:
            try:
                info_file_path = api.hf_hub_download(
                    repo_id=training_params.validation_dataset_name,
                    repo_type="dataset",
                    filename="meta/info.json",
                    force_download=True,
                )
                meta_folder_path = os.path.dirname(info_file_path)
                InfoModel.from_json(meta_folder_path=meta_folder_path)
            except Exception as e:
                logger.warning(f"Error accessing validation dataset info: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"Failed to download and parse validation meta/info.json.\n{e}",
                )

    # Send training request to modal API
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{tokens.MODAL_API_URL}/train",
            json=request.model_dump(),
            headers={"Authorization": f"Bearer {session.access_token}"},
        )

        # We need the token on modal so we raise an error if the token is not right
        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Token expired. Please login again.",
            )
        if response.status_code == 429:
            # Too many requests: this can happen if the user is trying to train too many models at once
            raise HTTPException(status_code=429, detail=response.text)

        if response.status_code == 422:
            raise HTTPException(
                status_code=422,
                detail=f"The training request is invalid. Please check your parameters. {response.text}",
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to start training on the backend: {response.text}",
            )

    response_data = response.json()

    return StartTrainingResponse(
        message=f"Training triggered successfully, find your model at: https://huggingface.co/{request.model_name}",
        training_id=response_data.get("training_id", None),
    )


@router.post(
    "/training/start-locally",
    response_model=StatusResponse,
    summary="Start training a model locally",
    description=(
        "Start training an ACT or gr00t model on the specified dataset. This will upload a trained model to the "
        "Hugging Face Hub using the main branch of the specified dataset."
    ),
)
async def start_training_locally(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse | HTTPException:
    """
    Start training a model locally.

    This will train a model locally using the specified dataset.
    """
    # Check that the given dataset has enough episodes to train
    api = HfApi(token=get_hf_token())
    try:
        info_file_path = api.hf_hub_download(
            repo_id=request.dataset_name,
            repo_type="dataset",
            filename="meta/info.json",
            force_download=True,
        )
        meta_folder_path = os.path.dirname(info_file_path)
        validated_info_model = InfoModel.from_json(meta_folder_path=meta_folder_path)
        if validated_info_model.total_episodes < 10:
            raise HTTPException(
                status_code=400,
                detail="The dataset has less than 10 episodes. Please record more episodes before training.",
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.warning(f"Error accessing dataset info: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Failed to download and parse meta/info.json.\n{e}",
        )

    # Build the training command and redirect to `start_custom_training`
    lerobot_root = os.path.dirname(lerobot.__file__)
    output_root = os.path.join(os.path.expanduser("~"), "navrim", "models")
    output_dir = os.path.join(output_root, request.model_name)
    training_command = " ".join(
        [
            shlex.quote(sys.executable),
            shlex.quote(os.path.join(lerobot_root, "scripts", "train.py")),
            shlex.quote("--env.type=aloha"),
            shlex.quote("--policy.push_to_hub=True"),
            shlex.quote(f"--batch_size={request.training_params.batch_size}"),
            shlex.quote(f"--dataset.repo_id={request.dataset_name}"),
            shlex.quote(f"--job_name={request.model_name.replace('/', '--')}"),
            shlex.quote(f"--policy.device={'cuda' if torch.cuda.is_available() else 'cpu'}"),
            shlex.quote(f"--policy.repo_id={request.model_name}"),
            shlex.quote(f"--policy.type={request.model_type.lower()}"),
            shlex.quote(f"--steps={request.training_params.steps}"),
            shlex.quote(f"--output_dir={output_dir}"),
        ]
    )
    logger.info(f"Training command: {training_command}")
    return await start_custom_training(
        request=CustomTrainingRequest(custom_command=training_command),
        background_tasks=background_tasks,
    )


def _prepare_dynamic_link_libraries_macos():
    av_root = os.path.dirname(av.__file__)
    av_dylibs_dir = os.path.join(av_root, ".dylibs")
    dylib_pattern = re.compile(r"^lib(.+?)\.(\d+)(\..+)?\.dylib$")
    for filename in os.listdir(av_dylibs_dir):
        matches = dylib_pattern.match(filename)
        if matches:
            target = f"lib{matches.group(1)}.{matches.group(2)}.dylib"
            source_path = os.path.join(av_dylibs_dir, filename)
            target_path = os.path.join(av_dylibs_dir, target)
            logger.info(f"symlink {source_path} -> {target_path}")
            if not os.path.exists(target_path):
                os.symlink(source_path, target_path)
    return av_dylibs_dir


def _prepare_dynamic_link_libraries_win():
    pass


def prepare_dynamic_link_libraries() -> dict[str, str]:
    env = os.environ.copy()
    if sys.platform == "darwin":
        dylibs_dir = _prepare_dynamic_link_libraries_macos()
        dyld_library_path = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{dylibs_dir}:{dyld_library_path}" if dyld_library_path else dylibs_dir
    elif sys.platform == "win32":
        _prepare_dynamic_link_libraries_win()
    # Currently we only handle MacOS and Windows
    return env


@router.post("/training/start-custom", response_model=StatusResponse)
async def start_custom_training(
    request: CustomTrainingRequest,
    background_tasks: BackgroundTasks,
) -> StatusResponse | HTTPException:
    # 1) Prepare log file
    log_file_name = f"training_{int(time.time())}.log"
    log_file_path = os.path.join(get_home_app_path(), "logs", log_file_name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 2) Prepare dynamic link libraries
    env = prepare_dynamic_link_libraries()

    # 3) Spawn the process using pipes
    command_tokens = shlex.split(request.custom_command)
    process = await asyncio.create_subprocess_exec(  # create_subprocess_shell causes issues
        command_tokens[0],
        *command_tokens[1:],
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError("Failed to create subprocess")
    reader = process.stdout

    training_id = int(time.time())
    with DatabaseManager.get_instance() as db:
        # Try to parse the command to get the arguments
        arguments = command_tokens[2:]
        args_mapping = {}
        for arg in arguments:
            arg = re.sub(r"^--", "", arg)
            if "=" in arg:
                key, value = arg.split("=", 1)
                args_mapping[key] = value
        # Insert the training into the database
        db.insert_training(
            {
                "id": training_id,
                "status": "running",
                "user_id": "default_user",
                "dataset_name": args_mapping.get("dataset.repo_id", "unknown/unknown"),
                "model_name": args_mapping.get("policy.repo_id", "unknown/unknown"),
                "requested_at": datetime.now().isoformat(),
                "terminated_at": "",
                "used_wandb": False,
                "model_type": args_mapping.get("policy.type", "unknown"),
                "training_params": {},
                "modal_function_call_id": "",
            }
        )

    def cancel_training_at_exit():
        if process.returncode is None:
            process.terminate()
            time.sleep(0.5)
            if process.returncode is None:
                process.kill()
            with DatabaseManager.get_instance() as db:
                db.update_training_status(training_id, "cancelled", datetime.now().isoformat())

    atexit.register(cancel_training_at_exit)

    # 4) Monitor task: read from the process output and write to your log file
    async def monitor_output(reader: asyncio.StreamReader, log_path: str):
        with open(log_path, "wb") as f:
            # header
            f.write(f"Custom training started at {time.ctime()}\n".encode())
            f.write(f"Command: {request.custom_command}\n\n".encode())
            f.flush()

            # stream everything, flushing as it arrives
            while True:
                chunk = await reader.read(1024)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()

            # when process exits, append return code
            await process.wait()
            footer = f"\nProcess completed with return code {process.returncode}\n"
            f.write(footer.encode())
            if process.returncode == 0:
                f.write(b"Training completed successfully!\n")
                with DatabaseManager.get_instance() as db:
                    db.update_training_status(training_id, "succeeded", datetime.now().isoformat())
            else:
                f.write(b"Training failed. See errors above.\n")
                with DatabaseManager.get_instance() as db:
                    db.update_training_status(training_id, "failed", datetime.now().isoformat())

    background_tasks.add_task(monitor_output, reader, log_file_path)

    return StatusResponse(message=log_file_name)


@router.get("/training/logs/{log_file}")
async def stream_logs(log_file: str):
    """Stream the logs from a log file"""
    log_path = os.path.join(get_home_app_path(), "logs", log_file)

    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")

    async def log_generator():
        """Generator to stream logs line by line as they are written"""
        with open(log_path, "rb") as f:
            # First, send all existing content
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            if file_size > 0:
                yield f.read()

            # Then, continue streaming as new content is added
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    # Check if process is still running by looking for completion message
                    if b"Process completed with return code" in line:
                        break
                    await asyncio.sleep(0.1)  # Small delay to avoid busy waiting

    return StreamingResponse(log_generator(), media_type="text/plain")


@router.post("/training/cancel", response_model=StatusResponse)
async def cancel_training(
    request: CancelTrainingRequest,
    session=Depends(user_is_logged_in),
) -> StatusResponse | HTTPException:
    """Cancel a training job"""
    logger.debug(f"Cancelling training request: {request}")
    tokens = get_tokens()
    if not tokens.MODAL_API_URL:
        raise HTTPException(
            status_code=400,
            detail="Modal API url not found. Please check your configuration.",
        )

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{tokens.MODAL_API_URL}/cancel",
            json=request.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {session.access_token}"},
        )

        if response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Token expired. Please login again.",
            )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to cancel training on the backend: {response.text}",
            )

    response_data = response.json()

    return StatusResponse(
        status=response_data.get("status", "error"),
        message=response_data.get("message", "No message"),
    )
