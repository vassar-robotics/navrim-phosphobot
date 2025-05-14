import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from huggingface_hub import HfApi
from loguru import logger

from phosphobot.am.base import TrainingRequest
from phosphobot.models import (
    StatusResponse,
    SupabaseTrainingModel,
    TrainingConfig,
)
from phosphobot.models.dataset import InfoModel
from phosphobot.supabase import get_client, user_is_logged_in
from phosphobot.utils import get_hf_token, get_home_app_path, get_tokens

router = APIRouter(tags=["training"])


@router.post("/training/models/read", response_model=TrainingConfig)
async def get_models(
    session=Depends(user_is_logged_in),
) -> TrainingConfig:
    """Get the list of models to be trained"""
    client = await get_client()
    user_id = session.user.id

    model_list = await (
        client.table("trainings")
        .select("*")
        .eq("user_id", user_id)
        .order("requested_at", desc=True)
        .limit(1000)
        .execute()
    )

    if not model_list.data or len(model_list.data) == 0:
        return TrainingConfig(
            models=[],
        )

    # Convert the list of models to a TrainingConfig object
    training_config = TrainingConfig(
        models=[
            SupabaseTrainingModel(
                **model,
            )
            for model in model_list.data
        ]
    )
    return training_config


@router.post(
    "/training/start",
    response_model=StatusResponse,
    summary="Start training a model",
    description="Start training an ACT or gr00t model on the specified dataset. This will upload a trained model to the Hugging Face Hub using the main branch of the specified dataset.",
)
async def start_training(
    request: TrainingRequest,
    session=Depends(user_is_logged_in),
) -> StatusResponse | HTTPException:
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
                detail="Token expired, please relogin.",
            )

        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="A training is already in progress. Please wait until it is finished.",
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Failed to start training on the backend.",
            )

    return StatusResponse(
        message=f"Training triggered successfully, find your model at: https://huggingface.co/{request.model_name}"
    )
