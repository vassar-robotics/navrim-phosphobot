import os
from pathlib import Path, PurePath

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from huggingface_hub import HfApi
from loguru import logger
from phosphobot.am.base import ActionModel
from phosphobot.configs import config
from phosphobot.models import Dataset
from phosphobot.utils import (
    get_home_app_path,
    parse_hf_username_or_orgid,
)

from teleop.models import (
    AdminSettingsRequest,
    AdminSettingsResponse,
    AdminSettingsTokenResponse,
    BrowseFilesResponse,
    BrowserFilesRequest,
    DatasetListResponse,
    DeleteEpisodeRequest,
    HuggingFaceTokenRequest,
    MergeDatasetsRequest,
    ModelVideoKeysRequest,
    ModelVideoKeysResponse,
    StatusResponse,
    VizSettingsResponse,
    WandBTokenRequest,
)
from teleop.utils import (
    get_dashboard_path,
    is_running_on_pi,
    list_directory_items,
    login_to_hf,
    sanitize_path,
    zip_folder,
)

router = APIRouter(tags=["pages"])

# Root directory for the file browser
ROOT_DIR = str(get_home_app_path() / "recordings")
INDEX_PATH = str(get_dashboard_path() / "dist" / "index.html")


# Optionally, if you want the dashboard to be served at the root endpoint:
@router.get("/auth", response_class=HTMLResponse)
@router.get("/sign-in", response_class=HTMLResponse)
@router.get("/sign-up", response_class=HTMLResponse)
@router.get("/auth/confirm", response_class=HTMLResponse)
@router.get("/auth/forgot-password", response_class=HTMLResponse)
@router.get("/auth/reset-password", response_class=HTMLResponse)
@router.get("/calibration", response_class=HTMLResponse)
@router.get("/admin", response_class=HTMLResponse)
@router.get("/viz", response_class=HTMLResponse)
@router.get("/network", response_class=HTMLResponse)
@router.get("/browse", response_class=HTMLResponse)
@router.get("/leader", response_class=HTMLResponse)
@router.get("/replay", response_class=HTMLResponse)
@router.get("/control", response_class=HTMLResponse)
@router.get("/inference", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    with open(INDEX_PATH, "r") as f:
        content = f.read()
    return HTMLResponse(content=content)


@router.get("/admin/settings", response_model=AdminSettingsResponse)
async def get_admin_settings():
    return AdminSettingsResponse(
        dataset_name=config.DEFAULT_DATASET_NAME,
        freq=config.DEFAULT_FREQ,
        episode_format=config.DEFAULT_EPISODE_FORMAT,
        video_codec=config.DEFAULT_VIDEO_CODEC,
        video_size=config.DEFAULT_VIDEO_SIZE,
        task_instruction=config.DEFAULT_TASK_INSTRUCTION,
        cameras_to_record=config.DEFAULT_CAMERAS_TO_RECORD,
    )


@router.post("/admin/settings/tokens", response_model=AdminSettingsTokenResponse)
async def get_admin_settings_token():
    # Return bool if the token is set and valid
    return AdminSettingsTokenResponse(
        huggingface=login_to_hf(revalidate=False),
        wandb=os.path.exists(str(get_home_app_path()) + "/wandb.token"),
    )


@router.get("/viz/settings", response_model=VizSettingsResponse)
def get_viz_settings(request: Request):
    """
    Page with an overview of the connected cameras. Open this page in the chrome browser.
    """
    # Downgrade the stream quality on raspberry pi
    if is_running_on_pi():
        quality = 8
        target_size = (320, 240)
    else:
        # We compress the video slightly but it should not be noticeable
        quality = 60
        target_size = (320, 240)

    return VizSettingsResponse(
        width=target_size[0], height=target_size[1], quality=quality
    )


@router.post("/files", response_model=BrowseFilesResponse)
async def files(query: BrowserFilesRequest):
    # Record page view
    safe_path = sanitize_path(query.path)
    root = Path(ROOT_DIR)
    full_path = os.path.join(ROOT_DIR, safe_path)

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Path not found")

    if os.path.isfile(full_path):
        return FileResponse(full_path)

    items_info = list_directory_items(path=safe_path, root_dir=ROOT_DIR)

    # Prepare episode data if in a dataset structure
    episode_paths = []
    episode_ids = []

    parts = PurePath(safe_path).parts

    if len(parts) > 1 and parts[-2] in {"json", "lerobot_v2", "lerobot_v2.1"}:
        data_dir = root / safe_path / "data"
        # look for chunk folders
        chunk0 = data_dir / "chunk-000"
        if chunk0.exists():
            # gather all parquet files
            for p in sorted(data_dir.glob("chunk-*/**/*.parquet")):
                # basename like "whatever_123.parquet"
                episode_paths.append(str(p))
                # extract the numeric ID
                stem = p.stem  # e.g. "whatever_123"
                try:
                    ep_id = int(stem.split("_")[-1])
                except ValueError:
                    continue
                episode_ids.append(ep_id)
            # sort descending for IDs
            episode_ids.sort(reverse=True)

    response_data = BrowseFilesResponse(
        directoryTitle=f"Directory: {safe_path}",
        items=items_info,
        episode_paths=episode_paths,
        episode_ids=episode_ids,
    )

    # Optionally add a tokenError if needed
    # response_data.tokenError = "Invalid Hugging Face token. Please update your token in the Admin page."
    return response_data


@router.post("/admin/huggingface/whoami")
async def whoami():
    # Fetch the Hugging Face token and try to run whoami to return the person id
    token_path = str(get_home_app_path()) + "/huggingface.token"
    if not os.path.exists(token_path):
        return {
            "status": "error",
        }
    try:
        with open(token_path, "r") as token_file:
            token = token_file.read().strip()
        api = HfApi(token=token)
        user_info = api.whoami(token=token)
        # Get the username or org that has write access
        username_or_orgid = parse_hf_username_or_orgid(user_info)
        logger.info(f"Token is valid for {username_or_orgid}")
        if username_or_orgid is None:
            return {
                "status": "error",
            }
        return {
            "status": "success",
            "username": username_or_orgid,
        }
    except Exception:
        return {
            "status": "error",
        }


@router.post("/admin/huggingface")
async def submit_token(query: HuggingFaceTokenRequest):
    # Try first to connect to HuggingFace with the token
    try:
        api = HfApi(token=query.token)
        user_info = api.whoami(token=query.token)
        # Get the username or org that has write access
        username_or_orgid = parse_hf_username_or_orgid(user_info)
        if username_or_orgid is None:
            return {
                "status": "error",
                "message": "The token does not have write access to any repository. Please add 'Write access to content/settings' in the token scope.",
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error connecting to Hugging Face: {str(e)}",
        }

    # Define the file path where the token will be saved
    file_path = str(get_home_app_path()) + "/huggingface.token"

    try:
        # Open the file in write mode and save the token
        with open(file_path, "w") as token_file:
            token_file.write(query.token)
        # Set secure file permissions (optional, but recommended)
        os.chmod(file_path, 0o600)  # Read and write for owner only
        # Change config
        logger.info("Token saved successfully!")
        # Auth again to HuggingFace
        login_to_hf()

        return {"status": "success", "message": "Token saved successfully!"}

    except Exception as e:
        logger.error(f"Error saving token: {str(e)}")
        return {"status": "error", "message": f"Error saving token: {str(e)}"}


@router.post("/admin/wandb")
async def submit_wandb_token(query: WandBTokenRequest):
    # For now, we don't perform any check on the token
    # TODO: make sure an invalid token won't crash the training process

    # Define the file path where the token will be saved
    file_path = str(get_home_app_path()) + "/wandb.token"

    try:
        # Open the file in write mode and save the token
        with open(file_path, "w") as token_file:
            token_file.write(query.token)
        # Set secure file permissions (optional, but recommended)
        os.chmod(file_path, 0o600)  # Read and write for owner only
        # Change config
        logger.info("Token saved successfully!")

        return {"status": "success", "message": "Token saved successfully!"}

    except Exception as e:
        logger.error(f"Error saving token: {str(e)}")
        return {"status": "error", "message": f"Error saving token: {str(e)}"}


@router.post("/admin/form/usersettings")
async def submit_user_settings(user_settings: AdminSettingsRequest):
    try:
        config.save_user_settings(user_settings.model_dump())

        return {"status": "success", "message": "User settings saved successfully!"}
    except Exception as e:
        logger.error(f"User Settings submission error: {e}")
        return {"status": "error", "message": f"Error saving user settings: {str(e)}"}


@router.post("/dataset/delete")
async def delete_dataset(request: Request, path: str):
    dataset_path = os.path.join(ROOT_DIR, path)
    # Check if the path exists and is a directory
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset {path} not found")

    # Test that the path contains "lerobot" or "json" to prevent accidental deletion
    if "json" not in path and "lerobot_v2" not in path and "lerobot_v2.1" not in path:
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset path. Please use the delete button provided in the admin page to delete dataset.",
        )

    dataset = Dataset(path=dataset_path)
    dataset.delete()

    return StatusResponse(status="ok")


@router.post("/dataset/merge")
async def merge_datasets(merge_request: MergeDatasetsRequest):
    """
    Merge two datasets into one.
    """
    # Validation
    # 1 - Check that there are more than 1 dataset
    # 2 - Check that the datasets are of the same type, v2 or v2.1
    # 3 - Check that the datasets are not empty
    # 4 - Check that the datasets are not the same

    # 1
    if len(merge_request.datasets) < 2:
        raise HTTPException(
            status_code=400,
            detail="You must select at least two datasets to merge.",
        )

    # 2
    dataset_type = None
    for dataset in merge_request.datasets:
        if dataset_type is None:
            dataset_type = dataset.split("/")[0]
        elif dataset.split("/")[0] != dataset_type:
            raise HTTPException(
                status_code=400,
                detail="You can only merge datasets of the same type (v2 or v2.1).",
            )

    # 3
    for dataset in merge_request.datasets:
        dataset_path = os.path.join(ROOT_DIR, dataset)
        if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
            raise HTTPException(
                status_code=404,
                detail=f"Dataset {dataset} not found",
            )
        # Check that the dataset is not empty
        if len(os.listdir(dataset_path)) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset {dataset} is empty.",
            )

    # 4
    if len(set(merge_request.datasets)) != len(merge_request.datasets):
        raise HTTPException(
            status_code=400,
            detail="You cannot merge the same dataset.",
        )

    # Merge the datasets

    # initial_dataset = Dataset(path=os.path.join(ROOT_DIR, merge_request.datasets[0]))
    # initial_dataset.merge_datasets(
    #     datasets=[
    #         os.path.join(ROOT_DIR, dataset) for dataset in merge_request.datasets[1:]
    #     ]
    # )

    return StatusResponse(status="ok")


@router.post("/dataset/list", response_model=DatasetListResponse)
async def list_datasets():
    """
    List all datasets that are both in Hugging Face and locally.
    """
    root_v2 = os.path.join(ROOT_DIR, "lerobot_v2")
    root_v2_1 = os.path.join(ROOT_DIR, "lerobot_v2.1")
    # if the folder does not exist, return an empty list
    if (
        not os.path.exists(root_v2)
        or not os.path.isdir(root_v2)
        or not os.path.exists(root_v2_1)
        or not os.path.isdir(root_v2_1)
    ):
        return DatasetListResponse(
            pushed_datasets=[],
            local_datasets=[],
        )
    # List all folders in the root directory. Keep only the directory names
    datasets_folders = [
        f for f in os.listdir(root_v2) if os.path.isdir(os.path.join(root_v2, f))
    ]
    datasets_folders += [
        f for f in os.listdir(root_v2_1) if os.path.isdir(os.path.join(root_v2_1, f))
    ]
    # Keep only directories
    api = HfApi()
    try:
        user_info = api.whoami()
    except Exception:
        logger.info("No Hugging Face token found.")
        return DatasetListResponse(
            pushed_datasets=[],
            local_datasets=[],
        )
    # Get the username or org that has write access
    username_or_orgid = parse_hf_username_or_orgid(user_info)

    # List HF datasets
    hf_datasets = list(
        api.list_datasets(author=username_or_orgid, limit=100, gated=False)
    )
    # Filter datasets that are in the local folder
    datasets = []
    for dataset in hf_datasets:
        only_name = dataset.id.split("/")[-1]
        if only_name in datasets_folders:
            datasets.append(dataset.id)

    return DatasetListResponse(
        pushed_datasets=datasets,
        local_datasets=datasets_folders,
    )


@router.post("/episode/delete")
async def delete_episode(query: DeleteEpisodeRequest):
    """
    Delete an episode from the dataset.
    Parameters:
    - episode_id: int: The episode ID to delete.
    - path: str: The path to the dataset folder.
    """

    logger.info(f"Deleting episode {query.episode_id} from {query.path}")

    try:
        dataset = Dataset(path=os.path.join(ROOT_DIR, query.path))
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="The dataset was not found or the dataset name is incorrect",
        )

    # Delete the data file
    dataset.delete_episode(episode_id=query.episode_id, update_hub=True)
    return StatusResponse(status="ok")


@router.post("/dataset/sync")
async def sync_dataset(path: str):
    # Extract dataset name et huggingface repo id from the path
    dataset = Dataset(path=os.path.join(ROOT_DIR, path))

    dataset.sync_local_to_hub()

    # Redirect to the parent directory after deletion
    return StatusResponse(status="ok")


@router.get("/dataset/download")
async def download_folder(folder_path: str):
    # Construct the full path
    full_path = os.path.join(ROOT_DIR, folder_path)

    # Check if the path exists and is a directory
    if not os.path.exists(full_path) or not os.path.isdir(full_path):
        raise HTTPException(status_code=404, detail="Folder not found")

    # Create a temporary ZIP file
    zip_path = f"{full_path}.zip"
    zip_folder(full_path, zip_path)

    # Return the ZIP file as a response
    return FileResponse(
        zip_path, filename=os.path.basename(zip_path), media_type="application/zip"
    )


@router.post("/model/video-keys", response_model=ModelVideoKeysResponse)
async def get_model_video_keys(
    request: ModelVideoKeysRequest,
) -> ModelVideoKeysResponse:
    """
    Fetch the model info from Hugging Face and return the video keys.
    """
    request.model_id = request.model_id.strip()

    from phosphobot.am import ACT, Gr00tN1

    model_type_to_class: dict[str, type[ActionModel]] = {
        "gr00t": Gr00tN1,
        "ACT": ACT,
    }
    model_class = model_type_to_class.get(request.model_type)
    if model_class is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {request.model_type}",
        )

    try:
        video_keys = model_class.fetch_and_get_video_keys(model_id=request.model_id)
        return ModelVideoKeysResponse(video_keys=video_keys)

    except Exception as e:
        logger.warning(f"No video keys found for {request.model_id}: {e}")
        return ModelVideoKeysResponse(video_keys=[])
