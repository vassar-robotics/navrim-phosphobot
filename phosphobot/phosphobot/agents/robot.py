import json
import requests
from phosphobot.workaround.db import DatabaseManager
from copilotkit import Action
from huggingface_hub import HfApi
from phosphobot.utils import get_hf_token
from phosphobot.main import get_local_ip
from phosphobot.configs import config
from loguru import logger


def get_service_url():
    return f"http://{get_local_ip()}:{config.PORT}"


def build_model_description(training_data: dict):
    api = HfApi(token=get_hf_token())
    info_file_path = api.hf_hub_download(
        repo_id=training_data["dataset_name"],
        repo_type="dataset",
        filename="meta/tasks.jsonl",
        force_download=True,
    )
    with open(info_file_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    tasks = [json.loads(line) for line in lines]
    return f"""
    The {training_data["model_name"]} model is a {training_data["model_type"]} model that was trained on the 
    {training_data["dataset_name"]} dataset. This model can perform the following tasks:
    {tasks}
    """


def build_robot_action(training_data: dict):
    return Action(
        name=training_data["model_name"].replace("/", "--"),
        handler=lambda: training_data,
        description=build_model_description(training_data),
        parameters=[],
    )


def build_robot_actions(_):
    actions = []
    trainings_data = requests.post(f"{get_service_url()}/training/models/read").json()
    logger.info(f"Trainings data: {trainings_data}")
    with DatabaseManager.get_instance() as db:
        trainings_data = db.get_trainings_by_user("default_user")[:1000]
        for training_data in trainings_data:
            if training_data["status"] == "succeeded" and training_data["dataset_name"] != "unknown/unknown":
                actions.append(build_robot_action(training_data))
    return actions
