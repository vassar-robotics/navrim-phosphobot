import json
import sys
from pathlib import Path
from threading import Lock
import subprocess
from typing import Optional
import time

from copilotkit import Action
from huggingface_hub import HfApi
from loguru import logger
import httpx

from phosphobot.configs import config
from phosphobot.main import get_local_ip
from phosphobot.utils import get_hf_token
from phosphobot.workaround.db import DatabaseManager


class InferenceServerGuard:
    def __init__(self):
        self.lock = Lock()
        self.model_id: Optional[str] = None
        self.inference_server_process: Optional[subprocess.Popen] = None
        self.inference_client_process: Optional[subprocess.Popen] = None

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock.release()


inference_server_guard = InferenceServerGuard()


def get_service_url():
    return f"http://{get_local_ip()}:{config.PORT}"


def build_model_handler(training_data: dict):
    def handler():
        with inference_server_guard:
            model_id = training_data["model_name"].replace("--", "/")
            if inference_server_guard.inference_server_process is None:
                return {
                    "status": "failure",
                    "message": f"Inference server not launched for model {model_id}",
                }
            if inference_server_guard.model_id != model_id:
                return {
                    "status": "failure",
                    "message": (
                        f"Inference server launched for model {inference_server_guard.model_id} "
                        f"Please stop the inference server for model {inference_server_guard.model_id} before "
                        f"using the model {model_id}"
                    ),
                }
            client_path = Path(__file__).parent / "client.py"
            inference_server_guard.inference_client_process = subprocess.Popen([sys.executable, client_path])
            return {
                "status": "success",
                "message": f"Inference client launched for model {model_id}",
            }

    return handler


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
        handler=build_model_handler(training_data),
        description=build_model_description(training_data),
    )


def launch_inference_server_action(model_id: str):
    with inference_server_guard:
        model_id = model_id.replace("--", "/")
        if (
            inference_server_guard.inference_server_process is not None
            and inference_server_guard.inference_server_process.poll() is None
        ):
            if inference_server_guard.model_id == model_id:
                return {
                    "status": "success",
                    "message": f"Inference server already launched for model {model_id}",
                }
            else:
                return {
                    "status": "failure",
                    "message": (
                        f"Inference server already launched for model {inference_server_guard.model_id}"
                        f"Please stop the inference server for model {inference_server_guard.model_id} before "
                        f"launching the inference server for model {model_id}"
                    ),
                }
        # Launch the inference server
        python = sys.executable
        server_path = Path(__file__).parent / "server.py"
        inference_server_guard.inference_server_process = subprocess.Popen(
            [python, server_path, "--model_id", model_id, "--port", "8080"]
        )
        inference_server_guard.model_id = model_id
        start_time = time.time()
        while time.time() - start_time < 5 * 60:
            try:
                response = httpx.get("http://localhost:8080/health", timeout=1.0)
                if response.status_code == 200:
                    break
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            time.sleep(1)
        else:
            # Timeout reached, server didn't start properly
            graceful_terminate_process(inference_server_guard.inference_server_process)
            inference_server_guard.inference_server_process = None
            inference_server_guard.model_id = None
            return {
                "status": "failure",
                "message": f"Inference server failed to start for model {model_id} within 5 minutes",
            }
    return {"status": "success", "message": f"Inference server launched for model {model_id}"}


def graceful_terminate_process(process: subprocess.Popen):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        finally:
            process = None


def stop_inference_server_action():
    with inference_server_guard:
        if inference_server_guard.inference_client_process is not None:
            graceful_terminate_process(inference_server_guard.inference_client_process)
            inference_server_guard.inference_client_process = None
        if inference_server_guard.inference_server_process is not None:
            graceful_terminate_process(inference_server_guard.inference_server_process)
            inference_server_guard.inference_server_process = None
        inference_server_guard.model_id = None
        return {"status": "success", "message": "Inference server stopped"}


def build_robot_actions(_):
    actions = [
        Action(
            name="launch_inference_server",
            handler=launch_inference_server_action,
            description=(
                "Launch the inference server for the model. Notice that this action must be called before "
                "starting the robot. The model id is the id of the model to launch the inference server for. "
                "This operation may take a few minutes to complete, because it needs to download the model from the "
                "Hugging Face Hub and start the inference server."
            ),
            parameters=[
                {
                    "name": "model_id",
                    "type": "string",
                    "description": "The id of the model to launch the inference server for.",
                }
            ],
        ),
        Action(
            name="stop_inference_server",
            handler=stop_inference_server_action,
            description="Stop the inference server.",
        ),
    ]

    # Use DatabaseManager directly instead of making HTTP request
    try:
        with DatabaseManager.get_instance() as db:
            trainings_data = db.get_trainings_by_user("default_user")[:1000]

        logger.info(f"Trainings data: {trainings_data}")

        for training_data in trainings_data:
            if training_data["status"] == "succeeded" and training_data["dataset_name"] != "unknown/unknown":
                logger.info(f"Building robot action for model {training_data['model_name']}")
                actions.append(build_robot_action(training_data))
    except Exception as e:
        logger.error(f"Error fetching trainings data: {e}")
        # Continue without trained model actions rather than failing completely

    logger.info(f"Actions: {actions}")
    return actions
