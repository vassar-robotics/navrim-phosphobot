"""
Integration tests for the Websocket.

```
make test_server
uv run pytest -s
```
"""

import asyncio
import json
import sys
import time

import numpy as np
import pytest
import requests  # type: ignore
import websockets
from loguru import logger

BASE_WS_URI = "ws://127.0.0.1:8080"
BASE_API_URL = "http://127.0.0.1:8080"
WEBSOCKET_TEST_TIME = 5  # seconds


# Ensure loguru logs appear in pytest output
@pytest.fixture(scope="module", autouse=True)
def configure_logger():
    """
    Plain-text logger (no ANSI colors), with a minimal format
    """
    logger.remove()
    logger.add(
        sys.stderr, colorize=False, level="DEBUG", format="{time} | {level} | {message}"
    )


@pytest.mark.asyncio
async def send_data(websocket, total_seconds, send_frequency):
    """
    Sends messages at `send_frequency` until `total_seconds` have elapsed.
    """
    sample_control_data = {
        "x": 0.001,
        "y": 0.0,
        "z": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
        "open": 0.0,
    }
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < total_seconds:
        # 1) Send the control data
        await websocket.send(json.dumps(sample_control_data))
        # 2) Sleep to maintain desired frequency
        await asyncio.sleep(1 / send_frequency)


@pytest.mark.asyncio
async def receive_data(websocket, total_seconds, nb_actions_history):
    """
    Continuously receives messages from the server until `total_seconds` have elapsed.
    Extracts nb_actions_received and appends it to `nb_actions_history`.
    """
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < total_seconds:
        try:
            # Wait for incoming message (short timeout so we keep reading often)
            status_message = await asyncio.wait_for(websocket.recv(), timeout=0.3)
            data = json.loads(status_message)

            nb_actions = data.get("nb_actions_received")
            if nb_actions is not None:
                nb_actions_history.append(nb_actions)
                logger.info(f"nb_actions_received={nb_actions}")

        except asyncio.TimeoutError:
            pass


@pytest.mark.asyncio
async def test_send_messages(send_frequency=30, total_seconds=WEBSOCKET_TEST_TIME):
    """
    - Connects to /move/teleop/ws endpoint.
    - Spawns two async tasks:
        1) Send messages at `send_frequency` for `total_seconds`.
        2) Continuously receive and parse nb_actions_received from the server.
    - Computes the mean of all nb_actions_received.
    """

    move_uri = BASE_WS_URI + "/move/teleop/ws"

    # Shared list for collecting nb_actions_received
    nb_actions_history = []

    async with websockets.connect(move_uri) as websocket:
        logger.success("[TEST] Connected to WebSocket")

        # Create and run tasks concurrently
        send_task = asyncio.create_task(
            send_data(websocket, total_seconds, send_frequency)
        )
        receive_task = asyncio.create_task(
            receive_data(websocket, total_seconds, nb_actions_history)
        )

        # Wait for both tasks to finish
        await asyncio.gather(send_task, receive_task)

    # After completion, compute average if we have data
    if nb_actions_history:
        avg_nb_actions_received = float(np.mean(nb_actions_history))
    else:
        avg_nb_actions_received = 0.0

    logger.info(
        f"[TEST_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz: {avg_nb_actions_received:.2f}"
    )

    assert avg_nb_actions_received > 5, (
        f"[TEST_FAILED] Average nb_actions_received per second: {avg_nb_actions_received}",
        "Expected an average of nb_actions_received above 5 per second",
    )

    logger.success("[TEST_SUCCESS] Websocket test completed successfully")


# Do the same with 500 Hz
@pytest.mark.asyncio
async def test_send_messages_500hz(
    send_frequency=500, total_seconds=WEBSOCKET_TEST_TIME
):
    await test_send_messages(send_frequency=send_frequency, total_seconds=total_seconds)


@pytest.mark.asyncio
async def test_send_messages_500hz_while_recording(
    send_frequency=500, total_seconds=WEBSOCKET_TEST_TIME
):
    """
    Performance test:
      1) Start recording (via HTTP POST /recording/start).
      2) Send data at 500Hz for total_seconds to /move/teleop/ws.
      3) Stop recording (via HTTP POST /recording/stop).
      4) Print the avegerage nb_actions_received per second.
    """

    ##################
    # 1) Start recording
    ##################
    dataset_name = "test_lerobot_ws_dataset"
    start_payload = {"dataset_name": dataset_name, "episode_format": "lerobot_v2"}

    # USE WEBSOCKET CONNEXION HERE
    start_response = requests.post(
        f"{BASE_API_URL}/recording/start", json=start_payload
    )
    assert (
        start_response.status_code == 200
    ), f"Failed to start recording: {start_response.text}"
    assert (
        start_response.json().get("status") == "ok"
    ), f"Recording start not ok: {start_response.text}"

    logger.info("[TEST] Recording started successfully")

    ##################
    # 2) WebSocket concurrency: send & receive
    ##################
    move_uri = BASE_WS_URI + "/move/teleop/ws"
    nb_actions_history = []

    async with websockets.connect(move_uri) as websocket:
        logger.info("[TEST] Connected to WebSocket")

        # Create tasks
        send_task = asyncio.create_task(
            send_data(websocket, total_seconds, send_frequency)
        )
        receive_task = asyncio.create_task(
            receive_data(websocket, total_seconds, nb_actions_history)
        )

        await asyncio.gather(send_task, receive_task)

    # Compute average if we have data
    if nb_actions_history:
        avg_nb_actions_received = float(np.mean(nb_actions_history))
    else:
        avg_nb_actions_received = 0.0

    logger.info(
        f"[TEST_RECORDING_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz while recording: "
        f"{avg_nb_actions_received:.2f}"
    )
    # Optional assertion for performance
    assert avg_nb_actions_received > 5, (
        f"[TEST_FAILED] Average nb_actions_received is {avg_nb_actions_received}, "
        "expected to be > 5"
    )

    ##################
    # 3) Stop recording
    ##################
    stop_payload = {"save": False, "episode_format": "lerobot_v2"}
    stop_response = requests.post(f"{BASE_API_URL}/recording/stop", json=stop_payload)
    assert (
        stop_response.status_code == 200
    ), f"Failed to stop recording: {stop_response.text}"
    logger.info("[TEST] Recording stopped successfully")

    ##################
    # 4) Print The mean of the nb_actions_received
    ##################
    logger.info(
        f"[TEST_RECORDING_PERFORMANCE_{send_frequency}Hz] Average nb_actions_received per second at {send_frequency} Hz while recording: "
        f"{avg_nb_actions_received:.2f}"
    )

    logger.success("[TEST_SUCCESS] Websocket test completed successfully")
