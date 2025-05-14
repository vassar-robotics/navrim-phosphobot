import asyncio
import socket
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Literal, Optional

import numpy as np
from fastapi import WebSocket
from loguru import logger

from phosphobot.hardware import BaseRobot
from phosphobot.models import AppControlData, RobotStatus
from phosphobot.robot import RobotConnectionManager


@dataclass
class RobotState:
    last_update: datetime = datetime.now()
    last_timestamp: float = 0.0
    gripped: bool = False


class TeleopManager:
    robot_id: int | None
    rcm: RobotConnectionManager
    states: Dict[Literal["left", "right"], RobotState]
    action_counter: int
    last_report: datetime
    MOVE_TIMEOUT: float = 1.0  # seconds

    def __init__(self, rcm: RobotConnectionManager, robot_id: int | None = None):
        self.rcm = rcm
        self.states: Dict[str, RobotState] = {
            "left": RobotState(),
            "right": RobotState(),
        }
        self.action_counter = 0
        self.last_report = datetime.now()
        self.robot_id = robot_id

    def get_robot(self, source: str) -> Optional[BaseRobot]:
        """Get the appropriate robot based on source"""
        if self.robot_id is not None:
            return self.rcm.get_robot(self.robot_id)

        # Otherwise, use the source
        if source == "right":
            return self.rcm.robots[0]
        elif source == "left" and len(self.rcm.robots) > 1:
            return self.rcm.robots[1]

        return None

    async def move_init(self):
        """
        Move the robot to the initial position.
        """
        for robot in self.rcm.robots:
            robot.init_config()
            robot.enable_torque()
            # For Agilex Piper, we need to connect after enabling torque
            if robot.name == "agilex-piper":
                robot.connect()
            zero_position = np.zeros(len(robot.SERVO_IDS))
            robot.set_motors_positions(zero_position)

        # Hard block the code to wait for the robot to reach the initial position
        if any(robot.name == "agilex-piper" for robot in self.rcm.robots):
            await asyncio.sleep(2.5)
        else:
            await asyncio.sleep(0.5)

        for robot in self.rcm.robots:
            initial_position, initial_orientation_rad = robot.forward_kinematics()

            robot.initial_effector_position = initial_position
            robot.initial_effector_orientation_rad = initial_orientation_rad

    async def process_control_data(self, control_data: AppControlData) -> bool:
        """Process control data and return if it was processed"""
        state = self.states[control_data.source]

        # Check timestamp freshness
        if control_data.timestamp is not None:
            if control_data.timestamp <= state.last_timestamp:
                return False

            state.last_timestamp = control_data.timestamp

        robot = self.get_robot(control_data.source)
        if not robot:
            return False

        # Initialize robot if needed
        if (
            robot.initial_effector_position is None
            or robot.initial_effector_orientation_rad is None
        ):
            await self.move_init()

        # Convert and execute command
        (
            target_pos,
            target_orient_deg,
            target_open,
        ) = control_data.to_robot(robot_name=robot.name)

        target_orientation_rad = (
            np.deg2rad(target_orient_deg) + robot.initial_effector_orientation_rad
        )
        target_position = robot.initial_effector_position + target_pos

        loop = asyncio.get_event_loop()
        try:
            # off-load blocking move_robot into threadpool + enforce timeout
            await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    robot.move_robot,
                    target_position,
                    target_orientation_rad,
                    False,  # interpolate_trajectory
                ),
                timeout=self.MOVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"move_robot timed out after {self.MOVE_TIMEOUT}s; skipping this command"
            )
            # skip gripper & counting if move failed
            return False

        robot.control_gripper(open_command=target_open)
        robot.update_object_gripping_status()

        self.action_counter += 1
        return True

    async def send_status_updates(self, websocket: Optional[WebSocket] = None):
        """Generate and optionally send status updates"""
        updates = []
        now = datetime.now()

        for source, state in self.states.items():
            robot = self.get_robot(source)
            if robot and (now - state.last_update).total_seconds() > 0.033:
                if state.gripped != robot.is_object_gripped:
                    state.gripped = robot.is_object_gripped
                    updates.append(
                        RobotStatus(
                            is_object_gripped=state.gripped,
                            is_object_gripped_source=source,
                            nb_actions_received=self.action_counter,
                        )
                    )
                state.last_update = now

        # Send periodic action count
        if (now - self.last_report).total_seconds() > 1:
            updates.append(RobotStatus(nb_actions_received=self.action_counter))
            self.action_counter = 0
            self.last_report = now

        # Send updates if websocket is provided
        if websocket:
            for update in updates:
                await websocket.send_text(update.model_dump_json())

        return updates


# UDP server implementation
class UDPServer:
    def __init__(self, rcm: RobotConnectionManager):
        self.rcm = rcm
        self.manager = TeleopManager(rcm)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5055))
        self.sock.setblocking(False)
        self.running = False

    async def start(self):
        self.running = True
        logger.info("Starting UDP server on port 5005")
        loop = asyncio.get_event_loop()

        while self.running:
            try:
                data, addr = await loop.sock_recv(self.sock, 1024)
                try:
                    control_data = AppControlData.model_validate_json(data)
                    await self.manager.process_control_data(control_data)
                    # Generate status updates and send back to client
                    updates = await self.manager.send_status_updates()
                    for update in updates:
                        json_data = update.model_dump_json()
                        encoded = json_data.encode("utf-8")
                        await loop.sock_sendto(self.sock, encoded, addr)
                except Exception as e:
                    logger.error(f"UDP processing error: {e}")
            except BlockingIOError:
                await asyncio.sleep(0.01)

    def stop(self):
        self.running = False
        self.sock.close()
