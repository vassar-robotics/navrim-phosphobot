import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import time
from typing import Dict, Literal, Optional, Tuple, cast

import numpy as np
from fastapi import WebSocket
from loguru import logger
from pydantic import ValidationError

from phosphobot.hardware import BaseRobot, BaseManipulator
from phosphobot.models import AppControlData, RobotStatus, UDPServerInformationResponse
from phosphobot.robot import RobotConnectionManager
from phosphobot.utils import get_local_network_ip


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
    MAX_INSTRUCTIONS_PER_SEC: int = 200

    def __init__(self, rcm: RobotConnectionManager, robot_id: int | None = None):
        self.rcm = rcm
        self.states: Dict[str, RobotState] = {
            "left": RobotState(),
            "right": RobotState(),
        }
        self.action_counter = 0
        self.last_report = datetime.now()
        self.robot_id = robot_id

        # rate limiting window
        self._window_start: datetime = datetime.now()
        self._instr_in_window: int = 0

    def allow_instruction(self) -> bool:
        """Simple 1-second sliding window rate limiter."""
        now = datetime.now()
        if (now - self._window_start).total_seconds() >= 1.0:
            self._window_start = now
            self._instr_in_window = 0

        if self._instr_in_window < self.MAX_INSTRUCTIONS_PER_SEC:
            self._instr_in_window += 1
            return True
        return False

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

    async def move_init(self, robot_id: int | None = None) -> None:
        """
        Move the robot to the initial position.
        """
        for i, robot in enumerate(self.rcm.robots):
            if robot_id is not None and i != robot_id:
                continue
            robot.init_config()
            robot.enable_torque()
            # For Agilex Piper, we need to connect after enabling torque
            if robot.name == "agilex-piper":
                robot.connect()
            await robot.move_to_initial_position()

        # Hard block the code to wait for the robot to reach the initial position
        if any(robot.name == "agilex-piper" for robot in self.rcm.robots):
            await asyncio.sleep(2.5)
        else:
            await asyncio.sleep(0.5)

        for i, robot in enumerate(self.rcm.robots):
            if robot_id is not None and i != robot_id:
                continue
            if hasattr(robot, "forward_kinematics"):
                initial_position, initial_orientation_rad = robot.forward_kinematics()
                robot.initial_position = initial_position
                robot.initial_orientation_rad = initial_orientation_rad

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
        if isinstance(robot, BaseManipulator):
            if robot.initial_position is None or robot.initial_orientation_rad is None:
                await self.move_init()

            # Convert and execute command
            (
                target_pos,
                target_orient_deg,
                target_open,
            ) = control_data.to_robot(robot_name=robot.name)

        target_orientation_rad = (
            np.deg2rad(target_orient_deg) + robot.initial_orientation_rad
        )
        target_position = robot.initial_position + target_pos

        # if robot.is_moving, wait for it to stop
        start_wait_time = time.perf_counter()
        while (
            robot.is_moving
            and time.perf_counter() - start_wait_time < self.MOVE_TIMEOUT
        ):
            await asyncio.sleep(0.0001)

        try:
            # off-load blocking move_robot into threadpool + enforce timeout
            await asyncio.wait_for(
                robot.move_robot_absolute(target_position, target_orientation_rad),
                timeout=self.MOVE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"move_robot timed out after {self.MOVE_TIMEOUT}s; skipping this command"
            )
            # skip gripper & counting if move failed
            return False

        if isinstance(robot, BaseManipulator):
            robot.control_gripper(open_command=target_open)
            robot.update_object_gripping_status()

        self.action_counter += 1
        return True

    async def send_status_updates(
        self, websocket: Optional[WebSocket] = None
    ) -> list[RobotStatus]:
        """Generate and optionally send status updates"""
        updates = []
        now = datetime.now()

        for source, state in self.states.items():
            robot = self.get_robot(source)
            if robot and (now - state.last_update).total_seconds() > 0.033:
                if isinstance(robot, BaseManipulator):
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


udp_server = None


class _TeleopProtocol(asyncio.DatagramProtocol):
    def __init__(self, manager: TeleopManager):
        self.manager = manager
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport):
        self.transport = cast(asyncio.DatagramTransport, transport)
        sockname = transport.get_extra_info("sockname")
        logger.info(f"UDP socket opened on {sockname}")

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        # fire-and-forget per-packet handling
        asyncio.create_task(self._handle(data, addr))

    async def _handle(self, data: bytes, addr: Tuple[str, int]):
        if self.transport is None:
            logger.error("Transport is None, cannot handle datagram")
            return

        if not self.manager.allow_instruction():
            err = {
                "error": "rate_limited",
                "detail": f"Exceeded {self.manager.MAX_INSTRUCTIONS_PER_SEC} msgs/sec",
            }
            self.transport.sendto(json.dumps(err).encode("utf-8"), addr)
            return

        # 1) decode
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            err = {"error": "invalid_encoding", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"Decoding error from {addr}: {e}")
            return

        # 2) parse JSON
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as e:
            err = {"error": "invalid_json", "detail": e.msg}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"JSON parse error from {addr}: {e.msg}")
            return

        # 3) validate schema
        try:
            control = AppControlData.model_validate(raw)
        except ValidationError as e:
            err = {"error": "validation_error", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.error(f"Schema validation failed from {addr}: {e}")
            return

        # 4) process and respond
        try:
            await self.manager.process_control_data(control)
            updates = await self.manager.send_status_updates()
            for u in updates:
                self.transport.sendto(u.model_dump_json().encode(), addr)
        except Exception as e:
            err = {"error": "internal_server_error", "detail": str(e)}
            self.transport.sendto(json.dumps(err).encode(), addr)
            logger.exception("Error while processing control data")


class UDPServer:
    def __init__(self, rcm: RobotConnectionManager):
        self.manager = TeleopManager(rcm)
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.protocol: Optional[_TeleopProtocol] = None
        self.bound_port: Optional[int] = None

    async def init(
        self, port: Optional[int] = None, restart: bool = False
    ) -> UDPServerInformationResponse:
        """
        Initialize (or re-init) the UDP server via asyncio.create_datagram_endpoint.
        Returns the bound host/port.
        """
        if self.transport is not None and not restart:
            host, bound_port = self.transport.get_extra_info("sockname")
            return UDPServerInformationResponse(host=host, port=bound_port)

        loop = asyncio.get_running_loop()
        local_ip = get_local_network_ip()

        # choose port
        if port is None:
            for p in range(5000, 6000):
                try:
                    transport, protocol = await loop.create_datagram_endpoint(
                        lambda: _TeleopProtocol(self.manager),
                        local_addr=(local_ip, p),
                    )
                    self.transport = transport
                    self.protocol = protocol
                    self.bound_port = p
                    logger.info(f"Bound UDP server to {local_ip}:{p}")
                    break
                except OSError:
                    continue
            else:
                raise RuntimeError("Could not bind to any port between 5000 and 6000")
        else:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _TeleopProtocol(self.manager),
                local_addr=(local_ip, port),
            )
            self.transport = transport
            self.protocol = protocol
            self.bound_port = port
            logger.info(f"Bound UDP server to {local_ip}:{port}")

        host, bound_port = self.transport.get_extra_info("sockname")
        return UDPServerInformationResponse(host=host, port=bound_port)

    def stop(self) -> None:
        """
        Close the transport; no more packets will be received.
        """
        if self.transport:
            self.transport.close()
            logger.info("UDP server transport closed")
            self.transport = None
            self.protocol = None
            self.bound_port = None


@lru_cache()
def get_udp_server() -> UDPServer:
    """
    Get the UDP server instance.
    If it doesn't exist, create it.
    """
    from phosphobot.robot import get_rcm

    global udp_server
    if udp_server is None:
        udp_server = UDPServer(get_rcm())
    return udp_server
