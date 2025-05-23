import asyncio
import httpx
import numpy as np

from phosphobot.hardware.base import BaseRobot
from loguru import logger

from phosphobot.models.robot import RobotConfigStatus


class RemotePhosphobot(BaseRobot):
    """
    Class to connect to another phosphobot server using HTTP
    """

    name = "phosphobot"

    def __init__(self, ip: str, port: int, robot_id: int, **kwargs):
        """
        Initialize connectio to phosphobot.

        Args:
            ip: IP address of the robot
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.ip = ip
        self.port = port
        self.client = httpx.Client(base_url=f"http://{ip}:{port}")
        self.async_client = httpx.AsyncClient(base_url=f"http://{ip}:{port}")
        self.current_position = np.zeros(3)
        self.current_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.robot_id = robot_id
        self.connect()

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """
        Set the connection status of the robot.

        Args:
            value: True if connected, False otherwise
        """
        self._is_connected = value

    def connect(self) -> None:
        """
        Initialize communication with the phosphobot server by calling /status

        Raises:
            Exception: If the connection fails
        """
        try:
            response = self.client.get("/status")
            response.raise_for_status()
            self.is_connected = True
            logger.info(f"Connected to remote phosphobot at {self.ip}:{self.port}")
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to remote phosphobot: {e}")
            raise Exception(f"Connection failed: {e}")

    def disconnect(self) -> None:
        """
        Close the connection to the robot.
        """
        try:
            self.client.close()
            asyncio.run(self.async_client.aclose())
            self.is_connected = False
            logger.info("Disconnected from remote phosphobot")
        except Exception as e:
            logger.error(f"Failed to disconnect from remote phosphobot: {e}")
            raise Exception(f"Disconnection failed: {e}")

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        Returns:
            - state: [x, y, z, roll, pitch, yaw, gripper_state]
            - joints_position: np.array of joint positions
        """

        end_effector_position = self.client.post("/end-effector/read").json()
        joints = self.client.post("/joints/read").json()
        state = np.array(
            [
                end_effector_position["x"],
                end_effector_position["y"],
                end_effector_position["z"],
                end_effector_position["rx"],
                end_effector_position["ry"],
                end_effector_position["rz"],
                end_effector_position["open"],
            ]
        )
        joints_position = np.array(joints["angles"])

        return state, joints_position

    def set_motors_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Set the motor positions of the robot.
        """

        self.client.post(
            f"/joints/write?robot_id={self.robot_id}",
            json={"angles": positions.tolist(), "unit": "rad"},
        )

    def get_info_for_dataset(self):
        """
        Not implemented
        """
        raise NotImplementedError

    async def move_robot_absolute(
        self,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray | None,
        **kwargs,
    ) -> None:
        """
        Move the robot to the target position and orientation asynchronously.

        Args:
            target_position: Target position as [x, y, z]
            target_orientation_rad: Target orientation in radians [roll, pitch, yaw]
            **kwargs: Additional arguments
        """
        if not self.is_connected:
            logger.error("Robot is not connected")
            return

        body = {
            "x": target_position[0],
            "y": target_position[1],
            "z": target_position[2],
        }
        if target_orientation_rad is not None:
            body = {
                **body,
                "rx": target_orientation_rad[0],
                "ry": target_orientation_rad[1],
                "rz": target_orientation_rad[2],
            }

        await self.async_client.post(
            f"/move/absolute?robot_id={self.robot_id}", json=body
        )

    def status(self) -> RobotConfigStatus:
        """
        Get the status of the robot.

        Returns:
            RobotConfigStatus object
        """
        return RobotConfigStatus(
            name=self.name,
            usb_port=self.ip,
        )

    async def move_to_initial_position(self) -> None:
        """
        Move the robot to its initial position.

        This puts the robot in a stand position ready for operation.
        """
        if not self.is_connected:
            logger.error("Robot is not connected")
            return

        await self.async_client.post(f"/move/init?robot_id={self.robot_id}")

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.

        This makes the robot sit down before potentially disconnecting.
        """
        if not self.is_connected:
            logger.error("Robot is not connected")
            return
        await self.async_client.post(f"/move/sleep?robot_id={self.robot_id}")
