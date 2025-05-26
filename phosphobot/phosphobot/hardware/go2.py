import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional
import concurrent.futures
import threading
import time

import numpy as np
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from loguru import logger

from phosphobot.hardware.base import BaseMobileRobot
from phosphobot.models import RobotConfigStatus


@dataclass
class MovementCommand:
    position: np.ndarray
    orientation: np.ndarray | None = None


class UnitreeGo2(BaseMobileRobot):
    name = "unitree-go2"

    UNITREE_MAC_PREFIXES = ["78:22:88"]

    def __init__(self, ip: str = "192.168.1.42", max_history_len: int = 100, **kwargs):
        """
        Initialize the UnitreeGo2 robot.

        Args:
            ip: IP address of the robot
            **kwargs: Additional keyword arguments
        """
        self.ip = ip
        self.conn = None
        self.current_position = np.zeros(3)  # [x, y, z]
        self.current_orientation = np.zeros(3)  # [roll, pitch, yaw]
        self._is_connected = False
        self._connection_loop = None
        self._connection_thread = None
        self._loop_thread = None
        self._shutdown_event = threading.Event()

        # Track movement instructions
        self.movement_queue: Deque[MovementCommand] = deque(maxlen=max_history_len)
        super().__init__(**kwargs)

    @property
    def is_connected(self) -> bool:
        """
        Check if the robot is connected.

        Returns:
            bool: True if connected, False otherwise
        """
        return (
            self._is_connected
            and self.conn is not None
            and getattr(self.conn, "isConnected", False)
        )

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        """
        Set the connection status of the robot.

        Args:
            value: True if connected, False otherwise
        """
        if value and self.conn is None:
            raise ValueError("Cannot set is_connected to True without a connection")
        self._is_connected = value

    def _create_event_loop_thread(self):
        """Create a dedicated event loop thread for WebRTC operations."""

        def loop_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._connection_loop = loop

            try:
                # Keep the loop running until shutdown
                loop.run_until_complete(self._loop_runner())
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()

                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )

                    loop.close()
                except Exception as e:
                    logger.error(f"Error cleaning up event loop: {e}")

        self._loop_thread = threading.Thread(target=loop_thread, daemon=True)
        self._loop_thread.start()

        # Wait for loop to be created
        max_wait = 5
        start_time = time.time()
        while self._connection_loop is None and (time.time() - start_time) < max_wait:
            time.sleep(0.1)

        if self._connection_loop is None:
            raise RuntimeError("Failed to create event loop")

    async def _loop_runner(self):
        """Keep the event loop alive until shutdown."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

    def _run_async_safely(self, coro):
        """
        Helper method to run async code safely in the dedicated event loop.

        Args:
            coro: The coroutine to run

        Returns:
            The result of the coroutine
        """
        if self._connection_loop is None:
            self._create_event_loop_thread()

        # Use the dedicated event loop
        future = asyncio.run_coroutine_threadsafe(coro, self._connection_loop)
        try:
            return future.result(timeout=30)  # 30 second timeout
        except concurrent.futures.TimeoutError:
            logger.error("Async operation timed out")
            raise
        except Exception as e:
            logger.error(f"Async operation failed: {e}")
            raise

    async def _connect_async(self):
        """
        Async implementation of the connection logic.
        """
        try:
            # Create connection and connect
            self.conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip)
            if self.conn is None:
                raise Exception("Failed to create WebRTC connection")

            await self.conn.connect()

            # Wait a bit for connection to stabilize
            await asyncio.sleep(2)

            # Check if connection is actually established
            if not getattr(self.conn, "isConnected", False):
                raise Exception("WebRTC connection failed to establish")

            # Check if in normal mode and switch if needed
            try:
                response = await asyncio.wait_for(
                    self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["MOTION_SWITCHER"], {"api_id": 1001}
                    ),
                    timeout=10.0,
                )

                if response["data"]["header"]["status"]["code"] == 0:
                    data = json.loads(response["data"]["data"])
                    current_motion_switcher_mode = data["name"]

                    if current_motion_switcher_mode != "normal":
                        logger.info(
                            f"Switching from {current_motion_switcher_mode} to 'normal' mode"
                        )
                        await asyncio.wait_for(
                            self.conn.datachannel.pub_sub.publish_request_new(
                                RTC_TOPIC["MOTION_SWITCHER"],
                                {"api_id": 1002, "parameter": {"name": "normal"}},
                            ),
                            timeout=10.0,
                        )
                        # Wait for mode switch
                        await asyncio.sleep(5)
                else:
                    logger.warning("Failed to get motion switcher status")
            except asyncio.TimeoutError:
                logger.warning("Timeout while checking/setting motion mode")
            except Exception as e:
                logger.warning(f"Error while setting motion mode: {e}")

        except Exception as e:
            # Clean up connection on failure
            if self.conn:
                try:
                    await asyncio.wait_for(self.conn.disconnect(), timeout=5.0)
                except:
                    pass
                self.conn = None
            raise e

    def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method creates a WebRTC connection to the robot.

        Raises:
            Exception: If the connection fails
        """
        try:
            # First verify the IP is reachable (basic network check)
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            try:
                # Try to connect to common robot ports
                result = sock.connect_ex(
                    (self.ip, 8081)
                )  # Common WebRTC signaling port
                if result != 0:
                    logger.warning(
                        f"Port 8081 not reachable on {self.ip}, but continuing..."
                    )
            except Exception as e:
                logger.warning(f"Network check failed: {e}, but continuing...")
            finally:
                sock.close()

            self._run_async_safely(self._connect_async())
            self.is_connected = True
            logger.info("Successfully connected to UnitreeGo2")

        except Exception as e:
            logger.error(f"Failed to connect to UnitreeGo2: {e}")
            self.is_connected = False
            raise Exception(f"Failed to connect to UnitreeGo2: {e}")

    def disconnect(self) -> None:
        """
        Close the connection to the robot.
        """
        try:
            # Set shutdown flag
            self._shutdown_event.set()

            if self.conn:
                try:
                    # Disconnect the WebRTC connection
                    if self._connection_loop and not self._connection_loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self.conn.disconnect(), self._connection_loop
                        )
                        future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"Error during WebRTC disconnect: {e}")

                self.conn = None

            # Clean up the event loop thread
            if self._loop_thread and self._loop_thread.is_alive():
                try:
                    self._loop_thread.join(timeout=2)
                except Exception as e:
                    logger.warning(f"Error joining loop thread: {e}")

            self.is_connected = False
            self._connection_loop = None
            self._loop_thread = None
            logger.info("Disconnected from UnitreeGo2")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            # Force cleanup
            self.conn = None
            self.is_connected = False
            self._connection_loop = None

    def get_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        Returns:
            - state: Last movement command position [x, y, z] and orientation [roll, pitch, yaw]
            - joints_position: Empty array as we're not tracking joint positions
        """
        # Return the last movement command as observation
        # Combine position and orientation into a 6D state vector
        state = np.concatenate([self.current_position, self.current_orientation])
        # Return empty array for joints since we're not tracking them
        joints_position = np.array([])

        return state, joints_position

    def set_motors_positions(
        self, positions: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Not implemented
        """
        pass

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
        if not self.is_connected or self.conn is None:
            logger.error("Robot is not connected")
            return

        try:
            self.movement_queue.append(
                MovementCommand(
                    position=target_position.copy(),
                    orientation=target_orientation_rad.copy()
                    if target_orientation_rad is not None
                    else None,
                )
            )

            self.current_position = target_position

            # Send move command with timeout
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {
                        "api_id": SPORT_CMD["Move"],
                        "parameter": {
                            "x": float(target_position[0]),
                            "y": float(target_position[1]),
                            "z": float(target_position[2]),
                        },
                    },
                ),
                timeout=10.0,
            )

            if target_orientation_rad is not None:
                self.current_orientation = target_orientation_rad
                target_orientation_deg = np.degrees(target_orientation_rad[2])

                # Send orientation command with timeout
                await asyncio.wait_for(
                    self.conn.datachannel.pub_sub.publish_request_new(
                        RTC_TOPIC["SPORT_MOD"],
                        {
                            "api_id": SPORT_CMD["Euler"],
                            "parameter": {
                                "x": float(target_orientation_rad[0]),
                                "y": float(target_orientation_rad[1]),
                                "z": float(target_orientation_deg),
                            },
                        },
                    ),
                    timeout=10.0,
                )

            logger.info(
                f"Moved to position {target_position} with orientation {target_orientation_rad}"
            )

        except asyncio.TimeoutError:
            logger.error("Move command timed out")
            raise
        except Exception as e:
            logger.error(f"Error during move: {e}")
            raise

    @classmethod
    def from_ip(cls, ip: str, **kwargs) -> Optional["UnitreeGo2"]:
        """
        Create a UnitreeGo2 instance from an IP address.

        Args:
            ip: IP address of the robot
            **kwargs: Additional arguments

        Returns:
            UnitreeGo2 instance or None if the connection fails
        """
        try:
            robot = cls(ip=ip, **kwargs)
            return robot
        except Exception as e:
            logger.error(f"Failed to connect to UnitreeGo2 at {ip}: {e}")
            return None

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
        if not self.is_connected or self.conn is None:
            logger.error("Robot is not connected")
            return

        try:
            # Stand up the robot
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StandUp"], "parameter": {"data": True}},
                ),
                timeout=15.0,
            )

            # Wait for the robot to stand up
            await asyncio.sleep(5)

            # Reset position tracking
            self.current_position = np.zeros(3)
            self.current_orientation = np.zeros(3)

            # Clear movement history
            self.movement_queue.clear()

            logger.info("Robot moved to initial position")

        except asyncio.TimeoutError:
            logger.error("Move to initial position timed out")
            raise
        except Exception as e:
            logger.error(f"Error moving to initial position: {e}")
            raise

    def move_to_initial_position_sync(self) -> None:
        """
        Synchronous wrapper for move_to_initial_position.
        Move the robot to its initial position.
        """
        self._run_async_safely(self.move_to_initial_position())

    async def move_to_sleep(self) -> None:
        """
        Move the robot to its sleep position.

        This makes the robot sit down before potentially disconnecting.
        """
        if not self.is_connected or self.conn is None:
            logger.error("Robot is not connected")
            return

        try:
            # Make the robot sit
            await asyncio.wait_for(
                self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Sit"], "parameter": {"data": True}},
                ),
                timeout=10.0,
            )

            # Wait for the robot to sit
            await asyncio.sleep(3)

            logger.info("Robot moved to sleep position")

        except asyncio.TimeoutError:
            logger.error("Move to sleep position timed out")
            raise
        except Exception as e:
            logger.error(f"Error moving to sleep position: {e}")
            raise

    def move_to_sleep_sync(self) -> None:
        """
        Synchronous wrapper for move_to_sleep.
        Move the robot to its sleep position.
        """
        self._run_async_safely(self.move_to_sleep())

    def move_robot_absolute_sync(
        self,
        target_position: np.ndarray,
        target_orientation_rad: np.ndarray | None,
        **kwargs,
    ) -> None:
        """
        Synchronous wrapper for move_robot_absolute.
        Move the robot to the target position and orientation.
        """
        self._run_async_safely(
            self.move_robot_absolute(target_position, target_orientation_rad, **kwargs)
        )

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if hasattr(self, "_is_connected") and self._is_connected:
                self.disconnect()
        except:
            pass
