import atexit
import json
import os
import asyncio
from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Union

import numpy as np
import pybullet as p  # type: ignore
from loguru import logger
from phosphobot.models import (
    BaseRobotConfig,
    BaseRobotInfo,
    FeatureDetails,
)
from scipy.spatial.transform import Rotation as R  # type: ignore
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.configs import SimulationMode, config
from phosphobot.models import RobotConfigStatus
from phosphobot.utils import (
    euler_from_quaternion,
    get_resources_path,
    step_simulation,
)


class AxisRobot:
    """
    Used to place the robots in a grid.
    """

    def __init__(self):
        # Create a grid of (x, y, 0) positions with a step of 0.4
        self.grid = []
        for x in np.arange(0, 10, 1):
            for y in np.arange(0, 10, 1):
                self.grid.append([x, y, 0])
        self.grid_index = 0

    def new_position(self):
        if self.grid_index >= len(self.grid):
            self.grid_index = 0
        axis = self.grid[self.grid_index]
        self.grid_index += 1
        return axis


axis_robot = AxisRobot()


class BaseRobot(ABC):
    """
    Abstract class for hardware interface.
    This class defines
    - abstract methods that must be implemented by the hardware interface.
    - common methods that can be used by the hardware interface.
    """

    name: str

    # Path to the URDF file of the robot
    URDF_FILE_PATH: str
    # Axis and orientation of the robot. This depends on the default
    # orientation of the URDF file
    AXIS_ORIENTATION: List[int]

    SERIAL_ID: str

    DEVICE_NAME: Optional[str]
    DEVICE_PID: int
    REGISTERED_SERIAL_ID: List[str]

    # List of servo IDs, used to write and read motor positions
    # They are in the same order as the joint links in the URDF file
    SERVO_IDS: List[int]

    CALIBRATION_POSITION: list[float]  # same size as SERVO_IDS
    SLEEP_POSITION: list[float] | None = None
    RESOLUTION: int
    # The effector is the gripper
    END_EFFECTOR_LINK_INDEX: int

    # calibration config: offsets, signs, pid values
    config: BaseRobotConfig

    # status variables
    is_connected: bool = False
    is_moving: bool = False
    _add_debug_lines: bool = False

    # Gripper status. This is the value of the last closing command.
    GRIPPER_JOINT_INDEX: int
    closing_gripper_value = 0.0
    is_object_gripped = False

    # Used to keep track of the calibration sequence
    calibration_current_step: int = 0
    calibration_max_steps: int = 3

    @abstractmethod
    def connect(self) -> None:
        """
        Initialize communication with the robot.

        This method is called after the __init__ method.

        raise: Exception if the setup fails. For example, if the robot is not plugged in.
            This Exception will be caught by the __init__ method.
        """
        raise NotImplementedError("The robot setup method must be implemented.")

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the robot.

        This method is called on __del__ to disconnect the robot.
        """
        raise NotImplementedError("The robot setup method must be implemented.")

    @abstractmethod
    def enable_torque(self) -> None:
        """
        Enable all motor torque.

        raise: Exception if the routine has not been implemented
        """
        raise NotImplementedError("The robot enable torque must be implemented.")

    @abstractmethod
    def disable_torque(self) -> None:
        """
        Disable all motor torque.

        raise: Exception if the routine has not been implemented
        """
        raise NotImplementedError("The robot enable torque must be implemented.")

    @abstractmethod
    def read_motor_torque(self, servo_id: int) -> float | None:
        """
        Read the torque of a motor

        raise: Exception if the routine has not been implemented
        """
        raise NotImplementedError("The robot enable torque must be implemented.")

    @abstractmethod
    def read_motor_voltage(self, servo_id: int) -> float | None:
        """
        Read the voltage of a motor

        raise: Exception if the routine has not been implemented
        """
        raise NotImplementedError("The robot enable torque must be implemented.")

    @abstractmethod
    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        """
        Move the motor to the specified position.

        Args:
            servo_id: The ID of the motor to move.
            units: The position to move the motor to. This is in the range 0 -> (self.RESOLUTION -1).
        Each position is mapped to an angle.
        """
        raise NotImplementedError("The robot write motor position must be implemented.")

    @abstractmethod
    def read_motor_position(self, servo_id: int, **kwargs) -> int | None:
        """
        Read the position of the motor. This should return the position in motor units.
        """
        raise NotImplementedError("The robot read motor position must be implemented.")

    @abstractmethod
    def calibrate_motors(self, **kwargs) -> None:
        """
        This is called during the calibration phase of the robot.
        It sets the offset of all motors to self.RESOLUTION/2.
        """
        raise NotImplementedError("calibrate_motors must be implemented.")

    def read_group_motor_position(self) -> np.ndarray:
        """
        Read the position of all motors in the group.
        """
        raise NotImplementedError("read_group_motor_position must be implemented.")

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool
    ) -> None:
        """
        Write the positions to the motors in the group.
        """
        raise NotImplementedError("write_group_motor_position must be implemented.")

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["BaseRobot"]:
        """
        Return the robot class from the port information.
        """
        logger.error(
            f"For automatic detection of {cls.__name__}, the method from_port must be implemented. Skipping autodetection."
        )
        return None

    def __init__(
        self,
        device_name: str | None = None,
        serial_id: str | None = None,
        only_simulation: bool = False,
        reset_simulation: bool = False,
        axis: List[float] | None = None,
        add_debug_lines: bool = False,
        show_debug_link_indices: bool = True,
    ):
        """
        Args:
            device_name: The path to the USB device. If None, the default value is used.
            test: Special flag used in the tests to avoid connecting to the robot.
            add_debug_lines: Add debug lines in the simulation to show the target position and orientation.
                Warning! This adds A BIG overhead to the simulation and should be used only for debugging.
                DISABLE THIS IN PRODUCTION.
        """

        if axis is None:
            axis = axis_robot.new_position()

        # When creating a new robot, you should add default values for these
        # These values depends on the hardware
        assert (
            self.CALIBRATION_POSITION is not None
        ), "CALIBRATION_POSITION must be defined in the class"
        assert self.RESOLUTION is not None, "RESOLUTION must be defined in the class"
        assert self.SERVO_IDS is not None, "SERVO_IDS must be defined in the class"

        if serial_id is not None:
            self.SERIAL_ID = serial_id
        else:
            logger.warning("No serial ID provided.")

        if device_name is not None:
            # Override the device name if provided
            self.DEVICE_NAME = device_name

        self._add_debug_lines = add_debug_lines

        if reset_simulation:
            p.resetSimulation()

        logger.info(f"Loading URDF file: {self.URDF_FILE_PATH}")
        if not os.path.exists(self.URDF_FILE_PATH):
            raise FileNotFoundError(
                f"URDF file not found: {self.URDF_FILE_PATH}\nCurrent path: {os.getcwd()}"
            )
        self.p_robot_id = p.loadURDF(
            self.URDF_FILE_PATH,
            axis,
            self.AXIS_ORIENTATION,
            useFixedBase=True,
            flags=p.URDF_MAINTAIN_LINK_ORDER,
        )

        # Find actuated joints (in case some are not)
        actuated_joints = []
        for i in range(p.getNumJoints(self.p_robot_id)):
            joint_info = p.getJointInfo(self.p_robot_id, i)
            joint_type = joint_info[2]
            # type_to_label = {
            #     p.JOINT_REVOLUTE: "revolute",
            #     p.JOINT_PRISMATIC: "prismatic",
            #     p.JOINT_SPHERICAL: "spherical",
            #     p.JOINT_FIXED: "fixed",
            # }
            # logger.debug(
            #     f"[{self.name}] Joint {i} index {joint_info[0]}: {joint_info[1]} - {type_to_label[joint_type]} - {joint_info[8]} - {joint_info[9]}"
            # )
            # Consider only revolute joints
            if joint_type in [p.JOINT_REVOLUTE]:
                actuated_joints.append(i)
        self.actuated_joints = actuated_joints

        # Set the motors to postion
        p.setJointMotorControlArray(
            self.p_robot_id,
            self.actuated_joints,
            p.POSITION_CONTROL,
        )

        # Display link indices
        if show_debug_link_indices:
            for i in range(20):
                link = p.getLinkState(self.p_robot_id, i)
                if link is None:
                    break
                logger.debug(
                    f"[{self.name}] Link {i}: position {link[0]} orientation {link[1]}"
                )
                p.addUserDebugText(
                    text=f"{i}",
                    textPosition=link[0],
                    textColorRGB=[1, 0, 0],
                    lifeTime=0,
                )

        self.num_actuated_joints = len(self.actuated_joints)
        # Gripper motor is the last one :
        self.gripper_servo_id = self.SERVO_IDS[-1]

        joint_infos = [
            p.getJointInfo(self.p_robot_id, i)
            for i in range(p.getNumJoints(self.p_robot_id))
        ]

        self.lower_joint_limits = [info[8] for info in joint_infos]
        self.upper_joint_limits = [info[9] for info in joint_infos]

        (
            self.initial_effector_position,
            self.initial_effector_orientation_rad,
        ) = self.forward_kinematics()

        self.gripper_initial_angle = p.getJointState(
            bodyUniqueId=self.p_robot_id,
            jointIndex=self.END_EFFECTOR_LINK_INDEX,
        )[0]

        try:
            if not only_simulation:
                self.connect()
                # Register the disconnect method to be called on exit
                atexit.register(self.move_to_sleep_sync)
            else:
                logger.info("Only simulation: Not connecting to the robot.")
                self.is_connected = False
        except Exception as e:
            logger.warning(
                f"""Error when connecting to robot {self.__class__.__name__}: {e}
Make sure the robot is connected and powered on.
Falling back to simulation mode.
"""
            )
            logger.info("Simulation mode enabled.")
            self.is_connected = False

        # Once connected, we read the calibration from the motors
        if self.is_connected:
            # Init calibration config
            self.init_config()
        else:
            logger.warning("No robot connected")
            config = self.get_default_base_robot_config(
                voltage="6V", raise_if_none=True
            )
            if config is None:
                raise FileNotFoundError(
                    "No default config found. Please provide a valid config file."
                )
            self.config = config

    def __del__(self):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.move_to_sleep())
            else:
                loop.run_until_complete(self.move_to_sleep())
        except RuntimeError:
            # If no event loop exists
            asyncio.run(self.move_to_sleep())

    @property
    def bundled_config_file(self) -> str:
        """
        The file where the bundled calibration config is stored.
        """
        if not hasattr(self, "SERIAL_ID"):
            return str(get_resources_path() / f"default/{self.name}.json")
        relative_path = f"calibration/{self.name}_{self.SERIAL_ID}_config.json"
        return str(get_resources_path() / relative_path)

    def get_default_base_robot_config(
        self,
        voltage: str,
        raise_if_none: bool = False,
    ) -> Union[BaseRobotConfig, None]:
        json_filename = get_resources_path() / "default" / f"{self.name}-{voltage}.json"
        try:
            with open(json_filename, "r") as f:
                data = json.load(f)
            return BaseRobotConfig(**data)
        except FileNotFoundError:
            if raise_if_none:
                logger.error(f"Default config file not found: {json_filename}")
                raise FileNotFoundError(
                    f"Default config file not found: {json_filename}"
                )
        except Exception as e:
            logger.error(f"Error loading default config: {e}")
            if raise_if_none:
                raise e

        return None

    def read_gripper_torque(self) -> np.int32:
        """
        Read the torque of the gripper
        Returns:
            gripper torque value as np.int32
        """
        # Read present position for each motor
        if self.is_connected:
            reading_gripper_torque = self.read_motor_torque(self.gripper_servo_id)
            if reading_gripper_torque is None:
                logger.warning("None torque value for gripper motor ")
                current_gripper_torque = np.int32(0)
            else:
                current_gripper_torque = np.int32(reading_gripper_torque)
            return current_gripper_torque

        # If the robot is not connected, we use the pybullet simulation
        # Retrieve joint angles using getJointStates
        joint_state: list = p.getJointState(
            bodyUniqueId=self.p_robot_id,
            jointIndex=self.END_EFFECTOR_LINK_INDEX,
        )
        # Joint torque is in the 4th element of the joint state tuple
        current_gripper_torque = joint_state[3]
        if not isinstance(current_gripper_torque, float):
            logger.warning("None torque value for gripper motor ")
            current_gripper_torque = np.int32(0)

        return current_gripper_torque

    async def move_to_sleep(self, disconnect: bool = True):
        """
        Cleanup the robot. This method is called on exit.
        """
        if self.is_connected:
            if self.SLEEP_POSITION:
                try:
                    self.set_motors_positions(
                        q_target_rad=np.array(self.SLEEP_POSITION), enable_gripper=True
                    )
                except Exception:
                    pass
            await asyncio.sleep(0.7)
            self.disable_torque()
            await asyncio.sleep(0.1)
            if disconnect:
                self.disconnect()

    def move_to_sleep_sync(self):
        asyncio.run(self.move_to_sleep())

    def _units_vec_to_radians(self, units: np.ndarray) -> np.ndarray:
        """
        Convert from motor discrete units (0 -> RESOLUTION) to radians
        """
        return (
            (units - self.config.servos_offsets[: len(units)])
            * self.config.servos_offsets_signs[: len(units)]
            * ((2 * np.pi) / (self.RESOLUTION - 1))
        )

    def _radians_vec_to_units(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert from radians to motor discrete units (0 -> RESOLUTION)

        Note: The result can exceed the resolution of the motor, in the case of a continuous rotation motor.
        """

        x = (
            radians
            * self.config.servos_offsets_signs[: len(radians)]
            * ((self.RESOLUTION - 1) / (2 * np.pi))
        ) + self.config.servos_offsets[: len(radians)]
        return x.astype(int)

    def _radians_motor_to_units(self, radians: float, servo_id: int) -> int:
        """
        Convert a single q position from radians to motor discrete units (0 -> RESOLUTION)

        Note: The result can exceed the resolution of the motor, in the case of a continuous rotation motor.
        """
        offset_id = self.SERVO_IDS.index(servo_id)

        x = (
            int(
                radians
                * self.config.servos_offsets_signs[offset_id]
                * ((self.RESOLUTION - 1) / (2 * np.pi))
            )
            + self.config.servos_offsets[offset_id]
        )

        return int(x)

    def inverse_kinematics(
        self,
        target_position_cartesian: np.ndarray,
        target_orientation_quaternions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute the inverse kinematics of the robot.
        Returns the joint angles in radians.

        If the IK with the orientation results in the robot not moving, we try without the orientation.
        """
        if self.name == "koch-v1.1":
            # In the URDF of Koch 1.1, the limits are fucked up. So we add
            # limits in the inverse kinematics to make it work.

            # In Koch 1.1, the gripper_opening joint in the URDF file is set to -1.74 ; 1.74 even tough it's
            # actually the yaw join link is supposed to have these limits.
            # You need to keep this otherwise the inverse kinematics will not work.
            target_q_rad = p.calculateInverseKinematics(
                self.p_robot_id,
                self.END_EFFECTOR_LINK_INDEX,
                targetPosition=target_position_cartesian,
                targetOrientation=target_orientation_quaternions,
                restPoses=[0] * len(self.lower_joint_limits),
                lowerLimits=self.lower_joint_limits,
                upperLimits=self.upper_joint_limits,
                jointRanges=[
                    abs(up - low)
                    for up, low in zip(self.upper_joint_limits, self.lower_joint_limits)
                ],
                maxNumIterations=50,
                residualThreshold=1e-9,
            )
        elif self.name == "wx-250s":
            # More joints means longer IK to find the right position
            target_q_rad = p.calculateInverseKinematics(
                self.p_robot_id,
                self.END_EFFECTOR_LINK_INDEX,
                targetPosition=target_position_cartesian,
                targetOrientation=target_orientation_quaternions,
                restPoses=[0] * len(self.lower_joint_limits),
                lowerLimits=self.lower_joint_limits,
                upperLimits=self.upper_joint_limits,
                jointRanges=[
                    abs(up - low)
                    for up, low in zip(self.upper_joint_limits, self.lower_joint_limits)
                ],
                maxNumIterations=250,
                residualThreshold=1e-9,
            )
        else:
            # We removed the limits because they made the inverse kinematics fail.
            # The robot couldn't go to its left.
            # The limits of the URDF are, however, already respected. Overall,
            # the robot moves more freely without the limits.
            # Let this be a lesson #ThierryBreton

            target_q_rad = p.calculateInverseKinematics(
                self.p_robot_id,
                self.END_EFFECTOR_LINK_INDEX,
                targetPosition=target_position_cartesian.tolist(),
                targetOrientation=target_orientation_quaternions,
                # jointDamping=[0.3, 0, 0, 0, 0, 0],
                jointDamping=[0.001] * len(self.lower_joint_limits),
                solver=p.IK_SDLS,
                restPoses=[0] * len(self.lower_joint_limits),
                lowerLimits=self.lower_joint_limits,
                upperLimits=self.upper_joint_limits,
                jointRanges=[
                    abs(up - low)
                    for up, low in zip(self.upper_joint_limits, self.lower_joint_limits)
                ],
                maxNumIterations=180,
                residualThreshold=1e-6,
            )

        return np.array(target_q_rad)[np.array(self.actuated_joints)]

    def forward_kinematics(
        self, sync_robot_pos: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of the robot
        Returns cartesian position and orientation in radians

        The position is the "URDF link frame" position, not the center of mass.
        This means a tip of the plastic part.
        """

        # Move the robot in simulation to the position of the motors to correct for desync
        if self.is_connected and sync_robot_pos:
            current_motor_positions = self.current_position(unit="rad", source="robot")
            p.setJointMotorControlArray(
                bodyIndex=self.p_robot_id,
                jointIndices=self.actuated_joints,
                targetPositions=current_motor_positions.tolist(),
                controlMode=p.POSITION_CONTROL,
            )
            # Update the simulation
            step_simulation()

        # Get the link state of the end effector
        end_effector_link_state = p.getLinkState(
            self.p_robot_id,
            self.END_EFFECTOR_LINK_INDEX,
            computeForwardKinematics=True,
        )

        # World position of the URDF link frame
        # Note: This is not the center of mass (LinkState[0])
        # because the inverseKinematics requires the link frame position, not the center of mass
        current_effector_position = np.array(end_effector_link_state[4])

        # orientation of the end effector in radians
        current_effector_orientation_rad = euler_from_quaternion(
            np.array(end_effector_link_state[1]), degrees=False
        )

        return (
            current_effector_position,
            current_effector_orientation_rad,
        )

    def get_end_effector_state(self):
        """
        Return the position and orientation in radians of the end effector and the gripper opening value.
        The gripper opening value between 0 and 1.
        """
        effector_position, effector_orientation_rad = self.forward_kinematics()
        return effector_position, effector_orientation_rad, self.closing_gripper_value

    def current_position(
        self,
        unit: Literal["rad", "motor_units", "degrees"] = "rad",
        source: Literal["sim", "robot"] = "robot",
    ) -> np.ndarray:
        """
        Read the current angles q of the joints of the robot.

        Args:
            unit: The unit of the output. Can be "rad", "motor_units" or "degrees".
                - "rad": radians
                - "motor_units": motor units (0 -> RESOLUTION)
                - "degrees": degrees
            source: The source of the data. Can be "sim" or "robot".
                - "sim": read from the simulation
                - "robot": read from the robot if connected. Otherwise, read from the simulation.
        """

        source_unit = "motor_units"

        if source == "robot" and self.is_connected and not self.is_moving:
            # Check if the method was implemented in the child class
            current_position = np.zeros(len(self.SERVO_IDS))
            if (
                self.read_group_motor_position.__qualname__
                != BaseRobot.read_group_motor_position.__qualname__
            ):
                # Read all the motors at once
                current_position = self.read_group_motor_position()
                if current_position.any() is None or np.isnan(current_position).any():
                    logger.warning("Position contains None value")
            else:
                # Read present position for each motor
                for i, servo_id in enumerate(self.SERVO_IDS):
                    joint_position = self.read_motor_position(servo_id)
                    if joint_position is not None:
                        current_position[i] = joint_position
                    else:
                        logger.warning("None value for joint ", servo_id)
            source_unit = "motor_units"
            output_position = current_position
        else:
            # If the robot is not connected, we use the pybullet simulation
            # Retrieve joint angles using getJointStates
            current_position_rad = np.zeros(self.num_actuated_joints)
            for idx, joint_id in enumerate(self.actuated_joints):
                joint_state = p.getJointState(
                    bodyUniqueId=self.p_robot_id,
                    jointIndex=joint_id,
                )
                current_position_rad[idx] = joint_state[0]  # in radians
            source_unit = "rad"
            output_position = current_position_rad

        if unit == "rad":
            if source_unit == "motor_units":
                # Convert from motor units to radians
                output_position = self._units_vec_to_radians(output_position)
        elif unit == "motor_units":
            if source_unit == "rad":
                # Convert from radians to motor units
                output_position = self._radians_vec_to_units(output_position)
        elif unit == "degrees":
            if source_unit == "motor_units":
                # Convert from motor units to radians
                output_position = self._units_vec_to_radians(output_position)
                # Convert from radians to degrees
                output_position = np.rad2deg(output_position)
            elif source_unit == "rad":
                # Convert from radians to degrees
                output_position = np.rad2deg(output_position)
        else:
            raise ValueError(
                f"Invalid unit: {unit}. Must be one of ['rad', 'motor_units', 'degrees']"
            )

        return output_position

    def set_motors_positions(
        self, q_target_rad: np.ndarray, enable_gripper: bool = False
    ) -> None:
        """
        Write the positions to the motors.

        If the robot is connected, the position is written to the motors.
        We always move the robot in the simulation.

        This does not control the gripper.

        q_target_rad is in radians.
        """
        if self.is_connected:
            q_target = self._radians_vec_to_units(q_target_rad)
            if (
                self.write_group_motor_position.__qualname__
                != BaseRobot.write_group_motor_position.__qualname__
            ):
                # Use the batched motor write if available
                self.write_group_motor_position(q_target, enable_gripper)
            else:
                # Otherwise loop through the motors
                for i, servo_id in enumerate(self.SERVO_IDS):
                    if servo_id == self.gripper_servo_id and not enable_gripper:
                        # The gripper is not controlled by the motors
                        # We skip it
                        continue
                    # Write goal position
                    self.write_motor_position(servo_id=servo_id, units=q_target[i])

        # Filter out the gripper_joint_index
        if not enable_gripper:
            joint_indices = [
                i for i in self.actuated_joints if i != self.GRIPPER_JOINT_INDEX
            ]
            target_positions = [
                q_target_rad[i] for i in joint_indices if i != self.GRIPPER_JOINT_INDEX
            ]
        else:
            joint_indices = self.actuated_joints
            target_positions = q_target_rad.tolist()

        p.setJointMotorControlArray(
            bodyIndex=self.p_robot_id,
            jointIndices=joint_indices,
            targetPositions=target_positions,
            controlMode=p.POSITION_CONTROL,
        )
        # Update the simulation
        step_simulation()

    def read_gripper_command(self) -> float:
        """
        Read if gripper is open or closed.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            return 0

        # Gripper is the last motor
        current_position = self.read_motor_position(servo_id=self.gripper_servo_id)
        # Since last motor ID might not be equal to the number of motors ( due to some shadowed motors)
        # We extract last motor calibration data for the gripper:
        open_position = self.config.servos_calibration_position[-1]
        close_position = self.config.servos_offsets[-1]

        if current_position is None:
            return 0

        command = (current_position - close_position) / (open_position - close_position)
        return command

    def write_joint_positions(
        self, angles: List[float], unit: Literal["rad", "motor_units", "degrees"]
    ) -> None:
        """
        Move the robot's joints to the specified angles.
        """
        if len(angles) != self.num_actuated_joints:
            raise ValueError(
                f"Number of joints {len(angles)} does not match the robot: {self.num_actuated_joints}"
            )

        # Convert to np
        np_angles = np.array(angles)
        if unit == "deg":
            np_angles = np.deg2rad(np_angles)
        if unit == "motor_units":
            np_angles = self._units_vec_to_radians(np_angles)

        self.set_motors_positions(q_target_rad=np_angles, enable_gripper=True)

    def write_gripper_command(self, command: float) -> None:
        """
        Open or close the gripper.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            return
        # Since last motor ID might not be equal to the number of motors ( due to some shadowed motors)
        # We extract last motor calibration data for the gripper:
        open_position = np.clip(
            self.config.servos_calibration_position[-1], 0, self.RESOLUTION
        )
        close_position = np.clip(self.config.servos_offsets[-1], 0, self.RESOLUTION)
        command = int(close_position + (open_position - close_position) * command)
        self.write_motor_position(
            self.gripper_servo_id, np.clip(command, 0, self.RESOLUTION)
        )
        self.update_object_gripping_status()

    def move_robot(
        self,
        target_position: np.ndarray,  # cartesian np.array
        target_orientation_rad: np.ndarray | None,  # rad np.array
        interpolate_trajectory: bool = False,
        steps: int = 10,
        **kwargs,
    ) -> None:
        """
        Move the robot to the absolute position and orientation.

        target_position: np.array cartesian position
        target_orientation_rad: np.array radian orientation
        interpolate_trajectory: if True, interpolate the trajectory
        steps: number of steps for the interpolation (unused)
        """

        if self._add_debug_lines:
            # Print debug point in pybullet
            p.addUserDebugPoints(
                pointPositions=[target_position],
                pointColorsRGB=[[1, 0, 0]],
                pointSize=4,
                lifeTime=3,
            )

            # Print a debug line in pybullet to show the target orientation
            start_point = target_position
            # End point is the target position + the orientation vector
            # Convert Euler angles to a rotation matrix
            rotation = R.from_euler("xyz", target_orientation_rad)
            rotation_matrix = rotation.as_matrix()
            # Extract the direction vector (e.g., the x-axis of the rotation matrix)
            # Assuming y-axis as forward direction
            direction_vector = rotation_matrix[:, 1]
            # Define a small delta
            delta = 0.02
            # Compute the end point
            end_point = target_position + delta * direction_vector
            p.addUserDebugLine(
                lineFromXYZ=start_point,
                lineToXYZ=end_point,
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=3,
            )

        if target_orientation_rad is not None:
            target_orientation_quaternion = np.array(
                p.getQuaternionFromEuler(target_orientation_rad)
            )
        else:
            target_orientation_quaternion = None

        goal_q_robot_rad = self.inverse_kinematics(
            target_position,
            target_orientation_quaternion,
        )

        self.is_moving = True
        if not interpolate_trajectory:
            self.set_motors_positions(goal_q_robot_rad)
        else:
            raise NotImplementedError("Interpolation not implemented yet")
        self.is_moving = False

        # reset gripping status when going to init position
        self.update_object_gripping_status()

    def relative_move_robot(
        self,
        delta_position: np.ndarray,
        delta_orientation_euler_rad: Optional[np.ndarray] = None,
    ):
        """We use the output of the OpenVLA model to move the robot in the simulation.
        The orientation output of Openvla is (roll, pitch, yaw) in radians.

        If the orientation is not provided, we assume it's zero.
        """

        if delta_orientation_euler_rad is None:
            delta_orientation_euler_rad = np.zeros(3)

        (
            current_effector_position,
            current_effector_orientation_euler,
        ) = self.forward_kinematics()

        # The Forward -> Inverse kinematics process is not idempotent
        # We want to diminish its effect by limiting the calculations if the delta is small

        target_position = current_effector_position + delta_position
        target_orientation_radian = (
            current_effector_orientation_euler + delta_orientation_euler_rad
        )

        return self.move_robot(target_position, target_orientation_radian)

    def set_simulation_positions(self, joints: np.ndarray) -> None:
        """
        Move robot joints to the specified positions in the simulation.
        """

        p.setJointMotorControlArray(
            bodyIndex=self.p_robot_id,
            jointIndices=self.actuated_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints,
        )
        # Update the simulation
        step_simulation()

    def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        """
        Compute and save offsets and signs for the motors.

        This method has to be called multiple time, moving the robot to the same position as in the simulation beforehand.
        """

        if not self.is_connected:
            self.calibration_current_step = 0
            logger.warning(
                "Robot is not connected. Cannot calibrate. Calibration sequence reset to 0."
            )
            return (
                "error",
                "Robot is not connected. Cannot calibrate. Calibration sequence reset to 0.",
            )

        voltage = self.current_voltage()
        if voltage is None:
            logger.warning("Cannot read voltage. Calibration sequence reset to 0.")
            self.calibration_current_step = 0
            return (
                "error",
                "Cannot read voltage. Plug your robot to power.",
            )

        self.config.servos_voltage = 12.0  # Assume default 12V motor
        motor_voltage = np.mean(voltage)

        if np.abs(motor_voltage - 12.0) > np.abs(motor_voltage - 6.0):
            self.config.servos_voltage = 6.0

        # Load default config
        voltage_as_str: str = "6V" if motor_voltage < 9.0 else "12V"
        default_config = self.get_default_base_robot_config(voltage=voltage_as_str)
        if default_config is not None:
            self.config.pid_gains = default_config.pid_gains
            self.config.gripping_threshold = int(default_config.gripping_threshold)
            self.config.non_gripping_threshold = int(
                default_config.non_gripping_threshold
            )

        self.disable_torque()

        # TODO: force pybullet to appear in headless to give the user instructions
        sim_helper_text = ""
        if config.SIM_MODE == SimulationMode.gui:
            sim_helper_text = "For reference, look at the simulation."
        else:
            sim_helper_text = "For reference, look in the instructions manual."

        if self.calibration_current_step == 0:
            # The first position is the initial position
            self.set_simulation_positions(np.zeros(self.num_actuated_joints))

            self.calibration_current_step += 1

            return (
                "in_progress",
                f"Step {self.calibration_current_step}/{self.calibration_max_steps}: Place the robot in POSITION 1. {sim_helper_text} Verify the gripper position.",
            )

        if self.calibration_current_step == 1:
            self.connect()
            # Set the offset to the middle of the motor range
            self.calibrate_motors()
            self.config.servos_offsets = self.current_position(
                unit="motor_units", source="robot"
            ).tolist()
            logger.info(
                f"Initial joint positions (motor units): {self.config.servos_offsets}"
            )
            # The second position is the calibration position
            self.set_simulation_positions(np.array(self.CALIBRATION_POSITION))
            self.calibration_current_step += 1

            return (
                "in_progress",
                f"Step {self.calibration_current_step}/{self.calibration_max_steps}: Place the robot in POSITION 2. {sim_helper_text} Verify the gripper position.",
            )

        if self.calibration_current_step == 2:
            self.config.servos_calibration_position = self.current_position(
                unit="motor_units", source="robot"
            ).tolist()
            logger.info(
                f"Current joint positions (motor units): {self.config.servos_calibration_position}"
            )
            self.config.servos_offsets_signs = np.sign(
                (
                    np.array(self.config.servos_calibration_position)
                    - np.array(self.config.servos_offsets)
                )
                / np.array(self.CALIBRATION_POSITION)
            ).tolist()
            logger.info(f"Motor signs computed: {self.config.servos_offsets_signs}")

            # Save to file
            path = self.config.save_local(serial_id=self.SERIAL_ID)
            self.calibration_current_step = 0

            return (
                "success",
                f"Step {self.calibration_max_steps}/{self.calibration_max_steps}: Calibration completed successfully. Offsets and signs saved to {path}",
            )

        raise ValueError(
            f"Invalid calibration step: {self.calibration_current_step}, must be between 0 and {self.calibration_max_steps - 1}"
        )

    def init_config(self) -> None:
        """
        Load the config file.

        1. Try to load specific configurations from the serial ID in the home directory.
        2. Try to load the default configuration from the bundled config file.
        3. If the robot is an so-100, we load default values from the motors.
        """

        # We check for serial id specific files in the home directory
        if hasattr(self, "SERIAL_ID"):
            config = BaseRobotConfig.from_serial_id(
                serial_id=self.SERIAL_ID, name=self.name
            )
            if config is not None:
                self.config = config
                logger.success("Loaded config from home directory phosphobot.")
                return

        # We check for bundled config files in resources
        config = BaseRobotConfig.from_json(filepath=self.bundled_config_file)
        if config is not None:
            self.config = config
            logger.success(f"Loaded config from {self.bundled_config_file}")
            return

        # We load default values
        current_voltage = self.current_voltage()
        if current_voltage is not None:
            motor_voltage = np.mean(current_voltage)
            voltage: str = "6V" if motor_voltage < 9.0 else "12V"
            config = self.get_default_base_robot_config(voltage=voltage)
            if config is not None:
                self.config = config
                logger.success(
                    f"Loaded default config for {self.name}, voltage {voltage}."
                )
                return

        logger.warning(
            f"Cannot find any config file for robot {self.name}. Perform calibration sequence."
        )
        self.SLEEP_POSITION = None
        self.config = BaseRobotConfig(
            name=self.name,
            servos_voltage=12.0,
            servos_offsets=[0] * len(self.SERVO_IDS),
            servos_offsets_signs=[1] * len(self.SERVO_IDS),
            servos_calibration_position=[0] * len(self.SERVO_IDS),
            gripping_threshold=0,
            non_gripping_threshold=0,
        )

    def control_gripper(
        self,
        open_command: float,  # Should be between 0 and 1
        **kwargs,
    ) -> None:
        """
        Open or close the gripper until object is gripped.
        open_command: 0 to close, 1 to open
        If the gripper already gripped the object, no higher closing command can be sent.
        """
        # logger.info(
        #     f"Control gripper: {open_command}. Object gripped: {self.is_object_gripped}. Closing gripper value: {self.closing_gripper_value}"
        # )
        # Clamp the command between 0 and 1
        self.update_object_gripping_status()
        if not self.is_object_gripped:
            open_command_clipped = np.clip(open_command, 0, 1)
        else:
            open_command_clipped = np.clip(open_command, self.closing_gripper_value, 1)

        # Only tighten if object is not gripped:
        if self.is_connected:
            self.write_gripper_command(open_command_clipped)
        self.closing_gripper_value = open_command_clipped

        ## Simulation side
        # Since last motor ID might not be equal to the number of motors ( due to some shadowed motors)
        # We extract last motor calibration data for the gripper:
        # Find which is the close position corresponds to the lower or upper limit joint of the gripper:
        close_position = self.gripper_initial_angle
        if abs(close_position - self.upper_joint_limits[-1]) < abs(
            close_position - self.lower_joint_limits[-1]
        ):
            open_position = self.lower_joint_limits[-1]
        else:
            open_position = self.upper_joint_limits[-1]

        # Update simulation only if the object has not been gripped:
        if not self.is_object_gripped:
            p.setJointMotorControl2(
                bodyIndex=self.p_robot_id,
                jointIndex=self.GRIPPER_JOINT_INDEX,
                controlMode=p.POSITION_CONTROL,
                targetPosition=close_position
                + (open_position - close_position) * open_command_clipped,
            )

    def get_observation(
        self, do_forward: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the observation of the robot.

        This method should return the observation of the robot.
        Will be used to build an observation in a Step of an episode.

        Returns:
            - state: np.array state of the robot (7D)
            - joints_position: np.array joints position of the robot
        """

        from phosphobot.endpoints.control import (
            ai_control_signal,
            leader_follower_control,
            vr_control_signal,
        )

        if (
            ai_control_signal.is_in_loop()
            or leader_follower_control.is_in_loop()
            or vr_control_signal.is_in_loop()
        ):
            joints_position = self.current_position(unit="rad", source="sim")
        else:
            joints_position = self.current_position(unit="rad", source="robot")

        if do_forward:
            effector_position, effector_orientation_euler_rad = (
                self.forward_kinematics()
            )
            state = np.concatenate((effector_position, effector_orientation_euler_rad))
        else:
            # Skip forward kinematics and return nan values
            state = np.full(6, np.nan)

        return state, joints_position

    def mimick_simu_to_robot(self):
        """
        Update simulation base on leader robot reading of joint position.
        """
        joints_position = self.current_position(unit="rad")
        gripper_command = self.read_gripper_command()

        # Update simulation
        # this take into account the leader that has less joints
        logger.debug(f"Moving to position: {joints_position}")
        p.setJointMotorControlArray(
            bodyIndex=self.p_robot_id,
            jointIndices=self.actuated_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints_position,
        )
        # Update the simulation
        step_simulation()
        self.control_gripper(gripper_command)

    def get_info(self) -> BaseRobotInfo:
        """
        Get information about a robot.

        This method returns an BaseRobotInfo object to initialize info.json during saving.
        This does not contain data about cameras that will be added during recording.

        Returns:
            - BaseRobotInfo: information of the robot
        """
        return BaseRobotInfo(
            robot_type=self.name,
            action=FeatureDetails(
                dtype="float32",
                shape=[len(self.SERVO_IDS)],
                names=[f"motor_{i}" for i in self.SERVO_IDS],
            ),
            observation_state=FeatureDetails(
                dtype="float32",
                shape=[len(self.SERVO_IDS)],
                names=[f"motor_{i}" for i in self.SERVO_IDS],
            ),
        )

    def current_voltage(self) -> np.ndarray | None:
        """
        Read the current voltage u of the joints of the robot.

        Returns :
            current_voltage : np.ndarray of the current torque of each joint
        """

        # Read present position for each motor
        if self.is_connected:
            current_voltage = np.zeros(len(self.SERVO_IDS))
            for i, servo_id in enumerate(self.SERVO_IDS):  # Controlling 3 joints
                joint_voltage = self.read_motor_voltage(servo_id)
                if joint_voltage is not None:
                    current_voltage[i] = joint_voltage
            return current_voltage

        # If the robot is not connected, error raised
        return None

    def is_powered_on(self) -> bool:
        """
        Return True if all voltage readings are above 0.1V and successful
        or if a movement is in progress.
        """
        if self.is_moving:
            return True

        for servo_id in self.SERVO_IDS:
            voltage = self.read_motor_voltage(servo_id)
            if voltage is not None and voltage < 0.1:
                logger.warning(
                    f"Robot {self.name} is not powered on. Read {voltage} voltage for servo {servo_id}"
                )
                return False
        return True

    def current_torque(self) -> np.ndarray:
        """
        Read the current torque q of the joints of the robot.

        Returns :
            current_torque : np.ndarray of the current torque of each joint
        """

        current_torque = np.zeros(self.num_actuated_joints)
        # Read present position for each motor
        if self.is_connected:
            for idx, servo_id in enumerate(self.actuated_joints):
                joint_torque = self.read_motor_torque(servo_id)
                if joint_torque is not None:
                    current_torque[idx] = joint_torque
                else:
                    logger.warning("None torque value for joint ", servo_id)
            return current_torque

        # If the robot is not connected, we use the pybullet simulation
        # Retrieve joint angles using getJointStates
        for idx, joint_id in enumerate(self.actuated_joints):
            joint_state = p.getJointState(
                bodyUniqueId=self.p_robot_id,
                jointIndex=idx,
            )
            # Joint torque is in the 4th element of the joint state tuple
            current_torque[idx] = joint_state[3]

        return current_torque

    def update_object_gripping_status(self):
        """
        Based on the torque value, update the object gripping status.

        If the torque is above the threshold, the object is gripped.
        If under the threshold, the object is not gripped.
        """
        gripper_torque = self.read_gripper_torque()
        # logger.info(
        #     f"Gripper torque: {gripper_torque}. Threshold: {self.config.gripping_threshold}. Non-gripping threshold: {self.config.non_gripping_threshold}"
        # )
        if gripper_torque >= self.config.gripping_threshold:
            self.is_object_gripped = True
        if gripper_torque <= self.config.non_gripping_threshold:
            self.is_object_gripped = False

    def status(self) -> RobotConfigStatus | None:
        # Check robot voltage
        if not self.is_powered_on():
            logger.warning("Robot is not powered on.")
            return None

        return RobotConfigStatus(
            name=self.name,
            usb_port=getattr(self, "SERIAL_ID", None),
        )
