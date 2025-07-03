import atexit
import json
import os
import asyncio
from abc import abstractmethod
from typing import List, Literal, Optional, Union

from fastapi import HTTPException
import numpy as np
from loguru import logger
from phosphobot.configs import config as cfg
from phosphobot.models import BaseRobot, BaseRobotConfig, BaseRobotInfo
from phosphobot.models.lerobot_dataset import FeatureDetails
from scipy.spatial.transform import Rotation as R  # type: ignore
from phosphobot.utils import (
    euler_from_quaternion,
    get_resources_path,
    get_quaternion_from_euler,
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


class BaseManipulator(BaseRobot):
    """
    Abstract class for a manipulator robot (single robot arm).
    E.g SO-100, SO-101, AgilexPiper, Kock 1.1, etc.
    """

    # Path to the URDF file of the robot
    URDF_FILE_PATH: str
    # Axis and orientation of the robot. This depends on the default
    # orientation of the URDF file
    AXIS_ORIENTATION: List[int]

    SERIAL_ID: str

    device_name: Optional[str]

    # List of servo IDs, used to write and read motor positions
    # They are in the same order as the joint links in the URDF file
    SERVO_IDS: List[int]

    CALIBRATION_POSITION: list[float]  # same size as SERVO_IDS
    SLEEP_POSITION: list[float] | None = None
    RESOLUTION: int
    # The effector is the gripper
    END_EFFECTOR_LINK_INDEX: int

    # calibration config: offsets, signs, pid values
    config: BaseRobotConfig | None = None

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

    # (x, y, z) position of the robot in the simulation in meters
    initial_orientation_rad: np.ndarray | None = None
    # (rx, ry, rz) orientation of the robot in the simulation
    initial_position: np.ndarray | None = None

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

    def __init__(
        self,
        device_name: str | None = None,
        serial_id: str | None = None,
        only_simulation: bool = False,
        reset_simulation: bool = False,
        axis: List[float] | None = None,
        add_debug_lines: bool = False,
        show_debug_link_indices: bool = True,
        **kwargs: Optional[dict[str, str]],
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
            self.device_name = device_name

        self._add_debug_lines = add_debug_lines

        # Since pybullet is removed, we cannot load URDF or use simulation
        logger.info(f"Note: PyBullet simulation has been removed. URDF file would be: {self.URDF_FILE_PATH}")
        
        # Set default values for properties that were previously set by pybullet
        self.actuated_joints = list(range(len(self.SERVO_IDS)))
        self.num_actuated_joints = len(self.actuated_joints)
        
        # Gripper motor is the last one
        self.gripper_servo_id = self.SERVO_IDS[-1]

        # Set default joint limits (these would normally come from URDF)
        # These are reasonable defaults for most robots
        self.lower_joint_limits = [-np.pi] * self.num_actuated_joints
        self.upper_joint_limits = [np.pi] * self.num_actuated_joints

        self.gripper_initial_angle = 0.0

        if not only_simulation:
            # Register the disconnect method to be called on exit
            atexit.register(self.move_to_sleep_sync)
        else:
            logger.info("Only simulation: Not connecting to the robot.")
            self.is_connected = False

        if only_simulation:
            config = self.get_default_base_robot_config(
                voltage="6V", raise_if_none=True
            )
            if config is None:
                raise FileNotFoundError(
                    f"Default config file not found for {self.name} at 6V."
                )
            self.config = config
        else:
            self.config = None

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

        # If the robot is not connected, return a default value
        # (previously used pybullet simulation)
        logger.warning("Robot not connected, returning default gripper torque value")
        return np.int32(0)

    async def move_to_initial_position(self):
        """
        Move the robot to its initial position.
        """
        self.init_config()
        self.enable_torque()
        zero_position = np.zeros(len(self.actuated_joints))
        self.set_motors_positions(zero_position, enable_gripper=True)
        # Wait for the robot to move to the initial position
        await asyncio.sleep(0.5)
        (
            self.initial_position,
            self.initial_orientation_rad,
        ) = self.forward_kinematics()

    async def move_to_sleep(self):
        """
        Move the robot to its sleep position and disable torque.
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

    def _units_vec_to_radians(self, units: np.ndarray) -> np.ndarray:
        """
        Convert from motor discrete units (0 -> RESOLUTION) to radians
        """
        if self.config is None:
            raise ValueError(
                "Robot configuration is not set. Run the calibration first."
            )
        return (
            (units - self.config.servos_offsets[: len(units)])
            * self.config.servos_offsets_signs[: len(units)]
            * ((2 * np.pi) / (self.RESOLUTION - 1))
        )

    def _radians_vec_to_motor_units(self, radians: np.ndarray) -> np.ndarray:
        """
        Convert from radians to motor discrete units (0 -> RESOLUTION)

        Note: The result can exceed the resolution of the motor, in the case of a continuous rotation motor.
        """
        if self.config is None:
            raise ValueError(
                "Robot configuration is not set. Run the calibration first."
            )

        x = (
            radians
            * self.config.servos_offsets_signs[: len(radians)]
            * ((self.RESOLUTION - 1) / (2 * np.pi))
        ) + self.config.servos_offsets[: len(radians)]
        return x.astype(int)

    def _radians_to_motor_units(self, radians: float, servo_id: int) -> int:
        """
        Convert a single q position from radians to motor discrete units (0 -> RESOLUTION)

        Note: The result can exceed the resolution of the motor, in the case of a continuous rotation motor.
        """
        offset_id = self.SERVO_IDS.index(servo_id)
        if self.config is None:
            raise ValueError(
                "Robot configuration is not set. Run the calibration first."
            )

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

        Note: PyBullet simulation has been removed from this project.
        This method now returns the current joint positions as a placeholder.
        For proper IK, a robotics library like ikpy or roboticstoolbox-python should be used.
        """
        logger.warning("Inverse kinematics called but pybullet removed - returning current joint positions")
        
        # Return current joint positions if connected, otherwise zeros
        if self.is_connected:
            return self.read_joints_position(unit="rad", source="robot")
        else:
            return np.zeros(self.num_actuated_joints)

    def forward_kinematics(
        self, sync_robot_pos: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the forward kinematics of the robot
        Returns cartesian position and orientation in radians

        Note: PyBullet simulation has been removed from this project.
        This method now returns the initial position/orientation if available,
        otherwise returns zeros.
        For proper FK, a robotics library like ikpy or roboticstoolbox-python should be used.
        """
        logger.warning("Forward kinematics called but pybullet removed - returning default values")
        
        # Return initial position/orientation if available
        if self.initial_position is not None and self.initial_orientation_rad is not None:
            return (self.initial_position, self.initial_orientation_rad)
        else:
            # Return default values
            return (np.zeros(3), np.zeros(3))

    def get_end_effector_state(self):
        """
        Return the position and orientation in radians of the end effector and the gripper opening value.
        The gripper opening value between 0 and 1.
        """
        effector_position, effector_orientation_rad = self.forward_kinematics()
        return effector_position, effector_orientation_rad, self.closing_gripper_value

    def read_joints_position(
        self,
        unit: Literal["rad", "motor_units", "degrees"] = "rad",
        source: Literal["sim", "robot"] = "robot",
        joints_ids: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Read the current angles q of the joints of the robot.

        Args:
            unit: The unit of the output. Can be "rad", "motor_units" or "degrees".
                - "rad": radians
                - "motor_units": motor units (0 -> RESOLUTION)
                - "degrees": degrees
            source: The source of the data. Can be "sim" or "robot".
                - "sim": read from the simulation (no longer supported)
                - "robot": read from the robot if connected.
        """

        source_unit = "motor_units"

        if self.is_connected and not self.is_moving:
            # Check if the method was implemented in the child class
            if joints_ids is None:
                joints_ids = self.SERVO_IDS

            current_position = np.zeros(len(joints_ids))

            if (
                # if we want to read all the motors at once
                joints_ids == self.SERVO_IDS
                and self.read_group_motor_position.__qualname__
                != BaseManipulator.read_group_motor_position.__qualname__
            ):
                # Read all the motors at once
                current_position = self.read_group_motor_position()
                if current_position.any() is None or np.isnan(current_position).any():
                    logger.warning("Position contains None value")
            else:
                # Read present position for each motor
                for i, servo_id in enumerate(joints_ids):
                    joint_position = self.read_motor_position(servo_id)
                    if joint_position is not None:
                        current_position[i] = joint_position
                    else:
                        logger.warning("None value for joint ", servo_id)
            source_unit = "motor_units"
            output_position = current_position
        else:
            # If the robot is not connected, return zeros
            # (previously used pybullet simulation)
            if joints_ids is None:
                joints_ids = list(range(self.num_actuated_joints))
            
            logger.warning("Robot not connected, returning zero joint positions")
            current_position_rad = np.zeros(len(joints_ids))
            source_unit = "rad"
            output_position = current_position_rad

        if unit == "rad":
            if source_unit == "motor_units":
                # Convert from motor units to radians
                output_position = self._units_vec_to_radians(output_position)
        elif unit == "motor_units":
            if source_unit == "rad":
                # Convert from radians to motor units
                output_position = self._radians_vec_to_motor_units(output_position)
        elif unit == "degrees":
            if source_unit == "motor_units":
                # Convert from motor units to radians
                output_position_rad = self._units_vec_to_radians(output_position)
                # Convert from radians to degrees
                output_position = np.rad2deg(output_position_rad)
            elif source_unit == "rad":
                # Convert from radians to degrees
                output_position = np.rad2deg(output_position)  # type: ignore
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

        This does not control the gripper.

        q_target_rad is in radians.
        """
        if self.is_connected:
            q_target = self._radians_vec_to_motor_units(q_target_rad)
            if (
                self.write_group_motor_position.__qualname__
                != BaseManipulator.write_group_motor_position.__qualname__
            ):
                # Use the batched motor write if available
                self.write_group_motor_position(q_target, enable_gripper)
            else:
                # Otherwise loop through the motors
                for i, servo_id in enumerate(self.actuated_joints):
                    if servo_id == self.gripper_servo_id and not enable_gripper:
                        # The gripper is not controlled by the motors
                        # We skip it
                        continue
                    # Write goal position
                    self.write_motor_position(servo_id=servo_id, units=q_target[i])

        # PyBullet simulation has been removed
        # Previously would update simulation here

    def read_gripper_command(self) -> float:
        """
        Read if gripper is open or closed.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            return 0
        if self.config is None:
            raise ValueError(
                "Robot configuration is not set. Run the calibration first."
            )

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
        self,
        angles: List[float],
        unit: Literal["rad", "motor_units", "degrees"],
        joints_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Move the robot's joints to the specified angles.
        """

        # Convert to np and radians
        np_angles_rad = np.array(angles)
        if unit == "deg":
            np_angles_rad = np.deg2rad(np_angles_rad)
        elif unit == "motor_units":
            np_angles_rad = self._units_vec_to_radians(np_angles_rad)

        if joints_ids is None:
            if len(np_angles_rad) == len(self.SERVO_IDS):
                # If the number of angles is equal to the number of motors, we set the angles
                # to the motors
                self.set_motors_positions(
                    q_target_rad=np_angles_rad, enable_gripper=True
                )
                return
            else:
                # Iterate over the angles and set the corresponding joint positions
                for i, angle in enumerate(np_angles_rad):
                    if i < len(self.SERVO_IDS):
                        motor_units = self._radians_to_motor_units(
                            angle, servo_id=self.SERVO_IDS[i]
                        )
                        self.write_motor_position(
                            servo_id=self.SERVO_IDS[i], units=motor_units
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Joint ID {i} is out of range for the robot.",
                        )

        else:
            # If we have joint ids, we get the current joint positions and edit the specified joints
            current_joint_positions = self.read_joints_position(unit=unit)
            for i, joint_id in enumerate(joints_ids):
                if joint_id in self.SERVO_IDS:
                    index = self.SERVO_IDS.index(joint_id)
                    current_joint_positions[index] = angles[i]
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Joint ID {joint_id} is out of range for the robot.",
                    )
            # Write the joint positions
            self.set_motors_positions(q_target_rad=np_angles_rad, enable_gripper=True)

    def write_gripper_command(self, command: float) -> None:
        """
        Open or close the gripper.

        command: 0 to close, 1 to open
        """
        if not self.is_connected:
            return
        if self.config is None:
            raise ValueError(
                "Robot configuration is not set. Run the calibration first."
            )
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

    async def move_robot_absolute(
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
        
        Note: PyBullet visualization has been removed.
        Debug lines and points are no longer displayed.
        """

        if target_orientation_rad is not None:
            # Convert Euler angles to quaternion using scipy
            target_orientation_quaternion = get_quaternion_from_euler(
                target_orientation_rad, degrees=False
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

    def set_simulation_positions(self, joints: np.ndarray) -> None:
        """
        Move robot joints to the specified positions in the simulation.
        
        Note: PyBullet simulation has been removed from this project.
        This method is now a no-op.
        """
        logger.debug("set_simulation_positions called but pybullet removed - no-op")
        pass

    async def calibrate(self) -> tuple[Literal["success", "in_progress", "error"], str]:
        raise NotImplementedError(
            "The calibrate method must be implemented in the child class."
        )

    def init_config(self) -> None:
        """
        Load the config file.

        1. Try to load specific configurations from the serial ID in the home directory.
        2. Try to load the default configuration from the bundled config file.
        3. If the robot is an so-100, we load default values from the motors.
        """

        # We do this for tests
        if cfg.ONLY_SIMULATION or not hasattr(self, "SERIAL_ID"):
            self.config = self.get_default_base_robot_config(
                voltage="6V", raise_if_none=True
            )
            return

        # We check for serial id specific files in the home directory
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
        self.config = None

    def control_gripper(
        self,
        open_command: float,  # Should be between 0 and 1
        **kwargs,
    ) -> None:
        """
        Open or close the gripper until object is gripped.
        open_command: 0 to close, 1 to open
        If the gripper already gripped the object, no higher closing command can be sent.
        
        Note: PyBullet simulation has been removed.
        Only controls physical robot if connected.
        """
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

        joints_position = self.read_joints_position(unit="rad", source="robot")

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
        
        Note: PyBullet simulation has been removed from this project.
        This method is now a no-op.
        """
        logger.debug("mimick_simu_to_robot called but pybullet removed - no-op")
        pass

    def get_info_for_dataset(self) -> BaseRobotInfo:
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

        # If the robot is not connected, return zeros
        # (previously used pybullet simulation)
        logger.warning("Robot not connected, returning zero torque values")
        return current_torque

    def update_object_gripping_status(self):
        """
        Based on the torque value, update the object gripping status.

        If the torque is above the threshold, the object is gripped.
        If under the threshold, the object is not gripped.
        """

        gripper_torque = self.read_gripper_torque()

        if self.config is None:
            logger.warning("Robot configuration is not set. Run the calibration first.")
            return

        if gripper_torque >= self.config.gripping_threshold:
            self.is_object_gripped = True
        if gripper_torque <= self.config.non_gripping_threshold:
            self.is_object_gripped = False


class BaseMobileRobot(BaseRobot):
    """
    Abstract class for a mobile robot
    E.g. LeKiwi, Unitree Go2
    """

    def __init__(
        self,
        only_simulation: bool = False,
    ):
        if not only_simulation:
            # Register the disconnect method to be called on exit
            atexit.register(self.move_to_sleep_sync)
        else:
            logger.info("Only simulation: Not connecting to the robot.")
            self.is_connected = False 