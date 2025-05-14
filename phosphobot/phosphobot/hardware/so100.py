import asyncio
import time
from typing import Optional

import numpy as np
import pybullet as p  # type: ignore
from loguru import logger
from phosphobot.control_signal import ControlSignal
from serial.tools.list_ports_common import ListPortInfo

from phosphobot.hardware.base import BaseRobot
from phosphobot.hardware.motors.feetech import FeetechMotorsBus  # type: ignore
from phosphobot.utils import step_simulation, get_resources_path


class SO100Hardware(BaseRobot):
    name = "so-100"

    URDF_FILE_PATH = str(
        get_resources_path() / "urdf" / "so-100" / "urdf" / "so-100.urdf"
    )

    DEVICE_PID: int = 21971

    AXIS_ORIENTATION = [0, 0, 1, 1]

    REGISTERED_SERIAL_ID = ["58CD176683"]

    # Control commands (refer to the Feetech SCServo manual)
    TORQUE_ENABLE = 0x01
    TORQUE_DISABLE = 0

    TORQUE_ADDRESS = 0x40

    COMMAND_WRITE = 0x03
    COMMAND_READ = 0x02

    END_EFFECTOR_LINK_INDEX = 4
    GRIPPER_JOINT_INDEX = 5

    # Dynamixel settings
    motors = {
        # name: (index, model)
        "shoulder_pan": [1, "sts3215"],
        "shoulder_lift": [2, "sts3215"],
        "elbow_flex": [3, "sts3215"],
        "wrist_flex": [4, "sts3215"],
        "wrist_roll": [5, "sts3215"],
        "gripper": [6, "sts3215"],
    }

    SERVO_IDS = [1, 2, 3, 4, 5, 6]
    BAUDRATE = 1000000  # Baud rate
    RESOLUTION = 4096  # 12-bit resolution

    # Measured offset on the servos
    CALIBRATION_POSITION = [
        np.pi / 2,
        np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        np.pi / 2,
    ]
    SLEEP_POSITION = [
        -0.09359567856848712,
        -1.6632412388236073,
        1.4683781047547897,
        0.5799863360473464,
        0.72268138697963,
        0.018412264636423696,
    ]

    # Tracking of motor communication errors
    motor_communication_errors: int = 0

    _gravity_task: Optional[asyncio.Task] = None

    @property
    def servo_id_to_motor_name(self):
        return {v[0]: k for k, v in self.motors.items()}

    @classmethod
    def from_port(cls, port: ListPortInfo, **kwargs) -> Optional["SO100Hardware"]:
        """
        Detect if the device is a SO-100 robot.π
        """
        # The Feetech UART board CH340 has PID 29987
        if port.pid == cls.DEVICE_PID or port.pid == 29987:
            # The serial number is not always available
            serial_number = port.serial_number or "no_serial"
            robot = cls(device_name=port.device, serial_id=serial_number)
            # Check if voltage is not None
            voltages = []
            for servo_id in robot.SERVO_IDS:
                voltage = robot.read_motor_voltage(servo_id)
                if voltage is not None and voltage > 0:
                    voltages.append(voltage)
                else:
                    logger.warning(
                        f"Robot {robot.name} has voltage {voltage} in servo {servo_id}. Please plug the robot to power and check cable connections."
                    )
                    robot_ports_without_power = kwargs.get("robot_ports_without_power")
                    if isinstance(robot_ports_without_power, set):
                        robot_ports_without_power.add(port.device)

                    return None
            return robot
        return None

    def connect(self):
        """
        Connect to the robot.
        """
        if not hasattr(self, "DEVICE_NAME"):
            logger.warning(
                "Can't connect: no plugged robot detected (no DEVICE_NAME). Please plug the robot, then restart the server."
            )
            return

        # Create serial connection
        self.motors_bus = FeetechMotorsBus(port=self.DEVICE_NAME, motors=self.motors)
        self.motors_bus.connect()
        self.is_connected = True

    def disconnect(self):
        """
        Disconnect the robot.
        """
        try:
            self.motors_bus.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting motors: {e}")
        self.is_connected = False

    def enable_torque(self):
        if not self.is_connected:
            return

        try:
            self.motors_bus.write("Torque_Enable", 1)
            for servo_id, c in enumerate(self.config.pid_gains):
                self._set_pid_gains_motors(
                    servo_id + 1, p_gain=c.p_gain, i_gain=c.i_gain, d_gain=c.d_gain
                )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Error enabling torque: {e}")
            self.update_motor_errors()
            return

    def _set_pid_gains_motors(
        self, servo_id: int, p_gain: int = 32, i_gain: int = 0, d_gain: int = 32
    ):
        """
        Set the PID gains for the Feetech servo.

        :servo_id: Joint ID (0-6)
        :param p_gain: Proportional gain (0-255)
        :param i_gain: Integral gain (0-255)
        :param d_gain: Derivative gain (0-255)
        """
        try:
            torque_status = self.motors_bus.read(
                "Torque_Enable", motor_names=list(self.motors.keys())
            )
        except Exception as e:
            logger.warning(f"Error reading torque status: {e}")
            return

        if torque_status.all() == 1:
            self.motors_bus.write(
                "P_Coefficient",
                p_gain,
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motors_bus.write(
                "I_Coefficient",
                i_gain,
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motors_bus.write(
                "D_Coefficient",
                d_gain,
                motor_names=self.servo_id_to_motor_name[servo_id],
            )

        else:
            logger.warning(
                "Motors torque is disabled. Motors must have torque enabled to change PID coefficients. Enable torque first."
            )

    def disable_torque(self):
        # Disable torque
        if not self.is_connected:
            return

        self.motors_bus.write("Torque_Enable", 0)

    def update_motor_errors(self):
        """
        Every time a motor communication error is detected, increment the error counter.
        If the counter reaches a certain threshold, disconnect the robot.
        """
        if not self.is_connected:
            return

        self.motor_communication_errors += 1
        if self.motor_communication_errors > 10:
            logger.error("Too many communication errors. Disconnecting robot.")
            self.disconnect()

    def read_motor_position(self, servo_id: int, **kwargs) -> int | None:
        """
        Read the position of a Feetech servo.
        """
        if not self.is_connected:
            return None
        try:
            position = self.motors_bus.read(
                "Present_Position",
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motor_communication_errors = 0
            return position
        except Exception as e:
            logger.warning(f"Error reading motor position: {e}")
            self.update_motor_errors()
            return None

    def write_motor_position(self, servo_id: int, units: int, **kwargs) -> None:
        """
        Write a position to a Feetech servo.
        """
        if not self.is_connected:
            return None

        try:
            self.motors_bus.write(
                "Goal_Position",
                values=[units],
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Error writing motor position: {e}")
            self.update_motor_errors()

    def write_group_motor_position(
        self, q_target: np.ndarray, enable_gripper: bool = True
    ) -> None:
        """
        Write a position to all motors of the robot.
        """
        if not self.is_connected:
            return None

        values = q_target.tolist()
        motor_names = list(self.motors.keys())
        if not enable_gripper:
            # Gripper is the last parameter of q_target (last motor)
            values = values[:-1]
            motor_names = motor_names[:-1]

        try:
            self.motors_bus.write(
                "Goal_Position", values=values, motor_names=motor_names
            )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Error writing motor position: {e}")
            self.update_motor_errors()

    def read_group_motor_position(self) -> np.ndarray:
        """
        Read the position of all motors of the robot.
        """
        if not self.is_connected:
            return np.ones(6) * np.nan

        motor_names = list(self.motors.keys())
        try:
            motor_positions = self.motors_bus.read(
                "Present_Position", motor_names=motor_names
            )
            self.motor_communication_errors = 0
        except Exception as e:
            logger.warning(f"Error reading motor position: {e}")
            self.update_motor_errors()
            motor_positions = None

        if motor_positions is None:
            return np.ones(6) * np.nan
        return motor_positions

    def read_motor_torque(self, servo_id: int, **kwargs) -> float | None:
        """
        Read the torque of a Feetech servo.
        """
        if not self.is_connected:
            return None
        try:
            torque = self.motors_bus.read(
                "Present_Current",
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motor_communication_errors = 0
            return torque
        except Exception as e:
            logger.warning(f"Error reading motor torque for servo {servo_id}: {e}")
            self.update_motor_errors()
            return None

    def read_motor_voltage(self, servo_id: int, **kwargs) -> float | None:
        """
        Read the voltage of a Feetech servo.
        """
        if not self.is_connected:
            return None
        try:
            voltage = self.motors_bus.read(
                "Present_Voltage",
                motor_names=self.servo_id_to_motor_name[servo_id],
            )
            self.motor_communication_errors = 0
            return voltage / 10.0  # unit is 0.1V
        except Exception as e:
            logger.warning(f"Error reading motor voltage for servo {servo_id}: {e}")
            self.update_motor_errors()
            return None

    def calibrate_motors(self, **kwargs) -> None:
        """
        This is called during the calibration phase of the robot.
        It sets the offset of all motors to self.RESOLUTION/2.
        """
        if not self.is_connected:
            logger.warning("Robot is not connected.")
            return None

        self.motors_bus.write("Torque_Enable", 128)
        time.sleep(1)

    async def gravity_compensation_loop(
        self,
        control_signal: ControlSignal,
    ):
        """
        Background task that implements gravity compensation control:
        - Applies gravity compensation to the robot
        """
        # Connect to PyBullet for gravity compensation calculations
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        # Set up PID gains for leader's gravity compensation
        current_voltage = self.current_voltage()
        if current_voltage is None:
            logger.warning(
                "Unable to read motor voltage. Check that your robot is plugged to power."
            )
            return
        motor_voltage = np.mean(current_voltage)
        voltage = "6V" if motor_voltage < 9.0 else "12V"

        # Define PID gains for all six motors
        p_gains = [3, 6, 6, 3, 3, 3]
        d_gains = [9, 9, 9, 9, 9, 9]
        default_p_gains = [12, 20, 20, 20, 20, 20]
        default_d_gains = [36, 36, 36, 32, 32, 32]
        alpha = np.array([0, 0.2, 0.2, 0.1, 0.2, 0.2])

        if voltage == "12V":
            p_gains = [int(p / 2) for p in p_gains]
            d_gains = [int(d / 2) for d in d_gains]
            default_p_gains = [6, 6, 6, 10, 10, 10]
            default_d_gains = [30, 15, 15, 30, 30, 30]

        # Enable torque if using gravity compensation
        self.enable_torque()

        # Apply custom PID gains to leader for all six motors
        for i in range(6):
            self._set_pid_gains_motors(
                servo_id=i + 1,
                p_gain=p_gains[i],
                i_gain=0,
                d_gain=d_gains[i],
            )
            await asyncio.sleep(0.05)

        # Control loop parameters
        num_joints = len(self.actuated_joints)
        joint_indices = list(range(num_joints))
        loop_period = 1 / 50

        # Main control loop
        while control_signal.is_in_loop():
            start_time = time.time()

            # Get leader's current joint positions
            pos_rad = self.current_position(unit="rad")

            # Update PyBullet simulation for gravity calculation
            for i, idx in enumerate(joint_indices):
                p.resetJointState(self.p_robot_id, idx, pos_rad[i])
            step_simulation()

            # Calculate gravity compensation torque
            positions = list(pos_rad)
            velocities = [0.0] * num_joints
            accelerations = [0.0] * num_joints
            tau_g = p.calculateInverseDynamics(
                self.p_robot_id,
                positions,
                velocities,
                accelerations,
            )

            # Apply gravity compensation to leader
            theta_des_rad = pos_rad + alpha[:num_joints] * np.array(tau_g)
            self.write_joint_positions(theta_des_rad, unit="rad")

            # Maintain loop frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, loop_period - elapsed)
            await asyncio.sleep(sleep_time)

        # Cleanup: Reset leader's PID gains to default for all six motors
        for i in range(6):  # Changed from 4 to 6
            self._set_pid_gains_motors(
                servo_id=i + 1,
                p_gain=default_p_gains[i],
                i_gain=0,
                d_gain=default_d_gains[i],
            )
            await asyncio.sleep(0.05)
        logger.info("Gravity control stopped")
