import time
import numpy as np
from loguru import logger
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

from phosphobot.robot import RobotConnectionManager

rcm = RobotConnectionManager()

follower_port = rcm.robots[0].DEVICE_NAME

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

robot = ManipulatorRobot(
    robot_type="koch",
    # No leader arms
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",  # Needs to be created with the calibration script
)

servos_offsets = np.array([90, 0, 180, 0, 180, 0])


def convert_robot_angles_to_phi(alpha: np.ndarray) -> np.ndarray:
    """
    Convert the robot angles (measured with the read position method) to phi angles (our robot representation)

    alpha is in degrees
    phi is in radians
    """
    signs = np.ones(6)

    phi_deg = signs * (alpha - servos_offsets)

    return np.deg2rad(phi_deg)


def convert_phi_to_robot_angles(phi: np.ndarray) -> np.ndarray:
    """
    Convert the phi angles to robot angles

    phi is in radians
    alpha is in degrees
    """

    signs = np.ones(6)
    phi_deg = signs * np.rad2deg(phi)

    alpha_deg = phi_deg + servos_offsets

    # Check that the angles are in the correct range
    return alpha_deg


def execute_sequence(target_pos: np.ndarray, duration: float):
    """
    target_pos: phi in radians
    """

    start_time = time.time()
    angle_target = convert_phi_to_robot_angles(target_pos)
    while time.time() - start_time < duration:
        robot.follower_arms["main"].write("Goal_Position", angle_target.tolist())

    logger.info("stop sequence")


robot.connect()
logger.info("Connected to the robot")

angle_target = np.array([90, 90, 90, 0, 180, 0])

# Go to the initial position
robot.follower_arms["main"].write("Goal_Position", angle_target.tolist())

start_time = time.time()

while time.time() - start_time < 2.0:
    robot.follower_arms["main"].write("Goal_Position", angle_target.tolist())

logger.info("Moving to part 2")
execute_sequence(np.array([3.14 / 4, 3.14 / 2, -3.14 / 2, 0, 0, 0]), 2.0)

logger.info("Moving to part 3")
execute_sequence(np.array([3.14 / 2, 3.14 / 2, -3.14 / 4, 0, 0, 0]), 2.0)

robot.disconnect()
