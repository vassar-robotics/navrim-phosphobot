from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
import numpy as np

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

robot.connect()
print("Connected to the robot")

# Disable torque
robot.follower_arms["main"].write("Torque_Enable", 0)
print("Torque disabled")


def transform_angles(joints_deg):
    # Copy the list
    recorded_angles = joints_deg.copy()

    # apply offsets
    joints_deg[0] = joints_deg[0] - 90
    joints_deg[1] = joints_deg[1] - 90
    joints_deg[2] = joints_deg[2] - 90
    # no offset for the idx 3
    joints_deg[4] = joints_deg[3] + 90
    # no offset for the idx 5

    index_mapping = {0: 5, 1: 2, 2: 3, 3: 4, 4: 0, 5: 1}

    # Zero-angle offsets for servos and IK joints
    servo_zero_offsets = [90, 90, 90, 0, -90, 0]

    # Rotation direction (1 for same, -1 for opposite)
    rotation_directions = [1, 1, -1, 1, -1, -1]

    transformed_angles = [0] * 6

    for servo_index, ik_index in index_mapping.items():
        adjusted_angle = recorded_angles[servo_index] - servo_zero_offsets[servo_index]
        final_angle = rotation_directions[servo_index] * adjusted_angle
        transformed_angles[ik_index] = final_angle

    return np.deg2rad(transformed_angles)


L1 = 0.105  # in meters
L2 = 0.24  # in meters
# Servos offsets
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


def polar_to_cartesian(r, theta, z):
    """
    Convert polar coordinates to cartesian coordinates
    """

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y, z


def forward_kinematics(phi: np.ndarray) -> np.ndarray:
    """
    Compute the forward kinematics of the robot

    phi is in radians
    """

    theta = phi[0]

    r = L1 * np.cos(phi[1]) + L2 * np.cos(phi[1] + phi[2])
    z = L1 * np.sin(phi[1]) + L2 * np.sin(phi[1] + phi[2])

    x, y, z = polar_to_cartesian(r, theta, z)

    return np.array([x, y, z])


while True:
    current_joints = robot.follower_arms["main"].read("Present_Position")
    phi = convert_robot_angles_to_phi(np.array(current_joints))
    print(forward_kinematics(phi))
