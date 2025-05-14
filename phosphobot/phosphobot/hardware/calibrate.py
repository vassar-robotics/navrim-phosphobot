from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus  # type: ignore
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot  # type: ignore

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
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
)
