from fastapi import Depends
from phosphobot.recorder import Recorder

from teleop.camera import get_all_cameras
from teleop.robot import RobotConnectionManager, get_rcm

recorder = None  # Global variable to store the recorder instance


def get_recorder(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> Recorder:
    """
    Return the global recorder instance.
    """
    global recorder

    if recorder is not None:
        return recorder
    else:
        robots = rcm.robots
        cameras = get_all_cameras()
        recorder = Recorder(
            robots=robots,  # type: ignore
            cameras=cameras,
        )
        return recorder
