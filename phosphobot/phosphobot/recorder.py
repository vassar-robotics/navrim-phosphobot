from fastapi import Depends
from phosphobot_old.recorder import Recorder

from phosphobot_old.camera import get_all_cameras
from phosphobot_old.robot import RobotConnectionManager, get_rcm

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
