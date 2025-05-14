from phosphobot_old.camera import AllCameras
from phosphobot_old.configs import config

cameras = None


def get_all_cameras() -> AllCameras:
    """
    Return the global AllCameras instance.
    """
    global cameras

    if not cameras:
        cameras = AllCameras(disabled_cameras=config.DEFAULT_CAMERAS_TO_DISABLE)

    return cameras
