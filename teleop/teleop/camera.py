from phosphobot.camera import AllCameras
from phosphobot.configs import config

cameras = None


def get_all_cameras() -> AllCameras:
    """
    Return the global AllCameras instance.
    """
    global cameras

    if not cameras:
        cameras = AllCameras(disabled_cameras=config.DEFAULT_CAMERAS_TO_DISABLE)

    return cameras
