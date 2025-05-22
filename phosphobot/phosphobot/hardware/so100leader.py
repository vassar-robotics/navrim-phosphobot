from phosphobot.hardware import SO100Hardware
import numpy as np


class SO100LeaderHardware(SO100Hardware):
    name = "so-100-leader"

    END_EFFECTOR_LINK_INDEX = 5

    # Measured offset on the servos
    CALIBRATION_POSITION = [
        np.pi / 2,
        np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        np.pi / 2,
    ]
