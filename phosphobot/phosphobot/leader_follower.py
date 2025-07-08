import asyncio
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

from phosphobot.control_signal import ControlSignal
from phosphobot.hardware import SO100Hardware, RemotePhosphobot


@dataclass
class RobotPair:
    leader: SO100Hardware | RemotePhosphobot
    follower: SO100Hardware | RemotePhosphobot


async def leader_follower_loop(
    robot_pairs: list[RobotPair],
    control_signal: ControlSignal,
    invert_controls: bool,
    enable_gravity_compensation: bool,
    compensation_values: dict[str, int] | None,
):
    """
    Background task that implements leader-follower control:
    - Makes the follower mirror the leader's current joint positions
    
    Note: Gravity compensation using PyBullet has been removed from this project.
    """
    logger.info("Starting leader-follower control.")
    logger.warning("Gravity compensation is no longer supported (pybullet removed). Forcing it to False.")
    enable_gravity_compensation = False
    loop_period = 1 / 150 if not enable_gravity_compensation else 1 / 60

    # Check if the initial position is set, otherwise move them
    wait_for_initial_position = False
    for pair in robot_pairs:
        for robot in [pair.leader, pair.follower]:
            if robot.initial_position is None or robot.initial_orientation_rad is None:
                logger.warning(
                    f"Initial position or orientation not set for {robot.name} {robot.device_name}"
                    "Moving to initial position before starting leader-follower control."
                )
                robot.enable_torque()
                await robot.move_to_initial_position()
                wait_for_initial_position = True
    if wait_for_initial_position:
        # Give some time for the robots to move to initial position
        await asyncio.sleep(1)

    # Enable torque if using gravity compensation, and on the follower
    for pair in robot_pairs:
        leader = pair.leader
        follower = pair.follower

        follower.enable_torque()
        if not enable_gravity_compensation:
            leader.disable_torque()
            p_gains = [12, 12, 12, 12, 12, 12]
            d_gains = [32, 32, 32, 32, 32, 32]
            default_p_gains = [12, 20, 20, 20, 20, 20]
            default_d_gains = [36, 36, 36, 32, 32, 32]

            for i in range(6):
                follower._set_pid_gains_motors(
                    servo_id=i + 1,
                    p_gain=p_gains[i],
                    i_gain=0,
                    d_gain=d_gains[i],
                )
                await asyncio.sleep(0.05)
        else:
            # Gravity compensation has been removed - pybullet dependency
            logger.warning("Gravity compensation is no longer supported (pybullet removed).")
            control_signal.stop()
            return

    # Main control loop
    while control_signal.is_in_loop():
        start_time = time.perf_counter()

        for pair in robot_pairs:
            leader = pair.leader
            follower = pair.follower

            # Control loop parameters
            num_joints = len(leader.actuated_joints)
            joint_indices = list(range(num_joints))

            # Get leader's current joint positions
            pos_rad = leader.read_joints_position(unit="rad")

            if any(np.isnan(pos_rad)):
                logger.warning(
                    "Leader joint positions contain NaN values. Skipping this iteration."
                )
                continue

            # Simple leader-follower control: follower mirrors leader's position
            if invert_controls:
                pos_rad[0] = -pos_rad[0]
            follower.write_joint_positions(list(pos_rad), unit="rad")

        # Maintain loop frequency
        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, loop_period - elapsed)
        await asyncio.sleep(sleep_time)

    # Cleanup: Reset leader's PID gains to default for all six motors
    for pair in robot_pairs:
        leader = pair.leader
        follower = pair.follower

        leader.enable_torque()
        default_p_gains = [12, 20, 20, 20, 20, 20]
        default_d_gains = [36, 36, 36, 32, 32, 32]
        for i in range(6):
            leader._set_pid_gains_motors(
                servo_id=i + 1,
                p_gain=default_p_gains[i],
                i_gain=0,
                d_gain=default_d_gains[i],
            )
            await asyncio.sleep(0.05)

        # Disable torque
        leader.disable_torque()
        follower.disable_torque()

    logger.info("Leader-follower control stopped")
