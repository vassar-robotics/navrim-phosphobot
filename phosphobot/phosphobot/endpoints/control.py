import asyncio
import datetime
import json
from copy import copy
from typing import Literal, cast

import json_numpy  # type: ignore
import numpy as np
from dateutil import parser  # type: ignore
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from loguru import logger
from scipy.spatial.transform import Rotation as R

from phosphobot.ai_control import CustomAIControlSignal, setup_ai_control
from phosphobot.camera import AllCameras, get_all_cameras
from phosphobot.control_signal import ControlSignal
from phosphobot.hardware.base import BaseManipulator
from phosphobot.leader_follower import RobotPair, leader_follower_loop
from phosphobot.models import (
    AIControlStatusResponse,
    AIStatusRequest,
    AIStatusResponse,
    AppControlData,
    CalibrateResponse,
    EndEffectorPosition,
    FeedbackRequest,
    JointsReadRequest,
    JointsReadResponse,
    JointsWriteRequest,
    MoveAbsoluteRequest,
    RelativeEndEffectorPosition,
    RobotConnectionRequest,
    SpawnStatusResponse,
    StartAIControlRequest,
    StartLeaderArmControlRequest,
    StartServerRequest,
    StatusResponse,
    TorqueControlRequest,
    TorqueReadResponse,
    UDPServerInformationResponse,
    VoltageReadResponse,
)
from phosphobot.robot import (
    RemotePhosphobot,
    RobotConnectionManager,
    SO100Hardware,
    get_rcm,
)
from phosphobot.api_supabase import get_client, user_is_logged_in
from phosphobot.teleoperation import (
    TeleopManager,
    UDPServer,
    get_teleop_manager,
    get_udp_server,
)
from phosphobot.utils import background_task_log_exceptions
from phosphobot.workaround.db import DatabaseManager

# This is used to send numpy arrays as JSON to OpenVLA server
json_numpy.patch()

router = APIRouter(tags=["control"])


# Object that controls the global /auto, /gravity state in a thread safe way
ai_control_signal = CustomAIControlSignal()
gravity_control = ControlSignal()
leader_follower_control = ControlSignal()
vr_control_signal = ControlSignal()


@router.post(
    "/move/init",
    response_model=StatusResponse,
    summary="Initialize Robot",
    description="Initialize the robot to its initial position before starting the teleoperation.",
)
async def move_init(
    robot_id: int | None = None,
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
):
    """
    Initialize the robot to its initial position before starting the teleoperation.
    """
    await teleop_manager.move_init(robot_id=robot_id)
    return StatusResponse()


# HTTP POST endpoint
@router.post(
    "/move/teleop",
    response_model=StatusResponse,
    summary="Teleoperation Control",
)
async def move_teleop_post(
    control_data: AppControlData,
    robot_id: int | None = None,
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
) -> StatusResponse:
    teleop_manager.robot_id = robot_id
    await teleop_manager.process_control_data(control_data)
    return StatusResponse()


# WebSocket endpoint
@router.websocket("/move/teleop/ws")
async def move_teleop_ws(
    websocket: WebSocket,
    rcm: RobotConnectionManager = Depends(get_rcm),
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
):
    teleop_manager.robot_id = None

    if not await rcm.robots:
        raise HTTPException(status_code=400, detail="No robot connected")

    await websocket.accept()

    vr_control_signal.start()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                control_data = AppControlData.model_validate_json(data)
                await teleop_manager.process_control_data(control_data)
                await teleop_manager.send_status_updates(websocket)
            except json.JSONDecodeError as e:
                logger.error(f"WebSocket JSON error: {e}")

    except WebSocketDisconnect:
        logger.warning("WebSocket client disconnected")

    vr_control_signal.stop()


@router.post("/move/teleop/udp", response_model=UDPServerInformationResponse)
async def move_teleop_udp(
    udp_server: UDPServer = Depends(get_udp_server),
    teleop_manager: TeleopManager = Depends(get_teleop_manager),
):
    """
    Start a UDP server to send and receive teleoperation data to the robot.
    """
    teleop_manager.robot_id = None
    udp_server_info = await udp_server.init()
    return udp_server_info


@router.post("/move/teleop/udp/stop", response_model=StatusResponse)
async def stop_teleop_udp(
    udp_server: UDPServer = Depends(get_udp_server),
):
    """
    Stop the UDP server main loop.
    """
    udp_server.stop()
    return StatusResponse()


@router.post(
    "/move/absolute",
    response_model=StatusResponse,
    summary="Move to Absolute Position",
    description="Move the robot to an absolute position specified by the end-effector (in centimeters and degrees). "
    + "Make sure to call `/move/init` before using this endpoint.",
)
async def move_to_absolute_position(
    query: MoveAbsoluteRequest,
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Data: position
    Update the robot position based on the received data
    """
    robot = await rcm.get_robot(robot_id)

    # Divide by 100 to convert from cm to m
    query.x = query.x / 100 if query.x is not None else 0
    query.y = query.y / 100 if query.y is not None else 0
    query.z = query.z / 100 if query.z is not None else 0

    if hasattr(robot, "control_gripper") and query.open is not None:
        # If the robot has a control_gripper method, use it to open/close the gripper
        background_tasks.add_task(
            background_task_log_exceptions(robot.control_gripper),
            open_command=query.open,
        )

    initial_position = getattr(robot, "initial_position", None)
    initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
    if initial_position is None or initial_orientation_rad is None:
        await robot.move_to_initial_position()
        initial_position = getattr(robot, "initial_position", None)
        initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
        if initial_position is None or initial_orientation_rad is None:
            raise HTTPException(
                status_code=400,
                detail=f"Robot {robot.name} .move_to_initial_position() did not set initial position or orientation: {initial_position=}, {initial_orientation_rad=}",
            )

    if hasattr(robot, "forward_kinematics"):
        # If the robot has a forward_kinematics method, use it to move more precisely to the target
        current_position, current_orientation = robot.forward_kinematics()

        target_controller_position = np.array([query.x, query.y, query.z])
        target_position = initial_position + target_controller_position
        position_residual = np.linalg.norm(current_position - target_position)

        # angle
        if query.rx is not None and query.ry is not None and query.rz is not None:
            if robot.name == "so100":
                # We invert rx and ry
                target_controller_orientation = np.array([query.ry, query.rx, query.rz])
            elif robot.name == "agilex-piper":
                rotation = R.from_euler("y", -90, degrees=True)
                target_controller_orientation = rotation.apply(
                    [query.rx, query.ry, query.rz]
                )
            else:
                target_controller_orientation = np.array([query.rx, query.ry, query.rz])

            # Convert from degrees to radians
            target_controller_orientation_rad = np.deg2rad(
                target_controller_orientation
            )

            target_orientation_rad = (
                initial_orientation_rad + target_controller_orientation_rad
            )
            use_angles = True
        else:
            target_orientation_rad = None
            use_angles = False

        orientation_residual: float
        if use_angles:
            orientation_residual = float(
                np.linalg.norm(current_orientation - target_orientation_rad)
            )
        else:
            orientation_residual = 0

        async def try_moving_to_target():
            nonlocal \
                current_position, \
                current_orientation, \
                position_residual, \
                orientation_residual

            num_trials = 0
            while (
                position_residual > query.position_tolerance
                or orientation_residual > query.orientation_tolerance
            ) and num_trials <= query.max_trials - 1:
                if num_trials > 0:
                    await asyncio.sleep(0.03 + 0.2 / (num_trials + 1))

                logger.debug(f"Trial {num_trials + 1}")
                num_trials += 1
                await robot.move_robot_absolute(
                    target_position=target_position,
                    target_orientation_rad=target_orientation_rad,
                    interpolate_trajectory=False,
                )
                current_position, current_orientation = robot.forward_kinematics()
                position_residual = np.linalg.norm(current_position - target_position)
                if use_angles:
                    orientation_residual = np.linalg.norm(
                        current_orientation - target_orientation_rad
                    )

        await try_moving_to_target()
    else:
        # Otherwise, run the move_robot_absolute method directly
        if query.rx is not None:
            query.rx = np.deg2rad(query.rx)
        if query.ry is not None:
            query.ry = np.deg2rad(query.ry)
        if query.rz is not None:
            query.rz = np.deg2rad(query.rz)
        await robot.move_robot_absolute(
            target_position=np.array([query.x, query.y, query.z]),
            target_orientation_rad=np.array([query.rx, query.ry, query.rz]),
        )

    return StatusResponse()


@router.post(
    "/move/relative",
    response_model=StatusResponse,
    summary="Move to Relative Position",
    description="Move the robot to a relative position based on received delta values (in centimeters and degrees).",
)
async def move_relative(
    data: RelativeEndEffectorPosition,
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Data: The delta sent by OpenVLA for example
    """

    # Convert units to meters
    data.x = data.x / 100 if data.x is not None else None
    data.y = data.y / 100 if data.y is not None else None
    data.z = data.z / 100 if data.z is not None else None

    robot = await rcm.get_robot(robot_id)

    if (
        data.x is None
        and data.y is None
        and data.z is None
        and data.rx is None
        and data.ry is None
        and data.rz is None
        and data.open is not None
    ):
        if hasattr(robot, "control_gripper"):
            # If the robot has a control_gripper method, use it to open/close the gripper
            robot.control_gripper(open_command=data.open)
            return StatusResponse()

    if hasattr(robot, "move_robot_relative"):
        # If the robot has a move_robot_relative method, use it
        target_orientation_rad = np.array(
            [
                np.deg2rad(u) if u is not None else None
                for u in [data.rx, data.ry, data.rz]
            ]
        )
        await robot.move_robot_relative(
            target_position=np.array([data.x, data.y, data.z]),
            target_orientation_rad=target_orientation_rad,
        )
        if hasattr(robot, "control_gripper") and data.open is not None:
            # If the robot has a control_gripper method, use it to open/close the gripper
            robot.control_gripper(open_command=data.open)
        return StatusResponse()

    # Call move_robot_absolute if the robot does not have move_robot_relative
    if not hasattr(robot, "forward_kinematics"):
        raise HTTPException(
            status_code=400,
            detail="Robot doesn't support move_robot_relative method or forward_kinematics method",
        )

    initial_position = getattr(robot, "initial_position", None)
    initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
    if initial_position is None or initial_orientation_rad is None:
        await robot.move_to_initial_position()
        initial_position = getattr(robot, "initial_position", None)
        initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
        if initial_position is None or initial_orientation_rad is None:
            raise HTTPException(
                status_code=400,
                detail=f"Robot {robot.name} .move_to_initial_position() did not set initial position or orientation: {initial_position=}, {initial_orientation_rad=}",
            )

    logger.info(f"Received relative data: {data}")
    delta_position = np.array([data.x, data.y, data.z])
    delta_orientation_euler_degrees = np.array([data.rx, data.ry, data.rz])
    open = data.open if data.open is not None else None

    # Call /move/absolute by adding the delta to the current position
    current_position, current_orientation = robot.forward_kinematics(
        sync_robot_pos=False
    )
    # Round to 3 decimals
    current_position = np.round(current_position, 3)
    current_orientation = np.round(current_orientation, 3)
    # Replace the None values in delta_position and delta_orientation_euler_degrees with 0
    delta_position = np.array([0 if v is None else v for v in delta_position])
    delta_orientation_euler_degrees = np.array(
        [0 if v is None else v for v in delta_orientation_euler_degrees]
    )

    target_position = current_position + delta_position - initial_position
    target_orientation = (
        np.rad2deg(current_orientation)
        + delta_orientation_euler_degrees
        - np.rad2deg(initial_orientation_rad)
    )

    # Round to 3 decimals
    target_position = np.round(target_position, 3)
    target_orientation = np.round(target_orientation, 3)

    logger.info(
        f"Target position: {target_position}. Target orientation: {target_orientation}"
    )

    await move_to_absolute_position(
        query=MoveAbsoluteRequest(
            x=target_position[0] * 100,
            y=target_position[1] * 100,
            z=target_position[2] * 100,
            rx=target_orientation[0],
            ry=target_orientation[1],
            rz=target_orientation[2],
            open=open,
            position_tolerance=1e-3,
            orientation_tolerance=1e-3,
            max_trials=1,
        ),
        background_tasks=background_tasks,
        robot_id=robot_id,
        rcm=rcm,
    )

    return StatusResponse()


@router.post(
    "/move/hello",
    response_model=StatusResponse,
    summary="Make the robot say hello (test endpoint)",
    description="Make the robot say hello by opening and closing its gripper. (Test endpoint)",
)
async def say_hello(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Make the robot say hello by opening and closing its gripper.
    """
    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "control_gripper"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support gripper control",
        )

    # Open and close the gripper
    robot.control_gripper(open_command=1)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=0.5)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=1)
    await asyncio.sleep(0.5)
    robot.control_gripper(open_command=0)

    return StatusResponse()


@router.post(
    "/move/sleep",
    response_model=StatusResponse,
    summary="Put the robot to its sleep position",
    description="Put the robot to its sleep position by giving direct instructions to joints. This function disables the torque.",
)
async def move_sleep(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Put the robot to its sleep position.
    """
    robot = await rcm.get_robot(robot_id)
    await robot.move_to_sleep()
    return StatusResponse()


@router.post(
    "/end-effector/read",
    response_model=EndEffectorPosition,
    summary="Read End-Effector Position",
    description="Retrieve the position, orientation, and open status of the robot's end effector. Only available for manipulators.",
)
async def end_effector_read(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> EndEffectorPosition:
    """
    Get the position, orientation, and open status of the end effector.
    """
    robot = await rcm.get_robot(robot_id)

    if not isinstance(robot, BaseManipulator):
        raise HTTPException(
            status_code=400,
            detail="Robot is not a manipulator and does not have an end effector",
        )

    initial_position = getattr(robot, "initial_position", None)
    initial_orientation_rad = getattr(robot, "initial_orientation_rad", None)
    if initial_position is None or initial_orientation_rad is None:
        raise HTTPException(
            status_code=400,
            detail=f"Call /move/init before using this endpoint for robot {robot.name}. ",
        )

    position, orientation, open_status = robot.get_end_effector_state()
    # Remove the initial position and orientation (used to zero the robot)
    position = position - initial_position
    orientation = orientation - initial_orientation_rad

    x, y, z = position
    rx, ry, rz = orientation

    # Convert position to centimeters
    x *= 100

    # Convert to degrees
    rx = np.rad2deg(rx)
    ry = np.rad2deg(ry)
    rz = np.rad2deg(rz)

    return EndEffectorPosition(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, open=open_status)


@router.post(
    "/voltage/read",
    response_model=VoltageReadResponse,
    summary="Read Voltage",
    description="Read the current voltage of the robot's motors.",
)
async def read_voltage(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
):
    """
    Read voltage of the robot.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "current_voltage"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading voltage",
        )

    voltage = robot.current_voltage()
    return VoltageReadResponse(
        current_voltage=voltage.tolist() if voltage is not None else None,
    )


@router.post(
    "/torque/read",
    response_model=TorqueReadResponse,
    summary="Read Torque",
    description="Read the current torque of the robot's joints.",
)
async def read_torque(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
):
    """
    Read torque of the robot.
    """
    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "current_torque"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading torque",
        )

    current_torque = robot.current_torque()

    # Replace NaN values with None and convert to list
    current_torque = [
        float(torque) if not np.isnan(torque) else None for torque in current_torque
    ]

    return TorqueReadResponse(current_torque=current_torque)


@router.post(
    "/torque/toggle",
    response_model=StatusResponse,
    summary="Toggle Torque",
    description="Enable or disable the torque of the robot.",
)
async def toggle_torque(
    request: TorqueControlRequest,
    robot_id: int | None = None,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Enable or disable the torque of the robot.
    """

    if robot_id is not None:
        robot = await rcm.get_robot(robot_id)

        if not hasattr(robot, "enable_torque") or not hasattr(robot, "disable_torque"):
            raise HTTPException(
                status_code=400,
                detail="Robot does not support torque control",
            )

        if request.torque_status:
            robot.enable_torque()
        else:
            robot.disable_torque()
        return StatusResponse()

    # If no robot_id is provided, toggle torque for all robots
    for robot in await rcm.robots:
        if not hasattr(robot, "enable_torque") or not hasattr(robot, "disable_torque"):
            logger.warning(
                f"Robot {robot.name} does not support torque control. Skipping."
            )
            continue
        else:
            if request.torque_status:
                robot.enable_torque()
            else:
                robot.disable_torque()

    return StatusResponse()


@router.post(
    "/joints/read",
    response_model=JointsReadResponse,
    summary="Read Joint Positions",
    description="Read the current positions of the robot's joints in radians and motor units.",
)
async def read_joints(
    request: JointsReadRequest | None = None,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> JointsReadResponse:
    """
    Read joint position.
    """
    if request is None:
        request = JointsReadRequest(unit="rad", joints_ids=None)

    robot = await rcm.get_robot(robot_id)

    if not hasattr(robot, "read_joints_position"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support reading joint positions",
        )

    current_units_position = robot.read_joints_position(
        unit=request.unit, joints_ids=request.joints_ids
    )
    # Replace NaN values with None and convert to list
    current_units_position = [
        float(angle) if not np.isnan(angle) else None
        for angle in current_units_position
    ]

    return JointsReadResponse(
        angles=current_units_position,
        unit=request.unit,
    )


@router.post(
    "/joints/write",
    response_model=StatusResponse,
    summary="Write Joint Positions",
    description="Move the robot's joints to the specified angles.",
)
async def write_joints(
    request: JointsWriteRequest,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Move the robot's joints to the specified angles.
    """
    robot = await rcm.get_robot(robot_id)
    if not hasattr(robot, "write_joint_positions"):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support writing joint positions",
        )

    robot = cast(BaseManipulator, robot)
    robot.write_joint_positions(
        angles=request.angles, unit=request.unit, joints_ids=request.joints_ids
    )

    return StatusResponse()


@router.post(
    "/calibrate",
    response_model=CalibrateResponse,
    summary="Calibrate Robot",
    description="Start the calibration sequence for the robot.",
)
async def calibrate(
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> CalibrateResponse:
    """
    Starts the calibration sequence of the robot.

    This endpoints disable torque. Move the robot to the positions you see in the simulator and call this endpoint again, until the calibration is complete.
    """
    robot = await rcm.get_robot(robot_id)

    if (
        not hasattr(robot, "calibrate")
        or not hasattr(robot, "calibration_current_step")
        or not hasattr(robot, "calibration_max_steps")
    ):
        raise HTTPException(
            status_code=400,
            detail="Robot does not support calibration",
        )

    if not robot.is_connected:
        raise HTTPException(status_code=400, detail="Robot is not connected")

    try:
        status, message = await robot.calibrate()
        current_step = robot.calibration_current_step
        total_nb_steps = robot.calibration_max_steps
        if status == "success":
            current_step = total_nb_steps
    except Exception as e:
        status = "error"
        current_step = getattr(robot, "calibration_current_step", 0)
        total_nb_steps = getattr(robot, "calibration_max_steps", 0)
        message = f"Calibration step {current_step}/{total_nb_steps} failed: {e}"

    return CalibrateResponse(
        calibration_status=status,
        message=message,
        current_step=current_step,
        total_nb_steps=total_nb_steps,
    )


@router.post(
    "/move/leader/start",
    response_model=StatusResponse,
    summary="Use the leader arm to control the follower arm",
    description="Use the leader arm to control the follower arm.",
)
async def start_leader_follower(
    request: StartLeaderArmControlRequest,
    background_tasks: BackgroundTasks,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Endpoint to start the leader-follower control.
    The first robot is the leader with gravity compensation enabled,
    and the second robot follows the leader's joint positions.
    """
    if leader_follower_control.is_in_loop():
        raise HTTPException(
            status_code=400,
            detail="Leader-follower control is already running. Call /move/leader/stop first.",
        )

    # Parse the robot IDs from the request
    robot_pairs: list[RobotPair] = []
    for i, robot_pair in enumerate(request.robot_pairs):
        if robot_pair.follower_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Follower ID is required for robot pair {i}.",
            )
        if robot_pair.leader_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Leader ID is required for robot pair {i}.",
            )
        leader = await rcm.get_robot(robot_pair.leader_id)
        follower = await rcm.get_robot(robot_pair.follower_id)

        if request.enable_gravity_compensation:
            # Only local SO100
            if not isinstance(leader, SO100Hardware):
                raise HTTPException(
                    status_code=400,
                    detail=f"Leader must be an instance of SO100Hardware for robot pair {i}.",
                )
            if not isinstance(follower, SO100Hardware):
                raise HTTPException(
                    status_code=400,
                    detail=f"Follower must be an instance of SO100Hardware for robot pair {i}.",
                )
        else:
            valid_robot_types = (
                SO100Hardware,
                RemotePhosphobot,
            )
            if not isinstance(leader, valid_robot_types):
                raise HTTPException(
                    status_code=400,
                    detail=f"Leader must be an instance of {valid_robot_types} for robot pair {i}.",
                )
            if not isinstance(follower, valid_robot_types):
                raise HTTPException(
                    status_code=400,
                    detail=f"Follower must be an instance of {valid_robot_types} for robot pair {i}.",
                )

        # TODO: Eventually add more config options individual for each pair
        robot_pairs.append(RobotPair(leader=leader, follower=follower))

    # Create control signal for managing the leader-follower operation
    leader_follower_control.start()

    # Add background task to run the control loop
    background_tasks.add_task(
        background_task_log_exceptions(leader_follower_loop),
        robot_pairs=robot_pairs,
        control_signal=leader_follower_control,
        invert_controls=request.invert_controls,
        enable_gravity_compensation=request.enable_gravity_compensation,
        compensation_values=request.gravity_compensation_values,
    )

    return StatusResponse(message="Leader-follower control started")


@router.post(
    "/move/leader/stop",
    response_model=StatusResponse,
    summary="Stop the leader-follower control",
    description="Stop the leader-follower control.",
)
async def stop_leader_follower(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Stop the leader-follower
    """
    if not leader_follower_control.is_in_loop():
        return StatusResponse(
            status="error", message="Leader-follower control is not running"
        )

    leader_follower_control.stop()
    return StatusResponse(message="Stopping leader-follower control")


@router.post("/gravity/start", response_model=StatusResponse)
async def start_gravity(
    background_tasks: BackgroundTasks,
    robot_id: int = 0,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Enable gravity compensation for the robot.
    """
    if gravity_control.is_in_loop():
        raise HTTPException(
            status_code=400, detail="Gravity control is already running"
        )

    if len(await rcm.robots) == 0:
        raise HTTPException(status_code=400, detail="No robot connected")

    robot = await rcm.get_robot(robot_id)
    if not isinstance(robot, SO100Hardware):
        raise HTTPException(
            status_code=400, detail="Gravity compensation is only for SO-100 robot"
        )

    gravity_control.start()

    # Add background task to run the control loop
    background_tasks.add_task(
        background_task_log_exceptions(robot.gravity_compensation_loop),
        control_signal=gravity_control,
    )
    return StatusResponse()


@router.post(
    "/gravity/stop",
    response_model=StatusResponse,
    summary="Stop the gravity compensation",
    description="Stop the gravity compensation.",
)
async def stop_gravity_compensation(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Stop the gravity compensation for all robots.
    """
    if not gravity_control.is_in_loop():
        return StatusResponse(status="error", message="Gravity control is not running")

    gravity_control.stop()
    return StatusResponse(message="Stopping gravity control")


@router.post(
    "/ai-control/status",
    response_model=AIStatusResponse,
    summary="Get the status of the auto control by AI",
    description="Get the status of the auto control by AI.",
)
async def fetch_auto_control_status(request: AIStatusRequest) -> AIStatusResponse:
    """
    Fetch the status of the auto control by AI
    """
    supabase_id: str | None = None
    supabase_status: Literal["stopped", "running", "paused", "waiting"] | None = None

    supabase_client = await get_client()
    try:
        user = await supabase_client.auth.get_user()
    except Exception as e:
        logger.warning(f"Failed to loggin: {e}")
        raise HTTPException(status_code=401, detail="Not authenticated")

    if user is not None:
        with DatabaseManager.get_instance() as db:
            session_data = db.get_latest_ai_control_session_by_user(user.user.id)
            if session_data is not None:
                supabase_id = session_data["id"]
                supabase_status = session_data["status"]
                if (
                    session_data["servers"] is not None
                    and session_data["servers"]["status"] == "stopped"
                ):
                    supabase_status = "stopped"
                    db.update_ai_control_session_status(supabase_id, supabase_status)

                if ai_control_signal.status == "stopped" and supabase_status == "running":
                    supabase_status = "stopped"
                    db.update_ai_control_session_status(supabase_id, supabase_status)
        # supabase_response = (
        #     await supabase_client.table("ai_control_sessions")
        #     .select("*, servers(status)")
        #     .eq("user_id", user.user.id)
        #     .order("created_at", desc=True)
        #     .limit(1)
        #     .execute()
        # )
        # if supabase_response.data:
        #     supabase_data = supabase_response.data[0]
        #     supabase_id = supabase_data["id"]
        #     supabase_status = supabase_data["status"]
        #     # If server status is stopped, set status to stopped
        #     if (
        #         supabase_data["servers"] is not None
        #         and supabase_data["servers"]["status"] == "stopped"
        #     ):
        #         # Set the backend status to stopped
        #         supabase_status = "stopped"
        #         # Update the status in the database
        #         await (
        #             supabase_client.table("ai_control_sessions")
        #             .update({"status": supabase_status})
        #             .eq("id", supabase_id)
        #             .execute()
        #         )
        #     # If ai-control signal is stopped but remote status is running, set the remote status to stopped
        #     if ai_control_signal.status == "stopped" and supabase_status == "running":
        #         supabase_status = "stopped"
        #         # Update the status in the database
        #         await (
        #             supabase_client.table("ai_control_sessions")
        #             .update({"status": supabase_status})
        #             .eq("id", supabase_id)
        #             .execute()
        #         )

    # Situation 1: There is already a different, running process in backend
    if (
        supabase_id is not None
        and supabase_id != ai_control_signal.id
        and (
            supabase_status == "running"
            or supabase_status == "paused"
            or supabase_status == "waiting"
        )
    ):
        # if started less than 10 minutes ago, return the backend status
        created_at = parser.isoparse(session_data["created_at"])
        # created_at = parser.isoparse(supabase_data["created_at"])
        if (
            datetime.datetime.now(datetime.timezone.utc) - created_at
        ).total_seconds() < 600:
            ai_control_signal.id = supabase_id
            return AIStatusResponse(id=supabase_id, status=supabase_status)

    # Situation 2: The backend says the process should be waiting or stopped
    if (
        supabase_id is not None
        and supabase_id == ai_control_signal.id
        and supabase_status == "stopped"
    ):
        # Stop the local process
        ai_control_signal.status = supabase_status
        return AIStatusResponse(id=supabase_id, status=supabase_status)

    # Situation 3: return the current local status
    return AIStatusResponse(id=ai_control_signal.id, status=ai_control_signal.status)


@router.post(
    "/ai-control/spawn",
    response_model=SpawnStatusResponse,
    summary="Start an inference server",
    description="Start an inference server and return the server info.",
)
async def spawn_inference_server(
    query: StartServerRequest,
    rcm: RobotConnectionManager = Depends(get_rcm),
    all_cameras: AllCameras = Depends(get_all_cameras),
    session=Depends(user_is_logged_in),
) -> SpawnStatusResponse:
    """
    Start an inference server and return the server info.
    """

    supabase_client = await get_client()
    await supabase_client.auth.get_user()

    robots_to_control = copy(await rcm.robots)
    for robot in await rcm.robots:
        if (
            hasattr(robot, "SERIAL_ID")
            and query.robot_serials_to_ignore is not None
            and robot.SERIAL_ID in query.robot_serials_to_ignore
        ):
            robots_to_control.remove(robot)
        if not isinstance(robot, BaseManipulator):
            logger.warning(
                f"Robot {robot.name} is not a manipulator and is not supported for AI control. Skipping."
            )
            robots_to_control.remove(robot)

    assert all(
        isinstance(robot, BaseManipulator) for robot in robots_to_control
    ), "All robots must be manipulators for AI control"

    # Get the modal host and port here
    _, _, server_info = await setup_ai_control(
        robots=robots_to_control,  # type: ignore
        all_cameras=all_cameras,
        model_id=query.model_id,
        init_connected_robots=False,
        model_type=query.model_type,
        ai_control_signal_id=ai_control_signal.id,
    )

    return SpawnStatusResponse(message="ok", server_info=server_info)


@router.post(
    "/ai-control/start",
    response_model=AIControlStatusResponse,
    summary="Start the auto control by AI",
    description="Start the auto control by AI.",
)
async def start_auto_control(
    query: StartAIControlRequest,
    background_tasks: BackgroundTasks,
    rcm: RobotConnectionManager = Depends(get_rcm),
    all_cameras: AllCameras = Depends(get_all_cameras),
    session=Depends(user_is_logged_in),
) -> AIControlStatusResponse:
    """
    Start the auto control by AI
    """
    if ai_control_signal.is_in_loop():
        return AIControlStatusResponse(
            status="error",
            message="Auto control is already running",
            ai_control_signal_id=ai_control_signal.id,
            ai_control_signal_status=ai_control_signal.status,
            server_info=None,
        )

    ai_control_signal.new_id()
    ai_control_signal.start()

    supabase_client = await get_client()
    user = await supabase_client.auth.get_user()
    # await (
    #     supabase_client.table("ai_control_sessions")
    #     .upsert(
    #         {
    #             "id": ai_control_signal.id,
    #             "user_id": user.user.id,
    #             "user_email": user.user.email,
    #             "model_type": query.model_type,
    #             "model_id": query.model_id,
    #             "prompt": query.prompt,
    #             "status": "waiting",
    #         }
    #     )
    #     .execute()
    # )

    # SQLite equivalent
    with DatabaseManager.get_instance() as db:
        db.insert_ai_control_session({
            "id": ai_control_signal.id,
            "user_id": user.user.id,
            "user_email": user.user.email,
            "model_type": query.model_type,
            "model_id": query.model_id,
            "prompt": query.prompt,
            "status": "waiting",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "setup_success": False,
        })

    robots_to_control = copy(await rcm.robots)
    for robot in await rcm.robots:
        if (
            hasattr(robot, "SERIAL_ID")
            and query.robot_serials_to_ignore is not None
            and robot.SERIAL_ID in query.robot_serials_to_ignore
        ):
            robots_to_control.remove(robot)
        if not isinstance(robot, BaseManipulator):
            logger.warning(
                f"Robot {robot.name} is not a manipulator and is not supported for AI control. Skipping."
            )
            robots_to_control.remove(robot)

    assert all(
        isinstance(robot, BaseManipulator) for robot in robots_to_control
    ), "All robots must be manipulators for AI control"

    # Get the modal host and port here
    model, model_spawn_config, server_info = await setup_ai_control(
        robots=robots_to_control,  # type: ignore
        all_cameras=all_cameras,
        model_type=query.model_type,
        model_id=query.model_id,
        cameras_keys_mapping=query.cameras_keys_mapping,
        ai_control_signal_id=ai_control_signal.id,
        verify_cameras=query.verify_cameras,
    )

    # Add a flag: successful setup
    # await (
    #     supabase_client.table("ai_control_sessions")
    #     .update(
    #         {
    #             "setup_success": True,
    #             "server_id": server_info.server_id,
    #         }
    #     )
    #     .eq("id", ai_control_signal.id)
    #     .execute()
    # )

    with DatabaseManager.get_instance() as db:
        db.update_ai_control_session_setup(
            ai_control_signal.id,
            setup_success=True,
            server_id=server_info.server_id
        )

    background_tasks.add_task(
        model.control_loop,
        robots=robots_to_control,
        control_signal=ai_control_signal,
        prompt=query.prompt,
        all_cameras=all_cameras,
        model_spawn_config=model_spawn_config,
        speed=query.speed,
        cameras_keys_mapping=query.cameras_keys_mapping,
        detect_instruction=query.prompt,
        selected_camera_id=query.selected_camera_id,
    )

    return AIControlStatusResponse(
        status="ok",
        message=f"Starting AI control with id: {ai_control_signal.id}",
        server_info=server_info,
        ai_control_signal_id=ai_control_signal.id,
        ai_control_signal_status="waiting",
    )


@router.post(
    "/ai-control/stop",
    response_model=StatusResponse,
    summary="Stop the auto control by AI",
    description="Stop the auto control by AI.",
)
async def stop_auto_control(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Stop the auto control by AI
    """
    if not ai_control_signal.is_in_loop():
        return StatusResponse(message="Auto control is not running")

    ai_control_signal.stop()
    return StatusResponse(message="Stopping auto control")


@router.post(
    "/ai-control/pause",
    response_model=StatusResponse,
    summary="Pause the auto control by AI",
    description="Pause the auto control by AI.",
)
async def pause_auto_control(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Pause the auto control by AI
    """
    if not ai_control_signal.is_in_loop():
        return StatusResponse(message="Auto control is not running")

    ai_control_signal.status = "paused"
    return StatusResponse(message="Pausing auto control")


@router.post(
    "/ai-control/resume",
    response_model=StatusResponse,
    summary="Resume the auto control by AI",
    description="Resume the auto control by AI.",
)
async def resume_auto_control(
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Resume the auto control by AI
    """
    if ai_control_signal.status == "running":
        return StatusResponse(message="Auto control is already running")

    ai_control_signal.status = "running"
    return StatusResponse(message="Resuming auto control")


@router.post(
    "/ai-control/feedback",
    response_model=StatusResponse,
    summary="Feedback about the auto control session",
)
async def feedback_auto_control(
    request: FeedbackRequest,
    session=Depends(user_is_logged_in),
) -> StatusResponse:
    """
    Feedback about the auto control session
    """
    supabase_client = await get_client()

    # await (
    #     supabase_client.table("ai_control_sessions")
    #     .update(
    #         {
    #             "feedback": request.feedback,
    #         }
    #     )
    #     .eq("id", request.ai_control_id)
    #     .execute()
    # )

    with DatabaseManager.get_instance() as db:
        db.update_ai_control_session_feedback(request.ai_control_id, request.feedback)

    return StatusResponse(message="Feedback sent")


@router.post("/robot/add-connection", response_model=StatusResponse)
async def add_robot_connection(
    query: RobotConnectionRequest,
    rcm: RobotConnectionManager = Depends(get_rcm),
) -> StatusResponse:
    """
    Manually add a robot connection to the robot manager.
    Useful for adding robot that are accessible only via WiFi, for example.
    """
    try:
        await rcm.add_connection(
            robot_name=query.robot_name,
            connection_details=query.connection_details,
        )
        return StatusResponse(
            status="ok", message=f"Robot connection to {query.robot_name} added"
        )
    except Exception as e:
        logger.error(f"Failed to add robot connection: {e}")
        raise HTTPException(
            status_code=400, detail=f"Failed to add robot connection: {e}"
        )
