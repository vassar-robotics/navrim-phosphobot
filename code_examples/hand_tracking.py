import sys
import cv2
import time
import signal
import requests
import numpy as np
import mediapipe as mp  # type: ignore

# Configurations
PI_IP: str = "127.0.0.1"
PI_PORT: int = 8080

# Initialize MediaPipe Hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mp_draw = mp.solutions.drawing_utils


# Handle Ctrl+C to exit the program gracefully
def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Function to call the API
def call_to_api(endpoint: str, data: dict = {}):
    response = requests.post(f"http://{PI_IP}:{PI_PORT}/move/{endpoint}", json=data)
    return response.json()


def calculate_hand_closure(hand_landmarks):
    """
    Calculate if the hand is closed based on thumb and index finger distance
    Returns a value between 0 (open) and 1 (closed)
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    distance = np.sqrt(
        (thumb_tip.x - index_tip.x) ** 2
        + (thumb_tip.y - index_tip.y) ** 2
        + (thumb_tip.z - index_tip.z) ** 2
    )

    # Normalize distance (these values might need adjustment based on the hand size)
    normalized = np.clip(1.0 - (distance * 5), 0, 1)
    return normalized


# 1 - Initialize the robot
call_to_api("init")
print("Initializing robot")
time.sleep(2)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Get camera frame dimensions
frame_width: float = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height: float = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the workspace boundaries (in meters)
WORKSPACE_Y = 0.4
WORKSPACE_Z = 0.2


def map_to_robot_coordinates(hand_x, hand_y):
    """
    Map normalized hand coordinates to robot workspace coordinates
    We match the hand x coordinate to the robot y coordinate
    And the hand y coordinate to the robot z coordinate
    """
    robot_y = ((0.5 - hand_x) * 2) * (WORKSPACE_Y / 2)
    robot_z = ((0.5 - hand_y) * 2) * (WORKSPACE_Z / 2)
    return robot_y, robot_z


# Previous position for smoothing, this helps make the robot movements less jerky
prev_pos = {"y": 0, "z": 0}
smoothing_factor = 0.5

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to capture frame")
            continue

        image = cv2.flip(image, 1)  # The front camera is inverted
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB image
        results = hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                palm = hand_landmarks.landmark[0]
                hand_closed = calculate_hand_closure(hand_landmarks)

                robot_y, robot_z = map_to_robot_coordinates(palm.x, palm.y)

                robot_y = prev_pos["y"] * smoothing_factor + robot_y * (
                    1 - smoothing_factor
                )
                robot_z = prev_pos["z"] * smoothing_factor + robot_z * (
                    1 - smoothing_factor
                )

                prev_pos = {"y": robot_y, "z": robot_z}

                call_to_api(
                    "absolute",
                    {
                        "x": -0.05,
                        "y": robot_y,
                        "z": robot_z,
                        "rx": 0,
                        "ry": 0,
                        "rz": 0,
                        "open": 1 - hand_closed,
                    },
                )

                cv2.putText(
                    image,
                    f"Position: (y:{robot_y:.3f}, z:{robot_z:.3f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    f"Grip: {'Closed' if hand_closed > 0.5 else 'Open'}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Hand Tracking", image)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nExiting gracefully...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
