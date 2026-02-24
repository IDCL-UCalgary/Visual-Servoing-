#!/usr/bin/env python3
"""
Kinova Orange Pick and Place with YOLO Detection
Combines visual servoing approach with Kinova arm control
"""

import collections
import collections.abc

# Fix for Python 3.10+ compatibility with old protobuf
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence
if not hasattr(collections, "MutableSet"):
    collections.MutableSet = collections.abc.MutableSet

import utilities
import sys
import os
import time
import threading
import numpy as np
import cv2
from ultralytics import YOLO

# Kinova imports
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
from kortex_api.Exceptions.KServerException import KServerException

# Local imports (your existing files)
from T import T
from Rx import Rx
from Ry import Ry
from Rz import Rz
from get_angles_pose import get_angles_pose
#from close_gripper import close_gripper
from trust_constr import trust_constr
from angle_guess import angle_guess
from smallest_angular_distance import smallest_angular_distance_limits, smallest_angular_distance_nolimits

from flask import Flask, Response
import threading

app = Flask(__name__)

latest_frame = None
frame_lock = threading.Lock()

cap = None
camera_running = True


# =============================================================================
# CONFIGURATION
# =============================================================================

GRASP_FORWARD_OFFSET = -0.065  # meters (start with 3 cm)

# YOLO Model
YOLO_WEIGHTS = "/home/eeel126/Desktop/last.pt"  # UPDATE THIS PATH
CONF_THRESHOLD = 0.6

# Camera Parameters (from your calibration)
CAMERA_MATRIX = np.array([[643.77840171, 0, 311.5361204],
                          [0, 643.99115635, 248.9306098],
                          [0, 0, 1]], dtype=np.float32)

DIST_COEFFS = np.array([0.01331069, 0.1154656, 0.00361715, 
                        -0.00244894, -1.04813852], dtype=np.float32)

# Orange Properties
ORANGE_DIAMETER = 0.035  # meters (average orange ~75mm)
GRIPPER_WIDTH_ORANGE = 0.034  # meters (65mm - slightly less than diameter)
GRIPPER_WIDTH_OPEN = 0.1  # meters (85mm - fully open)

# Fixed Viewing Location
# VIEW_POSITION = {
#     "x": 0.3264,
#     "y": -0.0058,
#     "z": 0.3219,
#     "theta_x": 134.60,
#     "theta_y": -3.92,
#     "theta_z": 83.39
# }

VIEW_JOINT_ANGLES = [
    12.5875825881958,
    337.91107177734375,
    173.00613403320312,
    219.7299346923828,
    346.29766845703125,
    344.22088623046875,
    95.72096252441406]

# Fixed Drop-off Location (MEASURE AND UPDATE THESE!) UPDATE==================================================================================================================================
DROP_OFF_POSITION = np.array([[0, 1, 0, 0.2557],   # X position
                               [1, 0, 0, -0.3963],   # Y position  
                               [0, 0, -1, 0.0379],  # Z height
                               [0, 0, 0, 1]])

DROP_OFF_JOINT_ANGLES = [
    49.6326,
    29.7240,
    190.3669,
    214.7743,
    356.8244,
    84.3642,
    89.3134
]


# Movement Parameters
TIMEOUT_DURATION = 30  # seconds
APPROACH_HEIGHT = 0.25  # meters - hover height before descending
PICKUP_HEIGHT_OFFSET = 0.02  # meters - how much above orange center to stop
MOVEMENT_TIME = 4.5  # seconds for joint speed movements

# Camera offset from end-effector (from your camera_coor function)
CAMERA_OFFSET = T(0, 0.054, -0.138)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def enable_servo_mode(base):
    servo_mode = Base_pb2.ServoingModeInformation()
    servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(servo_mode)

def check_for_end_or_abort(e):
    """Closure for action notification"""
    def check(notification, e=e):
        if notification.action_event == Base_pb2.ACTION_END \
           or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def mjpeg_generator():
    global latest_frame

    while True:
        with frame_lock:
            frame = latest_frame

        if frame is None:
            time.sleep(0.01)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


@app.route('/video')
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def camera_loop(model):
    global latest_frame, cap, camera_running

    #cap = cv2.VideoCapture("rtsp://192.168.1.10/color")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(
        "rtsp://192.168.1.10/color",
        cv2.CAP_FFMPEG
    )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("📷 Camera thread started")

    while camera_running:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Frame read failed — reconnecting...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture("rtsp://192.168.1.10/color", cv2.CAP_FFMPEG)
            continue


        # Run YOLO detection continuously
        detections = detect_oranges(frame, model)

        for det in detections:
            x1, y1, x2, y2, cx, cy, bbox_width, conf = det

            cv2.rectangle(frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0), 2)

            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            cv2.putText(frame,
                        f"Cylinder {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)


        # Always update stream frame
        with frame_lock:
            latest_frame = frame.copy()

    cap.release()



def move_to_home_position(base):
    """Move arm to home position"""
    print("Moving to home position...")
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle
            break
    
    if action_handle is None:
        print("ERROR: Cannot find Home position")
        return False
    
    e = threading.Event()
    notification_handle = None
    
    try:
        notification_handle = base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )
        
        base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        
        if finished:
            print("Home position reached")
        else:
            print("Timeout reaching home position")
            
    except Exception as ex:
        print(f"Error moving to home: {ex}")
        finished = False
    finally:
        # Safely unsubscribe only if handle was created
        if notification_handle is not None:
            try:
                base.Unsubscribe(notification_handle)
            except:
                pass  # Ignore unsubscribe errors
    
    return finished



# def move_cartesian(base, base_cyclic, x, y, z, theta_x, theta_y, theta_z):
#     """Move to Cartesian position"""
#     print(f"Moving to: x={x:.3f}, y={y:.3f}, z={z:.3f}")
    
#     action = Base_pb2.Action()
#     action.name = "Cartesian movement"
#     action.application_data = ""
    
#     cartesian_pose = action.reach_pose.target_pose
#     cartesian_pose.x = x
#     cartesian_pose.y = y
#     cartesian_pose.z = z
#     cartesian_pose.theta_x = theta_x
#     cartesian_pose.theta_y = theta_y
#     cartesian_pose.theta_z = theta_z
    
#     e = threading.Event()
#     notification_handle = None
    
#     try:
#         notification_handle = base.OnNotificationActionTopic(
#             check_for_end_or_abort(e),
#             Base_pb2.NotificationOptions()
#         )
        
#         base.ExecuteAction(action)
#         finished = e.wait(TIMEOUT_DURATION)
        
#         if finished:
#             print("✓ Movement completed")
#         else:
#             print("✗ Movement timeout")
            
#     except Exception as ex:
#         print(f"✗ Error during movement: {ex}")
#         finished = False
#     finally:
#         # Safely unsubscribe only if handle was created
#         if notification_handle is not None:
#             try:
#                 base.Unsubscribe(notification_handle)
#             except:
#                 pass  # Ignore unsubscribe errors
    
#     return finished

def move_cartesian_velocity(base, base_cyclic,
                            target_x, target_y, target_z,
                            theta_x, theta_y, theta_z,
                            max_linear_speed=0.35,
                            max_angular_speed=60.0):

    enable_servo_mode(base)

    dt = 0.01

    # Gains (critically damped style)
    Kp_pos = 2.2
    Kd_pos = 1.1

    Kp_ang = 3.6
    Kd_ang = 1.8

    tolerance_pos = 0.05
    tolerance_ang = 7.5

    # feedback = base_cyclic.RefreshFeedback()
    # print(dir(feedback.base))
    # exit()

    while True:

        feedback = base_cyclic.RefreshFeedback()

        # ---- Current Pose (CYCLIC ONLY) ----
        current_x = feedback.base.tool_pose_x
        current_y = feedback.base.tool_pose_y
        current_z = feedback.base.tool_pose_z

        current_theta_x = feedback.base.tool_pose_theta_x
        current_theta_y = feedback.base.tool_pose_theta_y
        current_theta_z = feedback.base.tool_pose_theta_z

        # ---- Current Velocities ----
        current_lin_vel = np.array([
            feedback.base.tool_twist_linear_x,
            feedback.base.tool_twist_linear_y,
            feedback.base.tool_twist_linear_z
        ])

        current_ang_vel = np.array([
            feedback.base.tool_twist_angular_x,
            feedback.base.tool_twist_angular_y,
            feedback.base.tool_twist_angular_z
        ])


        # ---- Position Error ----
        error_pos = np.array([
            target_x - current_x,
            target_y - current_y,
            target_z - current_z
        ])

        # ---- Orientation Error (WRAPPED) ----
        error_ang = np.array([
            theta_x - current_theta_x,
            theta_y - current_theta_y,
            theta_z - current_theta_z
        ])
        error_ang = (error_ang + 180) % 360 - 180

        if np.linalg.norm(error_pos) < tolerance_pos and \
           np.linalg.norm(error_ang) < tolerance_ang:
            break

        # ---- PD Law ----
        lin_vel = Kp_pos * error_pos - Kd_pos * current_lin_vel
        ang_vel = Kp_ang * error_ang - Kd_ang * current_ang_vel

        # ---- Clamp ----
        lin_vel = np.clip(lin_vel, -max_linear_speed, max_linear_speed)
        ang_vel = np.clip(ang_vel, -max_angular_speed, max_angular_speed)

        # ---- Send Command ----
        twist = Base_pb2.TwistCommand()
        twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist.duration = 0

        twist.twist.linear_x = lin_vel[0]
        twist.twist.linear_y = lin_vel[1]
        twist.twist.linear_z = lin_vel[2]

        twist.twist.angular_x = ang_vel[0]
        twist.twist.angular_y = ang_vel[1]
        twist.twist.angular_z = ang_vel[2]

        base.SendTwistCommand(twist)
        time.sleep(dt)


    base.SendTwistCommand(Base_pb2.TwistCommand())
    return True





# def move_to_view_position(base, base_cyclic):
#     print("Moving to VIEW position...")

#     return move_cartesian(
#         base, base_cyclic,
#         VIEW_POSITION["x"],
#         VIEW_POSITION["y"],
#         VIEW_POSITION["z"],
#         VIEW_POSITION["theta_x"],
#         VIEW_POSITION["theta_y"],
#         VIEW_POSITION["theta_z"]
#     )

# def move_to_joint_angles(base, joint_angles):
#     print("Moving to joint view position...")

#     action = Base_pb2.Action()
#     action.name = "Joint movement"

#     for i, angle in enumerate(joint_angles):
#         joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
#         joint_angle.joint_identifier = i
#         joint_angle.value = angle

#     e = threading.Event()

#     notification_handle = base.OnNotificationActionTopic(
#         check_for_end_or_abort(e),
#         Base_pb2.NotificationOptions()
#     )

#     base.ExecuteAction(action)
#     finished = e.wait(TIMEOUT_DURATION)

#     base.Unsubscribe(notification_handle)

#     return finished

def lift_in_joint_space(base, base_cyclic, lift_deg=30.0):
    """
    Lift by increasing shoulder joint angle.
    lift_deg: how many degrees to add
    """

    feedback = base_cyclic.RefreshFeedback()

    current_angles = [
        actuator.position for actuator in feedback.actuators
    ]

    target_angles = current_angles.copy()

    # Increase shoulder joint (index 1)
    target_angles[3] += lift_deg/2 +10
    #target_angles[6] += lift_deg/2

    return move_to_joint_angles_velocity(
        base,
        base_cyclic,
        target_angles
    )


def move_to_joint_angles_velocity(base, base_cyclic,
                                  target_angles,
                                  max_speed_deg=70.0,
                                  tolerance_deg=1.0):

    enable_servo_mode(base)
    dt = 0.01

    while True:

        feedback = base_cyclic.RefreshFeedback()

        current = np.array([
            actuator.position for actuator in feedback.actuators
        ])

        target = np.array(target_angles)

        error = target - current
        error = (error + 180) % 360 - 180

        if np.max(np.abs(error)) < tolerance_deg:
            break

        cmd = Base_pb2.JointSpeeds()

        for i in range(len(error)):

            joint_speed = cmd.joint_speeds.add()
            joint_speed.joint_identifier = i

            speed = 2.0 * error[i]
            speed = np.clip(speed, -max_speed_deg, max_speed_deg)

            joint_speed.value = speed
            joint_speed.duration = 0

        base.SendJointSpeedsCommand(cmd)
        time.sleep(dt)

    base.SendJointSpeedsCommand(Base_pb2.JointSpeeds())
    return True



#really autistic way of doing it but its just for video shoot
def move_to_view_position(base, base_cyclic):
    return move_to_joint_angles_velocity(base, base_cyclic, VIEW_JOINT_ANGLES)


def set_gripper_width(base, width_m):
    """
    Control Gen3 gripper using high-level API.
    width_m: opening width in meters (0.0 – 0.085)
    """

    # Clamp width
    width_m = max(0.0, min(0.085, width_m))

    # Convert meters → normalized position (Kinova: 0=open, 1=closed)
    normalized_position = 1.0 - (width_m / 0.085)

    gripper_command = Base_pb2.GripperCommand()
    gripper_command.mode = Base_pb2.GRIPPER_POSITION

    finger = gripper_command.gripper.finger.add()
    finger.finger_identifier = 1
    finger.value = normalized_position

    base.SendGripperCommand(gripper_command)


# def move_delta(base, base_cyclic, dx, dy, dz):
#     """Move relative to current position using cyclic feedback"""

#     feedback = base_cyclic.RefreshFeedback()

#     # Current Cartesian pose from cyclic feedback
#     current_x = feedback.base.tool_pose_x
#     current_y = feedback.base.tool_pose_y
#     current_z = feedback.base.tool_pose_z

#     current_theta_x = feedback.base.tool_pose_theta_x
#     current_theta_y = feedback.base.tool_pose_theta_y
#     current_theta_z = feedback.base.tool_pose_theta_z

#     return move_cartesian_velocity(
#         base,
#         base_cyclic,
#         current_x + dx,
#         current_y + dy,
#         current_z + dz,
#         current_theta_x,
#         current_theta_y,
#         current_theta_z
#     )



def camera_to_base_transform(pose):
    """Convert camera pose to base frame"""
    x, y, z = pose.x, pose.y, pose.z
    X, Y, Z = pose.theta_x, pose.theta_y, pose.theta_z
    
    rx = Rx(np.deg2rad(X))
    ry = Ry(np.deg2rad(Y))
    rz = Rz(np.deg2rad(Z))
    
    R = np.linalg.multi_dot([rz, ry, rx])
    M = np.eye(4)
    M[:3, :3] = R[:3, :3]
    M[:3, 3] = [x, y, z]
    
    return np.dot(M, CAMERA_OFFSET)


def rot2eul_zyx(R):
    """Convert rotation matrix to Euler angles (ZYX convention)"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees([x, y, z])

#typing typing typing abcedf yes he  must be one of the best postdoc


# =============================================================================
# YOLO DETECTION
# =============================================================================

def detect_oranges(frame, model):
    """
    Detect ALL oranges in frame using YOLO
    Returns: list of detections
    Each detection = (x1, y1, x2, y2, cx, cy, bbox_width, confidence)
    """

    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    if len(results) == 0 or results[0].boxes is None:
        return []

    boxes = results[0].boxes
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    
    #print(model.names)



    detections = []

    for i in range(len(conf)):
        if conf[i] < CONF_THRESHOLD:
            continue
        class_name = model.names[int(cls[i])]
        if class_name != 'cylinder':
            continue


        x1, y1, x2, y2 = xyxy[i]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bbox_width = x2 - x1

        detections.append(
            (x1, y1, x2, y2, cx, cy, bbox_width, float(conf[i]))
        )

    return detections



def estimate_orange_3d_position(cx, cy, bbox_w, bbox_h, base_cyclic):
    """
    Estimate 3D position of orange in robot base frame
    Returns: 4x4 transformation matrix or None
    """
    # Estimate depth using known orange size
    focal_length = CAMERA_MATRIX[0, 0]

    minor_axis_pixels = min(bbox_w, bbox_h)
    depth_z = (ORANGE_DIAMETER * focal_length) / (minor_axis_pixels*0.9) #==================================================================================================================

    #depth_z = (ORANGE_DIAMETER * focal_length) / (bbox_width*0.9) #ADJUST THIS BASED ON HOW GOOD IT IS AT PICKING UP
    
    # Convert pixel coordinates to normalized image coordinates
    point_2d = np.array([[[cx, cy]]], dtype=np.float32)
    point_undist = cv2.undistortPoints(point_2d, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
    
    cx_u = point_undist[0][0][0]
    cy_u = point_undist[0][0][1]
    
    # Convert to 3D camera coordinates
    x_cam = (cx_u - CAMERA_MATRIX[0, 2]) * depth_z / CAMERA_MATRIX[0, 0]
    y_cam = (cy_u - CAMERA_MATRIX[1, 2]) * depth_z / CAMERA_MATRIX[1, 1]
    
    # Create transformation matrix in camera frame
    orange_cam_frame = np.eye(4)
    orange_cam_frame[0, 3] = -x_cam
    orange_cam_frame[1, 3] = -y_cam
    orange_cam_frame[2, 3] = depth_z
    
    # Transform to robot base frame
    feedback = base_cyclic.RefreshFeedback()

    current_x = feedback.base.tool_pose_x
    current_y = feedback.base.tool_pose_y
    current_z = feedback.base.tool_pose_z

    current_theta_x = feedback.base.tool_pose_theta_x
    current_theta_y = feedback.base.tool_pose_theta_y
    current_theta_z = feedback.base.tool_pose_theta_z

    # Build pose-like object for your transform function
    class Pose:
        pass

    pose = Pose()
    pose.x = current_x
    pose.y = current_y
    pose.z = current_z
    pose.theta_x = current_theta_x
    pose.theta_y = current_theta_y
    pose.theta_z = current_theta_z

    camera_base_frame = camera_to_base_transform(pose)

    orange_base_frame = np.dot(camera_base_frame, orange_cam_frame)
    
    return orange_base_frame


def find_orange_with_vision(base_cyclic, model, max_attempts=150):
    """
    Continuously look for orange and return its position.
    Automatically switches to headless mode if cv2.imshow is unavailable.
    Returns: 4x4 transformation matrix or None
    """
    

    # Try GUI mode
    headless = False
    try:
        cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Test", np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("Test")
    except cv2.error:
        headless = True
        print("⚠️ OpenCV GUI not available. Running in headless mode (no window display).")

    print("Searching for orange...")
    detection_buffer = []

    for attempt in range(max_attempts):
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            time.sleep(0.05)
            continue
        
        # Detect orange
        detections = detect_oranges(frame, model)

        if detections:

            # Select highest confidence detection
            best_detection = max(detections, key=lambda x: x[6] * x[7])

            x1, y1, x2, y2, cx, cy, bbox_width, conf = best_detection
            

            bbox_w = x2 - x1
            bbox_h = y2 - y1




            # Estimate 3D position
            orange_pos = estimate_orange_3d_position(cx, cy, bbox_w, bbox_h, base_cyclic)
            detection_buffer.append(orange_pos)

            # Keep only last 5 detections
            if len(detection_buffer) > 5:
                detection_buffer.pop(0)

            # If we have enough detections, average them
            if len(detection_buffer) >= 3:
                avg_position = detection_buffer[-1].copy()

                translations = np.array([pose[:3, 3] for pose in detection_buffer])
                avg_translation = np.mean(translations, axis=0)

                avg_position[:3, 3] = avg_translation


                print(f"Orange found at: x={avg_position[0,3]:.3f}, "
                      f"y={avg_position[1,3]:.3f}, z={avg_position[2,3]:.3f}")

                if not headless:
                    # Draw detection and display
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Orange {conf:.2f}", (cx - 50, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, "LOCKED!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Orange Detection", frame)
                    cv2.waitKey(500)
                    cv2.destroyAllWindows()

                
                return avg_position


        if not headless:
            # Display live search progress
            cv2.putText(frame, f"Searching... {attempt+1}/{max_attempts}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            try:
                cv2.imshow("Orange Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except cv2.error:
                headless = True
                print("⚠️ Switched to headless mode during loop.")

    time.sleep(0.05)
    
    if not headless:
        cv2.destroyAllWindows()
    print("Orange not found")
    return None


# =============================================================================
# PICK AND PLACE SEQUENCE
# =============================================================================

def pick_orange(base, base_cyclic, router, router_real_time, orange_position):
    """
    Execute pick-up sequence for orange
    """
    print("\n=== PICK SEQUENCE ===")
    
    # Extract target position
    target_x = orange_position[0, 3]
    target_y = orange_position[1, 3]
    target_z = orange_position[2, 3]
    
    # Get orientation from transformation matrix
    theta = rot2eul_zyx(orange_position[:3, :3])
    
    # # 1. Move above orange
    # print("1. Moving above orange...")
    # success = move_cartesian(base, base_cyclic,
    #                         target_x, target_y, APPROACH_HEIGHT,
    #                         theta[0], theta[1], theta[2])
    # if not success:
    #     print("✗ Failed to move above orange")
    #     return False
    
    # time.sleep(0.5)
    
    # 2. Open gripper
    print("1. Opening gripper...")
    ###close_gripper(router, router_real_time, GRIPPER_WIDTH_OPEN)
    set_gripper_width(base, GRIPPER_WIDTH_OPEN)
    time.sleep(0.1)
    
    # # 3. Descend to orange
    # print("3. Descending to orange...")
    # pickup_z = target_z + PICKUP_HEIGHT_OFFSET
    # success = move_cartesian(base, base_cyclic,
    #                         target_x, target_y, pickup_z,
    #                         theta[0], theta[1], theta[2])
    # if not success:
    #     print("✗ Failed to descend to orange")
    #     return False
    
    # 1. Move directly to grasp pose (single motion)
    print("2. Moving directly to orange...")
    # Compute tool-frame forward offset
    R = orange_position[:3, :3]

    forward_offset = R @ np.array([0, 0, -GRASP_FORWARD_OFFSET])

    grasp_x = target_x + forward_offset[0]
    grasp_y = target_y + forward_offset[1]
    grasp_z = target_z + forward_offset[2] +0.01

    success = move_cartesian_velocity(
        base, base_cyclic,
        grasp_x,
        grasp_y,
        grasp_z,
        theta[0], theta[1], theta[2],
        max_linear_speed=0.6,     # was 0.35
        max_angular_speed=120.0   # was 60
    )



    if not success:
        print("Failed to move directly to orange")
        return False


    #print("BUG")
    #time.sleep(0.25)
    
    # 4. Close gripper on orange
    print("3. Grasping orange...")
    ###close_gripper(router, router_real_time, GRIPPER_WIDTH_ORANGE)
    set_gripper_width(base, GRIPPER_WIDTH_ORANGE)
    time.sleep(0.1)
    
    # 5. Lift orange
    print("4. Lifting orange...")
    lift_in_joint_space(base, base_cyclic, lift_deg=30.0)
    if not success:
        print("Failed to lift orange")
        return False
    
    print("Orange picked successfully!")
    return True

def test_vertical_influence(base, base_cyclic, joint_index, delta_deg=3.0):
    feedback = base_cyclic.RefreshFeedback()
    current_angles = [a.position for a in feedback.actuators]

    print("Current Z:",
          feedback.base.tool_pose_z)

    target = current_angles.copy()
    target[joint_index] += delta_deg

    move_to_joint_angles_velocity(base, base_cyclic, target)

    feedback = base_cyclic.RefreshFeedback()
    print("New Z:",
          feedback.base.tool_pose_z)


def place_orange(base, base_cyclic, router, router_real_time):
    print("\n=== PLACE SEQUENCE (JOINT BASED) ===")

    # 1. Move to joint drop position
    print("1. Moving to drop-off joint configuration...")
    success = move_to_joint_angles_velocity(
        base,
        base_cyclic,
        DROP_OFF_JOINT_ANGLES
    )

    if not success:
        print("Failed to reach drop joint position")
        return False

    time.sleep(0.2)

    # 2. Open gripper
    print("2. Releasing object...")
    set_gripper_width(base, GRIPPER_WIDTH_OPEN)
    time.sleep(0.2)

    # 3. Optional: small lift after release
    print("3. Small lift away...")
    lift_in_joint_space(base, base_cyclic, lift_deg=10)

    print("Drop-off complete")
    return True



# =============================================================================
# MAIN CONTROL LOOP
# =============================================================================

def main():
    """Main pick and place loop"""
    
    # Check YOLO weights path
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"ERROR: YOLO weights not found at {YOLO_WEIGHTS}")
        print("Please update YOLO_WEIGHTS in the configuration section")
        return 1
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(YOLO_WEIGHTS)
    print("Model loaded")
    
    # Start continuous camera + detection thread
    cam_thread = threading.Thread(target=camera_loop, args=(model,), daemon=True)
    cam_thread.start()


    # Setup utilities path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        
    # Parse connection arguments
    args = utilities.parseConnectionArguments()
    
    # Connect to robot
    print("Connecting to robot...")
    with utilities.DeviceConnection.createTcpConnection(args) as router, \
         utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
        
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        
        print("Connected to robot")
        
        def start_stream():
            app.run(host='0.0.0.0', port=5000, threaded=True)

        stream_thread = threading.Thread(target=start_stream, daemon=True)
        stream_thread.start()

        print("📡 MJPEG stream available at: http://<robot_ip>:5000/video")


        # Main loop
        while True:


            print("\n" + "="*60)
            print("KINOVA ORANGE PICK AND PLACE")
            print("="*60)
            print("\nOptions:")
            print("  1. Run pick and place cycle")
            print("  2. Move to home position")
            print("  3. Test orange detection only")
            print("  4. Open gripper")
            print("  5. Close gripper")
            print("  q. Quit")
            
            choice = input("\nEnter choice: ").strip()

            stop = Base_pb2.TwistCommand()
            stop.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
            stop.twist.linear_x = 0
            stop.twist.linear_y = 0
            stop.twist.linear_z = 0
            stop.twist.angular_x = 0
            stop.twist.angular_y = 0
            stop.twist.angular_z = 0
            

            
            if choice == 'q':
                print("Exiting...")
                break
            
            elif choice == '1':
                # Full pick and place cycle
                print("\n--- STARTING PICK AND PLACE CYCLE ---")
                
                # Move to home
                # if not move_to_home_position(base):
                #     print("Failed to reach home position")
                #     continue
                # Move to view position
                if not move_to_view_position(base, base_cyclic):
                    print("Failed to reach view position")
                    continue
                base.SendTwistCommand(stop)
                time.sleep(0.5)
                
                # Search for orange
                orange_pos = find_orange_with_vision(base_cyclic, model)
                if orange_pos is None:
                    print("Cannot find orange. Aborting.")
                    continue
                
                while orange_pos is not None:
                    # Pick orange
                    if not pick_orange(base, base_cyclic, router, router_real_time, orange_pos):
                        print("Pick failed. Aborting.")
                        move_to_home_position(base)
                        continue
                    
                    # Place orange
                    if not place_orange(base, base_cyclic, router, router_real_time):
                        print("Place failed.")
                        move_to_home_position(base)
                        continue

                    if not move_to_view_position(base, base_cyclic):
                        print("Failed to reach view position")
                        continue

                    base.SendTwistCommand(stop)
                    time.sleep(0.5)
                    orange_pos = find_orange_with_vision(base_cyclic, model)
                    continue
                
                # Return home
                move_to_home_position(base)
                print("\nCYCLE COMPLETE!")
            
            elif choice == '2':
                move_to_home_position(base)
            
            elif choice == '3':
                orange_pos = find_orange_with_vision(base_cyclic, model)
                if orange_pos is not None:
                    print(f"Orange position:\n{orange_pos}")
            
            elif choice == '4':
                print("Opening gripper...")
                ####close_gripper(router, router_real_time, GRIPPER_WIDTH_OPEN)
                set_gripper_width(base, GRIPPER_WIDTH_OPEN)
            
            elif choice == '5':
                print("Closing gripper...")
                ####close_gripper(router, router_real_time, GRIPPER_WIDTH_ORANGE)
                set_gripper_width(base, GRIPPER_WIDTH_ORANGE)
            
            elif choice == '6':
                test_vertical_influence(base, base_cyclic, 1)
                print("1")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 2)
                print("2")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 3)
                print("3")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 4)
                print("4")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 5)
                print("5")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 6)
                print("6")
                time.sleep(1.5)
                test_vertical_influence(base, base_cyclic, 7)
                print("7")



            else:
                print("Invalid choice")
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        exit(1)