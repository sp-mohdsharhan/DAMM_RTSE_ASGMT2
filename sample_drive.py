import socket
import threading
import struct
import cv2
import numpy as np
import time
#import keyboard
import select
import ctypes
#  sharhan edit
# Perception layer (HSV calibration, contour shape filters, lane offset,
# brightness, overlay rendering) lives in image_detection.py. This module
# owns the controller, RT scheduling glue, and the locked skeleton sections.
# Per the assignment poster the *game* is the authoritative state machine,
# so there is no shadow simulation here — we react to what cameras show.
from image_detection import (
    # constants
    PROC_W, PROC_H,
    RED_AVOID_AREA_FRAC, RED_AVOID_BAND_FRAC,
    RED_LANE_CHANGE_DURATION_S, RED_SETTLE_DURATION_S,
    YELLOW_AVOID_AREA_FRAC, CENTER_BAND_FRAC,
    GREEN_ATTRACT_GAIN, GREEN_ATTRACT_MIN_AREA,
    RED_AVOID_GAIN, YELLOW_AVOID_GAIN, LANE_GAIN,
    LANE_CURVE_GAIN, HILL_AREA_SCALE,
    # functions
    detect_front_objects, detect_rear, detect_lane_offset,
    detect_lane_curve, draw_lane_curve_debug,
    detect_low_brightness, detect_slope, draw_overlay,
    calibrate_step, calibration_done,
)

# Controller-policy throttle constants (NOT perception — live here).
CRUISE_THROTTLE = 0.8                    # normal forward cruise
LOW_BRIGHTNESS_THROTTLE = 0.4            # ease off when scene is dim (tokens may be invisible / all-yellow)
HILL_THROTTLE = 0.5                      # on a hill, don't accelerate — coast so we can react at the crest


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
# data_lock:  scoped to raw frame slots only (read by perception, written by camera tasks)
# state_lock: control-command mutations (kept separate to avoid blocking camera I/O)
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0,
    # --- control commands written by Processing, read by SendControls ---
    'steering_cmd': 0.0,
    'accel_cmd': 0.0,
    'perception_front': {},
    'perception_back': {},
}
data_lock = threading.Lock()
state_lock = threading.Lock()
is_running = True

# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
    
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
                
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
                
        if not (front_connected and back_connected):
            time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
    
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue

# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------

def read_single_camera(sock, window_name, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
        
    try:
        latest_frame_data = None
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes:
            return
            
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            latest_frame_data = received_bytes
            
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable:
                break
                
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes:
                return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet:
                    break
                received_bytes += packet
                
            if len(received_bytes) == image_length:
                latest_frame_data = received_bytes
                
        if latest_frame_data is not None:
            np_arr = np.frombuffer(latest_frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock:
                    shared_data[data_key] = frame
                
                # You may disable this if you don't need to display the frames / This could effect the fps
                frame_resized = cv2.resize(frame, (640, 480))
                cv2.imshow(window_name, frame_resized)
                cv2.waitKey(1)
                
    except Exception as e:
        pass

def read_front_camera_task():
    read_single_camera(front_camera_sock, "Front Camera", 'latest_front_frame')

def read_back_camera_task():
    read_single_camera(back_camera_sock, "Back Camera", 'latest_back_frame')

# =========================================================
# Controller (perception lives in image_detection.py)
# =========================================================
# The game itself owns score, speed, and event resolution. This controller
# observes the cameras frame-by-frame and emits (steering, accel). No
# parallel simulation of game rules is kept in software.

# Red avoidance latch: once a red is detected ahead, commit to a full
# lane-change away from it for RED_LANE_CHANGE_DURATION_S, then a brief
# counter-steer settle phase to straighten out in the new lane.
_red_avoid = {'until': 0.0, 'settle_until': 0.0, 'dir': 0}

# ---------------------------------------------------------------------------
# REMOVED behaviours (kept here for future reference)
# ---------------------------------------------------------------------------
# This game build has no police vehicle and no trailing/overtaking car, so the
# two rear-threat reactions that used to sit ABOVE the colour stack were removed:
#
#   * POLICE-SEEK MODE — when the rear camera saw police-blue and a red was
#     ahead, steering inverted to drive TOWARD the red centroid
#     (poster rule "catch the next red token or -50% speed"):
#         if police_seen and red and red.area_frac > RED_AVOID_AREA_FRAC:
#             return clip(RED_AVOID_GAIN * red.centroid_x_norm)
#
#   * TRAILING-CAR FORCED SWERVE — a rear contour growing frame-over-frame armed
#     a fixed ±LANE_CHANGE_STEER swerve for LANE_CHANGE_DURATION_S, direction
#     alternating each trigger (tracked via _lane_change_until / _lane_change_dir):
#         if force_lane_change:
#             return LANE_CHANGE_STEER * swerve_dir
#
# If a future build reintroduces police / trailing cars, restore detect_rear()
# consumption in processing_task, re-add the _lane_change_* timer, and re-insert
# these two branches at the TOP of _compute_steering (they are safety/penalty
# overrides and must outrank the colour priorities below). The matching
# constants (LANE_CHANGE_DURATION_S, LANE_CHANGE_STEER) still live in
# image_detection.py.
# ---------------------------------------------------------------------------

def _compute_steering(front_per, lane_offset, curve_bias, hill):
    """Steering priority stack (highest wins):
       1. GREEN seek   — literal override: green wins whenever it is visible.
       2. RED  evade   — committed lane change away from red, with latch + settle.
       3. YELLOW evade — swerve away when centred & close.
       4. Lane follow  — instantaneous offset + anticipated curve bias.
    On a hill the red/yellow trigger areas shrink (HILL_AREA_SCALE) so evasion
    fires earlier — the car "prepares for a lane change" before orbs crest fully.
    """
    now = time.monotonic()
    red    = front_per['red']    if front_per else None
    green  = front_per['green']  if front_per else None
    yellow = front_per['yellow'] if front_per else None

    # Hill: orbs crest into view late -> lower the trigger areas so we react sooner.
    red_area_thr    = RED_AVOID_AREA_FRAC    * (HILL_AREA_SCALE if hill else 1.0)
    yellow_area_thr = YELLOW_AVOID_AREA_FRAC * (HILL_AREA_SCALE if hill else 1.0)

    # 1) GREEN SEEK (priority #1) — steer toward green, anticipating the bend.
    if green is not None and green['area_frac'] > GREEN_ATTRACT_MIN_AREA:
        return float(np.clip(GREEN_ATTRACT_GAIN * green['centroid_x_norm']
                             + LANE_CURVE_GAIN * curve_bias, -1.0, 1.0))

    # 2) RED EVADE — commit to a lane change away from any red ahead.
    if red is not None and red['area_frac'] > red_area_thr \
            and abs(red['centroid_x_norm']) < RED_AVOID_BAND_FRAC:
        direction = -1 if red['centroid_x_norm'] >= 0 else 1
        _red_avoid['until'] = now + RED_LANE_CHANGE_DURATION_S
        _red_avoid['settle_until'] = _red_avoid['until'] + RED_SETTLE_DURATION_S
        _red_avoid['dir'] = direction
        return float(RED_AVOID_GAIN * direction)
    # 2b) Red-avoid latch (swerve phase): hold the lane change.
    #     If a new red appears on the side we're swerving toward, flip direction.
    if now < _red_avoid['until']:
        if red is not None and red['area_frac'] > red_area_thr:
            red_side = 1 if red['centroid_x_norm'] >= 0 else -1
            if red_side == _red_avoid['dir']:
                _red_avoid['dir'] = -_red_avoid['dir']
                _red_avoid['until'] = now + RED_LANE_CHANGE_DURATION_S
                _red_avoid['settle_until'] = _red_avoid['until'] + RED_SETTLE_DURATION_S
        return float(RED_AVOID_GAIN * _red_avoid['dir'])
    # 2c) Settle phase: brief counter-steer to straighten out.
    if now < _red_avoid['settle_until']:
        return float(-0.5 * _red_avoid['dir'])

    # 3) YELLOW EVADE — swerve away if in lane band & close.
    if yellow is not None and yellow['area_frac'] > yellow_area_thr \
            and abs(yellow['centroid_x_norm']) < CENTER_BAND_FRAC:
        return float(np.clip(-YELLOW_AVOID_GAIN * np.sign(yellow['centroid_x_norm'] or 1.0), -1, 1))

    # 4) Lane following + anticipated curve bias.
    if lane_offset is not None:
        return float(np.clip(LANE_GAIN * lane_offset + LANE_CURVE_GAIN * curve_bias, -1, 1))
    # No lane lines but a warped curve estimate is available -> follow the bend.
    if curve_bias:
        return float(np.clip(LANE_CURVE_GAIN * curve_bias, -1, 1))

    # 5) Default
    return 0.0

def processing_task():
    # Snapshot frame references under data_lock (fast), then release.
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    if front_frame is None and back_frame is None:
        return

    # Autonomous HSV calibration (warm-up only)
    if not calibration_done() and front_frame is not None:
        calibrate_step(front_frame)

    # Perception (lock-free)
    front_per   = detect_front_objects(front_frame) if front_frame is not None else None
    lane_offset = detect_lane_offset(front_frame)   if front_frame is not None else None
    # Rear perception still rendered for the REAR overlay panel, but no longer
    # consumed by steering (police-seek / trailing-car reactions removed — this
    # build has neither; see the REMOVED block above _compute_steering).
    rear_per    = detect_rear(back_frame)           if back_frame is not None else None
    low_light   = detect_low_brightness(front_frame) if front_frame is not None else False

    # Lane-curve readout (also drives the "Lane Curve" debug window below, so we
    # compute it once here and reuse it). curve_bias anticipates the bend.
    curve_dbg  = detect_lane_curve(front_frame) if front_frame is not None else None
    curve_bias = curve_dbg['curve_bias'] if curve_dbg else 0.0

    # Hill / slope: ease throttle and trigger evasion earlier when on a crest.
    slope = detect_slope(front_frame) if front_frame is not None else None
    hill  = bool(slope and slope['is_hill'])

    # Steering: pure reaction to perception (green-seek > red-evade > yellow-evade > lane).
    steering = _compute_steering(front_per, lane_offset, curve_bias, hill)

    # Throttle: constant cruise, eased on hills (don't accelerate at the crest)
    # and under low brightness (degraded token visibility -> buy reaction time).
    if hill:
        accel = HILL_THROTTLE
    elif low_light:
        accel = LOW_BRIGHTNESS_THROTTLE
    else:
        accel = CRUISE_THROTTLE

    with state_lock:
        shared_data['steering_cmd'] = steering
        shared_data['accel_cmd'] = accel
        shared_data['perception_front'] = front_per or {}
        shared_data['perception_back'] = rear_per or {}

    # Overlay (own window; does NOT edit locked read_single_camera).
    try:
        events_visible = []
        if hill:       events_visible.append('HILL')
        if low_light:  events_visible.append('LOW_LIGHT')
        hud = {
            'target': accel * 100.0, 'eff': accel * 100.0,
            'police': False, 'events': events_visible,
            'str': steering, 'acc': accel,
        }
        overlay = draw_overlay(front_per, rear_per, lane_offset, hud)
        cv2.imshow("Perception", overlay)
        # CL0-CL2 lane-curve debug window (reuses curve_dbg computed above).
        curve_panel = draw_lane_curve_debug(curve_dbg)
        if curve_panel is not None:
            cv2.imshow("Lane Curve", curve_panel)
        cv2.waitKey(1)
    except Exception:
        pass

# Last successfully-sent command — used as non-blocking fallback
_last_sent = {'steering': 0.0, 'accel': 0.0}

def send_controls_task():
    global control_conn
    if control_conn is None:
        return

    # Non-blocking acquire: if processing thread is mid-write, reuse last command
    if state_lock.acquire(blocking=False):
        try:
            steering_input = shared_data['steering_cmd']
            acceleration_input = shared_data['accel_cmd']
        finally:
            state_lock.release()
        _last_sent['steering'] = steering_input
        _last_sent['accel'] = acceleration_input
    else:
        steering_input = _last_sent['steering']
        acceleration_input = _last_sent['accel']

    try:
        data = struct.pack('ff', steering_input, acceleration_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None


# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing RTSE Sample Drive...")
    
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
    
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    # Priorities follow Deadline-Monotonic Scheduling (DMS): priority tracks
    # deadline criticality, not raw period. Camera reads call the skeleton's
    # read_*_camera_task directly — input corruption / delay is the game's
    # job, so we do not gate or throttle our own reads.
    #   ReadFrontCamera 5ms  HIGH   - collision-critical input (red orbs ahead); shortest deadline.
    #   SendControls   20ms  HIGH   - hard 50Hz actuator deadline; stale output = car drives blind.
    #   Processing     33ms  MEDIUM - the brain (perception -> steering/accel); above the rear
    #                                 camera so a slow rear frame can never starve it.
    #   ReadBackCamera 50ms  LOW    - rear threats (police / trailing car) evolve slowly:
    #                                 longest deadline -> lowest priority.
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH,  execute_func=read_front_camera_task)
    t_back_camera  = RTTask("ReadBackCamera",  period=0.050, priority=TaskPriority.LOW,   execute_func=read_back_camera_task)
    t_processing   = RTTask("Processing",      period=0.033, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls     = RTTask("SendControls",    period=0.020, priority=TaskPriority.HIGH,  execute_func=send_controls_task)
    
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
    
    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False

    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
    
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
