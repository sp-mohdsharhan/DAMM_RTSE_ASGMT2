import socket
import threading
import struct
import cv2
import numpy as np
import time
import keyboard
import select
import ctypes
import random


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

# Shared Resources with Mutex Lock for Concurrency
# data_lock: scoped to raw frame slots only (read by perception, written by camera tasks)
# state_lock: command/event state mutations (kept separate to avoid blocking camera I/O)
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 0.0,
    # --- perception/control state (added) ---
    'steering_cmd': 0.0,
    'accel_cmd': 0.0,
    'target_speed': 50.0,           # 0..100 internal speed model
    'police_active': False,         # halves throttle while True
    'active_events': {},            # {event_name: expiry_monotonic_ts}
    'last_hit_ts': {'red': 0.0, 'green': 0.0, 'yellow': 0.0},
    'swerve_dir': 1,                # alternates +1/-1 per force_lane_change
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
# Perception / Events / Control (OpenCV auto-detection)
# =========================================================

# --- Tunable constants -----------------------------------
PROC_W, PROC_H = 320, 240                # working resolution for perception
FRONT_ROI_TOP_FRAC = 0.40                # ignore top 40% of front frame
HIT_AREA_FRAC = 0.06                     # bbox/ROI area to count as "hit"
RED_AVOID_AREA_FRAC = 0.04               # bbox/ROI area to trigger avoid
CENTER_BAND_FRAC = 0.30                  # |x_norm| < this counts as "in path"
COOLDOWN_S = 1.0                         # per-color hit cooldown
EVENT_DURATION_S = 5.0                   # cam-degrade & most yellow events
LANE_CHANGE_DURATION_S = 1.5
LANE_CHANGE_STEER = 0.8
GREEN_ATTRACT_GAIN = 0.8
RED_AVOID_GAIN = 0.9
LANE_GAIN = 0.6
POLICE_THROTTLE_MULT = 0.5

CAM_BASE_PERIOD = 0.005                  # 200 Hz nominal
CAM_DEGRADED_PERIOD = 0.100              # 10 Hz while degraded

# HSV ranges (OpenCV uses H:0-179, S:0-255, V:0-255)
HSV_RED_1 = (np.array([0, 120, 80]),   np.array([10, 255, 255]))
HSV_RED_2 = (np.array([170, 120, 80]), np.array([179, 255, 255]))
HSV_GREEN = (np.array([40, 80, 60]),   np.array([85, 255, 255]))
HSV_YELLOW = (np.array([20, 120, 120]), np.array([35, 255, 255]))
HSV_POLICE_BLUE = (np.array([100, 120, 60]), np.array([130, 255, 255]))

YELLOW_EVENTS = [
    'front_cam_degraded',
    'back_cam_degraded',
    'speed_minus_5pct',
    'force_lane_change',
    'spawn_police',
]

# --- Perception helpers (pure) ---------------------------
def _color_mask(hsv, *ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask

def _largest_contour_info(mask, roi_area):
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50:
        return None
    x, y, w, h = cv2.boundingRect(c)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return {
        'bbox': (int(x), int(y), int(w), int(h)),
        'area_frac': float(area) / float(roi_area),
        'centroid_x_norm': (cx - PROC_W / 2.0) / (PROC_W / 2.0),  # -1..+1
        'centroid_y': float(cy),
    }

def detect_front_objects(frame):
    """Return dict {'red'|'green'|'yellow': info or None, 'roi_y0': int}."""
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_y0 = int(PROC_H * FRONT_ROI_TOP_FRAC)
    roi = small[roi_y0:, :]
    roi_area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return {
        'frame': small,
        'roi_y0': roi_y0,
        'red':    _largest_contour_info(_color_mask(hsv, HSV_RED_1, HSV_RED_2), roi_area),
        'green':  _largest_contour_info(_color_mask(hsv, HSV_GREEN), roi_area),
        'yellow': _largest_contour_info(_color_mask(hsv, HSV_YELLOW), roi_area),
    }

_prev_other_area = 0.0

def detect_rear(frame):
    """Return dict {'frame','police':{...},'other_car':{...}}."""
    global _prev_other_area
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_area = PROC_W * PROC_H
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    police_info = _largest_contour_info(_color_mask(hsv, HSV_POLICE_BLUE), roi_area)
    police_present = police_info is not None and police_info['area_frac'] > 0.01

    # Other car: high-saturation contour that is NOT police-blue.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    veh_mask = cv2.inRange(sat, 80, 255)
    veh_mask = cv2.bitwise_and(veh_mask, cv2.inRange(val, 40, 255))
    police_mask = _color_mask(hsv, HSV_POLICE_BLUE)
    if police_mask is not None:
        veh_mask = cv2.bitwise_and(veh_mask, cv2.bitwise_not(police_mask))
    veh_mask = cv2.morphologyEx(veh_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    other = _largest_contour_info(veh_mask, roi_area)
    growing = False
    if other is not None:
        growing = other['area_frac'] > _prev_other_area + 0.005 and other['area_frac'] > 0.02
        _prev_other_area = other['area_frac']
    else:
        _prev_other_area = 0.0

    return {
        'frame': small,
        'police': {'info': police_info, 'present': police_present},
        'other_car': {'info': other, 'growing': growing},
    }

def detect_lane_offset(frame):
    """Lightweight lane center estimation. Returns offset in -1..+1 or None."""
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_y0 = int(PROC_H * 0.55)
    roi = small[roi_y0:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=20)
    if lines is None:
        return None
    left_x, right_x = [], []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        if x2 == x1:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        if abs(slope) < 0.3:
            continue
        midx = (x1 + x2) / 2.0
        if slope < 0 and midx < PROC_W * 0.55:
            left_x.append(midx)
        elif slope > 0 and midx > PROC_W * 0.45:
            right_x.append(midx)
    if not left_x and not right_x:
        return None
    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2.0
    elif left_x:
        lane_center = np.mean(left_x) + PROC_W * 0.25
    else:
        lane_center = np.mean(right_x) - PROC_W * 0.25
    return (lane_center - PROC_W / 2.0) / (PROC_W / 2.0)

def _draw_obj(img, info, label, color, y_offset=0):
    if info is None:
        return
    x, y, w, h = info['bbox']
    cv2.rectangle(img, (x, y + y_offset), (x + w, y + h + y_offset), color, 2)
    cv2.putText(img, f"{label} {info['area_frac']*100:.1f}%",
                (x, max(10, y + y_offset - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def draw_overlay(front_per, rear_per, lane_offset, hud):
    front_img = front_per['frame'].copy() if front_per else np.zeros((PROC_H, PROC_W, 3), np.uint8)
    rear_img = rear_per['frame'].copy() if rear_per else np.zeros((PROC_H, PROC_W, 3), np.uint8)

    if front_per:
        y0 = front_per['roi_y0']
        cv2.rectangle(front_img, (0, y0), (PROC_W - 1, PROC_H - 1), (80, 80, 80), 1)
        _draw_obj(front_img, front_per['red'],    "RED",    (0, 0, 255), y0)
        _draw_obj(front_img, front_per['green'],  "GREEN",  (0, 255, 0), y0)
        _draw_obj(front_img, front_per['yellow'], "YELLOW", (0, 255, 255), y0)
        if lane_offset is not None:
            cx = int(PROC_W / 2 + lane_offset * PROC_W / 2)
            cv2.line(front_img, (PROC_W // 2, PROC_H - 5), (cx, PROC_H - 25), (255, 255, 255), 2)

    if rear_per:
        _draw_obj(rear_img, rear_per['police']['info'],  "POLICE", (255, 0, 0))
        _draw_obj(rear_img, rear_per['other_car']['info'], "CAR",    (200, 200, 0))

    # HUD strip
    hud_h = 60
    canvas = np.zeros((PROC_H + hud_h, PROC_W * 2 + 10, 3), np.uint8)
    canvas[:PROC_H, :PROC_W] = front_img
    canvas[:PROC_H, PROC_W + 10:] = rear_img
    cv2.putText(canvas, "FRONT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "REAR",  (PROC_W + 15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas,
                f"target={hud['target']:.0f} eff={hud['eff']:.0f} police={int(hud['police'])} "
                f"events={hud['events']}",
                (5, PROC_H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(canvas,
                f"str={hud['str']:+.2f} acc={hud['acc']:+.2f}",
                (5, PROC_H + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas

# --- Event engine ----------------------------------------
# Gating flags consulted by gated_read_front/back. Mutated under state_lock.
_cam_degraded = {'front': False, 'back': False}
_last_cam_read = {'front': 0.0, 'back': 0.0}

def _apply_color_hit(color, now):
    """Must be called with state_lock held."""
    if now - shared_data['last_hit_ts'][color] < COOLDOWN_S:
        return False
    shared_data['last_hit_ts'][color] = now

    if color == 'red':
        shared_data['target_speed'] -= 20
    elif color == 'green':
        shared_data['target_speed'] += 10
    elif color == 'yellow':
        _trigger_yellow_event(now)

    # Police-escape: any color hit clears police
    if shared_data['police_active']:
        shared_data['police_active'] = False

    shared_data['target_speed'] = max(0.0, min(100.0, shared_data['target_speed']))
    return True

def _trigger_yellow_event(now):
    """Must be called with state_lock held."""
    event = random.choice(YELLOW_EVENTS)
    if event == 'front_cam_degraded':
        shared_data['active_events']['front_cam_degraded'] = now + EVENT_DURATION_S
        _cam_degraded['front'] = True
    elif event == 'back_cam_degraded':
        shared_data['active_events']['back_cam_degraded'] = now + EVENT_DURATION_S
        _cam_degraded['back'] = True
    elif event == 'speed_minus_5pct':
        shared_data['target_speed'] *= 0.95
        shared_data['active_events']['speed_minus_5pct'] = now + 0.5  # short flash on HUD
    elif event == 'force_lane_change':
        shared_data['active_events']['force_lane_change'] = now + LANE_CHANGE_DURATION_S
        shared_data['swerve_dir'] = -shared_data['swerve_dir']
    elif event == 'spawn_police':
        shared_data['police_active'] = True
        shared_data['active_events']['spawn_police'] = now + 0.5  # flash; police_active persists

def _expire_events(now):
    """Must be called with state_lock held."""
    expired = [k for k, t in shared_data['active_events'].items() if now >= t]
    for k in expired:
        del shared_data['active_events'][k]
        if k == 'front_cam_degraded':
            _cam_degraded['front'] = False
        elif k == 'back_cam_degraded':
            _cam_degraded['back'] = False

# --- Gated camera reads (yellow-event productivity reduction) -----
def gated_read_front():
    now = time.monotonic()
    eff = CAM_DEGRADED_PERIOD if _cam_degraded['front'] else CAM_BASE_PERIOD
    if now - _last_cam_read['front'] < eff:
        return
    _last_cam_read['front'] = now
    read_front_camera_task()

def gated_read_back():
    now = time.monotonic()
    eff = CAM_DEGRADED_PERIOD if _cam_degraded['back'] else CAM_BASE_PERIOD
    if now - _last_cam_read['back'] < eff:
        return
    _last_cam_read['back'] = now
    read_back_camera_task()

# --- Controller ------------------------------------------
def _compute_steering(front_per, lane_offset, force_lane_change, swerve_dir):
    # 1) Forced lane change overrides everything
    if force_lane_change:
        return LANE_CHANGE_STEER * swerve_dir
    # 2) Red avoidance if centered & close
    red = front_per['red'] if front_per else None
    if red is not None and red['area_frac'] > RED_AVOID_AREA_FRAC \
            and abs(red['centroid_x_norm']) < CENTER_BAND_FRAC:
        return float(np.clip(-RED_AVOID_GAIN * np.sign(red['centroid_x_norm'] or 1.0), -1, 1))
    # 3) Green attraction if visible
    green = front_per['green'] if front_per else None
    if green is not None:
        return float(np.clip(GREEN_ATTRACT_GAIN * green['centroid_x_norm'], -1, 1))
    # 4) Lane following
    if lane_offset is not None:
        return float(np.clip(LANE_GAIN * lane_offset, -1, 1))
    # 5) Default
    return 0.0

def processing_task():
    # Snapshot frame references under data_lock (fast), then release.
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        back_frame = shared_data['latest_back_frame']

    if front_frame is None and back_frame is None:
        return

    # Perception (lock-free)
    front_per = detect_front_objects(front_frame) if front_frame is not None else None
    lane_offset = detect_lane_offset(front_frame) if front_frame is not None else None
    rear_per = detect_rear(back_frame) if back_frame is not None else None

    now = time.monotonic()

    with state_lock:
        _expire_events(now)

        # Apply color hits from front
        if front_per is not None:
            for color in ('red', 'green', 'yellow'):
                info = front_per[color]
                if info is not None and info['area_frac'] > HIT_AREA_FRAC:
                    _apply_color_hit(color, now)

        # Rear: police presence sustains debuff (only cleared by front color hit)
        if rear_per is not None and rear_per['police']['present']:
            shared_data['police_active'] = True

        # Rear: other car growing -> force lane change
        if rear_per is not None and rear_per['other_car']['growing']:
            if 'force_lane_change' not in shared_data['active_events']:
                shared_data['active_events']['force_lane_change'] = now + LANE_CHANGE_DURATION_S
                shared_data['swerve_dir'] = -shared_data['swerve_dir']

        force_lc = 'force_lane_change' in shared_data['active_events']
        swerve_dir = shared_data['swerve_dir']
        target_speed = shared_data['target_speed']
        police = shared_data['police_active']
        events_snapshot = list(shared_data['active_events'].keys())

        # Steering & throttle
        steering = _compute_steering(front_per, lane_offset, force_lc, swerve_dir)
        eff_speed = target_speed * (POLICE_THROTTLE_MULT if police else 1.0)
        accel = float(np.clip(eff_speed / 100.0, -1.0, 1.0))

        shared_data['steering_cmd'] = steering
        shared_data['accel_cmd'] = accel
        shared_data['perception_front'] = front_per or {}
        shared_data['perception_back'] = rear_per or {}

    # Build & show overlay (own window; does NOT edit locked read_single_camera)
    try:
        hud = {
            'target': target_speed, 'eff': eff_speed, 'police': police,
            'events': events_snapshot, 'str': steering, 'acc': accel,
        }
        overlay = draw_overlay(front_per, rear_per, lane_offset, hud)
        cv2.imshow("Perception", overlay)
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
    # NOTE: cameras use gated wrappers so yellow events can degrade their effective rate
    # without mutating RTTask.period. Periods chosen Rate-Monotonic compliant:
    # shorter period -> higher priority (cameras 5ms HIGH > controls 20ms HIGH > processing 33ms MEDIUM).
    t_front_camera = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=gated_read_front)
    t_back_camera = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=gated_read_back)
    t_processing = RTTask("Processing", period=0.033, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.020, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    
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
