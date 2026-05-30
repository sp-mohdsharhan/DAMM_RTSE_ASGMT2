"""
image_detection.py — OpenCV perception module for SpeedTrials2D autonomous driver.

Extracted from sample_drive.py so the controller / RT scheduling layer stays focused
on real-time concerns and the perception layer can be evolved / unit-tested in isolation.

PUBLIC SURFACE (everything below is consumed by sample_drive.py):
- Constants:
    PROC_W, PROC_H,
    RED_AVOID_AREA_FRAC, RED_AVOID_BAND_FRAC,
    RED_LANE_CHANGE_DURATION_S, RED_SETTLE_DURATION_S,
    YELLOW_AVOID_AREA_FRAC, CENTER_BAND_FRAC,
    LANE_CHANGE_DURATION_S, LANE_CHANGE_STEER,
    GREEN_ATTRACT_GAIN, GREEN_ATTRACT_MIN_AREA,
    RED_AVOID_GAIN, YELLOW_AVOID_GAIN, LANE_GAIN,
    LOW_BRIGHTNESS_THRESHOLD,
- Functions:
    detect_front_objects(frame) -> dict
    detect_rear(frame)          -> dict
    detect_lane_offset(frame)   -> float | None
    detect_low_brightness(frame) -> bool        (poster: "low brightness" event)
    draw_overlay(front_per, rear_per, lane_offset, hud) -> ndarray
    calibrate_step(frame)       -> None         (autonomous HSV warm-up)
    calibration_done()          -> bool
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PROC_W, PROC_H = 320, 240                # working resolution for perception
FRONT_ROI_TOP_FRAC = 0.28                # ignore top 28%: keep horizon margin for uphill/downhill slopes
FRONT_ROI_SIDE_FRAC = 0.15               # ignore leftmost/rightmost 15% (grass shoulders)

# Orb-shape filters (reject grass strips, road markings, curb dashes, etc.)
ORB_MAX_AREA_FRAC = 0.07                 # anything larger than this is environment
ORB_MIN_AREA_PX = 60                     # smallest detectable orb (raised: reject small dashes)
ORB_MIN_ASPECT = 0.70                    # near-square only (rejects dash rectangles)
ORB_MAX_ASPECT = 1.45
ORB_MIN_CIRCULARITY = 0.60               # 4*pi*A/P^2 — true orbs ~0.75+, dashes <0.5
ORB_MIN_FILL_RATIO = 0.65                # area / bbox_area; circles fill ~0.78, dashes <0.5

# Steering / lane-change thresholds (consumed by the controller in sample_drive.py).
# No shadow-state thresholds (hit cooldown, event durations, cam-degrade periods, etc.)
# live here — the game itself owns those rules; we just react to what we see.
RED_AVOID_AREA_FRAC = 0.010              # detect red even further away (commit lane change early)
RED_AVOID_BAND_FRAC = 0.70               # wider than CENTER_BAND_FRAC: any red roughly ahead triggers lane change
RED_LANE_CHANGE_DURATION_S = 1.6         # matches LANE_CHANGE_DURATION_S — guaranteed full lane cross
RED_SETTLE_DURATION_S = 0.35             # brief counter-steer to straighten out after the swerve
YELLOW_AVOID_AREA_FRAC = 0.02            # bbox/ROI area to trigger yellow avoidance
CENTER_BAND_FRAC = 0.55                  # |x_norm| < this counts as "in path"
LANE_CHANGE_DURATION_S = 1.5             # trailing-car defensive swerve
LANE_CHANGE_STEER = 0.8

GREEN_ATTRACT_GAIN = 0.5                 # gentle pull toward green; don't get dragged into reds
GREEN_ATTRACT_MIN_AREA = 0.005           # ignore tiny far-away greens (false attractors)
RED_AVOID_GAIN = 1.0                     # full-lock swerve when red is in path
YELLOW_AVOID_GAIN = 0.9
LANE_GAIN = 0.6

# Low-brightness event detection (poster: "low brightness — turn light on or all tokens yellow")
LOW_BRIGHTNESS_THRESHOLD = 50            # mean V channel below this -> consider it dim


# ---------------------------------------------------------------------------
# HSV ranges & auto-calibration state
# ---------------------------------------------------------------------------
# OpenCV: H:0-179, S:0-255, V:0-255
HSV_RED_1 = (np.array([0, 120, 80]),    np.array([10, 255, 255]))
HSV_RED_2 = (np.array([170, 120, 80]),  np.array([179, 255, 255]))
HSV_GREEN = (np.array([40, 80, 60]),    np.array([85, 255, 255]))
HSV_YELLOW = (np.array([20, 120, 120]), np.array([35, 255, 255]))
HSV_POLICE_BLUE = (np.array([100, 120, 60]), np.array([130, 255, 255]))

# Mutable active HSV ranges (list-of-(lo,hi) per color), consulted by detectors.
# Auto-calibration replaces entries it learns; buckets without samples keep defaults.
_hsv_active = {
    'red':    [HSV_RED_1, HSV_RED_2],
    'green':  [HSV_GREEN],
    'yellow': [HSV_YELLOW],
    'police': [HSV_POLICE_BLUE],
}

# --- Auto-calibration constants & state ---
CALIB_FRAMES = 90                                  # ~3s at 30 Hz
CALIB_KMEANS_K = 6
CALIB_MIN_PIXELS = 200
CALIB_SUBSAMPLE = 5000
HSV_MARGIN = np.array([10, 60, 60], dtype=np.int16)
_HUE_BUCKETS = [
    ('red',    [(0, 12), (168, 179)]),
    ('yellow', [(18, 36)]),
    ('green',  [(38, 88)]),
    ('police', [(95, 135)]),
]
_calib_state = {
    'frames_seen': 0,
    'samples': {'red': [], 'green': [], 'yellow': [], 'police': []},
    'done': False,
}


# ---------------------------------------------------------------------------
# Pure perception helpers
# ---------------------------------------------------------------------------
def _color_mask(hsv, *ranges):
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _largest_contour_info(mask, roi_area, roi_x0=0):
    """Return the largest orb-shaped contour, or None.
    Filters out grass strips / road markings via aspect ratio, circularity, max-area cap, fill ratio.
    roi_x0 is added to bbox x and centroid for correct global coords when ROI is horizontally cropped.
    """
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < ORB_MIN_AREA_PX:
            continue
        area_frac = float(area) / float(roi_area)
        if area_frac > ORB_MAX_AREA_FRAC:
            continue                                  # too big -> environment
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if aspect < ORB_MIN_ASPECT or aspect > ORB_MAX_ASPECT:
            continue                                  # too elongated -> grass strip
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < ORB_MIN_CIRCULARITY:
            continue                                  # not blob-like
        bbox_area = float(w * h)
        if bbox_area <= 0 or (area / bbox_area) < ORB_MIN_FILL_RATIO:
            continue                                  # sparse/hollow (dashed stripe)
        if area > best_area:
            best = (c, area, area_frac, x, y, w, h)
            best_area = area
    if best is None:
        return None
    _, area, area_frac, x, y, w, h = best
    cx = x + w / 2.0 + roi_x0
    cy = y + h / 2.0
    return {
        'bbox': (int(x + roi_x0), int(y), int(w), int(h)),
        'area_frac': area_frac,
        'centroid_x_norm': (cx - PROC_W / 2.0) / (PROC_W / 2.0),  # -1..+1
        'centroid_y': float(cy),
    }


def detect_front_objects(frame):
    """Return dict {'frame','roi_y0','roi_x0','red','green','yellow'}."""
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_y0 = int(PROC_H * FRONT_ROI_TOP_FRAC)
    roi_x0 = int(PROC_W * FRONT_ROI_SIDE_FRAC)
    roi_x1 = PROC_W - roi_x0
    roi = small[roi_y0:, roi_x0:roi_x1]
    roi_area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return {
        'frame': small,
        'roi_y0': roi_y0,
        'roi_x0': roi_x0,
        'red':    _largest_contour_info(_color_mask(hsv, *_hsv_active['red']), roi_area, roi_x0),
        'green':  _largest_contour_info(_color_mask(hsv, *_hsv_active['green']), roi_area, roi_x0),
        'yellow': _largest_contour_info(_color_mask(hsv, *_hsv_active['yellow']), roi_area, roi_x0),
    }


_prev_other_area = 0.0


def detect_rear(frame):
    """Return dict {'frame','police':{...},'other_car':{...}}."""
    global _prev_other_area
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_area = PROC_W * PROC_H
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    police_info = _largest_contour_info(_color_mask(hsv, *_hsv_active['police']), roi_area)
    police_present = police_info is not None and police_info['area_frac'] > 0.01

    # Other car: high-saturation contour that is NOT police-blue.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    veh_mask = cv2.inRange(sat, 80, 255)
    veh_mask = cv2.bitwise_and(veh_mask, cv2.inRange(val, 40, 255))
    police_mask = _color_mask(hsv, *_hsv_active['police'])
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


def detect_low_brightness(frame):
    """True if the scene is dim (poster's 'low brightness' event).
    Uses mean V on a centre crop so HUD overlays don't bias the result."""
    if frame is None:
        return False
    h, w = frame.shape[:2]
    crop = frame[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean()) < LOW_BRIGHTNESS_THRESHOLD


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
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
        x0 = front_per.get('roi_x0', 0)
        cv2.rectangle(front_img, (x0, y0), (PROC_W - 1 - x0, PROC_H - 1), (80, 80, 80), 1)
        _draw_obj(front_img, front_per['red'],    "RED",    (0, 0, 255), y0)
        _draw_obj(front_img, front_per['green'],  "GREEN",  (0, 255, 0), y0)
        _draw_obj(front_img, front_per['yellow'], "YELLOW", (0, 255, 255), y0)
        if lane_offset is not None:
            cx = int(PROC_W / 2 + lane_offset * PROC_W / 2)
            cv2.line(front_img, (PROC_W // 2, PROC_H - 5), (cx, PROC_H - 25), (255, 255, 255), 2)

    if rear_per:
        _draw_obj(rear_img, rear_per['police']['info'],    "POLICE", (255, 0, 0))
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


# ---------------------------------------------------------------------------
# Autonomous HSV calibration (warm-up only; no human input)
# ---------------------------------------------------------------------------
def _bucket_for_hue(h):
    for name, ranges in _HUE_BUCKETS:
        for lo, hi in ranges:
            if lo <= h <= hi:
                return name
    return None


def calibrate_step(frame):
    """Sample dominant colored clusters via k-means; update _calib_state.
    Fully autonomous - no human input. Runs only until CALIB_FRAMES is reached."""
    if _calib_state['done'] or frame is None:
        return
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi = small[int(PROC_H * FRONT_ROI_TOP_FRAC):, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Keep only saturated, bright pixels (ignore road/sky/dark)
    mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)
    pixels = hsv[mask]
    _calib_state['frames_seen'] += 1
    if len(pixels) >= CALIB_MIN_PIXELS:
        samples = pixels.astype(np.float32)
        if len(samples) > CALIB_SUBSAMPLE:
            idx = np.random.choice(len(samples), CALIB_SUBSAMPLE, replace=False)
            samples = samples[idx]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, _, centers = cv2.kmeans(samples, CALIB_KMEANS_K, None,
                                       criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        except cv2.error:
            centers = []
        for c in centers:
            h, s, v = float(c[0]), float(c[1]), float(c[2])
            if s < 80 or v < 60:
                continue
            name = _bucket_for_hue(h)
            if name is not None:
                _calib_state['samples'][name].append(c)
    if _calib_state['frames_seen'] >= CALIB_FRAMES:
        _finalize_calibration()


def _finalize_calibration():
    """Convert accumulated cluster centers into HSV ranges and update _hsv_active."""
    updated = []
    for name, samples in _calib_state['samples'].items():
        if not samples:
            continue
        arr = np.array(samples, dtype=np.float32)
        mean = arr.mean(axis=0)
        lo = np.clip(mean.astype(np.int16) - HSV_MARGIN, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        hi = np.clip(mean.astype(np.int16) + HSV_MARGIN, [0, 0, 0], [179, 255, 255]).astype(np.uint8)
        if name == 'red' and (mean[0] < 15 or mean[0] > 165):
            # Hue wraparound for red: split into two sub-ranges
            _hsv_active['red'] = [
                (np.array([0, lo[1], lo[2]], np.uint8),
                 np.array([min(15, int(hi[0])), hi[1], hi[2]], np.uint8)),
                (np.array([max(165, int(lo[0])), lo[1], lo[2]], np.uint8),
                 np.array([179, hi[1], hi[2]], np.uint8)),
            ]
        else:
            _hsv_active[name] = [(lo, hi)]
        updated.append(name)
    _calib_state['done'] = True
    if updated:
        print(f"[HSV Calib] Auto-calibrated colors: {updated}. Others kept defaults.")
    else:
        print("[HSV Calib] No dominant colors found; keeping all defaults.")


def calibration_done():
    return _calib_state['done']
