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
    detect_lane_curve(frame)    -> dict | None   (CL0-CL2: bird's-eye + sliding-window;
                                                  VISUALIZATION-ONLY for now, not consumed
                                                  by the steering controller)
    draw_lane_curve_debug(dbg)  -> ndarray | None (composite warp/mask/windows panel)
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

    # NEW: area-equivalent circle (tighter than minEnclosingCircle)

    M = cv2.moments(c)

    if M["m00"] > 0:
        circle_x = M["m10"] / M["m00"]
        circle_y = M["m01"] / M["m00"]
    else:
        circle_x = x + w / 2.0
        circle_y = y + h / 2.0

    radius = np.sqrt(area / np.pi)

    return {
        'bbox': (int(x + roi_x0), int(y), int(w), int(h)),

        'circle': (
            int(circle_x + roi_x0),
            int(circle_y),
            int(radius)
        ),

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


# ---------------------------------------------------------------------------
# Curve-aware lane detection (CL0-CL2): bird's-eye warp, lane-pixel mask,
# sliding-window pixel search.
#
# STATUS: visualization-only. The steering controller still consumes
# detect_lane_offset(). CL3 (polynomial fit + curvature) and CL4 (look-ahead
# offset / heading error) will plug these pixel collections into a richer
# struct in a later phase. See task0.md / plan.md.
# ---------------------------------------------------------------------------

# CL0 -- IPM (Inverse Perspective Mapping) constants.
# Source quad picked on the working-resolution (PROC_W x PROC_H = 320 x 240)
# frame. Trapezoid: narrow at the horizon (y ~= 0.58 * H, i.e. just below
# FRONT_ROI_TOP_FRAC=0.28 + a margin to keep slopes safe) and wide at the
# bottom. Numbers are first-pass; tune against screenshot/ before any
# controller hook-up.
IPM_SRC_PTS = np.float32([
    [int(PROC_W * 0.10), PROC_H - 1],            # bottom-left
    [int(PROC_W * 0.90), PROC_H - 1],            # bottom-right
    [int(PROC_W * 0.60), int(PROC_H * 0.58)],    # top-right (horizon)
    [int(PROC_W * 0.40), int(PROC_H * 0.58)],    # top-left  (horizon)
])
IPM_DST_W, IPM_DST_H = 200, 240
IPM_DST_PTS = np.float32([
    [0,           IPM_DST_H - 1],
    [IPM_DST_W - 1, IPM_DST_H - 1],
    [IPM_DST_W - 1, 0],
    [0,           0],
])

# CL1 -- lane-pixel mask thresholds.
LANE_SOBEL_KSIZE = 3
LANE_SOBEL_THRESH = 40        # |Sobel-x| > this after normalize-to-255
LANE_VALUE_THRESH = 150       # V (HSV) > this -> not asphalt

# CL2 -- sliding window search.
N_WINDOWS = 9
WINDOW_MARGIN = 30            # half-width of each search window, in warped px
MIN_WINDOW_PIX = 30           # min pixels to recenter the window
HIST_BOTTOM_FRAC = 1.0 / 3.0  # use bottom 1/3 of mask for base-x histogram
BASE_MIN_SEPARATION = 40      # px between left/right base picks

# Cached perspective matrices (computed once).
_IPM_M = None
_IPM_MINV = None


def _get_ipm_matrices():
    """Return (M, Minv) for the perspective warp, computing once and caching."""
    global _IPM_M, _IPM_MINV
    if _IPM_M is None:
        _IPM_M = cv2.getPerspectiveTransform(IPM_SRC_PTS, IPM_DST_PTS)
        _IPM_MINV = cv2.getPerspectiveTransform(IPM_DST_PTS, IPM_SRC_PTS)
    return _IPM_M, _IPM_MINV


def _lane_pixel_mask(warp_bgr):
    """CL1: cheap, lighting-tolerant binary mask of likely lane-marking pixels.

    Combines two signals:
      - Sobel-x on grayscale (vertical-ish edges -> lane markings appear
        as bright edge pixels after warping)
      - V channel from HSV thresholded high (filters out dark asphalt)
    """
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=LANE_SOBEL_KSIZE)
    sobel = np.absolute(sobel)
    smax = sobel.max() if sobel.size else 1.0
    if smax < 1e-3:
        sobel_u8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        sobel_u8 = np.uint8(255.0 * sobel / smax)
    edge_mask = cv2.inRange(sobel_u8, LANE_SOBEL_THRESH, 255)

    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv[:, :, 2], LANE_VALUE_THRESH, 255)

    mask = cv2.bitwise_or(edge_mask, bright_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def _histogram_base(mask):
    """Return (left_base_x, right_base_x) or (None, None) using the bottom
    HIST_BOTTOM_FRAC of the mask as a column-sum histogram."""
    h = mask.shape[0]
    y0 = int(h * (1.0 - HIST_BOTTOM_FRAC))
    hist = mask[y0:, :].sum(axis=0)
    if hist.max() == 0:
        return None, None
    midpoint = mask.shape[1] // 2
    left_base = int(np.argmax(hist[:midpoint]))
    right_rel = int(np.argmax(hist[midpoint:]))
    right_base = midpoint + right_rel
    # Require both peaks to actually carry signal and to be separated.
    if hist[left_base] < MIN_WINDOW_PIX:
        left_base = None
    if hist[right_base] < MIN_WINDOW_PIX:
        right_base = None
    if (left_base is not None and right_base is not None
            and (right_base - left_base) < BASE_MIN_SEPARATION):
        # Two peaks too close -> probably the same rail; keep the stronger one.
        if hist[left_base] >= hist[right_base]:
            right_base = None
        else:
            left_base = None
    return left_base, right_base


def _sliding_window_collect(mask, base_x):
    """Walk N_WINDOWS bottom-to-top from base_x, collecting nonzero pixel
    coords inside each window. Returns (xs, ys, window_rects)."""
    if base_x is None:
        return np.empty(0, np.int32), np.empty(0, np.int32), []
    h, w = mask.shape
    win_h = h // N_WINDOWS
    nonzero = mask.nonzero()
    nz_y = nonzero[0]
    nz_x = nonzero[1]
    cur_x = int(base_x)
    xs, ys = [], []
    rects = []
    for i in range(N_WINDOWS):
        y_hi = h - i * win_h
        y_lo = max(0, y_hi - win_h)
        x_lo = max(0, cur_x - WINDOW_MARGIN)
        x_hi = min(w, cur_x + WINDOW_MARGIN)
        rects.append((x_lo, y_lo, x_hi, y_hi))
        good = ((nz_y >= y_lo) & (nz_y < y_hi)
                & (nz_x >= x_lo) & (nz_x < x_hi)).nonzero()[0]
        if good.size > 0:
            xs.append(nz_x[good])
            ys.append(nz_y[good])
            if good.size >= MIN_WINDOW_PIX:
                cur_x = int(nz_x[good].mean())
    if xs:
        return np.concatenate(xs), np.concatenate(ys), rects
    return np.empty(0, np.int32), np.empty(0, np.int32), rects


def detect_lane_curve(frame):
    """CL0-CL2 entry point. Visualization-only for now.

    Returns dict with keys (or None if frame is None):
        'warp'       : warped BGR (IPM_DST_H x IPM_DST_W x 3)
        'mask'       : binary lane-pixel mask (same HxW as warp, uint8 0/255)
        'left_pts'   : (xs, ys) ndarray pair of left-rail pixels in warp space
        'right_pts'  : (xs, ys) ndarray pair of right-rail pixels in warp space
        'left_rects' : list of (x_lo, y_lo, x_hi, y_hi) windows (for overlay)
        'right_rects': same for right side
        'left_base'  : seed x or None
        'right_base' : seed x or None
    """
    if frame is None:
        return None
    small = cv2.resize(frame, (PROC_W, PROC_H))
    M, _ = _get_ipm_matrices()
    warp = cv2.warpPerspective(small, M, (IPM_DST_W, IPM_DST_H),
                               flags=cv2.INTER_LINEAR)
    mask = _lane_pixel_mask(warp)
    left_base, right_base = _histogram_base(mask)
    lx, ly, lrects = _sliding_window_collect(mask, left_base)
    rx, ry, rrects = _sliding_window_collect(mask, right_base)
    return {
        'warp': warp,
        'mask': mask,
        'left_pts':  (lx, ly),
        'right_pts': (rx, ry),
        'left_rects':  lrects,
        'right_rects': rrects,
        'left_base':  left_base,
        'right_base': right_base,
    }


def draw_lane_curve_debug(dbg):
    """Composite debug panel: [warp | mask-as-BGR | warp-with-windows].
    Returns None if dbg is None."""
    if dbg is None:
        return None
    warp = dbg['warp']
    mask = dbg['mask']
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    annotated = warp.copy()
    # Tint detected lane pixels: left=red, right=blue
    lx, ly = dbg['left_pts']
    rx, ry = dbg['right_pts']
    if lx.size:
        annotated[ly, lx] = (0, 0, 255)
    if rx.size:
        annotated[ry, rx] = (255, 0, 0)
    # Draw sliding windows
    for (x_lo, y_lo, x_hi, y_hi) in dbg['left_rects']:
        cv2.rectangle(annotated, (x_lo, y_lo), (x_hi - 1, y_hi - 1), (0, 255, 0), 1)
    for (x_lo, y_lo, x_hi, y_hi) in dbg['right_rects']:
        cv2.rectangle(annotated, (x_lo, y_lo), (x_hi - 1, y_hi - 1), (0, 255, 255), 1)
    # Base-x markers along the bottom row
    if dbg['left_base'] is not None:
        cv2.circle(annotated, (dbg['left_base'], IPM_DST_H - 4), 3, (0, 0, 255), -1)
    if dbg['right_base'] is not None:
        cv2.circle(annotated, (dbg['right_base'], IPM_DST_H - 4), 3, (255, 0, 0), -1)

    gap = 6
    panel = np.zeros((IPM_DST_H + 20, IPM_DST_W * 3 + gap * 2, 3), np.uint8)
    panel[:IPM_DST_H, 0:IPM_DST_W] = warp
    panel[:IPM_DST_H, IPM_DST_W + gap: IPM_DST_W * 2 + gap] = mask_bgr
    panel[:IPM_DST_H, IPM_DST_W * 2 + gap * 2:] = annotated
    cv2.putText(panel, "WARP", (4, IPM_DST_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "MASK", (IPM_DST_W + gap + 4, IPM_DST_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "WINDOWS", (IPM_DST_W * 2 + gap * 2 + 4, IPM_DST_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return panel


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

    # Draw circle instead of rectangle
    cx, cy, radius = info['circle']

    cv2.circle(
        img,
        (cx, cy + y_offset),
        radius,
        color,
        2
    )

    # Draw center point
    cv2.circle(
        img,
        (cx, cy + y_offset),
        3,
        color,
        -1
    )

    # Label
    cv2.putText(
        img,
        f"{label} {info['area_frac']*100:.1f}%",
        (
            cx - radius,
            max(15, cy - radius - 5 + y_offset)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1
    )


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
