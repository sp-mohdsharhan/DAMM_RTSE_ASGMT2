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
    GREEN_LANE_CHANGE_BAND, GREEN_SEEK_GAIN, GREEN_SEEK_HOLD_S,
    RED_AVOID_GAIN, YELLOW_AVOID_GAIN, LANE_GAIN,
    LANE_CURVE_GAIN, HILL_AREA_SCALE,
    LOW_BRIGHTNESS_THRESHOLD,
- Functions:
    detect_front_objects(frame) -> dict          (orbs + 'nearest' + 'police' (Ch.3, front))
    detect_rear(frame)          -> dict          (V2.0 Ch.2: teal chasing car, rear camera)
    detect_lane_offset(frame)   -> float | None
    detect_lane_curve(frame)    -> dict | None   (CL0-CL2 bird's-eye + sliding-window for the
                                                  debug panel; also returns 'curve_bias' (-1..+1)
                                                  consumed by the steering controller, plus a
                                                  'lane_grid' labelling lanes 1..N)
    estimate_lane_grid(mask)    -> dict | None   (CL5: label road lanes 1..N left-to-right in
                                                  bird's-eye space for hard lane targeting)
    detect_golden_lane(frame)   -> dict          (Phase 17: orange 'LANE N - ALL GREEN! (Xs)'
                                                  banner -> {'active','lane','remaining_s'};
                                                  logs to logs/golden_lane.log)
    infer_golden_lane_from_tokens(front_per, lane_grid) -> dict
                                                (camera-only fallback: infer the golden lane from
                                                 green-token concentration per lane)
    draw_lane_curve_debug(dbg)  -> ndarray | None (composite warp/mask/windows panel)
    detect_low_brightness(frame) -> bool        (V2.0 Challenge 1: low-light recovery)
    detect_slope(frame)         -> dict | None  (hill / pitch estimate; 'is_hill' drives
                                                  earlier evasion)
    draw_overlay(front_per, rear_per, lane_offset, hud) -> ndarray
    calibrate_step(frame)       -> None         (autonomous HSV warm-up)
    calibration_done()          -> bool
"""

import cv2
import numpy as np
from ultralytics import YOLO

import os
import time

# ---------------------------------------------------------------------------
# Detection mode
# ---------------------------------------------------------------------------

DETECTION_MODE = "HSV"      # "HSV" or "YOLO"
YOLO_MODEL_PATH = "yolo/best.pt"

_yolo_frame_counter = 0
_last_yolo_result = None

_yolo_model = None

def _get_yolo_model():
    global _yolo_model

    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_MODEL_PATH)

    return _yolo_model

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PROC_W, PROC_H = 320, 240                # working resolution for perception
FRONT_ROI_TOP_FRAC = 0.28                # ignore top 28%: keep horizon margin for uphill/downhill slopes
FRONT_ROI_SIDE_FRAC = 0.10               # ignore leftmost/rightmost 10% (grass shoulders); was 0.15,
                                         # narrowed so side-lane orbs (esp. green to chase) stay in view.
                                         # Shape filters below still reject grass strips that leak in.

# Orb-shape filters (reject grass strips, road markings, curb dashes, etc.)
ORB_MAX_AREA_FRAC = 0.07                 # anything larger than this is environment
ORB_MIN_AREA_PX = 60                     # smallest detectable orb (raised: reject small dashes)
ORB_MIN_ASPECT = 0.70                    # near-square only (rejects dash rectangles)
ORB_MAX_ASPECT = 1.45
ORB_MIN_CIRCULARITY = 0.60               # 4*pi*A/P^2 — true orbs ~0.75+, dashes <0.5
ORB_MIN_FILL_RATIO = 0.65                # area / bbox_area; circles fill ~0.78, dashes <0.5
# Solidity = contour_area / convex_hull_area. An orb (even a perspective-squashed
# oval) is convex -> ~0.92-0.98; red curb strips / L-bends along the lane edge
# have notches & concavities -> well below this. Convexity survives perspective,
# so the SAME floor is used for the strict and the relaxed road gate below.
ORB_MIN_SOLIDITY = 0.88                  # reject red lane-edge / curb false positives

# Relaxed gate for ON-ROAD orbs. Once detection is restricted to the asphalt
# region (grass is excluded by the road mask, not by strict shape rules), we can
# accept the big, close, perspective-squashed OVAL orbs that the strict gate
# above used to reject — these are exactly the close orbs about to be hit.
ORB_ROAD_MAX_AREA_FRAC = 0.45            # allow large close orbs (was 0.07 -> dropped them)
ORB_ROAD_MIN_ASPECT = 0.45              # accept ovals (taller-than-wide)
ORB_ROAD_MAX_ASPECT = 2.10              # accept ovals (wider-than-tall)
ORB_ROAD_MIN_CIRCULARITY = 0.45         # ellipses score lower than circles
ORB_ROAD_MIN_FILL_RATIO = 0.55          # ellipse fills its bbox a bit less than a circle
ORB_ROAD_MIN_SOLIDITY = 0.88            # convexity floor (same as strict): kills curb strips
RED_CURB_ASPECT_MAX = 1.25              # red curb chunks are usually stretched, not orb-like
RED_CURB_CIRCULARITY_MIN = 0.58         # reject perspective-cut red/white lane-edge pieces
RED_CURB_EDGE_X_FRAC = 0.16             # extra strict on red near road/camera edges
RED_CURB_EDGE_Y_FRAC = 0.45

# Road-region mask thresholds (asphalt = low-saturation grey).
# Measured asphalt: S~0-75, V~60-100. SAT_MAX 85 captures near-road (S~75) while
# orbs (bright, V>185 -> excluded by VAL_MAX) and grass (S~120) stay out.
ROAD_SAT_MAX = 85
ROAD_VAL_MIN = 30
ROAD_VAL_MAX = 185

# Imminence-first nearest-orb (consumed by the controller in sample_drive.py).
# Each orb gets a bird's-eye ground 'distance' (warp units, IPM_DST_H=240 tall;
# smaller = nearer). The controller reacts to the nearest orb within ACT_DISTANCE
# before falling back to colour priority.
ORB_ACT_DISTANCE = 120.0                  # only override colour priority when nearest orb is within this
ORB_TIE_MARGIN = 25.0                     # if a red/yellow is ~this close to the nearest, dodge it (safety > points)

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

GREEN_ATTRACT_GAIN = 0.6                 # fine-track gain once green is lined up ahead
GREEN_ATTRACT_MIN_AREA = 0.004           # act on greens a touch earlier (commit the lane change in time)
# Green pursuit: a gentle proportional pull can't cross a lane for an
# adjacent-lane green (far greens have a tiny centroid offset). So commit to a
# full-strength steer toward any green sitting clearly off-centre (= another
# lane), hold briefly to finish the crossing, then fall back to fine-tracking.
GREEN_LANE_CHANGE_BAND = 0.92            # |centroid_x_norm| above this => green is in another lane -> commit
GREEN_SEEK_GAIN = 1.0                    # committed steer magnitude toward an off-lane green
GREEN_SEEK_HOLD_S = 1.0                  # bridge frames where green flickers / leaves ROI mid-cross
# (Keep-LEFT / keep-RIGHT home-lane bias removed — the car now centres in the
#  lane and reacts to orbs. See plan.md Phase 5 to restore a home-lane hug.)
RED_AVOID_GAIN = 0.7                     # full-lock swerve when red is in path
YELLOW_AVOID_GAIN = 0.5
LANE_GAIN = 0.6

# Lane-curve steering bias: how strongly the anticipated bend from
# detect_lane_curve() biases steering on top of the instantaneous lane offset.
# Lets the car steer into a curve before the near lane offset has moved.
LANE_CURVE_GAIN = 0.4

# Hill / slope handling (consumed by the controller in sample_drive.py).
# On a crest the road horizon shifts up and orbs appear with little reaction
# distance, so the controller (a) eases throttle and (b) shrinks the red/yellow
# trigger areas via HILL_AREA_SCALE so evasion fires earlier ("prepare to swerve").
SLOPE_HILL_DEV = 0.06                    # |horizon deviation from flat baseline| to call it a hill
SLOPE_EMA_ALPHA = 0.05                   # EMA rate for the self-calibrating flat-road baseline
HILL_AREA_SCALE = 0.5                    # scale red/yellow trigger area thresholds on a hill (evade sooner)
ASPHALT_MAX_SAT = 60                     # road asphalt is low-saturation gray
ASPHALT_MIN_VAL = 30
ASPHALT_MAX_VAL = 170
ROAD_ROW_THRESH = 0.20                   # fraction of center-band asphalt for a row to count as "road"
SLOPE_CENTER_BAND = (0.30, 0.70)         # center column fraction used to locate the road horizon
SLOPE_GAP_TOL = 8                        # rows of non-road (lane dashes) tolerated before the road top

# Low-brightness event detection (poster: "low brightness — turn light on or all tokens yellow")
LOW_BRIGHTNESS_THRESHOLD = 45            # mean V channel below this -> consider it dim

# ---------------------------------------------------------------------------
# HSV ranges & auto-calibration state
# ---------------------------------------------------------------------------
# OpenCV: H:0-179, S:0-255, V:0-255
# Red orb sprite colours (measured): light #F9ACB9=H175,S79,V249; mid #E66179=H175,S147,V230;
# dark #C12742=H175,S203,V193. Hue pinned ~175 (high wraparound side), always BRIGHT (V>=193).
# Red is PINNED (excluded from auto-calibration). Two sub-ranges keep the hue wraparound safe.
HSV_RED_1 = (np.array([0, 60, 120]),    np.array([8, 255, 255]))    # low-side wraparound spill
HSV_RED_2 = (np.array([168, 60, 120]),  np.array([179, 255, 255]))  # main orb red (~H175)
# Green orb sprite colours (measured): light #BDF3B5=H56,S65,V243; mid #87E27E=H57,S113,V226;
# dark #5AB853=H58,S140,V184. Tight hue ~56-58, always BRIGHT (V>=184); grass is dark (V~65).
# Range below covers all three with margin and still excludes grass via the V floor.
# Green is PINNED (excluded from auto-calibration) so this measured range is what's used.
HSV_GREEN = (np.array([48, 30, 130]),   np.array([66, 255, 255]))
# Yellow/gold orb sprite colours (measured): light #FEEA75=H26,S138,V254; mid #DFA214=H21,S232,V223;
# dark #9D700C=H21,S236,V157. Hue ~21-26 (gold), bright & saturated. Yellow is PINNED.
HSV_YELLOW = (np.array([16, 100, 120]), np.array([32, 255, 255]))

# Police car colour (V2.0 Challenge 3, rear camera). The police car is the
# red+blue split livery; the BLUE half is the reliable discriminator (red would
# clash with red tokens / other cars). Measured from game rule/police.jpg:
# blue half H~125-127, S~245, V~110. Range below covers it; teal cars (H~85) and
# the sky (lower saturation) are excluded. The rear sky band is also masked off
# (REAR_SKY_FRAC) so blue sky/buildings can't be read as police.
HSV_POLICE = (np.array([112, 210, 60]), np.array([135, 255, 255]))  # S>=210: police blue S~245 vs sky S~200
REAR_SKY_FRAC = 0.35                      # ignore top 35% of the rear frame (sky/buildings)

# Chasing car (V2.0 Challenge 2): the TEAL/cyan car. Measured H~86, S~190 (S/V vary
# with lighting). Teal (H 78-98) is distinct from sky/buildings (H~120), police
# blue (H 112-135), and every orb (green 57 / yellow 21 / red 175) — so pinning
# it kills the "any saturated blob" false positives that fired on sky and orbs.
HSV_CHASING = (np.array([78, 120, 40]), np.array([98, 255, 255]))

# Police-car front avoidance (Challenge 3): if the police blob ahead is this big
# and roughly centred, dodge it (collision = game over) instead of seeking a token.
POLICE_DODGE_AREA = 0.04
POLICE_DODGE_BAND = 0.50

# Police detection hardening (Challenge 3).
POLICE_MIN_AREA_PX = 120             # smallest pixel area to accept (reject distant specks)
POLICE_MIN_AREA_FRAC = 0.008         # roi-fraction floor (rejects tiny / far-off blobs)
POLICE_RED_VERIFY_RATIO = 0.15       # dark-red pixels in search box must be >= this * blue_area
POLICE_VERIFY_PAD_X_FRAC = 1.0       # search-box width  = +/- 1.0 * blue bbox width
POLICE_VERIFY_PAD_Y_FRAC = 0.5       # search-box height = +/- 0.5 * blue bbox height

# Chasing-car detection hardening (Challenge 2).
CHASING_MIN_AREA_FRAC = 0.010        # below this the car is too far away to matter
CHASING_CENTER_BAND_FRAC = 0.90      # ignore teal blobs only on the very far shoulder
                                     # (was 0.75: dismissed car approaching on adj. lane)
CHASING_GROW_DELTA = 0.003           # total area_frac increase across smoothing window
                                     # (was 0.006: too large, car slips through fast)
CHASING_HIST_LEN = 4                 # frames of area history used for smoothed growth
CHASING_MIN_GROW_FRAMES = 2          # minimum history frames before "growing" can fire
                                     # (was 3: ~150ms lag at 50ms/frame; now 100ms)
CHASING_EMERGENCY_AREA_FRAC = 0.045  # skip growth check if car is already this large

# Mutable active HSV ranges (list-of-(lo,hi) per color), consulted by detectors.
# All orb colours are pinned (calibration off); police is pinned too.
_hsv_active = {
    'red':    [HSV_RED_1, HSV_RED_2],
    'green':  [HSV_GREEN],
    'yellow': [HSV_YELLOW],
    'police':  [HSV_POLICE],    # FRONT camera (Challenge 3)
    'chasing': [HSV_CHASING],   # REAR camera (Challenge 2)
}

# --- Auto-calibration constants & state ---
CALIB_FRAMES = 90                                  # ~3s at 30 Hz
CALIB_KMEANS_K = 6
CALIB_MIN_PIXELS = 200
CALIB_SUBSAMPLE = 5000
HSV_MARGIN = np.array([10, 60, 60], dtype=np.int16)
# All three orb colours are now PINNED to measured sprite ranges (HSV_GREEN /
# HSV_RED_* / HSV_YELLOW), so auto-calibration is fully disabled: done=True from
# the start and no hue buckets to learn, so calibrate_step() is never invoked.
_HUE_BUCKETS = []
_calib_state = {
    'frames_seen': 0,
    'samples': {},
    'done': True,
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


def _road_region_mask(hsv):
    """Filled mask (ROI coords) of the drivable road region, used to gate orb
    detection so only orbs sitting ON the asphalt count (grass excluded).
    Asphalt is low-saturation grey; a morphological close + convex-hull fill
    bridge the holes punched by lane markings and the orbs themselves; a final
    dilation lets orbs straddling the road edge still qualify. None if no road."""
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    road = ((s < ROAD_SAT_MAX) & (v > ROAD_VAL_MIN) & (v < ROAD_VAL_MAX)).astype(np.uint8) * 255
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < 0.05 * road.shape[0] * road.shape[1]:
        return None                                   # no plausible road visible
    filled = np.zeros_like(road)
    cv2.drawContours(filled, [cv2.convexHull(biggest)], -1, 255, -1)
    # Modest dilation: lets an orb straddling a lane-edge still count, without
    # bleeding the gate out onto the grass shoulder (we want off-road orbs excluded).
    return cv2.dilate(filled, np.ones((9, 9), np.uint8))


def _ground_distance(px, py):
    """Bird's-eye distance proxy for a full-frame image point (px, py): project
    it through the IPM homography and return IPM_DST_H - warped_y (smaller =
    nearer). Returns +inf for points above the horizon / off the road plane."""
    M, _ = _get_ipm_matrices()
    pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
    wy = float(cv2.perspectiveTransform(pt, M)[0, 0][1])
    if wy < 0.0 or wy > IPM_DST_H:
        return float('inf')
    return float(IPM_DST_H - wy)


def _orb_contours_info(mask, roi_area, roi_x0=0, roi_y0=0, road_mask=None,
                       max_area_frac=ORB_MAX_AREA_FRAC,
                       min_aspect=ORB_MIN_ASPECT, max_aspect=ORB_MAX_ASPECT,
                       min_circularity=ORB_MIN_CIRCULARITY,
                       min_fill=ORB_MIN_FILL_RATIO,
                       min_solidity=ORB_MIN_SOLIDITY,
                       reject_red_curb=False):
    """Return a LIST of orb dicts (one per qualifying contour), or [].
    Shape filters (aspect, circularity, fill, max-area) reject grass strips /
    road markings; road_mask (ROI coords) keeps only orbs whose centroid is on
    the asphalt, which is what lets the thresholds relax for big/close/oval orbs.
    Each dict carries a bird's-eye 'distance' (smaller = nearer) from projecting
    the orb's ground-contact point. roi_x0 / roi_y0 map ROI coords back to the
    full PROC_W x PROC_H frame.
    """
    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rh = road_mask.shape[0] if road_mask is not None else 0
    rw = road_mask.shape[1] if road_mask is not None else 0
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < ORB_MIN_AREA_PX:
            continue
        area_frac = float(area) / float(roi_area)
        if area_frac > max_area_frac:
            continue                                  # too big -> environment
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        if aspect < min_aspect or aspect > max_aspect:
            continue                                  # too elongated -> grass strip
        perim = cv2.arcLength(c, True)
        if perim <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perim * perim)
        if circularity < min_circularity:
            continue                                  # not blob-like
        full_cx_norm = ((x + w / 2.0 + roi_x0) - PROC_W / 2.0) / (PROC_W / 2.0)
        full_cy_frac = (y + h / 2.0 + roi_y0) / float(PROC_H)
        if reject_red_curb:
            # Red/white lane-edge stripes can pass the road gate when their
            # centroid falls on asphalt. They are stretched/cut by perspective
            # and usually sit near the lower side edges, unlike round tokens.
            if aspect > RED_CURB_ASPECT_MAX or circularity < RED_CURB_CIRCULARITY_MIN:
                continue
            if abs(full_cx_norm) > (1.0 - RED_CURB_EDGE_X_FRAC) and full_cy_frac > RED_CURB_EDGE_Y_FRAC:
                continue
        bbox_area = float(w * h)
        if bbox_area <= 0 or (area / bbox_area) < min_fill:
            continue                                  # sparse/hollow (dashed stripe)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or (area / hull_area) < min_solidity:
            continue                                  # concave -> red curb / lane-edge strip
        if road_mask is not None:                     # must sit ON the road
            cxr = min(rw - 1, max(0, int(x + w / 2.0)))
            cyr = min(rh - 1, max(0, int(y + h / 2.0)))
            if road_mask[cyr, cxr] == 0:
                continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            circle_x = m["m10"] / m["m00"]
            circle_y = m["m01"] / m["m00"]
        else:
            circle_x = x + w / 2.0
            circle_y = y + h / 2.0
        cx = x + w / 2.0 + roi_x0
        cy = y + h / 2.0
        # Ground-contact point (bottom-centre) in full-frame coords -> bird's-eye distance.
        dist = _ground_distance(x + w / 2.0 + roi_x0, y + h + roi_y0)
        out.append({
            'bbox': (int(x + roi_x0), int(y), int(w), int(h)),
            'circle': (int(circle_x + roi_x0), int(circle_y), int(np.sqrt(area / np.pi))),
            'area_frac': area_frac,
            'centroid_x_norm': (cx - PROC_W / 2.0) / (PROC_W / 2.0),  # -1..+1
            'centroid_y': float(cy),
            'distance': dist,
        })
    return out


def detect_front_objects(frame):

    if DETECTION_MODE == "YOLO":
        return detect_front_objects_yolo(frame)

    return detect_front_objects_hsv(frame)

def detect_front_objects_yolo(frame):

    global _yolo_frame_counter
    global _last_yolo_result

    _yolo_frame_counter += 1

    # Run YOLO every 2 frames
    if _yolo_frame_counter % 2 != 0 and _last_yolo_result is not None:
        return _last_yolo_result

    model = _get_yolo_model()

    small = cv2.resize(frame, (PROC_W, PROC_H))

    results = model.predict(
        small,
        conf=0.40,
        verbose=False
    )

    reds = []
    greens = []
    yellows = []
    police_list = []

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            area_frac = (w * h) / float(PROC_W * PROC_H)

            obj = {
                'bbox': (x1, y1, w, h),
                'circle': (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                    int(max(w, h) / 2)
                ),
                'area_frac': area_frac,
                'centroid_x_norm':
                    (((x1 + x2) / 2) - PROC_W / 2.0)
                    / (PROC_W / 2.0),
                'centroid_y': float((y1 + y2) / 2),

                # closer objects appear lower in image
                'distance': float(PROC_H - y2),

                'color': cls_name
            }

            if cls_name == "red":
                reds.append(obj)

            elif cls_name == "green":
                greens.append(obj)

            elif cls_name == "yellow":
                yellows.append(obj)

            elif cls_name == "police":
                police_list.append(obj)

    all_orbs = reds + greens + yellows

    all_orbs.sort(key=lambda x: x['distance'])

    return {
        'frame': small,
        'roi_y0': 0,
        'roi_x0': 0,
        'road_mask': None,

        'red':
            min(reds, key=lambda x: x['distance'])
            if reds else None,

        'green':
            min(greens, key=lambda x: x['distance'])
            if greens else None,

        'yellow':
            min(yellows, key=lambda x: x['distance'])
            if yellows else None,

        'police':
            min(police_list, key=lambda x: x['distance'])
            if police_list else None,

        'orbs': all_orbs,

        'nearest':
            all_orbs[0]
            if all_orbs else None
    }

def detect_front_objects_hsv(frame):
    """Return dict {'frame','roi_y0','roi_x0','road_mask','red','green','yellow','orbs','nearest'}.

    Orbs are detected ON the asphalt only (road-region mask gates the colour
    contours). Per-colour entries ('red'/'green'/'yellow') are the NEAREST orb of
    that colour by bird's-eye distance; 'orbs' is every on-road orb (all colours)
    sorted nearest-first; 'nearest' is the closest overall. Falls back to strict
    grass-safe thresholds if no road is found.
    """
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_y0 = int(PROC_H * FRONT_ROI_TOP_FRAC)
    roi_x0 = int(PROC_W * FRONT_ROI_SIDE_FRAC)
    roi_x1 = PROC_W - roi_x0
    roi = small[roi_y0:, roi_x0:roi_x1]
    roi_area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    road = _road_region_mask(hsv)

    if road is not None:
        gate = dict(road_mask=road,
                    max_area_frac=ORB_ROAD_MAX_AREA_FRAC,
                    min_aspect=ORB_ROAD_MIN_ASPECT, max_aspect=ORB_ROAD_MAX_ASPECT,
                    min_circularity=ORB_ROAD_MIN_CIRCULARITY,
                    min_fill=ORB_ROAD_MIN_FILL_RATIO,
                    min_solidity=ORB_ROAD_MIN_SOLIDITY)
    else:
        gate = {}                                     # strict defaults (grass-safe)

    def _orbs(color):
        lst = _orb_contours_info(_color_mask(hsv, *_hsv_active[color]),
                                 roi_area,
                                 roi_x0,
                                 roi_y0,
                                 reject_red_curb=(color == 'red'),
                                 **gate)
        for o in lst:
            o['color'] = color
        return lst

    reds, greens, yellows = _orbs('red'), _orbs('green'), _orbs('yellow')
    all_orbs = sorted(reds + greens + yellows, key=lambda o: o['distance'])

    def _nearest(lst):
        return min(lst, key=lambda o: o['distance']) if lst else None

    # Police car ahead (Challenge 3): verify by the unique red+blue split livery
    # (road-gated, min-area floored, dark-red adjacency confirmed).
    police = _detect_police(hsv, road, roi_area, roi_x0)

    return {
        'frame': small,
        'roi_y0': roi_y0,
        'roi_x0': roi_x0,
        'road_mask': road,
        'red':    _nearest(reds),
        'green':  _nearest(greens),
        'yellow': _nearest(yellows),
        'orbs':    all_orbs,
        'nearest': all_orbs[0] if all_orbs else None,
        'police':  police,
    }


def _largest_blob(mask, roi_area, roi_x0=0, min_area_px=ORB_MIN_AREA_PX):
    """Largest contour by area with only a minimal size gate (NO orb-shape
    filter) — for cars (chasing car / police car are not circular). roi_x0 maps
    a horizontally-cropped ROI back to full-frame x (same convention as orbs: x
    is full-frame, y is ROI-relative so the overlay's y_offset works). Returns an
    info dict {bbox, circle, area_frac, centroid_x_norm, centroid_y} or None."""
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area_px and a > best_area:
            best, best_area = c, a
    if best is None:
        return None
    x, y, w, h = cv2.boundingRect(best)
    cx, cy = x + w / 2.0 + roi_x0, y + h / 2.0
    return {
        'bbox': (int(x + roi_x0), int(y), int(w), int(h)),
        'circle': (int(cx), int(cy), int(np.sqrt(best_area / np.pi))),
        'area_frac': float(best_area) / float(roi_area),
        'centroid_x_norm': (cx - PROC_W / 2.0) / (PROC_W / 2.0),
        'centroid_y': float(cy),
    }


def _detect_police(hsv, road_mask, roi_area, roi_x0=0):
    """Confirm a police car by its unique red+blue split livery.

    The blue half is the primary detection; we then verify that dark-red pixels
    sit immediately adjacent to it. This rejects sky / blue signs / red orbs /
    tail-lights — none of which co-occur with both halves simultaneously.
    Also gates the blue blob on the road mask so sky patches can't win."""
    blue_mask = _color_mask(hsv, *_hsv_active['police'])
    if blue_mask is None:
        return None
    if road_mask is not None:                        # must sit on the asphalt
        blue_mask = cv2.bitwise_and(blue_mask, road_mask)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    for c in contours:
        a = cv2.contourArea(c)
        if a >= POLICE_MIN_AREA_PX and a > best_area:
            best, best_area = c, a
    if best is None or (best_area / roi_area) < POLICE_MIN_AREA_FRAC:
        return None

    bx, by, bw, bh = cv2.boundingRect(best)

    # Verify dark-red half is adjacent to the blue blob.
    # Police red is dark (V <= ~170) vs bright red orbs (V >= 193), so a
    # low-V red mask is used — it won't mis-fire on grabbable orbs.
    pad_x = int(bw * POLICE_VERIFY_PAD_X_FRAC)
    pad_y = int(bh * POLICE_VERIFY_PAD_Y_FRAC)
    sx = max(0, bx - pad_x)
    sy = max(0, by - pad_y)
    ex = min(hsv.shape[1], bx + bw + pad_x)
    ey = min(hsv.shape[0], by + bh + pad_y)
    region = hsv[sy:ey, sx:ex]
    rh = region[:, :, 0]
    rs = region[:, :, 1]
    rv = region[:, :, 2]
    dark_red = (((rh <= 10) | (rh >= 165)) & (rs >= 120) & (rv >= 40) & (rv <= 170))
    if dark_red.sum() < best_area * POLICE_RED_VERIFY_RATIO:
        return None

    cx = bx + bw / 2.0 + roi_x0
    cy = by + bh / 2.0
    return {
        'bbox': (int(bx + roi_x0), int(by), int(bw), int(bh)),
        'circle': (int(cx), int(cy), int(np.sqrt(best_area / np.pi))),
        'area_frac': float(best_area) / float(roi_area),
        'centroid_x_norm': (cx - PROC_W / 2.0) / (PROC_W / 2.0),
        'centroid_y': float(cy),
    }


# Chasing-car smoothed area history for robust growth detection.
_chasing_area_hist = []


def detect_rear(frame):

    if DETECTION_MODE == "YOLO":
        return detect_rear_yolo(frame)

    return detect_rear_hsv(frame)

def detect_rear_yolo(frame):

    model = _get_yolo_model()

    small = cv2.resize(frame, (PROC_W, PROC_H))

    results = model.predict(
        small,
        conf=0.40,
        verbose=False
    )

    other = None

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            #
            # Rear-camera only cares about chasing cars
            #
            if cls_name not in ("car", "other_car", "chasing_car"):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            area_frac = (w * h) / float(PROC_W * PROC_H)

            candidate = {
                'bbox': (x1, y1, w, h),
                'circle': (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                    int(max(w, h) / 2)
                ),
                'area_frac': area_frac,
                'centroid_x_norm':
                    (((x1 + x2) / 2) - PROC_W / 2.0)
                    / (PROC_W / 2.0),
                'centroid_y': float((y1 + y2) / 2),
                'distance': float(PROC_H - y2),
            }

            #
            # Keep nearest/largest car
            #
            if other is None or candidate['area_frac'] > other['area_frac']:
                other = candidate

    # Same growth/emergency arming logic as the HSV path so the controller sees
    # an identical {'frame', 'other_car': {'info', 'growing'}} contract.
    global _chasing_area_hist
    growing = False
    if (other is not None
            and other['area_frac'] > CHASING_MIN_AREA_FRAC
            and abs(other['centroid_x_norm']) < CHASING_CENTER_BAND_FRAC):
        # Emergency trigger: car is already close/large — swerve immediately without
        # waiting for growth history (handles fast approach & late detection).
        if other['area_frac'] >= CHASING_EMERGENCY_AREA_FRAC:
            growing = True
            _chasing_area_hist.clear()
        else:
            _chasing_area_hist.append(other['area_frac'])
            if len(_chasing_area_hist) > CHASING_HIST_LEN:
                _chasing_area_hist.pop(0)
            if len(_chasing_area_hist) >= CHASING_MIN_GROW_FRAMES:
                growing = (_chasing_area_hist[-1] - _chasing_area_hist[0]) > CHASING_GROW_DELTA
    elif other is None:
        # Only reset history when the car is fully lost; keep accumulating if it
        # shifted outside the center band (e.g., approaching on adjacent lane).
        _chasing_area_hist.clear()

    return {
        'frame': small,
        'other_car': {'info': other, 'growing': growing},
    }


def detect_rear_hsv(frame):
    """Rear-camera CHASING CAR (V2.0 Challenge 2). Returns
       {'frame', 'other_car': {'info', 'growing'}}.

    Hardening over the previous version:
      * Crops the top REAR_SKY_FRAC band so sky / buildings can't fire as teal.
      * Requires CHASING_MIN_AREA_FRAC -> distant specks are ignored.
      * Requires |centroid_x_norm| < CHASING_CENTER_BAND_FRAC -> far-shoulder
        teal blobs (road barriers, signs) do not trigger a swerve.
      * 'growing' uses a CHASING_HIST_LEN-frame smoothed area window so a single
        noisy frame can't arm it and a single missed frame can't reset it.
    """
    global _chasing_area_hist
    small = cv2.resize(frame, (PROC_W, PROC_H))
    roi_y0 = int(PROC_H * REAR_SKY_FRAC)            # drop sky / building band
    roi = small[roi_y0:, :]
    roi_area = roi.shape[0] * roi.shape[1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    teal = _color_mask(hsv, *_hsv_active['chasing'])
    if teal is not None:
        teal = cv2.morphologyEx(teal, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
        teal = cv2.morphologyEx(teal, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    other = _largest_blob(teal, roi_area)

    # Shift y coords back to full-frame so overlay draws correctly.
    if other is not None:
        bx, by, bw, bh = other['bbox']
        other['bbox'] = (bx, by + roi_y0, bw, bh)
        ccx, ccy, cr = other['circle']
        other['circle'] = (ccx, ccy + roi_y0, cr)
        other['centroid_y'] = float(other['centroid_y'] + roi_y0)

    growing = False
    if (other is not None
            and other['area_frac'] > CHASING_MIN_AREA_FRAC
            and abs(other['centroid_x_norm']) < CHASING_CENTER_BAND_FRAC):
        # Emergency trigger: car is already close/large — swerve immediately without
        # waiting for growth history (handles fast approach & late detection).
        if other['area_frac'] >= CHASING_EMERGENCY_AREA_FRAC:
            growing = True
            _chasing_area_hist.clear()
        else:
            _chasing_area_hist.append(other['area_frac'])
            if len(_chasing_area_hist) > CHASING_HIST_LEN:
                _chasing_area_hist.pop(0)
            if len(_chasing_area_hist) >= CHASING_MIN_GROW_FRAMES:
                growing = (_chasing_area_hist[-1] - _chasing_area_hist[0]) > CHASING_GROW_DELTA
    elif other is None:
        # Only reset history when the car is fully lost; keep accumulating if it
        # shifted outside the center band (e.g., approaching on adjacent lane).
        _chasing_area_hist.clear()

    return {
        'frame': small,
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

# CL5 -- lane-grid labelling (Phase 17 / Golden Lane).
# The track has N_LANES road lanes, numbered 1..N left-to-right from the
# driver's view. We detect the road span in bird's-eye space, then smooth it so
# lane numbering can follow the car without jumping on sparse lane markings.
N_LANES = 5
LANE_COL_MIN_PIX = 8          # min column-sum (bottom band) to count as road signal
LANE_EDGE_BAND_FRAC = 0.5     # use bottom half of the mask to find road edges
LANE_MIN_ROAD_FRAC = 0.30     # road span must cover >= this frac of warp width
LANE_GRID_EMA_ALPHA = 0.25
LANE_GRID_MAX_EDGE_JUMP_FRAC = 0.20
_lane_grid_state = {'left': None, 'right': None}

# Golden Lane camera fallback. Since the "LANE N - ALL GREEN" banner is drawn on
# the main game view, not the camera feed, infer the event from a lane that has a
# strong concentration of green tokens.
GOLDEN_INFER_MIN_GREEN_COUNT = 3
GOLDEN_INFER_MIN_SCORE = 2.6
GOLDEN_INFER_DOMINANCE_RATIO = 1.45
GOLDEN_INFER_GREEN_WEIGHT = 1.0
GOLDEN_INFER_NON_GREEN_PENALTY = 0.45
GOLDEN_INFER_MAX_DISTANCE = 220.0
GOLDEN_INFER_MIN_LANE_CONFIDENCE = 0.45
GOLDEN_INFER_BOUNDARY_MARGIN_FRAC = 0.15

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


def _curve_bias_from_pixels(lx, ly, rx, ry):
    """Estimate how much the lane bends ahead, in -1..+1 (positive => bends right).

    Fits x = f(y) for whichever rails carry enough pixels, evaluates the lane
    centre near the car (warp bottom) vs far ahead (warp top), and returns the
    normalized horizontal drift between them plus the near lane-centre offset.
    Returns (curve_bias, lane_center_norm); curve_bias is 0.0 and
    lane_center_norm is None when there isn't enough signal."""
    y_near = IPM_DST_H - 1
    y_far = int(IPM_DST_H * 0.15)

    def fit_eval(xs, ys):
        if xs.size < MIN_WINDOW_PIX:
            return None, None
        deg = 2 if xs.size > 200 else 1
        coeffs = np.polyfit(ys, xs, deg)
        return float(np.polyval(coeffs, y_near)), float(np.polyval(coeffs, y_far))

    lnear, lfar = fit_eval(lx, ly)
    rnear, rfar = fit_eval(rx, ry)

    if lnear is not None and rnear is not None:
        c_near = (lnear + rnear) / 2.0
        c_far = (lfar + rfar) / 2.0
    elif lnear is not None:
        c_near, c_far = lnear, lfar
    elif rnear is not None:
        c_near, c_far = rnear, rfar
    else:
        return 0.0, None

    half = IPM_DST_W / 2.0
    curve_bias = float(np.clip((c_far - c_near) / half, -1.0, 1.0))
    lane_center_norm = float(np.clip((c_near - half) / half, -1.0, 1.0))
    return curve_bias, lane_center_norm


def estimate_lane_grid(mask):
    """CL5: label the road's N_LANES lanes (1..N, left->right) in warp space.

    Finds the road's left/right extent from the lane-pixel mask, smooths it
    across frames, slices that span into 5 lanes, and locates which lane contains
    the car at the warp bottom-centre. This is used for Golden Lane targeting
    and for validating which lane the car is currently in.

    Returns dict (all x-values in warp pixels; *_norm in -1..+1 about warp
    centre, positive => right of car) or None if there is not enough signal:
        'n_lanes'            : N_LANES
        'road_left'          : left road-edge x
        'road_right'         : right road-edge x
        'lane_width'         : per-lane width in warp px
        'lane_centers'       : list[N_LANES] of lane-centre x (left->right)
        'lane_centers_norm'  : list[N_LANES] of lane-centre offsets, -1..+1
        'lane_bounds'        : list[N_LANES+1] of lane-boundary x
        'car_lane'           : 1..N lane index under the car, or None
    """
    if mask is None or mask.size == 0:
        return None
    h, w = mask.shape
    y0 = int(h * (1.0 - LANE_EDGE_BAND_FRAC))
    hist = mask[y0:, :].sum(axis=0) // 255
    cols = np.nonzero(hist >= LANE_COL_MIN_PIX)[0]
    if cols.size == 0:
        return None

    measured_left = float(cols.min())
    measured_right = float(cols.max())
    measured_span = measured_right - measured_left
    if measured_span < float(w * LANE_MIN_ROAD_FRAC):
        return None

    prev_left = _lane_grid_state['left']
    prev_right = _lane_grid_state['right']
    max_jump = w * LANE_GRID_MAX_EDGE_JUMP_FRAC
    jump_ratio = 0.0
    held_previous = False
    if prev_left is None or prev_right is None:
        road_left = measured_left
        road_right = measured_right
    else:
        left_jump = abs(measured_left - prev_left)
        right_jump = abs(measured_right - prev_right)
        jump_ratio = max(left_jump, right_jump) / max(1.0, w)
        if left_jump > max_jump or right_jump > max_jump:
            road_left = prev_left
            road_right = prev_right
            held_previous = True
        else:
            road_left = (1.0 - LANE_GRID_EMA_ALPHA) * prev_left + LANE_GRID_EMA_ALPHA * measured_left
            road_right = (1.0 - LANE_GRID_EMA_ALPHA) * prev_right + LANE_GRID_EMA_ALPHA * measured_right

    _lane_grid_state['left'] = road_left
    _lane_grid_state['right'] = road_right
    span = road_right - road_left
    if span < float(w * LANE_MIN_ROAD_FRAC):
        return None

    lane_width = span / float(N_LANES)
    lane_bounds = [road_left + i * lane_width for i in range(N_LANES + 1)]
    lane_centers = [road_left + (i + 0.5) * lane_width for i in range(N_LANES)]

    half = w / 2.0
    lane_centers_norm = [float(np.clip((c - half) / half, -1.0, 1.0))
                         for c in lane_centers]

    # The car sits at the warp bottom-centre; find which lane bin contains it.
    car_x = half
    car_lane = None
    if road_left <= car_x <= road_right:
        car_lane = int((car_x - road_left) / lane_width) + 1
        car_lane = max(1, min(N_LANES, car_lane))

    lane_confidence = 1.0
    lane_confidence -= min(0.45, jump_ratio * 1.8)
    if held_previous:
        lane_confidence -= 0.25
    lane_confidence = float(np.clip(lane_confidence, 0.0, 1.0))

    return {
        'n_lanes': N_LANES,
        'road_left': int(round(road_left)),
        'road_right': int(round(road_right)),
        'measured_left': int(round(measured_left)),
        'measured_right': int(round(measured_right)),
        'lane_width': float(lane_width),
        'lane_centers': [float(c) for c in lane_centers],
        'lane_centers_norm': lane_centers_norm,
        'lane_bounds': [float(b) for b in lane_bounds],
        'car_lane': car_lane,
        'confidence': lane_confidence,
        'edge_confidence': lane_confidence,
        'curve_penalty': 0.0,
        'held_previous': bool(held_previous),
    }


def _orb_ground_warp_x(orb):
    """Project an orb's bottom-centre contact point into bird's-eye x."""
    bbox = orb.get('bbox')
    if not bbox:
        return None
    x, y, w, h = bbox
    px = float(x + w / 2.0)
    py = float(y + h)
    M, _ = _get_ipm_matrices()
    pt = np.array([[[px, py]]], dtype=np.float32)
    wx, wy = cv2.perspectiveTransform(pt, M)[0, 0]
    if wx < 0.0 or wx > IPM_DST_W or wy < 0.0 or wy > IPM_DST_H:
        return None
    return float(wx)


def _lane_for_warp_x(lane_grid, warp_x):
    bounds = lane_grid.get('lane_bounds') if lane_grid else None
    if not bounds or warp_x is None:
        return None
    for idx in range(len(bounds) - 1):
        if bounds[idx] <= warp_x <= bounds[idx + 1]:
            width = max(1.0, bounds[idx + 1] - bounds[idx])
            margin = width * GOLDEN_INFER_BOUNDARY_MARGIN_FRAC
            if (warp_x - bounds[idx]) < margin or (bounds[idx + 1] - warp_x) < margin:
                return None
            return idx + 1
    return None


def infer_golden_lane_from_tokens(front_per, lane_grid):
    """Infer Golden Lane from green-token concentration in one road lane.

    The Golden Lane text is not present in the camera feed, so this is the
    camera-only trigger for the existing hard-steer path. It returns the same
    shape as detect_golden_lane(): active/lane/remaining_s. Extra score fields
    are included for logging/debugging but ignored by the controller.
    """
    inactive = {
        'active': False,
        'lane': None,
        'remaining_s': None,
        'source': 'tokens',
        'confidence': 0.0,
    }
    if not front_per or not lane_grid:
        return inactive
    if lane_grid.get('confidence', 1.0) < GOLDEN_INFER_MIN_LANE_CONFIDENCE:
        return inactive

    lane_count = int(lane_grid.get('n_lanes') or N_LANES)
    lane_green_counts = [0] * lane_count
    lane_scores = [0.0] * lane_count

    for orb in front_per.get('orbs', []):
        if orb.get('distance', float('inf')) > GOLDEN_INFER_MAX_DISTANCE:
            continue
        lane = _lane_for_warp_x(lane_grid, _orb_ground_warp_x(orb))
        if lane is None:
            continue

        idx = lane - 1
        # Farther tokens are still useful, but closer tokens are more likely to
        # match the lane the car can actually reach before the 5 s timer ends.
        distance = max(0.0, float(orb.get('distance', GOLDEN_INFER_MAX_DISTANCE)))
        proximity = 1.0 + max(0.0, GOLDEN_INFER_MAX_DISTANCE - distance) / GOLDEN_INFER_MAX_DISTANCE
        if orb.get('color') == 'green':
            lane_green_counts[idx] += 1
            lane_scores[idx] += GOLDEN_INFER_GREEN_WEIGHT * proximity
        else:
            lane_scores[idx] -= GOLDEN_INFER_NON_GREEN_PENALTY

    best_idx = int(np.argmax(lane_scores)) if lane_scores else 0
    best_score = lane_scores[best_idx] if lane_scores else 0.0
    best_green_count = lane_green_counts[best_idx] if lane_green_counts else 0
    other_scores = [s for i, s in enumerate(lane_scores) if i != best_idx]
    next_score = max(other_scores) if other_scores else 0.0
    dominance_ok = best_score >= max(GOLDEN_INFER_MIN_SCORE, next_score * GOLDEN_INFER_DOMINANCE_RATIO)

    if best_green_count < GOLDEN_INFER_MIN_GREEN_COUNT or not dominance_ok:
        return inactive

    confidence = float(np.clip(best_score / max(GOLDEN_INFER_MIN_SCORE, 1.0), 0.0, 1.0))
    return {
        'active': True,
        'lane': best_idx + 1,
        'remaining_s': None,
        'source': 'tokens',
        'confidence': confidence,
        'lane_scores': [float(s) for s in lane_scores],
        'green_counts': lane_green_counts,
    }


def detect_lane_curve(frame):
    """CL0-CL2 entry point + lightweight curve-bias readout (CL3-lite).

    The bird's-eye warp / mask / sliding-window outputs remain visualization
    fodder (draw_lane_curve_debug); the controller now also consumes the
    'curve_bias' scalar to anticipate bends (see LANE_CURVE_GAIN).

    Returns dict with keys (or None if frame is None):
        'warp'            : warped BGR (IPM_DST_H x IPM_DST_W x 3)
        'mask'            : binary lane-pixel mask (same HxW as warp, uint8 0/255)
        'left_pts'        : (xs, ys) ndarray pair of left-rail pixels in warp space
        'right_pts'       : (xs, ys) ndarray pair of right-rail pixels in warp space
        'left_rects'      : list of (x_lo, y_lo, x_hi, y_hi) windows (for overlay)
        'right_rects'     : same for right side
        'left_base'       : seed x or None
        'right_base'      : seed x or None
        'curve_bias'      : -1..+1 anticipated bend (positive => bends right), 0.0 if unknown
        'lane_center_norm': -1..+1 near lane-centre offset, or None if unknown
        'lane_grid'       : estimate_lane_grid() dict (lanes 1..N labelled), or None
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
    curve_bias, lane_center_norm = _curve_bias_from_pixels(lx, ly, rx, ry)
    lane_grid = estimate_lane_grid(mask)
    if lane_grid is not None:
        curve_penalty = min(0.45, abs(curve_bias) * 0.45)
        lane_grid['edge_confidence'] = float(lane_grid.get('confidence', 1.0))
        lane_grid['curve_penalty'] = float(curve_penalty)
        lane_grid['confidence'] = float(np.clip(
            lane_grid.get('confidence', 1.0) - curve_penalty,
            0.0,
            1.0,
        ))
    return {
        'warp': warp,
        'mask': mask,
        'left_pts':  (lx, ly),
        'right_pts': (rx, ry),
        'left_rects':  lrects,
        'right_rects': rrects,
        'left_base':  left_base,
        'right_base': right_base,
        'curve_bias': curve_bias,
        'lane_center_norm': lane_center_norm,
        'lane_grid': lane_grid,
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


# Running baseline of the scene's bright-level (90th-pct V), adapted on bright
# frames only so a sustained dark event can't drag it down.
_brightness_baseline = {'p90': None}


def scene_brightness_p90(frame):
    """90th-percentile V of a centre crop (the scene's 'bright level'). Used for
    both the low-light decision and the HUD readout."""
    if frame is None:
        return 255.0
    h, w = frame.shape[:2]
    crop = frame[h // 4: h * 3 // 4, w // 4: w * 3 // 4]
    v = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(np.percentile(v, 90))


def detect_low_brightness(frame):
    """True if the scene is dim (poster's 'low brightness' event).
    Uses mean V on a centre crop so HUD overlays don't bias the result."""
    if frame is None:
        return False

    h, w = frame.shape[:2]
    crop = frame[h // 4: h * 3 // 4, w // 4: w * 3 // 4]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_v = float(hsv[:, :, 2].mean())

    is_low = mean_v < LOW_BRIGHTNESS_THRESHOLD

    print(
        f"[LOW_BRIGHTNESS] Mean V={mean_v:.1f} "
        f"Threshold={LOW_BRIGHTNESS_THRESHOLD} "
        f"Detected={is_low}"
    )

    return is_low


# ---------------------------------------------------------------------------
# Golden Lane banner detection (Phase 17)
# ---------------------------------------------------------------------------
# When a Golden Lane event is active the patched game shows a thin orange/yellow
# strip at the very top of the front-camera feed with dark text:
# "LANE N - ALL GREEN! (Xs)".
HSV_GOLDEN_BANNER = (np.array([8, 80, 120]), np.array([35, 255, 255]))
GOLDEN_BANNER_Y0_FRAC = 0.00      # patched banner touches the top edge
GOLDEN_BANNER_Y1_FRAC = 0.08
GOLDEN_BANNER_X0_FRAC = 0.00
GOLDEN_BANNER_X1_FRAC = 1.00
GOLDEN_BANNER_MIN_PIX = 120       # orange/yellow strip pixels needed to call active
GOLDEN_TEXT_MIN_PIX = 8           # dark text pixels needed before digit OCR
GOLDEN_DIGIT_MIN_SCORE = 0.52     # pixel-font bitmap match score
GOLDEN_DIGIT_MARGIN = 0.03        # best match must beat 2nd-best by this
GOLDEN_BANNER_UPSCALE = 4         # enlarge the banner mask before OCR
# Expected horizontal positions of the lane digit and the countdown digit, as a
# fraction of the dark text's bounding-box width. The message format is fixed
# ("LANE N - ALL GREEN! (Xs)") so these positions are stable.
GOLDEN_LANE_DIGIT_XFRAC = 0.27
GOLDEN_COUNT_DIGIT_XFRAC = 0.89
GOLDEN_DIGIT_TEMPLATE_SIZE = (5, 7)

# Golden Lane detection log. Appends a line whenever the banner state changes
# (active<->inactive) or the read lane/countdown changes, so a run can be
# reviewed afterwards. Set GOLDEN_LOG_ENABLED = False to disable.
GOLDEN_LOG_ENABLED = True
GOLDEN_LOG_PATH = os.path.join('logs', 'golden_lane.log')
_golden_log_state = {'last_key': None, 'init': False}

GOLDEN_DEBUG_ENABLED = True
GOLDEN_DEBUG_INTERVAL_S = 0.5
GOLDEN_DEBUG_LOG_PATH = os.path.join('logs', 'golden_lane_debug.log')
GOLDEN_DEBUG_ROI_PATH = os.path.join('logs', 'golden_lane_roi.png')
GOLDEN_DEBUG_MASK_PATH = os.path.join('logs', 'golden_lane_mask.png')
_golden_debug_state = {'last_sample_at': 0.0, 'init': False}

_golden_digit_templates = None


def _golden_log(active, lane, remaining_s, banner_pix):
    """Append a Golden Lane detection line to GOLDEN_LOG_PATH on state change.

    Only writes when (active, lane, remaining_s) differs from the last logged
    tuple, so the file stays a readable event history instead of a per-frame
    dump. Failures are swallowed so logging never breaks perception."""
    if not GOLDEN_LOG_ENABLED:
        return
    key = (active, lane, remaining_s)
    if key == _golden_log_state['last_key']:
        return
    _golden_log_state['last_key'] = key
    try:
        os.makedirs(os.path.dirname(GOLDEN_LOG_PATH), exist_ok=True)
        ts = time.strftime('%H:%M:%S')
        lane_str = str(lane) if lane is not None else '?'
        rem_str = f"{remaining_s:.0f}s" if remaining_s is not None else '?'
        mode = 'w' if not _golden_log_state['init'] else 'a'
        with open(GOLDEN_LOG_PATH, mode, encoding='utf-8') as fh:
            if not _golden_log_state['init']:
                fh.write(f"# Golden Lane detection log (session started {ts})\n")
                fh.write("# time  active  lane  remaining  banner_pixels\n")
                _golden_log_state['init'] = True
            fh.write(
                f"{ts}  active={int(bool(active))}  lane={lane_str}  "
                f"remaining={rem_str}  banner_pix={banner_pix}\n"
            )
    except Exception:
        pass


def _golden_debug_sample(active, lane, remaining_s, banner_pix, roi, mask):
    """Rate-limited Golden Lane diagnostics for ROI/HSV threshold tuning."""
    if not GOLDEN_DEBUG_ENABLED:
        return

    now = time.monotonic()
    if now - _golden_debug_state['last_sample_at'] < GOLDEN_DEBUG_INTERVAL_S:
        return
    _golden_debug_state['last_sample_at'] = now

    try:
        os.makedirs(os.path.dirname(GOLDEN_DEBUG_LOG_PATH), exist_ok=True)

        # Save the exact ROI and binary mask currently used by detection.
        cv2.imwrite(
            GOLDEN_DEBUG_ROI_PATH,
            cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
        )
        cv2.imwrite(
            GOLDEN_DEBUG_MASK_PATH,
            cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST),
        )

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        colourful = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 100)
        colour_pix = int(np.count_nonzero(colourful))
        if colour_pix:
            hue_values = hsv[:, :, 0][colourful]
            hue_min = int(hue_values.min())
            hue_mean = float(hue_values.mean())
            hue_max = int(hue_values.max())
        else:
            hue_min, hue_mean, hue_max = None, None, None

        ts = time.strftime('%H:%M:%S')
        lane_str = str(lane) if lane is not None else '?'
        rem_str = f"{remaining_s:.0f}s" if remaining_s is not None else '?'
        hue_str = (
            f"{hue_min}/{hue_mean:.1f}/{hue_max}"
            if hue_mean is not None else "?/?/?"
        )
        mode = 'w' if not _golden_debug_state['init'] else 'a'
        with open(GOLDEN_DEBUG_LOG_PATH, mode, encoding='utf-8') as fh:
            if not _golden_debug_state['init']:
                fh.write(f"# Golden Lane debug log (session started {ts})\n")
                fh.write(
                    "# time  active  lane  remaining  banner_pixels  "
                    "colour_pixels  hue_min/mean/max  roi_wh\n"
                )
                _golden_debug_state['init'] = True
            fh.write(
                f"{ts}  active={int(bool(active))}  lane={lane_str}  "
                f"remaining={rem_str}  banner_pix={banner_pix}  "
                f"colour_pix={colour_pix}  hue={hue_str}  "
                f"roi={roi.shape[1]}x{roi.shape[0]}\n"
            )
    except Exception:
        pass


def _get_golden_digit_templates():
    """Binary 5x7 pixel-font templates for Golden Lane digits."""
    global _golden_digit_templates
    if _golden_digit_templates is None:
        patterns = {
            0: [
                "01110",
                "10001",
                "10011",
                "10101",
                "11001",
                "10001",
                "01110",
            ],
            1: [
                "00100",
                "01100",
                "00100",
                "00100",
                "00100",
                "00100",
                "01110",
            ],
            2: [
                "01110",
                "10001",
                "00001",
                "00010",
                "00100",
                "01000",
                "11111",
            ],
            3: [
                "11110",
                "00001",
                "00001",
                "01110",
                "00001",
                "00001",
                "11110",
            ],
            4: [
                "00010",
                "00110",
                "01010",
                "10010",
                "11111",
                "00010",
                "00010",
            ],
            5: [
                "11111",
                "10000",
                "10000",
                "11110",
                "00001",
                "00001",
                "11110",
            ],
            6: [
                "01110",
                "10000",
                "10000",
                "11110",
                "10001",
                "10001",
                "01110",
            ],
            7: [
                "11111",
                "00001",
                "00010",
                "00100",
                "01000",
                "01000",
                "01000",
            ],
            8: [
                "01110",
                "10001",
                "10001",
                "01110",
                "10001",
                "10001",
                "01110",
            ],
            9: [
                "01110",
                "10001",
                "10001",
                "01111",
                "00001",
                "00001",
                "01110",
            ],
        }
        templates = {}
        for d, rows in patterns.items():
            templates[d] = np.array(
                [[255 if ch == "1" else 0 for ch in row] for row in rows],
                dtype=np.uint8,
            )
        _golden_digit_templates = templates
    return _golden_digit_templates


def _match_golden_digit(glyph_bin):
    """Return (best_digit, score, margin) for a binary glyph crop.

    Uses IoU between the candidate and 5x7 pixel-font digit templates. margin is
    best_score - 2nd_score.
    """
    if glyph_bin is None or glyph_bin.size == 0:
        return None, 0.0, 0.0
    cand = cv2.resize(glyph_bin, GOLDEN_DIGIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    cand = (cand > 0).astype(np.float32)
    cand_sum = float(cand.sum())
    scores = []
    for d, tmpl in _get_golden_digit_templates().items():
        t = (tmpl > 0).astype(np.float32)
        intersection = float((cand * t).sum())
        union = cand_sum + float(t.sum()) - intersection + 1e-6
        scores.append((intersection / union, d))
    scores.sort(reverse=True)
    best_score, best_d = scores[0]
    margin = best_score - scores[1][0] if len(scores) > 1 else best_score
    return best_d, best_score, margin


def _golden_glyph_near(mask, x_target):
    """Binary crop of the orange glyph whose centre x is nearest x_target, or
    None. Tiny noise blobs (short height) are ignored."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = mask.shape[0]
    best, best_dx = None, 1e9
    for c in contours:
        x, y, w, hh = cv2.boundingRect(c)
        if hh < max(3, h * 0.06):
            continue
        dx = abs((x + w / 2.0) - x_target)
        if dx < best_dx:
            best, best_dx = (x, y, w, hh), dx
    if best is None:
        return None
    x, y, w, hh = best
    return mask[y:y + hh, x:x + w]


def _read_golden_digit(mask, x_target, allowed=None):
    """Confident digit at x_target in the banner mask, or None."""
    glyph = _golden_glyph_near(mask, x_target)
    if glyph is None:
        return None
    d, score, margin = _match_golden_digit(glyph)
    if (
        d is not None
        and (allowed is None or d in allowed)
        and score >= GOLDEN_DIGIT_MIN_SCORE
        and margin >= GOLDEN_DIGIT_MARGIN
    ):
        return int(d)
    return None


def _golden_text_mask(roi, banner_mask):
    """Dark text pixels inside the top orange/yellow Golden Lane strip."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = cv2.inRange(gray, 0, 95)
    if banner_mask is not None:
        text = cv2.bitwise_and(text, cv2.dilate(banner_mask, np.ones((3, 3), np.uint8)))
    text = cv2.morphologyEx(text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return text


def detect_golden_lane(frame):
    """Detect the Golden Lane banner and read its lane number + countdown.

    Returns dict:
        'active'      : True when the orange "LANE N - ALL GREEN!" banner strip shows
        'lane'        : golden lane number (1..N_LANES) when read confidently, else None
        'remaining_s' : countdown seconds read from "(Xs)" when confident, else None

    The patched game draws dark text on an orange/yellow top strip. The lane /
    countdown digits are best-effort OCR. When 'active' is True but 'lane' is
    None the caller should fall back to green-token lane inference.
    """
    inactive = {'active': False, 'lane': None, 'remaining_s': None}
    if frame is None:
        return inactive

    small = cv2.resize(frame, (PROC_W, PROC_H))
    y0 = int(PROC_H * GOLDEN_BANNER_Y0_FRAC)
    y1 = int(PROC_H * GOLDEN_BANNER_Y1_FRAC)
    x0 = int(PROC_W * GOLDEN_BANNER_X0_FRAC)
    x1 = int(PROC_W * GOLDEN_BANNER_X1_FRAC)
    roi = small[y0:y1, x0:x1]
    if roi.size == 0:
        return inactive

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *HSV_GOLDEN_BANNER)
    banner_pix = int(cv2.countNonZero(mask))
    if banner_pix < GOLDEN_BANNER_MIN_PIX:
        _golden_debug_sample(False, None, None, banner_pix, roi, mask)
        _golden_log(False, None, None, banner_pix)
        return inactive

    # Banner is active. Read the dark text printed on the orange/yellow strip.
    text_mask = _golden_text_mask(roi, mask)
    if cv2.countNonZero(text_mask) < GOLDEN_TEXT_MIN_PIX:
        _golden_debug_sample(True, None, None, banner_pix, roi, text_mask)
        _golden_log(True, None, None, banner_pix)
        return {'active': True, 'lane': None, 'remaining_s': None}

    up = cv2.resize(text_mask, None, fx=GOLDEN_BANNER_UPSCALE, fy=GOLDEN_BANNER_UPSCALE,
                    interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(up)
    lane, remaining_s = None, None
    if xs.size:
        tx0, tx1 = int(xs.min()), int(xs.max())
        tw = max(1, tx1 - tx0)
        lane = _read_golden_digit(
            up,
            tx0 + GOLDEN_LANE_DIGIT_XFRAC * tw,
            allowed=set(range(1, N_LANES + 1)),
        )
        countdown = _read_golden_digit(up, tx0 + GOLDEN_COUNT_DIGIT_XFRAC * tw)
        if countdown is not None:
            remaining_s = float(countdown)
    _golden_debug_sample(True, lane, remaining_s, banner_pix, roi, mask)
    _golden_log(True, lane, remaining_s, banner_pix)
    return {'active': True, 'lane': lane, 'remaining_s': remaining_s}


# ---------------------------------------------------------------------------
# Hill / slope detection (heuristic)
# ---------------------------------------------------------------------------
# Self-calibrating baseline of the "flat-road" horizon row. Updated only on
# (near-)flat frames so a sustained hill can't drag the baseline toward itself.
_slope_state = {'baseline': None}


def detect_slope(frame):
    """Heuristic hill / pitch estimate from how high the drivable asphalt reaches.

    On flat road the asphalt narrows to the horizon at a roughly constant frame
    row. A crest (uphill) cuts the view short so the road top sits higher in the
    frame; a dip lets the road reach lower. We walk up the center column band
    from the bottom to find the road's top edge, track an EMA of the flat
    baseline, and flag a hill when the current horizon deviates from it.

    Returns dict {'horizon_frac','deviation','is_hill','uphill'} or None if no
    road is visible / frame is None.
        horizon_frac : road-top row / PROC_H  (0=top .. 1=bottom)
        deviation    : baseline - horizon_frac (positive => road ends higher => uphill crest)
        is_hill      : |deviation| > SLOPE_HILL_DEV
        uphill       : deviation > 0

    NOTE: single-frame heuristic; thresholds (SLOPE_HILL_DEV, ASPHALT_* ,
    ROAD_ROW_THRESH) are first-pass and should be tuned against live runs.
    """
    if frame is None:
        return None
    small = cv2.resize(frame, (PROC_W, PROC_H))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    asphalt = (s < ASPHALT_MAX_SAT) & (v > ASPHALT_MIN_VAL) & (v < ASPHALT_MAX_VAL)
    x0 = int(PROC_W * SLOPE_CENTER_BAND[0])
    x1 = int(PROC_W * SLOPE_CENTER_BAND[1])
    row_frac = asphalt[:, x0:x1].mean(axis=1)          # fraction of asphalt per row, top->bottom

    # Walk up from the bottom while rows keep reading as road, tolerating short
    # gaps (lane dashes / bright markings break the asphalt run for a few rows).
    road_top = PROC_H - 1
    found = False
    misses = 0
    for r in range(PROC_H - 1, -1, -1):
        if row_frac[r] > ROAD_ROW_THRESH:
            road_top = r
            found = True
            misses = 0
        elif found:
            misses += 1
            if misses > SLOPE_GAP_TOL:
                break
    if not found:
        return None

    horizon_frac = road_top / float(PROC_H)
    base = _slope_state['baseline']
    if base is None:
        base = horizon_frac
    deviation = base - horizon_frac
    is_hill = abs(deviation) > SLOPE_HILL_DEV
    if not is_hill:                                    # adapt baseline on flat frames only
        base = (1.0 - SLOPE_EMA_ALPHA) * base + SLOPE_EMA_ALPHA * horizon_frac
    _slope_state['baseline'] = base
    return {
        'horizon_frac': float(horizon_frac),
        'deviation': float(deviation),
        'is_hill': bool(is_hill),
        'uphill': bool(deviation > 0),
    }


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
    """Front + rear perception overlay. Rear panel shows V2.0 threats (police /
    chasing car). rear_per may be None."""
    front_img = front_per['frame'].copy() if front_per else np.zeros((PROC_H, PROC_W, 3), np.uint8)
    rear_img = rear_per['frame'].copy() if rear_per else np.zeros((PROC_H, PROC_W, 3), np.uint8)

    if front_per:
        y0 = front_per['roi_y0']
        x0 = front_per.get('roi_x0', 0)
        cv2.rectangle(front_img, (x0, y0), (PROC_W - 1 - x0, PROC_H - 1), (80, 80, 80), 1)
        _draw_obj(front_img, front_per['red'],    "RED",    (0, 0, 255), y0)
        _draw_obj(front_img, front_per['green'],  "GREEN",  (0, 255, 0), y0)
        _draw_obj(front_img, front_per['yellow'], "YELLOW", (0, 255, 255), y0)
        _draw_obj(front_img, front_per.get('police'), "POLICE", (255, 0, 0), y0)  # Ch.3 (front)
        # Highlight the NEAREST orb (imminence target) with a white ring.
        near = front_per.get('nearest')
        if near is not None:
            ncx, ncy, nr = near['circle']
            cv2.circle(front_img, (ncx, ncy + y0), nr + 4, (255, 255, 255), 1)
        if lane_offset is not None:
            cx = int(PROC_W / 2 + lane_offset * PROC_W / 2)
            cv2.line(front_img, (PROC_W // 2, PROC_H - 5), (cx, PROC_H - 25), (255, 255, 255), 2)

    if rear_per:
        _draw_obj(rear_img, rear_per['other_car']['info'], "CAR", (0, 255, 255))  # teal chasing car (Ch.2)

    # HUD strip — front | rear panels
    hud_h = 60
    canvas = np.zeros((PROC_H + hud_h, PROC_W * 2 + 10, 3), np.uint8)
    canvas[:PROC_H, :PROC_W] = front_img
    canvas[:PROC_H, PROC_W + 10:] = rear_img
    cv2.putText(canvas, "FRONT", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "REAR",  (PROC_W + 15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas,
                f"target={hud['target']:.0f} eff={hud['eff']:.0f} police={int(hud.get('police', 0))} "
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
    # Keep only saturated, BRIGHT pixels. V floor raised 60 -> 110 so dark grass /
    # trees (green-hued but V~65) are NOT sampled as a colour — orbs are bright
    # (V>180). This was the root cause of grass being learned into the green range.
    mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 110)
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
