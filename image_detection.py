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
                                                  consumed by the steering controller)
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
GREEN_LANE_CHANGE_BAND = 0.12            # |centroid_x_norm| above this => green is in another lane -> commit
GREEN_SEEK_GAIN = 0.9                    # committed steer magnitude toward an off-lane green
GREEN_SEEK_HOLD_S = 0.5                  # bridge frames where green flickers / leaves ROI mid-cross
# (Keep-LEFT / keep-RIGHT home-lane bias removed — the car now centres in the
#  lane and reacts to orbs. See plan.md Phase 5 to restore a home-lane hug.)
RED_AVOID_GAIN = 1.0                     # full-lock swerve when red is in path
YELLOW_AVOID_GAIN = 0.9
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
CHASING_MIN_AREA_FRAC = 0.012        # below this the car is too far away to matter
CHASING_CENTER_BAND_FRAC = 0.75      # ignore teal blobs on the far shoulder
CHASING_GROW_DELTA = 0.006           # total area_frac increase across smoothing window
CHASING_HIST_LEN = 4                 # frames of area history used for smoothed growth
CHASING_MIN_GROW_FRAMES = 3          # minimum history frames before "growing" can fire

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
                       min_solidity=ORB_MIN_SOLIDITY):
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
                                 roi_area, roi_x0, roi_y0, **gate)
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
        _chasing_area_hist.append(other['area_frac'])
        if len(_chasing_area_hist) > CHASING_HIST_LEN:
            _chasing_area_hist.pop(0)
        if len(_chasing_area_hist) >= CHASING_MIN_GROW_FRAMES:
            growing = (_chasing_area_hist[-1] - _chasing_area_hist[0]) > CHASING_GROW_DELTA
    else:
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
