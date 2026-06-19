"""Perception orchestration for the SpeedTrials2D controller.

Raw image-processing algorithms remain in image_detection.py. This module owns
the order in which they are called and the compact data bundle consumed by the
driving policy and display layer.
"""

import os
import time

from image_detection import (
    boost_low_light,
    calibrate_step,
    calibration_done,
    detect_front_objects,
    detect_golden_lane,
    detect_lane_curve,
    detect_lane_offset,
    detect_low_brightness,
    detect_rear,
    detect_slope,
    infer_golden_lane_from_tokens,
)
from object_tracking import track_front, track_rear
from perception_types import PerceptionResult


LANE_GRID_LOG_ENABLED = True
LANE_GRID_LOG_INTERVAL_S = 0.5
LANE_GRID_LOG_PATH = os.path.join('logs', 'lane_grid.log')
_lane_grid_log_state = {'last_at': 0.0, 'init': False}


def _log_lane_grid(lane_grid, golden, curve_bias):
    """Rate-limited lane-numbering log for Golden Lane validation."""
    if not LANE_GRID_LOG_ENABLED:
        return

    now = time.monotonic()
    if now - _lane_grid_log_state['last_at'] < LANE_GRID_LOG_INTERVAL_S:
        return
    _lane_grid_log_state['last_at'] = now

    try:
        os.makedirs(os.path.dirname(LANE_GRID_LOG_PATH), exist_ok=True)
        ts = time.strftime('%H:%M:%S')
        mode = 'w' if not _lane_grid_log_state['init'] else 'a'
        with open(LANE_GRID_LOG_PATH, mode, encoding='utf-8') as fh:
            if not _lane_grid_log_state['init']:
                fh.write(f"# Lane grid log (session started {ts})\n")
                fh.write(
                    "# time  grid  car_lane  road_left/right  centers_norm  "
                    "confidence  measured_left/right  held  golden_active  "
                    "golden_lane  golden_source  curve_bias\n"
                )
                _lane_grid_log_state['init'] = True

            if lane_grid:
                centers = ','.join(f"{c:+.2f}" for c in lane_grid.get('lane_centers_norm', []))
                road = f"{lane_grid.get('road_left', '?')}/{lane_grid.get('road_right', '?')}"
                measured = f"{lane_grid.get('measured_left', '?')}/{lane_grid.get('measured_right', '?')}"
                car_lane = lane_grid.get('car_lane')
                confidence = lane_grid.get('confidence', 0.0)
                held = int(bool(lane_grid.get('held_previous')))
                grid_on = 1
            else:
                centers = '?'
                road = '?/?'
                measured = '?/?'
                car_lane = '?'
                confidence = 0.0
                held = 0
                grid_on = 0

            golden = golden or {}
            fh.write(
                f"{ts}  grid={grid_on}  car_lane={car_lane}  road={road}  "
                f"centers=[{centers}]  conf={confidence:.2f}  measured={measured}  "
                f"held={held}  golden_active={int(bool(golden.get('active')))}  "
                f"golden_lane={golden.get('lane') or '?'}  "
                f"golden_source={golden.get('source') or '?'}  "
                f"curve_bias={curve_bias:+.2f}\n"
            )
    except Exception:
        pass


def run_perception(front_frame, back_frame) -> PerceptionResult:
    """Run one lock-free perception cycle.

    Returns:
        (front_per, rear_per, lane_offset, curve_dbg, curve_bias, low_light, hill)
    """
    if front_frame is None:
        return None, None, None, None, 0.0, False, False

    if not calibration_done():
        calibrate_step(front_frame)

    # Detect the low-light event on the ORIGINAL frame, then brighten the frame
    # used for detection so HSV/lane detection survives the dark (Challenge 1;
    # toggle in image_detection.LOW_LIGHT_BOOST_ENABLED). No-op in normal light.
    low_light = detect_low_brightness(front_frame)
    detect_frame = boost_low_light(front_frame, low_light)

    front_per = detect_front_objects(detect_frame)
    lane_offset = detect_lane_offset(detect_frame)
    rear_per = detect_rear(back_frame) if back_frame is not None else None

    # Multi-object tracking (toggle in object_tracking.TRACKING_ENABLED). Purely
    # additive: enriches the orb / chasing-car dicts with stable track ids and
    # velocity; a no-op when disabled.
    front_per = track_front(front_per)
    rear_per = track_rear(rear_per)

    curve_dbg = detect_lane_curve(detect_frame)
    curve_bias = curve_dbg['curve_bias'] if curve_dbg else 0.0

    # Phase 17: Golden Lane. The banner is drawn on the main game view, not the
    # camera feed, so the camera fallback infers the lane from green-token
    # concentration when text detection is inactive or cannot read the lane.
    if front_per is not None:
        lane_grid = curve_dbg.get('lane_grid') if curve_dbg else None
        banner_golden = detect_golden_lane(front_frame)
        token_golden = infer_golden_lane_from_tokens(front_per, lane_grid)
        front_per['golden'] = (
            banner_golden
            if banner_golden.get('active') and banner_golden.get('lane')
            else token_golden
        )
        front_per['lane_grid'] = lane_grid
        _log_lane_grid(lane_grid, front_per['golden'], curve_bias)

    slope = detect_slope(front_frame)
    hill = bool(slope and slope['is_hill'])

    return front_per, rear_per, lane_offset, curve_dbg, curve_bias, low_light, hill
