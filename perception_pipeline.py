"""Perception orchestration for the SpeedTrials2D controller.

Raw image-processing algorithms remain in image_detection.py. This module owns
the order in which they are called and the compact data bundle consumed by the
driving policy and display layer.
"""

from image_detection import (
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
from perception_types import PerceptionResult


def run_perception(front_frame, back_frame) -> PerceptionResult:
    """Run one lock-free perception cycle.

    Returns:
        (front_per, rear_per, lane_offset, curve_dbg, curve_bias, low_light, hill)
    """
    if front_frame is None:
        return None, None, None, None, 0.0, False, False

    if not calibration_done():
        calibrate_step(front_frame)

    front_per = detect_front_objects(front_frame)
    lane_offset = detect_lane_offset(front_frame)
    low_light = detect_low_brightness(front_frame)
    rear_per = detect_rear(back_frame) if back_frame is not None else None

    curve_dbg = detect_lane_curve(front_frame)
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

    slope = detect_slope(front_frame)
    hill = bool(slope and slope['is_hill'])

    return front_per, rear_per, lane_offset, curve_dbg, curve_bias, low_light, hill
