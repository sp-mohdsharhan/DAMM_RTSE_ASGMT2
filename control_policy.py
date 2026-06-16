"""Driving policy for the SpeedTrials2D autonomous controller.

This module owns steering/throttle decisions and short-lived manoeuvre latches.
It deliberately does not touch sockets, locks, OpenCV windows, or shared_data.
"""

import numpy as np

from image_detection import (
    CENTER_BAND_FRAC,
    GREEN_ATTRACT_GAIN,
    GREEN_ATTRACT_MIN_AREA,
    GREEN_LANE_CHANGE_BAND,
    GREEN_SEEK_GAIN,
    GREEN_SEEK_HOLD_S,
    HILL_AREA_SCALE,
    LANE_CHANGE_DURATION_S,
    LANE_CHANGE_STEER,
    LANE_CURVE_GAIN,
    ORB_ACT_DISTANCE,
    ORB_TIE_MARGIN,
    POLICE_DODGE_AREA,
    POLICE_DODGE_BAND,
    RED_AVOID_AREA_FRAC,
    RED_AVOID_BAND_FRAC,
    RED_AVOID_GAIN,
    RED_LANE_CHANGE_DURATION_S,
    RED_SETTLE_DURATION_S,
    YELLOW_AVOID_AREA_FRAC,
    YELLOW_AVOID_GAIN,
)
from perception_types import FrontPerception, RearPerception


CRUISE_THROTTLE = 0.8
LOW_BRIGHTNESS_THROTTLE = -1.0

# Red avoidance latch: once a red is detected ahead, commit to a full
# lane-change away from it, then counter-steer briefly to settle.
_red_avoid = {'until': 0.0, 'settle_until': 0.0, 'dir': 0}

# Green pursuit latch: bridge frames where a green flickers / leaves ROI
# mid-crossing so the lane change still completes.
_green_seek = {'until': 0.0, 'dir': 0}

# Chasing-car forced-swerve latch. Direction alternates each trigger.
_lane_change_until = 0.0
_lane_change_dir = 1


def _compute_steering(
    front_per: FrontPerception | None,
    curve_bias: float,
    hill: bool,
    now: float,
    force_lane_change: bool = False,
    swerve_dir: int = 1,
) -> float:
    """Return steering in -1..+1.

    Priority: front police, rear chasing-car swerve, imminent nearest orb,
    fallback colour priority. The game owns score/speed/event state; this
    function reacts only to current perception and local manoeuvre latches.
    """
    red = front_per['red'] if front_per else None
    green = front_per['green'] if front_per else None
    yellow = front_per['yellow'] if front_per else None
    orbs = front_per.get('orbs', []) if front_per else []
    nearest = front_per.get('nearest') if front_per else None
    police = front_per.get('police') if front_per else None

    # Police ahead: dodge if close/centred, otherwise grab a red token to escape.
    if police is not None:
        pcx = police['centroid_x_norm']
        if police['area_frac'] > POLICE_DODGE_AREA and abs(pcx) < POLICE_DODGE_BAND:
            return float(RED_AVOID_GAIN * (-1 if pcx >= 0 else 1))

        reds_in_view = [o for o in orbs if o['color'] == 'red']
        target_red = None
        if reds_in_view:
            police_side = 1 if pcx >= 0 else -1
            opposite = [
                o for o in reds_in_view
                if (1 if o['centroid_x_norm'] >= 0 else -1) != police_side
            ]
            target_red = (
                min(opposite, key=lambda o: o['distance'])
                if opposite else min(reds_in_view, key=lambda o: o['distance'])
            )
        if target_red is not None:
            rcx = target_red['centroid_x_norm']
            if abs(rcx) > GREEN_LANE_CHANGE_BAND:
                return float(np.clip(GREEN_SEEK_GAIN * (1 if rcx > 0 else -1), -1.0, 1.0))
            return float(np.clip(GREEN_ATTRACT_GAIN * rcx, -1.0, 1.0))

        return float(0.5 * (-1 if pcx >= 0 else 1))

    if force_lane_change:
        return float(LANE_CHANGE_STEER * swerve_dir)

    red_area_thr = RED_AVOID_AREA_FRAC * (HILL_AREA_SCALE if hill else 1.0)
    yellow_area_thr = YELLOW_AVOID_AREA_FRAC * (HILL_AREA_SCALE if hill else 1.0)

    # Imminence: nearest on-road orb wins if it is close enough.
    if nearest is not None and nearest['distance'] < ORB_ACT_DISTANCE:
        hazards = [
            o for o in orbs
            if o['color'] in ('red', 'yellow')
            and o['distance'] <= nearest['distance'] + ORB_TIE_MARGIN
        ]
        target = min(hazards, key=lambda o: o['distance']) if hazards else nearest
        cx = target['centroid_x_norm']
        if target['color'] == 'red':
            direction = -1 if cx >= 0 else 1
            _red_avoid['until'] = now + RED_LANE_CHANGE_DURATION_S
            _red_avoid['settle_until'] = _red_avoid['until'] + RED_SETTLE_DURATION_S
            _red_avoid['dir'] = direction
            return float(RED_AVOID_GAIN * direction)
        if target['color'] == 'yellow':
            return float(np.clip(-YELLOW_AVOID_GAIN * np.sign(cx or 1.0), -1, 1))
        if abs(cx) > GREEN_LANE_CHANGE_BAND:
            _green_seek['dir'] = 1 if cx > 0 else -1
            _green_seek['until'] = now + GREEN_SEEK_HOLD_S
            return float(np.clip(
                GREEN_SEEK_GAIN * _green_seek['dir'] + LANE_CURVE_GAIN * curve_bias,
                -1.0,
                1.0,
            ))
        _green_seek['until'] = 0.0
        return float(np.clip(
            GREEN_ATTRACT_GAIN * cx + LANE_CURVE_GAIN * curve_bias,
            -1.0,
            1.0,
        ))

    # Fallback colour priority: green > red > yellow > straight.
    if green is not None and green['area_frac'] > GREEN_ATTRACT_MIN_AREA:
        cx = green['centroid_x_norm']
        if abs(cx) > GREEN_LANE_CHANGE_BAND:
            _green_seek['dir'] = 1 if cx > 0 else -1
            _green_seek['until'] = now + GREEN_SEEK_HOLD_S
            return float(np.clip(
                GREEN_SEEK_GAIN * _green_seek['dir'] + LANE_CURVE_GAIN * curve_bias,
                -1.0,
                1.0,
            ))
        _green_seek['until'] = 0.0
        return float(np.clip(
            GREEN_ATTRACT_GAIN * cx + LANE_CURVE_GAIN * curve_bias,
            -1.0,
            1.0,
        ))

    if now < _green_seek['until']:
        return float(GREEN_SEEK_GAIN * _green_seek['dir'])

    if (red is not None and red['area_frac'] > red_area_thr
            and abs(red['centroid_x_norm']) < RED_AVOID_BAND_FRAC):
        direction = -1 if red['centroid_x_norm'] >= 0 else 1
        _red_avoid['until'] = now + RED_LANE_CHANGE_DURATION_S
        _red_avoid['settle_until'] = _red_avoid['until'] + RED_SETTLE_DURATION_S
        _red_avoid['dir'] = direction
        return float(RED_AVOID_GAIN * direction)

    if now < _red_avoid['until']:
        if red is not None and red['area_frac'] > red_area_thr:
            red_side = 1 if red['centroid_x_norm'] >= 0 else -1
            if red_side == _red_avoid['dir']:
                _red_avoid['dir'] = -_red_avoid['dir']
                _red_avoid['until'] = now + RED_LANE_CHANGE_DURATION_S
                _red_avoid['settle_until'] = _red_avoid['until'] + RED_SETTLE_DURATION_S
        return float(RED_AVOID_GAIN * _red_avoid['dir'])

    if now < _red_avoid['settle_until']:
        return float(-0.5 * _red_avoid['dir'])

    if (yellow is not None and yellow['area_frac'] > yellow_area_thr
            and abs(yellow['centroid_x_norm']) < CENTER_BAND_FRAC):
        return float(np.clip(
            -YELLOW_AVOID_GAIN * np.sign(yellow['centroid_x_norm'] or 1.0),
            -1,
            1,
        ))

    return 0.0


def compute_control(
    front_per: FrontPerception | None,
    rear_per: RearPerception | None,
    curve_bias: float,
    hill: bool,
    low_light: bool,
    now: float,
) -> tuple[float, float, list[str]]:
    """Return (steering, accel, events_visible) for the current perception frame."""
    global _lane_change_until, _lane_change_dir

    if rear_per is not None and rear_per['other_car']['growing'] and now >= _lane_change_until:
        _lane_change_until = now + LANE_CHANGE_DURATION_S
        _lane_change_dir = -_lane_change_dir

    force_lc = now < _lane_change_until
    police_seen = bool(front_per and front_per.get('police'))

    if low_light:
        steering = 0.0
        accel = LOW_BRIGHTNESS_THROTTLE
    else:
        steering = _compute_steering(
            front_per,
            curve_bias,
            hill,
            now,
            force_lc,
            _lane_change_dir,
        )
        accel = CRUISE_THROTTLE

    events_visible = []
    if police_seen:
        events_visible.append('POLICE->GRAB_RED')
    if force_lc:
        events_visible.append('CHASING_CAR')
    if hill:
        events_visible.append('HILL')
    near = front_per.get('nearest') if front_per else None
    if near is not None:
        events_visible.append(f"NEAR:{near['color'][0].upper()} d={near['distance']:.0f}")

    return steering, accel, events_visible
