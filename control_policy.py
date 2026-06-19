"""Driving policy for the SpeedTrials2D autonomous controller.

This module owns steering/throttle decisions and short-lived manoeuvre latches.
It deliberately does not touch sockets, locks, OpenCV windows, or shared_data.
"""

import numpy as np

from image_detection import (
    CENTER_BAND_FRAC,
    GREEN_ATTRACT_GAIN,
    GREEN_ATTRACT_MIN_AREA,
    GREEN_COMMIT_DISTANCE,
    GREEN_COMMIT_MAX_STEER,
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
from perception_types import FrontPerception, RearPerception, TacticalSnapshot


CRUISE_THROTTLE = 0.8
LOW_BRIGHTNESS_THROTTLE = -1.0

# Golden Lane (Phase 17): aggressive steering gain used to snap the car onto a
# target lane within the 5 s window. Higher than the normal attract gains so the
# manoeuvre commits ("hard steer").
GOLDEN_LANE_STEER_GAIN = 1.6
GOLDEN_LANE_HOLD_BAND = 0.06   # |lane offset| under this = already in the lane
GOLDEN_LANE_MIN_CONFIDENCE = 0.25
GOLDEN_LANE_MIN_GAIN_SCALE = 0.45
GOLDEN_LANE_CURVE_SOFTEN = 0.45

# Golden Lane lock window (s). detect_golden_lane() reads the lane number N from
# the orange "LANE N - ALL GREEN!" banner in the camera frame, but the banner's
# (Xs) countdown OCR is unreliable, so we do not know the true remaining time.
# Detection also flickers (banner re-reads, greens get collected, or a yellow
# debuff hides tokens). So once a lane is announced we latch the lane number and
# hold it for this fixed window instead of re-deciding per frame: a flicker
# mid-crossing still completes and we stay in the lane. Matches the game's 5 s
# Golden Lane duration.
GOLDEN_LANE_LOCK_S = 5.0

# Golden Lane lock latch: remembers the inferred lane and the time the lock
# expires. Re-arms only on a fresh detection (different lane, or after expiry).
_golden_lock = {'until': 0.0, 'lane': None}

# Red avoidance latch: once a red is detected ahead, commit to a full
# lane-change away from it, then counter-steer briefly to settle.
_red_avoid = {'until': 0.0, 'settle_until': 0.0, 'dir': 0}

# Green pursuit latch: bridge frames where a green flickers / leaves ROI
# mid-crossing so the lane change still completes.
_green_seek = {'until': 0.0, 'dir': 0}

# Chasing-car forced-swerve latch. Direction alternates each trigger.
_lane_change_until = 0.0
_lane_change_dir = 1


def _near_green_commit(cx, distance, curve_bias):
    """Once a green is within GREEN_COMMIT_DISTANCE, drive gently onto it instead
    of starting a lane change. A near green's centroid offset magnifies as it
    drops to the frame bottom and would otherwise cross GREEN_LANE_CHANGE_BAND
    and swerve us off it just before contact (only a problem when slow). Returns
    a clamped gentle steer, or None if the green is not near enough to commit."""
    if distance is None or distance >= GREEN_COMMIT_DISTANCE:
        return None
    _green_seek['until'] = 0.0          # cancel any pending lane-change latch
    _green_seek['dir'] = 0
    steer = GREEN_ATTRACT_GAIN * cx + LANE_CURVE_GAIN * curve_bias
    return float(np.clip(steer, -GREEN_COMMIT_MAX_STEER, GREEN_COMMIT_MAX_STEER))


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

    # A green was just collected (object_tracking saw the token driven over and
    # vanish). Drop the green-seek bridging latch so we don't keep steering at
    # the now-empty lane — the logic below retargets the next green this frame.
    if front_per and front_per.get('collected_green'):
        _green_seek['until'] = 0.0
        _green_seek['dir'] = 0

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
        # Green: if it's nearly on us, commit to collecting (no late lane change).
        committed = _near_green_commit(cx, target.get('distance'), curve_bias)
        if committed is not None:
            return committed
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
        committed = _near_green_commit(cx, green.get('distance'), curve_bias)
        if committed is not None:
            return committed
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


def steer_to_lane(
    lane_grid: dict | None,
    target_lane: int | None,
    curve_bias: float = 0.0,
) -> float | None:
    """Hard-steer command (-1..+1) to drive the car onto ``target_lane`` (1..N).

    Uses the bird's-eye lane grid from estimate_lane_grid(): each lane has a
    signed centre offset (``lane_centers_norm``, positive => lane is to the
    right of the car). Returns a committed steering value toward the target
    lane's centre, or None when the grid/target is unavailable (so the caller
    can fall back to normal lane keeping). Returns 0.0 when the car is already
    within GOLDEN_LANE_HOLD_BAND of the target lane centre.

    This is the primitive the Golden Lane handler uses to snap onto the inferred
    golden lane within the 5 s window.
    """
    if not lane_grid or target_lane is None:
        return None
    centers = lane_grid.get('lane_centers_norm') or []
    idx = int(target_lane) - 1
    if idx < 0 or idx >= len(centers):
        return None
    confidence = float(lane_grid.get('confidence', 1.0))
    if confidence < GOLDEN_LANE_MIN_CONFIDENCE:
        return None
    offset = centers[idx]
    if abs(offset) < GOLDEN_LANE_HOLD_BAND:
        return 0.0
    confidence_scale = GOLDEN_LANE_MIN_GAIN_SCALE + (1.0 - GOLDEN_LANE_MIN_GAIN_SCALE) * confidence
    curve_scale = 1.0 - min(GOLDEN_LANE_CURVE_SOFTEN, abs(curve_bias) * GOLDEN_LANE_CURVE_SOFTEN)
    gain = GOLDEN_LANE_STEER_GAIN * confidence_scale * curve_scale
    return float(np.clip(gain * offset, -1.0, 1.0))


def _police_collision_risk(front_per: FrontPerception | None) -> bool:
    police = front_per.get('police') if front_per else None
    if police is None:
        return False
    return (
        police['area_frac'] > POLICE_DODGE_AREA
        and abs(police['centroid_x_norm']) < POLICE_DODGE_BAND
    )


def _steer_to_nearest_red(front_per: FrontPerception | None, curve_bias: float) -> float | None:
    """Aggressive red-token seek for EV2 police pass."""
    if not front_per:
        return None

    reds = [o for o in front_per.get('orbs', []) if o.get('color') == 'red']
    target = min(reds, key=lambda o: o['distance']) if reds else front_per.get('red')
    if target is None:
        return None

    cx = target['centroid_x_norm']
    if abs(cx) > GREEN_LANE_CHANGE_BAND:
        return float(np.clip(GREEN_SEEK_GAIN * (1 if cx > 0 else -1), -1.0, 1.0))
    return float(np.clip(GREEN_ATTRACT_GAIN * cx + LANE_CURVE_GAIN * curve_bias, -1.0, 1.0))


def compute_control(
    front_per: FrontPerception | None,
    rear_per: RearPerception | None,
    curve_bias: float,
    hill: bool,
    low_light: bool,
    now: float,
    tactical: TacticalSnapshot | None = None,
) -> tuple[float, float, list[str]]:
    """Return (steering, accel, events_visible) for the current perception frame."""
    global _lane_change_until, _lane_change_dir

    rear_other = rear_per.get('other_car') if rear_per else None
    chasing_seen = bool(rear_other and rear_other.get('info'))

    if chasing_seen and now >= _lane_change_until:
        _lane_change_until = now + LANE_CHANGE_DURATION_S
        _lane_change_dir = -_lane_change_dir

    force_lc = now < _lane_change_until
    police_seen = bool(front_per and front_per.get('police'))
    golden = front_per.get('golden') if front_per else None

    if police_seen:
        # Front police collision is game-ending: dodge first when close/centred,
        # otherwise seek a red token immediately to pass EV2.
        steering = _compute_steering(
            front_per,
            curve_bias,
            hill,
            now,
            False,
            _lane_change_dir,
        )
        accel = CRUISE_THROTTLE
    elif force_lc:
        # Rear chasing car is collision-critical: once detected, commit to the
        # evasive lane change regardless of front-token or Golden Lane goals.
        steering = float(LANE_CHANGE_STEER * _lane_change_dir)
        accel = CRUISE_THROTTLE
    elif low_light:
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

    if (
        not police_seen
        and not force_lc
        and tactical
        and tactical.get('police_active')
        and not _police_collision_risk(front_per)
    ):
        red_seek = _steer_to_nearest_red(front_per, curve_bias)
        if red_seek is not None:
            steering = red_seek

    # Golden Lane: latch the inferred lane and lock onto it for GOLDEN_LANE_LOCK_S
    # seconds. Token inference cannot read the 5 s banner countdown and flickers,
    # so once a lane is declared we re-arm only on a fresh detection (a different
    # lane, or after the previous lock expired) and otherwise let the timer run
    # down from the first detection -- continued/flickering detection does not
    # refresh it.
    if golden and golden.get('active') and golden.get('lane'):
        lane = int(golden['lane'])
        if lane != _golden_lock['lane'] or now >= _golden_lock['until']:
            _golden_lock['lane'] = lane
            _golden_lock['until'] = now + GOLDEN_LANE_LOCK_S

    golden_locked = now < _golden_lock['until'] and _golden_lock['lane'] is not None

    # The lock overrides normal front-camera steering, but not police or the rear
    # chasing-car escape above. It holds even when the current frame has no golden
    # evidence, so a flicker mid-crossing still completes the lane change.
    if not police_seen and not force_lc and golden_locked:
        golden_steer = steer_to_lane(
            front_per.get('lane_grid') if front_per else None,
            _golden_lock['lane'],
            curve_bias,
        )
        if golden_steer is not None:
            steering = golden_steer

    events_visible = []
    if golden_locked:
        remaining = _golden_lock['until'] - now
        source = golden.get('source') if golden else None
        tag = 'tok' if source == 'tokens' else 'banner'
        events_visible.append(
            f"GOLDEN_LOCK({tag})->L{_golden_lock['lane']} {remaining:.1f}s"
        )
    elif golden and golden.get('active') and not golden.get('lane'):
        events_visible.append('GOLDEN(?)')
    if tactical:
        if tactical.get('police_active'):
            time_left = tactical.get('police_time_left')
            if time_left is not None:
                events_visible.append(f"POLICE_RED:{time_left:.1f}s")
            else:
                events_visible.append('POLICE_RED')
        elif tactical.get('passed_police'):
            events_visible.append('POLICE_PASS')
        elif tactical.get('police_timeout'):
            events_visible.append('POLICE_TIMEOUT')
    if police_seen:
        events_visible.insert(0, 'POLICE_PRIORITY')
        events_visible.append('POLICE->GRAB_RED')
    if force_lc and not police_seen:
        events_visible.insert(0, 'CHASING_CAR_PRIORITY')
    elif force_lc:
        events_visible.append('CHASING_CAR_PRIORITY')
    if hill:
        events_visible.append('HILL')
    near = front_per.get('nearest') if front_per else None
    if near is not None:
        events_visible.append(f"NEAR:{near['color'][0].upper()} d={near['distance']:.0f}")

    return steering, accel, events_visible
