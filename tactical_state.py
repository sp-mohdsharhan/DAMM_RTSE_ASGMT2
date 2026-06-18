"""V3.0 tactical event bookkeeping.

This module tracks lightweight competition state that is not part of the Unity
physics simulation: event pass flags, deadlines, and approximate token hits.
"""

from __future__ import annotations

from perception_types import FrontPerception, TacticalSnapshot


POLICE_RED_DEADLINE_S = 5.0
RED_COLLECT_DISTANCE = 35.0
RED_COLLECT_AREA_FRAC = 0.06


_state = {
    'run_start_time': None,
    'elapsed_game_s': 0.0,
    'police_active': False,
    'police_started_at': None,
    'police_deadline': None,
    'passed_police': False,
    'police_timeout': False,
    'red_hits': 0,
}

_red_was_collectable = False


def is_red_collectable(front_per: FrontPerception | None) -> bool:
    """True when a red token is close enough to count as collected/passed.

    This is a camera-side approximation for the EV2 pass condition. The Unity
    game remains authoritative; this state is for controller decisions and HUD.
    """
    red = front_per.get('red') if front_per else None
    if red is None:
        return False

    distance = red.get('distance')
    if distance is not None and distance <= RED_COLLECT_DISTANCE:
        return True

    return red.get('area_frac', 0.0) >= RED_COLLECT_AREA_FRAC


def _update_run_timer(now: float) -> None:
    if _state['run_start_time'] is None:
        _state['run_start_time'] = now
    _state['elapsed_game_s'] = now - _state['run_start_time']


def update_tactical_state(front_per: FrontPerception | None, now: float) -> TacticalSnapshot:
    """Update V3.0 tactical state from current perception and return a snapshot."""
    global _red_was_collectable

    _update_run_timer(now)

    police_seen = bool(front_per and front_per.get('police'))
    if police_seen and not _state['police_active'] and not _state['passed_police']:
        _state['police_active'] = True
        _state['police_started_at'] = now
        _state['police_deadline'] = now + POLICE_RED_DEADLINE_S
        _state['police_timeout'] = False

    red_collectable = is_red_collectable(front_per)
    red_collected_now = red_collectable and not _red_was_collectable
    _red_was_collectable = red_collectable

    if red_collected_now:
        _state['red_hits'] += 1
        if _state['police_active'] and not _state['passed_police']:
            _state['passed_police'] = True
            _state['police_active'] = False

    if _state['police_active'] and _state['police_deadline'] is not None:
        if now > _state['police_deadline']:
            _state['police_active'] = False
            _state['police_timeout'] = True

    police_time_left = None
    if _state['police_active'] and _state['police_deadline'] is not None:
        police_time_left = max(0.0, _state['police_deadline'] - now)

    return {
        'elapsed_game_s': float(_state['elapsed_game_s']),
        'police_active': bool(_state['police_active']),
        'police_time_left': police_time_left,
        'passed_police': bool(_state['passed_police']),
        'police_timeout': bool(_state['police_timeout']),
        'red_hits': int(_state['red_hits']),
    }
