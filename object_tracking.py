"""object_tracking.py — lightweight multi-object tracking for SpeedTrials2D.

Adds temporal consistency on top of the per-frame detectors in
image_detection.py (works with BOTH the HSV and the YOLO detection paths, since
it consumes their common output dicts, not raw pixels).

Each tracked detection is enriched IN PLACE with:
    'track_id' : stable integer id that persists across frames
    'vx', 'vy' : centroid velocity in PROC-space pixels/frame (smoothed by a
                 constant-velocity Kalman filter; +x = right, +y = down/closer)
    'speed'    : sqrt(vx^2 + vy^2), pixels/frame

For the rear chasing car, the tracked 'info' dict additionally gets:
    'closing_speed' : how fast the car is approaching (px/frame, +ve = closing)

DESIGN: purely ADDITIVE. With TRACKING_ENABLED = False the track_* functions
return their input untouched, so the controller/overlay see exactly the current
contract. With it True, the same dicts simply carry extra keys; nothing is
forced to consume them.

Toggle here (mirrors DETECTION_MODE in image_detection.py):
    TRACKING_ENABLED = True   # set False to fully disable tracking
"""

import os
import time

import numpy as np
import cv2

from image_detection import PROC_W, PROC_H

# ---------------------------------------------------------------------------
# Toggle
# ---------------------------------------------------------------------------
TRACKING_ENABLED = True

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
# Association gate: a detection can only attach to a predicted track if it is
# within this many PROC-space pixels of the prediction. A fast-approaching token
# moves a long way DOWN the frame per frame; if the gate is tighter than that
# jump the token spawns a fresh track every frame, never confirms, and is never
# counted as collected. So the gate must comfortably exceed per-frame motion.
# (Kalman prediction separates a continuing token from a new one at the top, so
# a wide gate does not cause id-swaps once velocity is learned.)
TRACK_GATE_PX = 100.0
TRACK_MAX_MISSES = 6          # drop a track after this many unmatched frames
TRACK_MIN_HITS = 2           # frames before a track is considered "confirmed"

# Kalman process / measurement noise. Higher process noise = tracker trusts new
# measurements more (snappier, noisier velocity); higher measurement noise =
# smoother but laggier. Tuned for the game's fast token motion.
KALMAN_PROCESS_VAR = 1.0
KALMAN_MEAS_VAR = 4.0

# --- Collected-green detection ---------------------------------------------
# A token is "collected" when the car drives over it: a confirmed green track
# that, AT SOME POINT IN ITS LIFE, came NEAR (small bird's-eye distance) while
# roughly CENTERED — then vanished. We judge the closest approach over the whole
# track, NOT the final frame, because detection often drops a token a frame or
# two before it reaches the car (so the last-seen point is unreliable). A token
# that only ever appears off to the side / far away never sets the flag.
# Classified when the track first goes unmatched for COLLECT_MISSES frames (fast,
# past 1-frame flicker) so the controller can drop the green-seek latch.
COLLECT_DETECTION_ENABLED = True
COLLECT_MISSES = 2            # unmatched frames before classifying a vanish
COLLECT_MAX_DISTANCE = 145.0 # closest-approach distance to count as reached (smaller = nearer)
COLLECT_CENTER_BAND = 0.72   # |centroid_x_norm| at closest approach = in the car's path

# Backend diagnostic log: one line per confirmed-green that vanished, recording
# whether it was classified collected and the last-seen values, plus the running
# total. Lets the detected count be compared against the actual game score so the
# thresholds above can be tuned. No on-screen overlay — backend only.
COLLECT_LOG_ENABLED = True
COLLECT_LOG_PATH = os.path.join('logs', 'collected.log')
_collect_log_init = False


def _full_x(det):
    """Detection centroid x back in full PROC_W pixel space (overlay coords)."""
    return det['centroid_x_norm'] * (PROC_W / 2.0) + PROC_W / 2.0


def _log_collect(track, collected, total):
    """Append one diagnostic line per confirmed-green vanish (backend only)."""
    global _collect_log_init
    if not COLLECT_LOG_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(COLLECT_LOG_PATH), exist_ok=True)
        ts = time.strftime('%H:%M:%S')
        o = track.obs or {}
        mode = 'w' if not _collect_log_init else 'a'
        with open(COLLECT_LOG_PATH, mode, encoding='utf-8') as fh:
            if not _collect_log_init:
                fh.write(f"# Collected-green log (session started {ts})\n")
                fh.write("# time  id  collected  reached  last_dist  last_xn  last_yfrac  hits  total\n")
                _collect_log_init = True
            fh.write(
                f"{ts}  id={track.id}  collected={int(bool(collected))}  "
                f"reached={int(bool(track.reached))}  "
                f"dist={o.get('dist', float('inf')):.0f}  xn={o.get('xn', 0.0):+.2f}  "
                f"yfrac={o.get('yfrac', 0.0):.2f}  hits={track.hits}  total={total}\n"
            )
    except Exception:
        pass


def _new_kalman(x, y):
    """Constant-velocity Kalman filter (state [x, y, vx, vy]) seeded at (x, y)."""
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KALMAN_PROCESS_VAR
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KALMAN_MEAS_VAR
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf


class _Track:
    __slots__ = ('id', 'kf', 'color', 'hits', 'misses', 'pred',
                 'obs', 'reached', 'collected_emitted')

    def __init__(self, tid, x, y, color):
        self.id = tid
        self.kf = _new_kalman(x, y)
        self.color = color
        self.hits = 1
        self.misses = 0
        self.pred = (x, y)
        self.obs = None             # last matched detection's key fields
        self.reached = False        # came near+centered at any point in its life
        self.collected_emitted = False

    def predict(self):
        p = self.kf.predict()                       # (4, 1) column vector
        self.pred = (float(p[0, 0]), float(p[1, 0]))
        return self.pred

    def update(self, x, y):
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        self.hits += 1
        self.misses = 0

    def observe(self, det):
        """Record this frame's detection and update the closest-approach flag."""
        xn = det.get('centroid_x_norm', 0.0)
        dist = det.get('distance', float('inf'))
        self.obs = {
            'xn': xn,
            'dist': dist,
            'yfrac': det.get('centroid_y', 0.0) / float(PROC_H),
        }
        # Latch the moment this token came near + into the car's path. Judging the
        # whole life (not the final frame) survives detection dropping the token
        # a frame early.
        if dist <= COLLECT_MAX_DISTANCE and abs(xn) <= COLLECT_CENTER_BAND:
            self.reached = True

    @property
    def velocity(self):
        # statePost is a (4, 1) column vector; index [row, 0] for a true scalar
        # (NumPy 2.0 rejects float() on a 1-element non-0-d array).
        s = self.kf.statePost
        return float(s[2, 0]), float(s[3, 0])


class MultiObjectTracker:
    """Greedy nearest-neighbour tracker over detection centroids.

    One instance per camera stream (front orbs, rear car). Colour-aware: a track
    only matches detections of the same colour label so a red token can't steal a
    green token's id. Cars (no colour) match purely on proximity.
    """

    def __init__(self, gate_px=TRACK_GATE_PX):
        self._tracks = []
        self._next_id = 1
        self._gate = gate_px
        self.collected_this_frame = 0   # greens classified as collected this frame
        self.collected_total = 0        # running session total

    def update(self, detections):
        """Advance one frame. Enriches each detection dict in place with
        'track_id', 'vx', 'vy', 'speed' and returns the same list. Also tallies
        greens that vanished after being driven over (collected_this_frame)."""
        self.collected_this_frame = 0
        for t in self._tracks:
            t.predict()

        # Greedy association: nearest (track, detection) pair within the gate,
        # same colour, each used at most once.
        unmatched = list(range(len(detections)))
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di in unmatched:
                d = detections[di]
                if t.color != d.get('color'):
                    continue
                dx = _full_x(d) - t.pred[0]
                dy = d['centroid_y'] - t.pred[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= self._gate:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])

        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self._tracks[ti]
            d = detections[di]
            t.update(_full_x(d), d['centroid_y'])
            t.observe(d)
            self._annotate(d, t)

        # Unmatched detections -> new tracks.
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            t = _Track(self._next_id, _full_x(d), d['centroid_y'], d.get('color'))
            self._next_id += 1
            t.observe(d)
            self._tracks.append(t)
            self._annotate(d, t)

        # Age unmatched tracks; classify each vanished confirmed-green once (and
        # log it for diagnostics); cull the stale ones.
        for ti, t in enumerate(self._tracks):
            if ti in used_t:
                continue
            t.misses += 1
            if (COLLECT_DETECTION_ENABLED
                    and not t.collected_emitted
                    and t.misses >= COLLECT_MISSES):
                if t.color == 'green' and t.hits >= TRACK_MIN_HITS:
                    collected = self._is_collected(t)
                    if collected:
                        self.collected_this_frame += 1
                        self.collected_total += 1
                    _log_collect(t, collected, self.collected_total)
                t.collected_emitted = True          # classified once, don't repeat
        self._tracks = [t for t in self._tracks if t.misses <= TRACK_MAX_MISSES]
        return detections

    @staticmethod
    def _is_collected(track):
        """True if a vanished track looks driven-over: a confirmed green that came
        near + into the car's path at some point during its life."""
        return bool(
            track.color == 'green'
            and track.hits >= TRACK_MIN_HITS
            and track.reached
        )

    @staticmethod
    def _annotate(det, track):
        vx, vy = track.velocity
        det['track_id'] = track.id
        det['vx'] = vx
        det['vy'] = vy
        det['speed'] = float((vx * vx + vy * vy) ** 0.5)
        det['track_confirmed'] = track.hits >= TRACK_MIN_HITS


# ---------------------------------------------------------------------------
# Stream-level helpers (one persistent tracker per stream)
# ---------------------------------------------------------------------------
_front_tracker = MultiObjectTracker()
_rear_tracker = MultiObjectTracker(gate_px=TRACK_GATE_PX * 1.4)  # cars move more


def track_front(front_per):
    """Enrich front orbs with track ids + velocity, and flag collected greens.
    No-op if disabled. We only ADD fields, so the existing 'nearest'/per-colour
    entries stay valid (distances are unchanged).

    Always advances the tracker — even with zero orbs this frame — so a green
    that was the last orb still ages out and gets classified as collected.
    Sets on front_per:
        'collected_green' : greens driven-over this frame (int, 0 if none)
        'collected_total' : running session total (int)
    """
    if not TRACKING_ENABLED or not front_per:
        return front_per
    _front_tracker.update(front_per.get('orbs') or [])
    front_per['collected_green'] = _front_tracker.collected_this_frame
    front_per['collected_total'] = _front_tracker.collected_total
    return front_per


def track_rear(rear_per):
    """Track the chasing car and add 'closing_speed' (px/frame, +ve = approaching)
    to its info dict. No-op if disabled or no car this frame."""
    if not TRACKING_ENABLED or not rear_per:
        return rear_per
    other = rear_per.get('other_car') or {}
    info = other.get('info')
    if info is not None:
        _rear_tracker.update([info])
        # A car growing in the rear view moves DOWN the frame (+vy) as it nears.
        info['closing_speed'] = float(info.get('vy', 0.0))
    return rear_per
