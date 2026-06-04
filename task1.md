# Task 1 — Perception Confidence Gating + Temporal Filtering

**Owner:** Perception developer
**Files touched:** image_detection.py, sample_drive.py
**Goal:** Stop the controller from acting on noisy / low-confidence detections.
Today the `Perception` overlay shows boxes with `GREEN 2.3%`, `POLICE 0.1%`,
`CAR 0.5%` — those are essentially noise but they still reach the controller.

---

## 1. Background (read first)

- `detect_front_objects(frame)` in image_detection.py returns
  `{'red': info|None, 'green': info|None, 'yellow': info|None, ...}`.
- `_largest_contour_info(...)` already filters by shape (area, aspect,
  circularity, fill ratio). It returns a dict with `area_frac`,
  `centroid_x_norm`, `centroid_y`, `bbox`.
- `detect_rear(frame)` returns `{'police': {...}, 'other_car': {...}}`.
- `sample_drive.py::processing_task` reads these dicts and computes
  `steering_cmd` / `accel_cmd`.

There is currently **no confidence value** on detections — presence ==
"contour passed the shape filter". The overlay percentage you see in the
screenshot is computed elsewhere (likely `draw_overlay`) but is **not** used
as a gate.

---

## 2. Definition of done

1. Every detection dict carries a numeric `confidence ∈ [0, 1]`.
2. A configurable per-class threshold drops detections below it **before**
   they reach the controller.
3. A short temporal filter (N-of-M) requires a class to be seen in at least
   `N` of the last `M` frames before it counts as "confirmed".
4. The overlay shows the gated/confirmed state, not raw detections.
5. Controller logic in `processing_task` only consumes confirmed detections.

---

## 3. Step-by-step

### Step 3.1 — Add a confidence score in `_largest_contour_info`

In image_detection.py, inside `_largest_contour_info`, after the best
contour is chosen, compute a composite confidence from features already
calculated:

```python
# circularity in [0,1], fill_ratio in [0,1], area_frac small but positive
conf_shape = min(1.0, (circularity / 0.9)) * min(1.0, (area / bbox_area) / 0.85)
conf_size  = min(1.0, area_frac / 0.02)   # bigger orb -> more confident
confidence = float(0.6 * conf_shape + 0.4 * conf_size)
```

Add `'confidence': confidence` to the returned dict. Do the same in the
rear vehicle/police path inside `detect_rear`.

> Keep the existing shape rejection — confidence is an **additional** signal,
> not a replacement.

### Step 3.2 — Add per-class thresholds (constants)

At the top of image_detection.py, near the other tunables:

```python
CONF_THRESHOLD = {
    'red':    0.45,
    'green':  0.40,
    'yellow': 0.45,
    'police': 0.50,
    'other_car': 0.40,
}
```

Export them via the existing import block in sample_drive.py.

### Step 3.3 — Temporal N-of-M filter

Create a small helper class in image_detection.py:

```python
from collections import deque

class DetectionConfirmer:
    """Confirms a class only when seen in >= N of last M frames."""
    def __init__(self, classes, window=5, min_hits=3):
        self.window = window
        self.min_hits = min_hits
        self.hist = {c: deque(maxlen=window) for c in classes}

    def update(self, observed_classes_set):
        confirmed = {}
        for c, h in self.hist.items():
            h.append(1 if c in observed_classes_set else 0)
            confirmed[c] = sum(h) >= self.min_hits
        return confirmed
```

Instantiate it once at module scope:

```python
_confirmer = DetectionConfirmer(
    ['red', 'green', 'yellow', 'police', 'other_car'],
    window=5, min_hits=3,
)
```

### Step 3.4 — Gating function

Add to image_detection.py:

```python
def gate_detections(front_per, rear_per):
    """Drop low-confidence detections, then apply temporal confirmation.
    Mutates copies, returns (front_gated, rear_gated, confirmed_map)."""
    front = dict(front_per)
    rear  = dict(rear_per)

    for cls in ('red', 'green', 'yellow'):
        d = front.get(cls)
        if d is not None and d.get('confidence', 0.0) < CONF_THRESHOLD[cls]:
            front[cls] = None

    pol = rear.get('police', {}).get('info')
    if pol and pol.get('confidence', 0.0) < CONF_THRESHOLD['police']:
        rear['police'] = {'info': None, 'present': False}

    oc = rear.get('other_car', {}).get('info')
    if oc and oc.get('confidence', 0.0) < CONF_THRESHOLD['other_car']:
        rear['other_car'] = {'info': None, 'growing': False}

    observed = set()
    for cls in ('red', 'green', 'yellow'):
        if front.get(cls) is not None: observed.add(cls)
    if rear.get('police', {}).get('info') is not None:    observed.add('police')
    if rear.get('other_car', {}).get('info') is not None: observed.add('other_car')
    confirmed = _confirmer.update(observed)

    for cls in ('red', 'green', 'yellow'):
        if not confirmed.get(cls, False):
            front[cls] = None
    if not confirmed.get('police', False):
        rear['police'] = {'info': None, 'present': False}
    if not confirmed.get('other_car', False):
        rear['other_car'] = {'info': None, 'growing': False}

    return front, rear, confirmed
```

### Step 3.5 — Wire into sample_drive.py

In `processing_task`, immediately after the existing calls to
`detect_front_objects` and `detect_rear`:

```python
from image_detection import gate_detections   # add to import block

front_per = detect_front_objects(front_frame)
rear_per  = detect_rear(back_frame)
front_per, rear_per, _confirmed = gate_detections(front_per, rear_per)
```

All downstream code already treats `None` as "not detected", so no further
controller changes are required.

### Step 3.6 — Overlay

In `draw_overlay`, render the confidence next to each label
(`f"{cls.upper()} {info['confidence']*100:.0f}%"`) and skip drawing classes
that are `None` after gating. Boxes for un-confirmed classes must not be
drawn.

---

## 4. Verification checklist

- [ ] Run sample_drive.py; the `Perception` window no longer shows
  sub-10% boxes.
- [ ] Cover the front camera (e.g. swerve into grass): detections must
  disappear within ~`window` frames, not flicker.
- [ ] Print/log the rate of dropped detections per minute — should be > 0
  on noisy frames.
- [ ] No regression in normal driving distance vs. baseline run.

---

## 5. Out of scope

- Retraining any model — this task is pure post-processing.
- Multi-object tracking (SORT/ByteTrack) — leave a TODO; that is Task 1.b.
- Changing HSV calibration logic.
