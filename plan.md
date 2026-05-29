# Plan: OpenCV Auto-Detection for SpeedTrials2D (RTSE Competition)

Integrate OpenCV perception + autonomous control inside the **editable** regions of `sample_drive.py`. Front camera handles color-object detection + lane following with green-attractor steering; back camera detects police (debuff source) and other cars (lane-change trigger).

## Locked vs Editable (from skeleton comments)

- **DO NOT TOUCH**: `Configuration`, `Real-Time Scheduling Framework` (`TaskPriority`, `RTTask`), `Network Connection Setup` (`setup_cameras`, `setup_control_server`, globals), body of `read_single_camera`, `__main__` shutdown block.
- **EDITABLE**: imports (add only), `shared_data` keys, `processing_task`, `send_controls_task`, `RTTask(...)` period/priority args in `__main__`, new helpers added under "Task Implementations".
- We open a new `"Perception"` window from `processing_task` rather than editing `read_single_camera`'s built-in imshow.

## Game-Rule Mapping

**Front camera — colored objects**

| Color | Effect on hit | Steering role |
|---|---|---|
| Red | `target_speed -= 20` | AVOID if centroid near image center & close |
| Green | `target_speed += 10` | ATTRACT — steer toward centroid (lane-change to score +10) |
| Yellow | trigger ONE random yellow-event | Ignored for steering (risky) |

**Yellow event pool** (random pick, 1 s cooldown):
1. Front-cam productivity reduced (effective read period 5 ms → 100 ms for 5 s)
2. Back-cam productivity reduced (same)
3. `target_speed *= 0.95`
4. `force_lane_change` (1.5 s hard swerve, direction alternates)
5. **`spawn_police`** → police vehicle appears in **rear view**; persistent `eff_speed *= 0.5`

**Police escape rule**
- Police detected only in **back camera** (hardcoded police-color HSV; blue assumed; tunable).
- While `police_active`, throttle halved.
- To clear: hit the **next color object** in the front view (red/green/yellow). The hit applies its normal effect (which may re-spawn police via yellow), then `police_active = False`.

**Rear camera — two threat classes**
- **Police** (blue HSV): sets/maintains `police_active` while bbox present. Cleared only via "next color hit" rule.
- **Other car** (vehicle-shaped non-police contour, growing area frame-over-frame): triggers `force_lane_change`. Per spec rear car is always 10% faster → we change lane, don't try to out-accelerate.

**Hit definition**: front object bbox area > `HIT_AREA_FRAC` (6%) of front ROI, per-color 1 s cooldown.

**Throttle**: `accel = clip(target_speed * (0.5 if police_active else 1.0) / 100, -1, 1)`.

## Steering Priority Stack (highest wins)
1. `force_lane_change` active → fixed ±0.8 for 1.5 s (alternating direction)
2. RED avoidance (centroid in center band AND area_frac > avoid threshold) → steer away
3. GREEN attraction (any green visible) → P-controller on `green_x_offset` from center
4. Lane P-controller (Canny + HoughLinesP on bottom ROI)
5. Default 0.0

## Steps

### Phase 1 — Shared state & constants
1. Extend `shared_data` with: `steering_cmd` (0.0), `accel_cmd` (0.0), `target_speed` (50.0), `police_active` (False), `active_events` (dict {name: expiry_ts}), `last_hit_ts` (dict per color), `swerve_dir` (+1/−1 alternating), `perception_front`, `perception_back`.
2. Add `state_lock = threading.Lock()`; keep `data_lock` scoped to frames only.
3. Constants block: HSV ranges (RED1, RED2, GREEN, YELLOW, POLICE_BLUE), ROI fractions, `HIT_AREA_FRAC=0.06`, `RED_AVOID_AREA_FRAC=0.04`, `CENTER_BAND_FRAC=0.3`, `COOLDOWN_S=1.0`, event durations, P-gains, base/degraded camera periods.

### Phase 2 — Perception helpers (pure functions)
4. `_color_mask(hsv, ranges)` → binary mask.
5. `detect_front_objects(frame)` → resize 320×240, HSV; for {red, green, yellow}: mask → largest contour → `{bbox, area_frac, centroid_x_norm}`.
6. `detect_rear(frame, prev_other_area)` → `{'police': {bbox, present}, 'other_car': {bbox, area_frac, growing}}`.
7. `detect_lane_offset(frame)` → Canny + HoughLinesP on bottom ROI → normalized offset −1..+1 (or None).
8. `draw_overlay(front, rear, perception, hud)` → bboxes with labels, lane lines, green-attract arrow, red-avoid marker, HUD `target / eff / police / events / str / acc`.

### Phase 3 — Event engine
9. `apply_color_hit(color, now)` — apply delta (red −20, green +10, yellow rolls event); if `police_active` set False; update `last_hit_ts[color]`; clamp `target_speed` to [0, 100].
10. `trigger_yellow_event(now)` — `random.choice` over 5 events; set expiry in `active_events`; flip cam-gating flags or set police-armed.
11. `expire_events(now)` — drop expired entries; restore cam gating when degrade flags clear.

### Phase 4 — Dynamic cam-rate via gating
`RTTask.period` is immutable post-construction, so:
12. Keep `ReadFrontCamera` / `ReadBackCamera` RTTask periods at **0.005 s** HIGH.
13. Wrap reads as `gated_read_front()` / `gated_read_back()`: early-return unless `now - last_read_ts >= effective_period` (5 ms normal, 100 ms while degraded).
14. Register wrappers as `execute_func`. Originals remain callable.

### Phase 5 — Controller integration
15. New `processing_task` body:
    - Snapshot frames under `data_lock`, release immediately.
    - Run `detect_front_objects`, `detect_lane_offset`, `detect_rear` lock-free.
    - Under `state_lock`: `expire_events`; apply color hits; update `police_active` from rear + yellow-spawn flag minus just-cleared-by-hit; if `rear.other_car.growing` trigger `force_lane_change`.
    - Compute steering via priority stack; compute accel via throttle formula.
    - Write cmds + perception + state to `shared_data`; `cv2.imshow("Perception", overlay)` + `cv2.waitKey(1)`.
16. New `send_controls_task` body: try non-blocking `state_lock.acquire`; on success read cmds, on failure use last sent; `struct.pack('ff', s, a)`; send; preserve skeleton error handling.

### Phase 6 — Periods/priorities in `__main__`
17. Edit only the numbers in existing `RTTask(...)` lines:
    - `ReadFrontCamera`: 0.005 / HIGH (gated)
    - `ReadBackCamera`:  0.005 / HIGH (gated)
    - `Processing`:      0.033 / MEDIUM (~30 Hz)
    - `SendControls`:    0.020 / HIGH (50 Hz)
18. Add `import random` to imports. No new packages.

## RT Scheduling & Threading Improvements (editable areas only)

The skeleton uses identical 5 ms periods and mostly HIGH priorities — wasteful and risks priority inversion / lock contention.

### A. Period & priority retuning (Rate-Monotonic compliant)
| Task | Skeleton | Proposed | Rationale |
|---|---|---|---|
| `ReadFrontCamera` | 5 ms / HIGH | **5 ms / HIGH** (gated) | Tight loop drains backlog; gating handles yellow degrade |
| `ReadBackCamera` | 5 ms / HIGH | **5 ms / HIGH** (gated) | Same |
| `Processing` | 5 ms / MEDIUM | **33 ms / MEDIUM** | CV at 200 Hz is waste; 30 Hz matches camera FPS |
| `SendControls` | 5 ms / HIGH | **20 ms / HIGH** | 50 Hz is enough; cuts socket syscalls |

Shorter period ⇒ higher priority → RM-correct.

### B. Lock-granularity fix
Split skeleton's single `data_lock`: keep it for **frame slots only**, add `state_lock` for steering/throttle/event state. Camera threads never block on control state.

### C. Frame-snapshot pattern
`processing_task` grabs frame reference under `data_lock`, releases immediately, then runs the ~10–20 ms CV pipeline lock-free. No `.copy()` needed (camera thread rebinds atomically).

### D. Gated reads (instead of mutable RTTask periods)
`RTTask.period` is fixed post-construction. `gated_read_*` wrappers early-return when not enough time elapsed — RTTask still wakes at 5 ms, body is a no-op when degraded. Honors yellow cam-degrade events without breaking the locked framework.

### E. Non-blocking control fallback
`send_controls_task` uses `state_lock.acquire(blocking=False)`; if busy, re-send last `(steering, accel)`. Classic RT priority-inversion mitigation — 50 Hz deadline stays solid.

### F. Monotonic clocks
Use `time.monotonic()` in our cooldown / event-expiry logic (skeleton's `time.time()` unchanged).

### G. `cv2.waitKey` overhead
Skeleton's per-camera `waitKey(1)` at 200 Hz costs ~400 ms/s. Our `"Perception"` window only runs at 30 Hz → ~30 ms/s.

## Summary of Changes in Editable Sections
1. **`shared_data` extended** — adds cmd/state/perception keys; existing keys preserved.
2. **New `state_lock`** — splits command/state from frame I/O.
3. **New constants block** — HSV ranges, hit threshold, cooldowns, P-gains, periods (single tunable surface).
4. **New perception helpers** — `_color_mask`, `detect_front_objects`, `detect_rear`, `detect_lane_offset`, `draw_overlay`.
5. **New event engine** — `apply_color_hit`, `trigger_yellow_event`, `expire_events`, police-clear rule.
6. **New gated read wrappers** — registered as `RTTask` `execute_func`; originals untouched.
7. **Rewritten `processing_task`** — snapshot-under-lock, priority-stack steering, throttle, `"Perception"` overlay window.
8. **Rewritten `send_controls_task`** — non-blocking `state_lock`, last-command fallback, skeleton error handling preserved.
9. **RM-tuned `RTTask(...)` args in `__main__`** — only the numbers change.
10. **`import random`** added; `time.monotonic()` used in new helpers.

## Relevant Files
- `sample_drive.py` — only file modified (imports, `shared_data`, new helpers, two task bodies, RTTask args).
- `requirements.txt` — no edits (`opencv-python`, `numpy` already present).
- `test_communication.py` — reference only for `struct.pack('ff', steering, accel)`.

## Verification
1. Start SpeedTrials2D, then `python sample_drive.py`. Confirm 4 "Started" lines, camera + control connection lines.
2. `"Perception"` window opens with HUD `target=50 eff=50 police=0 events=[] str=0.00 acc=0.50`.
3. Pass GREEN close → HUD `target=60`; car steers toward green while visible.
4. Pass RED → HUD `target=40`; if centered+close, car briefly steers away.
5. Pass YELLOW → exactly one of: cam-degraded visible slowdown for 5 s, `target_speed *= 0.95`, 1.5 s swerve, or `police=1` with bbox in back-cam HUD.
6. While `police=1`: throttle halved. Hit next color → `police=0` and color's normal effect applies.
7. Approach a car behind → `events=[force_lane_change]`, swerve fires; direction alternates across triggers.
8. Bypass test: hard-code `target_speed=60`, skip perception → car drives at ~0.6 throttle (validates control path).
9. Ctrl+C → "System terminated cleanly." within ~1 s.

## Decisions / Assumptions
- `target_speed` 0–100 with police multiplier 0.5; initial value 50.
- Yellow = random single event; police only spawns via yellow event 5.
- Police identified only in rear (blue HSV, tunable).
- Police cleared by next color hit; that color's effect still applies.
- Other-car detected via rear contour growth; always change lane.
- Green = steering attractor; Red = avoider only when centered+close; Yellow ignored for steering.
- `force_lane_change` = 1.5 s ±0.8 steer, alternating per trigger.
- Cam productivity reduced via gated reads (RTTask period unchanged).
- Locked skeleton sections untouched.

## Out of Scope
ML detectors, disk logging, tuning GUI, edits to Configuration / Scheduling / Network blocks, edits to `read_single_camera` body.

## Further Considerations
1. **Police color** — assumed blue; confirm or override with actual game color.
2. **Green attractor strength** — recommend P-gain 0.8 overriding lane until green is hit.
3. **Red avoid trigger** — only when in center band; else ignored.
4. **Initial `target_speed`** — 50 default; prefer 70 for faster start?
