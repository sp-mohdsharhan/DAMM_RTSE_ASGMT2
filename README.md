# DAMM_RTSE_ASGMT2 - SpeedTrials2D Autonomous Driver

**SECJ 4423 - REAL-TIME SOFTWARE ENGINEERING**

**Semester II, Academic Session 2025/2026 - Group Assignment 2**

**Lecturer**: Prof. Ts. Dr. Dayang Norhayati Bte. Abang Jawawi

**Group**: DAMM

## Group Members

| Name | Matric No. |
| --- | --- |
| Auni Dalilah Binti Mohd Zain | SX170101CSJS04 |
| Siti Dzin Norsyafika Binti Mohd Isa | SX220330ECJHS04 |
| Muhammad Dzul Ifraan Bin Ab Rahman | SX231715ECJHF04 |
| Mohd Sharhan Bin Abdul Ghani | SX232315ECJHF04 |

---

## Project Overview

This project implements an autonomous driving controller for the SpeedTrials2D competition game. The current implementation is a pure camera-reactive controller: it does not keep a shadow copy of the game's score, speed, cooldowns, or event state. The Unity simulator remains the authoritative state machine, and the Python controller reacts to the current front and rear camera frames.

The driver is built around the four real-time software engineering requirements:

1. **Concurrency** - separate threads for front camera input, rear camera input, perception/decision making, and control output.
2. **Task Periods** - fixed task periods with each task run in a periodic loop.
3. **Task Priorities** - Windows thread priorities assigned using Deadline-Monotonic reasoning.
4. **Shared-Resource Synchronisation** - split locks for frame data and control state to reduce blocking between camera I/O and actuation.

The car receives two virtual camera streams over TCP, processes them with OpenCV, and sends `(steering, acceleration)` commands back to the simulator at 50 Hz.

---

## Current Algorithm

The controller uses a rule-based real-time perception and steering stack. It is deliberately deterministic and lightweight so the processing task can run near 30 Hz.

### Perception Techniques

| Component | Technique used |
| --- | --- |
| Image pre-processing | Resize camera frames to `320x240` for bounded processing cost. |
| Colour segmentation | Pinned HSV ranges measured from game assets for red, green, yellow, police blue, and chasing-car teal. HSV auto-calibration code remains in the module but is currently disabled because the measured ranges are more stable. |
| Road gating | Low-saturation asphalt mask filters colour contours so tokens must sit on the road, not on grass or sky. |
| Orb detection | HSV mask plus contour filters: area, aspect ratio, circularity, fill ratio, and solidity. A relaxed on-road gate accepts large perspective-squashed close orbs. |
| Orb distance | Inverse Perspective Mapping projects each orb's ground-contact point into a bird's-eye space. Smaller projected distance means the orb is nearer. |
| Front police detection | Detects the police car ahead by the blue half of the red/blue livery, then verifies adjacent dark-red pixels. This rejects sky, signs, red tokens, and other blue noise. |
| Rear chasing-car detection | Detects the teal chasing car from the rear camera, crops out the sky band, applies morphology, gates by area and centre band, then confirms it is catching up using smoothed area growth across recent frames. |
| Lane offset | Canny + HoughLinesP on the lower front ROI estimates lane-centre offset for overlay/debug. The current steering policy does not use this as the default fallback. |
| Lane curve | Bird's-eye warp, Sobel/V-channel lane-pixel mask, histogram bases, and sliding-window pixel collection estimate a `curve_bias` used as a small anticipatory steering bias while seeking green tokens. |
| Lane grid (1-5) | From the same bird's-eye lane-pixel mask, the road span is sliced into `N_LANES = 5` equal lanes numbered left-to-right. `estimate_lane_grid()` returns each lane's centre offset plus the car's current lane, giving the controller concrete lane targets for hard steering. |
| Golden Lane | The patched game draws the orange `LANE N - ALL GREEN! (Xs)` banner in the front-camera frame. `detect_golden_lane()` reads the lane number `N` by OCR; the `(Xs)` countdown OCR is unreliable, so the controller ignores it. When the banner lane is read, `control_policy` latches that lane and holds it for a fixed `GOLDEN_LANE_LOCK_S = 5.0 s` (the event duration). When the banner is active but the digit is unread (or no banner is seen), `infer_golden_lane_from_tokens()` falls back to the lane with the strongest green-token concentration. |
| Hill/slope detection | Tracks the asphalt horizon using an EMA baseline. When the horizon deviates enough, the controller treats the road as a hill and triggers red/yellow evasion earlier. |
| Low brightness | Mean V channel on a centre crop detects the low-light challenge. |

### Steering and Control Priority

The steering decision is ordered from highest priority to lowest:

1. **Front police car** - if police is detected, immediately take over steering. If it is close and centred, dodge away from it; otherwise seek a red token, preferring one on the opposite side of the police car. The HUD shows `POLICE_PRIORITY`.
2. **Rear chasing car** - if the rear teal car is detected, immediately commit to a forced lane change for `1.5 s` regardless of front-camera goals. The HUD shows `CHASING_CAR_PRIORITY`.
3. **Low brightness recovery** - if the front frame is dim, steering is set to `0.0` and acceleration is set to `-1.0` to reverse/recover visibility.
4. **Golden Lane** - when the orange banner is read (or token clustering infers the lane), latch that lane and hard-steer onto it via `steer_to_lane()` and the lane grid, holding for a fixed `5.0 s` lock. The HUD shows `GOLDEN_LOCK(banner)->L# Xs` (or `(tok)` for the token-inference fallback), where `Xs` counts down the local lock, not the game timer. This overrides the lower-priority steering below.
5. **Nearest imminent orb** - if any orb is within the IPM action distance (`ORB_ACT_DISTANCE = 120`), the nearest orb drives the action. If a red/yellow hazard is almost as near as a green (`ORB_TIE_MARGIN = 25`), hazard avoidance wins.
6. **Green token seek** - steer toward green. If the green is clearly in another lane, commit to a stronger lane change and hold briefly so flicker does not cancel the manoeuvre.
7. **Red token avoidance** - when a red token is ahead, commit to a full lane-change away from it for `1.6 s`, then counter-steer briefly for `0.35 s` to settle.
8. **Yellow token avoidance** - dodge yellow only when it is sufficiently close and inside the centre path band.
9. **Default** - go straight with cruise throttle.

Throttle policy:

| Situation | Acceleration command |
| --- | --- |
| Normal driving | `0.8` |
| Low brightness recovery | `-1.0` |

Hill policy does not reduce throttle. It scales red/yellow trigger thresholds by `HILL_AREA_SCALE = 0.5`, causing earlier evasive steering when tokens appear late over a crest.

---

## Game Rules and Implemented Reactions

| Game object/event | Camera signal | Current reaction |
| --- | --- | --- |
| Green token | Green on-road orb in front camera | Seek/grab using proportional steering or committed lane change. |
| Red token | Red on-road orb in front camera | Avoid with committed lane change, except during police handling where red is sought to escape. |
| Yellow token | Yellow on-road orb in front camera | Avoid only when close and centred. |
| Chasing car | Teal car in rear camera | Highest-priority forced lane change, alternating direction per trigger. |
| Police car | Red/blue police livery in front camera | Highest-priority steering: dodge if collision risk is high; otherwise seek a red token. |
| Golden Lane | Orange `LANE N - ALL GREEN!` banner read from the front camera (lane digit via OCR; countdown unreliable), with green-token concentration as fallback | Steer onto lane `N` via the lane grid and hold it with a fixed 5 s lock (the event duration), since the countdown cannot be read reliably. |
| Low brightness | Low mean V channel in front centre crop | Reverse with straight steering to recover. |
| Hill/crest | Asphalt horizon deviates from flat-road EMA baseline | Trigger red/yellow avoidance earlier. |

---

## Real-Time Architecture

### Tasks

| Task | Period | Priority | Role |
| --- | --- | --- | --- |
| `ReadFrontCamera` | 5 ms | HIGH | Decode the latest front TCP frame; collision-critical input. |
| `SendControls` | 20 ms | HIGH | Send `(steering, accel)` at 50 Hz; actuator deadline. |
| `Processing` | 33 ms | MEDIUM | Run perception and compute steering/throttle at about 30 Hz. |
| `ReadBackCamera` | 50 ms | LOW | Decode rear frames for the slower chasing-car signal. |

Priorities follow Deadline-Monotonic scheduling: shorter and more critical deadlines get higher priority. The rear camera is lowest priority because rear threats evolve more slowly and its deadline is longest.

### Synchronisation

The controller uses two locks:

| Lock | Protects | Reason |
| --- | --- | --- |
| `data_lock` | Latest front/rear frame slots | Camera tasks can update frames without touching command state. |
| `state_lock` | Steering, acceleration, and perception snapshots | Control output can read commands separately from camera I/O. |

`processing_task` snapshots frame references under `data_lock`, releases the lock, then runs the OpenCV pipeline lock-free. `send_controls_task` uses `state_lock.acquire(blocking=False)`; if the processing task is mid-write, it reuses the last successfully sent command to preserve the 50 Hz output deadline.

---

## Repository Layout

```text
DAMM_RTSE_ASGMT2/
|-- README.md                 - this file
|-- sample_drive.py           - runnable entrypoint, RT scheduling, TCP I/O, locks, task wiring
|-- control_policy.py         - steering/throttle decisions and manoeuvre latches
|-- perception_pipeline.py    - perception call order and data bundle for the controller
|-- display_overlay.py        - OpenCV perception and lane-curve debug windows
|-- image_detection.py        - raw OpenCV perception algorithms
|-- test_communication.py     - WASD manual-control reference client
|-- requirements.txt          - Python dependencies
|-- task1.md                  - planned confidence-gating/temporal-filtering task
|-- plan.md                   - design notes and tuning history
|-- imagedetection.md         - perception improvement notes
|-- game rule/
|   |-- RTSE_Poster_game.pdf  - assignment/game rule poster
|   `-- police.jpg            - police colour reference image
|-- screenshot/               - runtime perception screenshots
|-- video/                    - captured frame sequences used for tuning
|-- SpeedTrials2D/            - current Unity simulator build
|-- SpeedTrials2D_v1/         - older simulator build
`-- SpeedTrials2D_v2/         - alternate simulator build
```

---

## How to Run

**Prerequisites:** Python 3.10+ on Windows.

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the simulator
.\SpeedTrials2D\SpeedTrials2D.exe

# 3. In a separate terminal, launch the autonomous driver
python sample_drive.py
```

The driver opens these OpenCV windows:

| Window | Purpose |
| --- | --- |
| `Front Camera` | Raw front-camera stream from the skeleton reader. |
| `Back Camera` | Raw rear-camera stream from the skeleton reader. |
| `Perception` | Front/rear perception overlay, detected objects, nearest-orb marker, and command HUD. |
| `Lane Curve` | Bird's-eye lane-curve debug panel. |

Example HUD:

```text
target=80 eff=80 police=0 events=['NEAR:G d=92']
str=+0.54 acc=+0.80
```

`events` can include `POLICE_PRIORITY`, `POLICE->GRAB_RED`, `CHASING_CAR_PRIORITY`, `HILL`, and `NEAR:<colour> d=<distance>`.

Runtime lane numbering can be checked in `logs/lane_grid.log`. It records whether
the `1..5` lane grid was found, the current `car_lane`, lane centre offsets, and
the current Golden Lane source/lane. The `conf=` value is lowered on curves,
edge jumps, or held edge estimates; low confidence causes Golden Lane token
inference to ignore uncertain lane assignments and softens hard lane steering.

Press `Ctrl+C` in the terminal to shut down cleanly.

### Manual-Control Fallback

To test the communication path without the autonomous controller:

```powershell
python test_communication.py
```

Controls: `W/S` accelerate/brake, `A/D` steer, `Q` quit.

---

## Editable vs Locked Code Regions

Per the assignment skeleton, the following sections of `sample_drive.py` are treated as locked:

- `Configuration` constants
- `Real-Time Scheduling Framework` (`TaskPriority`, `RTTask`)
- `Network Connection Setup` (`setup_cameras`, `setup_control_server`)
- Body of `read_single_camera`
- `__main__` shutdown block

Our implementation lives in the editable regions and supporting modules:

- `shared_data` key extensions for frame slots, control commands, and perception snapshots
- `image_detection.py` for raw perception algorithms
- `control_policy.py` for autonomous driving decisions
- `perception_pipeline.py` for perception orchestration
- `display_overlay.py` for UI/debug rendering
- `processing_task` and `send_controls_task` integration in `sample_drive.py`
- `RTTask(...)` period and priority arguments

---

## 4-Person Ownership Split

| Owner | Main files | Responsibility |
| --- | --- | --- |
| Person 1 | `sample_drive.py` | RT scheduling, task periods/priorities, sockets, locks, shared-state integration, and `send_controls_task`. |
| Person 2 | `image_detection.py` | Raw OpenCV algorithms: HSV masks, orb detection, police/chasing-car detection, lane, hill, and brightness detection. |
| Person 3 | `perception_pipeline.py`, `display_overlay.py` | Perception orchestration, HUD packaging, and OpenCV debug windows. |
| Person 4 | `control_policy.py` | Steering/throttle policy, police handling, chasing-car swerve, nearest-orb priority, and manoeuvre latches. |

---

## Key Engineering Decisions

- **Pure perception model** - no duplicated game-state simulation. The controller reacts only to camera evidence.
- **Low-risk module split** - `sample_drive.py` remains the assignment entrypoint while editable logic is divided by ownership.
- **Pinned HSV ranges** - measured game-asset colours are more stable than live auto-calibration for this simulator.
- **Road-gated orb detection** - colour alone is not trusted; token contours must be plausible on-road blobs.
- **Imminence-first steering** - bird's-eye distance lets the nearest dangerous object override simple colour priority.
- **Committed manoeuvre latches** - red avoidance, green seeking, and chasing-car lane changes hold long enough to complete a lane transition despite momentary detection flicker.
- **Front-camera police handling** - the current V2.0 police challenge is handled as a front obstacle, not a rear event.
- **Low-light reverse recovery** - dim frames cause a straight reverse command instead of merely slowing down.
- **Split locks and non-blocking control send** - reduces priority-inversion risk and keeps actuator output periodic.

### Known Limitations

- **Golden Lane countdown is not read; the lock is a fixed 5 s.** `detect_golden_lane()` reads the lane number from the banner, but the `(Xs)` countdown OCR is unreliable, so the controller cannot know the true remaining time. It instead holds the lane for a fixed `GOLDEN_LANE_LOCK_S = 5.0 s` anchored to the first detection. Because detection can lag the real event start, the lock may end slightly before or after the game's actual window. A future main-window OCR path could supply the true deadline.
- **Token-inference fallback can be blocked by yellow-token debuffs.** If the banner lane digit is unread and the car hits a yellow token before/during Golden Lane, the game can apply a negative camera effect where tokens disappear or appear white/unknown. During that window the front camera may lack green-token evidence for `infer_golden_lane_from_tokens()`, so the fallback trigger can be delayed or missed. Once any lane is latched, though, the 5 s lock holds the manoeuvre through such flicker.
