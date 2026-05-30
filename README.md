# DAMM_RTSE_ASGMT2 — SpeedTrials2D Autonomous Driver

> **SECJ 4423 — REAL-TIME SOFTWARE ENGINEERING**
> **Semester II 2025/2026 — Group Assignment 2**
> **Lecturer:** Prof. Ts. Dr. Dayang Norhayati Bte. Abang Jawawi
> **Group:** DAMM

## Group Members

| Name | Matric No. |
| --- | --- |
| Auni Dalilah Binti Mohd Zain | SX170101CSJS04 |
| Siti Dzin Norsyafika Binti Mohd Isa | SX220330ECJHS04 |
| Muhammad Dzul Ifraan Bin Ab Rahman | SX231715ECJHF04 |
| Mohd Sharhan Bin Abdul Ghani | SX232315ECJHF04 |

---

## Project Overview

This project implements an **autonomous driving controller** for the **SpeedTrials2D** competition game, built around the four pillars of real-time software engineering:

1. **Concurrency** — multiple cooperating threads (camera readers, perception, control sender)
2. **Task Periods** — deterministic execution rates per task (each period set to its deadline)
3. **Task Priorities** — Windows thread priorities tuned to deadline criticality (Deadline-Monotonic)
4. **Shared-Resource Synchronisation** — split locks (`data_lock` for frames, `state_lock` for control state) to avoid priority inversion

The car perceives the world through **two virtual cameras** (front + rear) streamed over TCP, runs an OpenCV vision pipeline, and sends `(steering, acceleration)` commands back to the simulator at 50 Hz.

Per the official poster (`game rule/RTSE_Poster_game.pdf`), the **Unity simulator is the authoritative state machine** — it owns score, speed, and event resolution. Our controller therefore keeps no shadow simulation; it reacts to what the cameras show, frame by frame.

---

## Game Rules (per poster)

**Tokens (front-camera observations)**

| Colour | Game effect | Our steering reaction |
| --- | --- | --- |
| Green | +10 % speed | **Attract** — gentle P-controller toward centroid |
| Red | −20 % speed | **Avoid** — hard lane-change away (inverted to **seek** in police mode) |
| Yellow | Random 1-of-5 corruption (next token hidden, tokens invisible 5 s, camera input delay 5 s, action output delay 5 s, corrupted camera input 5 s) | Avoid only if dead-centre — most yellow effects are temporary input corruption, not catastrophic |

**Events (poster-listed, all observed directly from camera)**

| Event | Signal | Reaction |
| --- | --- | --- |
| **Trailing car** | Growing high-saturation contour in rear cam | 1.5 s defensive swerve, direction alternates each trigger |
| **Police** | Blue contour in rear cam | Flip red behaviour to **seek the next red token** (poster: "catch next red or −50 % speed") |
| **Low brightness** | Mean V channel of front frame falls below threshold | Reduce throttle (`0.8 → 0.4`) — token visibility is degraded |

Our software does **not** track score, speed, hit cooldowns, or yellow-event outcomes — those are the game's job.

---

## Real-Time Architecture

### Tasks (Deadline-Monotonic schedule)

| Task | Period | Priority | Role |
| --- | --- | --- | --- |
| `ReadFrontCamera` | 5 ms | HIGH | TCP frame decode — collision-critical input |
| `SendControls` | 20 ms | HIGH | Non-blocking send of `(steering, accel)` (50 Hz) — hard actuator deadline |
| `Processing` | 33 ms | MEDIUM | OpenCV perception + steering decision (~30 Hz) — the "brain" |
| `ReadBackCamera` | 50 ms | LOW | TCP frame decode; rear threats (police / trailing car) evolve slowly |

Priority tracks **deadline criticality**, not raw period (Deadline-Monotonic). The rear
camera runs at 20 Hz so its `LOW` priority is the *longest-deadline* task — keeping the
schedule coherent (longest deadline → lowest priority) and avoiding priority inversion on
`data_lock` against the `HIGH` front camera. Camera reads call the skeleton's
`read_*_camera_task` directly — input corruption / delay is the game's responsibility,
so we do not throttle our own reads on top of it.

### Steering Priority Stack (highest wins)
1. **Police-seek** — if rear cam sees blue **and** front cam has a red, steer toward the red (poster: "catch next red or −50 % speed")
2. **Red avoid swerve** — early-trigger (`area ≥ 1.0 %`, wide band 70 %), 1.6 s commit + 0.35 s counter-steer settle, direction-flip if re-threatened
3. **Trailing-car forced swerve** — single timer armed when rear cam shows a growing high-saturation contour; alternates direction each trigger
4. **Yellow avoid** — only if centred (`area ≥ 2.0 %`, band 55 %)
5. **Green attract** — P-controller on centroid offset
6. **Lane follow** — Canny + HoughLinesP on bottom ROI
7. Default `0.0`

Throttle is a one-line policy: `LOW_BRIGHTNESS_THROTTLE (0.4)` if the front frame is dim, otherwise `CRUISE_THROTTLE (0.8)`.

### Perception Pipeline (`image_detection.py`)
- **HSV auto-calibration** via k-means clustering over the first ~3 s of front-cam frames (no hand-tuned colour constants)
- **Orb shape gate** — area / aspect-ratio / circularity (`4πA/P² ≥ 0.60`) / fill-ratio (`≥ 0.65`) filters reject lane markings, grass, sky
- **ROI tightening** — top `0.28`, sides `0.15` to handle uphill orbs while ignoring shoulders
- **Low-brightness detector** — mean V channel on a centre crop (HUD-bias-free)
- **Frame-snapshot pattern** — `processing_task` grabs frame refs under `data_lock`, releases, then runs the ~10–20 ms CV pipeline lock-free

---

## Repository Layout

```
DAMM_RTSE_ASGMT2/
├── README.md                — this file
├── plan.md                  — design plan + tuning iteration log (historical, see Phase 3 postscript)
├── imagedetection.md        — perception improvement backlog (grounded in screenshot/)
├── requirements.txt         — opencv-python, numpy, keyboard
├── sample_drive.py          — controller + RT scheduling (editable + locked sections)
├── image_detection.py       — OpenCV perception module (HSV calib, contour filters, lane, brightness, overlay)
├── test_communication.py    — WASD manual-control reference client
├── SpeedTrials2D/           — Unity build of the competition game
│   └── SpeedTrials2D.exe
├── game rule/
│   └── RTSE_Poster_game.pdf — official rules poster (authoritative spec)
└── screenshot/              — runtime captures of the Perception HUD
```

---

## How to Run

**Prerequisites:** Python 3.10+, Windows (game is a Unity Windows build).

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the simulator (opens TCP camera + control sockets)
.\SpeedTrials2D\SpeedTrials2D.exe

# 3. In a separate terminal, launch the autonomous driver
python sample_drive.py
```

A `Perception` window opens showing the front/rear view, detected orbs, lane lines, and a HUD line:

```
target=80 eff=80 police=0 events=[] str=0.00 acc=0.80
```

`target` / `eff` are the chosen throttle × 100 (`CRUISE_THROTTLE = 0.8`, dropping to `0.4` under low brightness). `events` lists what perception is currently reacting to: `TRAILING_CAR`, `POLICE`, `LOW_LIGHT`.

Press `Ctrl+C` in the terminal to shut down cleanly.

### Manual-control fallback

To drive with the keyboard for testing the communication path:

```powershell
python test_communication.py
```

Controls: `W/S` accelerate/brake, `A/D` steer, `Q` quit.

---

## Editable vs Locked Code Regions

Per the assignment skeleton, the following sections of `sample_drive.py` are **locked** and must not be modified:
- `Configuration` constants
- `Real-Time Scheduling Framework` (`TaskPriority`, `RTTask`)
- `Network Connection Setup` (`setup_cameras`, `setup_control_server`)
- Body of `read_single_camera`
- `__main__` shutdown block

Our work lives in the **editable** regions:
- `shared_data` key extensions (kept minimal — just frame slots + control commands + perception snapshots)
- New module `image_detection.py` containing all perception
- Body of `processing_task` and `send_controls_task` in `sample_drive.py`
- Period / priority arguments inside `RTTask(...)` constructors

---

## Key Engineering Decisions

- **Pure-perception model** — no shadow simulation of game state (no software-tracked score, speed, cooldowns, or yellow-event outcomes). The Unity game is authoritative; the controller only reacts to what cameras show this frame. Removed the previous event-engine layer (~80 lines) once the official poster confirmed this is the intended model.
- **Perception in its own module** — `image_detection.py` owns HSV calibration, contour shape filters, lane offset, brightness check, and overlay rendering, so the controller stays focused on RT concerns.
- **Split locks** — `data_lock` scoped to frame slots only; `state_lock` protects command writes. Camera threads never block on control state.
- **Non-blocking control fallback** — `send_controls_task` uses `state_lock.acquire(blocking=False)`; if busy, it re-sends the last `(steering, accel)`. Classic RT priority-inversion mitigation that keeps the 50 Hz deadline solid.
- **Monotonic clocks** — `time.monotonic()` for the trailing-car swerve timer and red-avoid latch (skeleton's `time.time()` left untouched).
- **One perception window at 30 Hz** instead of the skeleton's per-camera `imshow` at 200 Hz — cuts `cv2.waitKey` overhead from ~400 ms/s to ~30 ms/s.

See [`plan.md`](./plan.md) for design history (note: phases 1–2 describe the previous shadow-state design — superseded by the Phase 3 postscript), and [`imagedetection.md`](./imagedetection.md) for the perception improvement backlog grounded in `screenshot/`.
