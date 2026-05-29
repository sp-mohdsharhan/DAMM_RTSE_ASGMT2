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
2. **Task Periods** — deterministic execution rates per task (Rate-Monotonic compliant)
3. **Task Priorities** — Windows thread priorities tuned to deadline criticality
4. **Shared-Resource Synchronisation** — split locks (`data_lock` for frames, `state_lock` for control state) to avoid priority inversion

The car perceives the world through **two virtual cameras** (front + rear) streamed over TCP, runs an OpenCV vision pipeline, and sends `(steering, acceleration)` commands back to the simulator at 50 Hz.

---

## Game Rules Implemented

**Front camera — coloured orbs**

| Colour | Effect on hit | Steering role |
| --- | --- | --- |
| Red | `target_speed −= 20` | **Avoid** — hard lane-change away |
| Green | `target_speed += 10` | **Attract** — steer toward (score booster) |
| Yellow | Random event (1 of 5) | Avoid only if dead-centre |

**Yellow event pool:** front-cam degrade, back-cam degrade, `target_speed *= 0.95`, forced lane-change, **spawn police**.

**Rear camera — two threats**
- **Police (blue)** — `eff_speed *= 0.5` while present; cleared only by hitting the next front orb.
- **Other car** — growing rear contour triggers a forced lane-change (rear cars are 10 % faster, so we yield).

---

## Real-Time Architecture

### Tasks (Rate-Monotonic schedule)

| Task | Period | Priority | Role |
| --- | --- | --- | --- |
| `ReadFrontCamera` | 5 ms | HIGH | TCP frame decode (gated to match degrade events) |
| `ReadBackCamera` | 5 ms | HIGH | TCP frame decode (gated) |
| `Processing` | 33 ms | MEDIUM | OpenCV perception + steering decision (~30 Hz) |
| `SendControls` | 20 ms | HIGH | Non-blocking send of `(steering, accel)` (50 Hz) |

Shorter-period tasks get higher priority → **RM-correct**.

### Steering Priority Stack (highest wins)
1. **Red avoid swerve** — early-trigger (`area ≥ 1.0 %`, wide band 70 %), 1.6 s commit + 0.35 s counter-steer settle, direction-flip if re-threatened
2. **Force-lane-change** event (yellow-spawned)
3. **Yellow avoid** — only if centred (`area ≥ 2.0 %`, band 55 %)
4. **Green attract** — P-controller on centroid offset
5. **Lane follow** — Canny + HoughLinesP on bottom ROI
6. Default `0.0`

### Perception Pipeline
- **HSV auto-calibration** via k-means clustering over the first ~3 s of front-cam frames (no hand-tuned colour constants)
- **Orb shape gate** — area / aspect-ratio / circularity (`4πA/P² ≥ 0.60`) / fill-ratio (`≥ 0.65`) filters reject lane markings, grass, sky
- **ROI tightening** — top `0.28`, sides `0.15` to handle uphill orbs while ignoring shoulders
- **Frame-snapshot pattern** — `processing_task` grabs frame refs under `data_lock`, releases, then runs the ~10–20 ms CV pipeline lock-free

---

## Repository Layout

```
DAMM_RTSE_ASGMT2/
├── README.md                — this file
├── plan.md                  — full design plan + tuning iteration log
├── requirements.txt         — opencv-python, numpy, keyboard
├── sample_drive.py          — main controller (editable + locked sections)
├── test_communication.py    — WASD manual-control reference client
├── SpeedTrials2D/           — Unity build of the competition game
│   └── SpeedTrials2D.exe
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
target=70 eff=70 police=0 events=[] str=0.00 acc=0.50
```

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
- `shared_data` key extensions
- New constants, perception helpers, and event-engine functions
- Bodies of `processing_task` and `send_controls_task`
- Period / priority arguments inside `RTTask(...)` constructors

---

## Key Engineering Decisions

- **Split locks** — `data_lock` scoped to frame slots only; `state_lock` protects command/event state. Camera threads never block on control state.
- **Gated reads instead of mutable periods** — `RTTask.period` is fixed after construction, so cam-degrade yellow events are implemented by early-returning from the read function until the degraded period elapses.
- **Non-blocking control fallback** — `send_controls_task` uses `state_lock.acquire(blocking=False)`; if busy, it re-sends the last `(steering, accel)`. Classic RT priority-inversion mitigation that keeps the 50 Hz deadline solid.
- **Monotonic clocks** — `time.monotonic()` everywhere in our cooldown / expiry logic (skeleton's `time.time()` left untouched).
- **One perception window at 30 Hz** instead of the skeleton's per-camera `imshow` at 200 Hz — cuts `cv2.waitKey` overhead from ~400 ms/s to ~30 ms/s.

See [`plan.md`](./plan.md) for the full design rationale, tuning iteration log, and pending improvement backlog.
