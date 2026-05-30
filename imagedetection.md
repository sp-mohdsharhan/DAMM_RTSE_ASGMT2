# Image Detection — Observations & Improvement Plan

Derived from the eight runtime captures in `screenshot/`. The detection logic itself lives in `image_detection.py` (extracted from `sample_drive.py` so it can be evolved in isolation).

---

## 1. Evidence captured

| File | View | Notable content | What it tells us |
| --- | --- | --- | --- |
| `Screenshot 2026-05-29 234925.jpg` | **Perception window** (front+rear+HUD) | HUD `target=90 eff=45 police=1 events=[] str=-0.29 acc=+0.45`. Front: a `GREEN 2.3%` bbox is drawn over a pink/red orb on the left. Rear: a tiny `POLICE 0.1%` bbox on a small blue post next to the road, and a `CAR 0.x%` cyan bbox on a far yellow object. | Two failure modes visible at once: **(a)** GREEN classifier is firing on a red-leaning orb (HSV bucket overlap / calibration drift), **(b)** rear detector keeps drawing a sub-1% "POLICE" contour on what is a background prop. |
| `Screenshot 2026-05-30 064833.jpg` | Front Camera (raw) | 3-lane straight, multiple orbs ahead, multiplier badge `0.50x`, distance `7827 m`. | Baseline scene — confirms the standard front layout the perception ROI must cover. |
| `Screenshot 2026-05-30 064924.jpg` | Front Camera | Orb cluster mid-distance + a large yellow coin sitting in our lane. | The yellow coin would trigger YELLOW avoidance — confirms the avoidance band logic matters. |
| `Screenshot 2026-05-30 065008.jpg` | Front Camera | **Uphill crest** — orbs ride above the flat-horizon line; a row of **red chevron arrow signs** lines the right shoulder. | Two known stressors in one frame: (a) the ROI top must keep enough headroom for uphill orbs, (b) shoulder chevrons are red and roughly bounded — a shape filter must reject them. |
| `Screenshot 2026-05-30 065027.jpg` | Front Camera | Three orbs (red/green/red) bunched together; more red chevrons to the right. | **Single-largest-contour** picks one of the three — the others vanish from the controller's view. |
| `Screenshot 2026-05-30 065044.jpg` | Front Camera | Big yellow coin slightly off-centre; many red chevrons left shoulder; row of orbs at horizon. | Lane-edge red chevrons are bright, near-circular at distance — circularity+fill filters are the only thing stopping false RED. |
| `Screenshot 2026-05-30 065116.jpg` | Front Camera | A big yellow coin centred in lane; row of green orbs at horizon. | Yellow is dead-centre → should fire YELLOW avoidance; small distant greens are below `GREEN_ATTRACT_MIN_AREA`, correctly ignored. |
| `Screenshot 2026-05-30 065131.jpg` | Front Camera | Dense row of 5 alternating red/green orbs at mid-distance. | Demonstrates why **multi-candidate detection** matters — picking the largest red may ignore a closer red that's actually in our path. |

---

## 2. Confirmed defects

### 2.1 Rear detector draws sub-threshold POLICE bboxes
**Evidence:** Screenshot 1 shows `POLICE 0.1%` labelled on a tiny blue background sprite.

**Why it happens:** `_largest_contour_info` returns the largest shape that passes the ORB shape gates. The overlay draws *every* returned `info`. With the current pure-perception controller, `police_seen = rear_per['police']['present']` is set frame-by-frame and gates entry into police-seek mode — a sub-1% false positive currently keeps `present` at `False` (the floor is `> 0.01`), so steering behaviour is correct, but the **HUD still shows a misleading box** and the floor is fragile.

**Fix:** Two layers —
1. Add a per-color `MIN_PRESENT_AREA_FRAC` and refuse to *return* `info` below it (so the overlay never draws it).
2. In the rear pipeline, raise the police floor: `POLICE_MIN_AREA_FRAC = 0.015` and `POLICE_MIN_AREA_PX = 200` — by the time a police car is actually a threat it occupies more than 1.5 % of the frame; sub-1 % blue is always background.

> **Note:** Earlier drafts of this document discussed `police_active=True` being set by a software-side yellow event. That code path has been removed (see `plan.md` Phase 3 postscript). Police state is now purely a function of what the rear camera shows this frame.

### 2.2 GREEN bbox covering a pink/red orb
**Evidence:** Screenshot 1 has `GREEN 2.3%` over a clearly pink/red orb.

**Why it happens:** HSV auto-calibration (`calibrate_step` in `image_detection.py`) runs k-means on saturated pixels and assigns cluster centres to colour buckets by hue alone. If the calibration window catches a sprite at a transition hue (pinkish-red bleeding into orange-yellow into green at the antialiased orb edge), the GREEN bucket can absorb hue centres that don't actually correspond to true green pixels. The mean ± `HSV_MARGIN` then over-widens.

**Fix:** Three combined changes —
1. **Validate cluster purity** before accepting a centre: require its samples to be ≥80% within the canonical bucket's hue range (reject ambiguous edge clusters).
2. **Tighten `HSV_MARGIN`** from `[10,60,60]` to `[6,40,40]` so a single bad centre poisons less of HSV space.
3. **Post-hoc bbox confirmation:** after `_largest_contour_info` returns, sample the central 3×3 pixels of the bbox and confirm the mean hue is still inside the active range; reject otherwise.

### 2.3 Rear ROI is the whole frame
**Evidence:** `detect_rear` calls `cv2.resize(frame, (PROC_W, PROC_H))` and runs detection on the full image — sky included. Night sky in the game is dark blue, which sits at the edge of the POLICE HSV bucket. Screenshot 1's `POLICE 0.1%` false positive is consistent with a sky/silhouette blob slipping past.

**Fix:** Mirror the front cropping —
```python
REAR_ROI_TOP_FRAC = 0.25     # crop sky
REAR_ROI_SIDE_FRAC = 0.10    # crop trees / buildings on shoulders
```
Then build `hsv` from the cropped ROI and pass `roi_x0` through `_largest_contour_info` (same pattern as `detect_front_objects`).

### 2.4 Single-largest-contour discards multi-orb scenes
**Evidence:** Screenshots 5, 7, 8 each show 3–5 same-colour orbs in view; we keep only one.

**Why it matters:** Lane-change decisions depend on which red is in our *path*, not which red has the biggest bbox. A larger but lane-clear red can cause us to ignore a smaller red dead ahead.

**Fix:** Have `_largest_contour_info` become `_orb_candidates_info` returning up to `K=5` per colour, sorted by area. The controller then picks the "most threatening" red (largest `area_frac` where `|centroid_x_norm| < RED_AVOID_BAND_FRAC`) and the most-attractive green (smallest `|centroid_x_norm|` above `GREEN_ATTRACT_MIN_AREA`).

### 2.5 No temporal stabilisation
**Evidence:** Not directly visible in stills, but the 0.1% rear police box appearing in only one frame is exactly the symptom — a one-frame false positive.

**Fix:** Add a 3-frame ring buffer per colour. Only accept a detection if it appeared in ≥2 of the last 3 frames (IoU > 0.3 across frames). Cheap, dramatic FP reduction.

### 2.6 Uphill horizon clipping
**Evidence:** Screenshot 4 (065008) — orbs sit *above* where the road meets the sky on flat sections.

**Current mitigation:** `FRONT_ROI_TOP_FRAC = 0.28` (down from 0.40) gives headroom for uphills but admits more sky on flats.

**Fix:** Estimate the lane vanishing-point each frame from `detect_lane_offset`'s line set; set `roi_y0` dynamically `max(0.18 * PROC_H, vp_y - 20px)`. Caps headroom on flats, expands it on uphills.

---

## 3. Hardening backlog (prioritised)

### Tier A — low-risk, high-impact (do these first)

| # | Change | File / function | Expected effect |
| --- | --- | --- | --- |
| A1 | Per-colour `MIN_PRESENT_AREA_FRAC` + suppress draw below it | `_largest_contour_info`, `_draw_obj` | Removes spurious sub-1% boxes from HUD; cleaner debug |
| A2 | Add `REAR_ROI_TOP_FRAC = 0.25` + `REAR_ROI_SIDE_FRAC = 0.10` | `detect_rear` | Kills sky-blue POLICE false positives |
| A3 | Multi-candidate per colour (top-K=5) + path-aware controller selection | `_largest_contour_info` → `_orb_candidates_info`; `_compute_steering` in `sample_drive.py` | Correct lane decisions in cluster scenes (screenshots 5/7/8) |
| A4 | 3-frame temporal vote (IoU ≥ 0.3, ≥2 of 3 frames) | new `_TemporalFilter` class wrapping detectors | ~3× FP reduction without raising thresholds |
| A5 | Post-hoc bbox-centre hue validation | end of `_largest_contour_info` | Fixes GREEN-over-red labelling (screenshot 1) |

### Tier B — calibration robustness

| # | Change | File / function |
| --- | --- | --- |
| B1 | Tighten `HSV_MARGIN` to `[6,40,40]` and require cluster purity ≥80% before accepting a centre | `_finalize_calibration`, `calibrate_step` |
| B2 | Restrict calibration sampling to **lower 60% of front ROI** (road region, away from sky/grass) | `calibrate_step` |
| B3 | Extend calibration if no samples collected for a bucket after 90 frames (instead of silently keeping defaults) | `_finalize_calibration` |
| B4 | Continuous re-calibration: every N=600 frames, run one calibration pass and blend new ranges with current (`α=0.3`) — handles lighting drift across tracks | new `recalibrate_blend()` |

### Tier C — shape filter refinements

| # | Change | File / function |
| --- | --- | --- |
| C1 | Add **convexity** check: `area / convexHull(area) ≥ 0.85` — rejects chevron signs and stars | `_largest_contour_info` |
| C2 | Add **min-enclosing-circle ratio**: `area / (π·r²) ≥ 0.70` — stricter circularity than the perimeter-based one | `_largest_contour_info` |
| C3 | Optional secondary verification via `cv2.HoughCircles` on the bbox patch when `area_frac` is close to the avoidance threshold (marginal cases only — too slow for every contour) | new `_verify_circle(roi_patch)` |

### Tier D — instrumentation

| # | Change | File / function |
| --- | --- | --- |
| D1 | `IMAGE_DETECTION_DEBUG = True` toggle that dumps per-frame masks + per-contour shape-filter rejection reasons to `debug/`, throttled to 1 frame/s | new `_debug_dump()` |
| D2 | JSONL telemetry: `(t, front_candidates, rear_candidates, steer, accel)` for offline A/B replay | new `_log_jsonl()` (called from `processing_task`) |
| D3 | Stats counters: per-colour detections / rejections per filter stage, printed on `Ctrl+C` shutdown | module-level `_stats` dict |

### Tier E — exclusion masks for known HUD regions

Pre-build a static binary mask that zeroes out:
- Top-left score badge region (`[0:18%, 0:14%]`) — kills numeric digit false detections.
- Top-right multiplier/distance badge (`[0:18%, 86:100%]`) — `0.50x` is red text.
- Lower car silhouette region in front (`[88:100%, 32:68%]`) — own car bumper.

Apply once after `cv2.resize` and before HSV conversion. Cheap and bulletproof for the locked game layout.

---

## 4. Quick validation procedure for any change

For each Tier A change in isolation:

1. Start the game and `python sample_drive.py`.
2. Drive **3 full laps**.
3. Record `(distance, red_hits, green_hits, yellow_hits, police_spawns)` from the in-game score panel.
4. Compare to the current baseline (see `plan.md` § "Tuning iteration history").
5. **Keep the change only if:** `distance ↑` AND `red_hits` does not increase.

Pair the screenshots in this doc with mask dumps from D1 before/after — visual confirmation that the targeted false positive class is actually gone is faster than relying on aggregated lap stats alone.

---

## 5. Relevant files

- `image_detection.py` — module that owns detection, calibration, overlay
- `sample_drive.py` — controller, event engine, RT scheduling; imports from `image_detection`
- `plan.md` — broader project plan + tuning log (this doc focuses specifically on perception)
- `screenshot/` — source evidence
