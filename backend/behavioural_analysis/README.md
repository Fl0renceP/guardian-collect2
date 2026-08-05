# Behavioural Risk Analysis

A second, independent signal alongside facial recognition and licence-plate
recognition. It watches **how people move**, not **who they are**.

---

## The one-minute version

Facial recognition fails in two directions, and each failure hurts someone
different:

| Failure | What happens | Who pays |
|---|---|---|
| **False positive** | An innocent person is matched to an offender record | Someone who did nothing |
| **False negative** | A real offender is missed — face covered, poor light, or simply not in the database | The household |

Behaviour is an independent check on both:

- **Face matched, but the person is behaving completely normally?**
  The score is pushed **down**. A match alone is not a reason to escalate
  someone who is doing nothing unusual.
- **No face match at all, but the behaviour is unusual?**
  It still reaches a human. This is the case facial recognition *cannot* cover.

Both signals are combined into one **composite risk score**. If that score
crosses a threshold, the event goes into a **queue for a person to review**.

**Nothing else happens.** No alert to police. No "offender" label. No automatic
anything. The only decision this module can make is *whether a human should
look*.

---

## Quick start

```bash
cd guardian-collect2/backend/behavioural_analysis

# One-time: install the two extra dependencies
pip install -r requirements-behavioural.txt

# The demo — opens a window showing boxes, skeletons, zones and live scores
python run_sample.py --clip 2

# Any video, with the real (not demo) thresholds
python main.py --source "path/to/clip.mp4" --show

# A webcam
python main.py --source 0 --show

# Prove the logic without any video (31 tests, a few seconds)
python tests/test_heuristics.py
```

On first run it downloads two model files (~12 MB total): YOLOv8-nano and the
MediaPipe pose bundle. After that it works offline.

**Speed.** On a laptop with no GPU, YOLO manages roughly 1–3 frames per second.
`config.yaml` therefore processes every 3rd frame by default. Timestamps come
from the video rather than the wall clock, so skipping frames costs temporal
resolution but **never changes what a threshold means** — a 45-second dwell is
45 seconds of footage whether the laptop keeps up or not.

### Two config files

| File | Use |
|---|---|
| `config.yaml` | **The real one.** Thresholds sized for a real camera — 45s before loitering is even considered. |
| `config.demo.yaml` | For the 18–33 second sample clips. Same structure, same safeguards, shorter *durations* so the mechanism is visible inside a short clip. Every shortened value is annotated with its real counterpart. |

```bash
python main.py --source "../behaviour analysis/<clip>.mp4" --config config.demo.yaml --show
```

---

## What it detects

Eight behaviours, each a separate function in `heuristics.py` with its own
thresholds and its own plain-English explanation:

| # | Behaviour | What it actually measures |
|---|---|---|
| 1 | **Loitering / casing** | Dwelling near a boundary, or repeatedly passing it |
| 2 | **Perimeter probing** | Approaching a gate, *stopping*, sometimes reaching toward it |
| 3 | **Climbing posture** | Hands above shoulders at a wall, with a raised knee, asymmetric reach, or feet leaving the ground |
| 4 | **Concealment approach** | Face not visible to the camera *while closing on a property* |
| 5 | **Crouched at a vehicle** | Sustained low posture beside a car |
| 6 | **Tampering motion** | Repetitive short-range hand movement at a vehicle, body otherwise still |
| 7 | **Group coordination** | One person static at a distance while another is active at the target |
| 8 | **Fleeing** | An abrupt change from walking to running, away from the target |

Every one returns `(triggered, confidence, explanation)`. The explanation is
**required** — there are no opaque scores anywhere in this module.

A real example, straight from a run:

> *Track held a crouched posture — 62% of their own standing height, threshold
> 72% — beside a car for 9.4s, rather than passing at walking height.*

---

## How the composite score works

### Step 1 — combine the behaviours (`noisy-OR`)

```
B = 1 − Π(1 − weightᵢ × confidenceᵢ)
```

Two moderate signals add up to more than either alone, the total can never
exceed 1, and **no single heuristic can reach 1.0 by itself** — its weight caps
it. One behaviour is never enough on its own.

### Step 2 — hot-spot context

```
B' = B × (1 + zone_weight × zone_risk)
```

A suburb's claims history can **amplify** evidence but never **create** it: at
`B = 0` the lift is zero no matter how bad the area's history. Treating people
as more suspicious for where they are is redlining, and in South Africa that
maps onto apartheid spatial geography almost exactly.

### Step 3 — fuse with the face module

```
composite = 0.50 × behaviour  +  0.35 × face  +  0.15 × (behaviour × face)
```

Half behaviour, about a third face, and a bonus when the two **agree** — two
independent signals pointing the same way are worth more than either alone.

### Step 4 — the rules that are the point of the whole module

| Situation | What happens | Why |
|---|---|---|
| Confident face match, **ordinary** behaviour | composite × **0.60** | The false-positive brake |
| Face matched as a **known resident** | composite × **0.35** | Someone at their own car is not an event |
| Strong behaviour, **no face match** | **review anyway** | The false-negative catch |

### Step 5 — the only decision

```
requires_human_review = composite ≥ 0.50
```

Every step above appends a sentence to a `reasoning` list that ships inside the
event, so the arithmetic can be read back in English. Every number lives in
`config.yaml`.

---

## Defining zones

Zones are polygons drawn on the camera's view, in **normalised coordinates**
(0–1), so one config survives a resolution change.

The easiest way is to trace them on a real frame:

```bash
python tools/draw_zones.py --source "path/to/your/clip.mp4"
```

Click the corners, press `ENTER` to close a zone, `t` to change its type, `s` to
save. It prints a block you paste straight into `config.yaml`.

| Zone type | Meaning |
|---|---|
| `property_boundary` | The area someone should walk *past*, not into |
| `gate` | Gate, door or window — the probing target |
| `vertical_structure` | Wall or fence — the climbing target |
| `vehicle_zone` | Driveway or parking bay |
| `street_frontage` | Public pavement outside the property |
| `exempt` | **Somewhere it is normal to stand still** — a taxi rank, a bus stop, a bench. Loitering is *never* evaluated here. |

Add an `exempt` zone anywhere your camera overlooks a place people legitimately
wait. It is the single most effective bias mitigation available to whoever
configures this.

---

## Responsible-AI design

This module flags people for how they move. Every system that has ever done that
has, left unchecked, ended up paying disproportionate attention to people who
were doing nothing wrong. What has been done about it:

**No autonomous action, ever.** The only output verb is
`requires_human_review`. There is no alert function, no `offender` or `suspect`
label, and no path into `services/alerts_service.py`. A behavioural flag is a
request to look, not an identification.

**No identity data.** The module works on anonymous per-run track IDs and pose
geometry. It receives one number from the face module — a confidence — plus a
coarse label, and never a name, ID, embedding or image. Track IDs map to no
person, and are discarded when the person leaves the frame.

**Every trigger explains itself.** No score is emitted without a sentence a
reviewer can read and disagree with.

**Full audit trail.** Every event is written to SQLite *and* JSONL with the
numbers it was computed from and the thresholds in force, so a decision can be
reconstructed weeks later without the footage. Raw pose skeletons are **not**
stored by default — the audit needs the numbers behind the decision, not the
body geometry.

**Posture is self-referenced.** "Crouched" means *lower than this person's own
standing height*, learned from their own first upright frames — never a
population average. A wheelchair user, a short person and someone with a stoop
are each compared only to themselves. There is no population norm anywhere in
the codebase.

**Thresholds are configuration, not code.** `heuristics.py` contains no magic
numbers. Every value is in `config.yaml`, annotated with why it is what it is,
so it can be reviewed and argued about by someone who does not read Python.

**The known bias risks are written down.** The top of `heuristics.py` carries a
notice covering loitering vs. disability and ordinary waiting, gait differences,
the acute risk of the concealment heuristic, running as a neutral act, and
hot-spot redlining — each with the specific mitigation applied. Read it before
changing a threshold.

**`--explain` shows the misses.** Running with `--explain` reports why each
heuristic did *not* fire ("dwell was 3.4s against a 45s threshold"). Being able
to see what the system nearly did is what makes tuning honest.

---

## Files

| File | Stage |
|---|---|
| `frame_ingest.py` | 1. Video/webcam → frames (timestamps from the video, not the clock) |
| `detector.py` | 2. YOLOv8 + ByteTrack → people and vehicles with persistent IDs |
| `pose_extractor.py` | 3. MediaPipe Pose → 33 body landmarks per person |
| `trajectory_tracker.py` | 4. Per-track history: dwell, speed, zone proximity |
| `heuristics.py` | 5. The eight detectors **← the bias notice lives here** |
| `risk_fusion.py` | 6. Noisy-OR, hot-spot lift, fusion with the face module |
| `api_output.py` | 7. JSON schema + `push_to_flask_api()` stub |
| `main.py` | 8. CLI |
| `zones.py` | Polygon geometry |
| `audit_log.py` | SQLite + JSONL audit trail |
| `debug_overlay.py` | The OpenCV debug window |
| `pipeline.py` | Orchestration (importable — no CLI concerns) |
| `run_sample.py` | The demo script |
| `tools/draw_zones.py` | Trace zones onto a real frame |
| `tests/test_heuristics.py` | 31 tests on synthetic fixtures — no video needed |

### Everything is in body heights

There is no camera calibration, so pixels mean nothing across cameras: 50px is a
long stride on a street view and a hand twitch on a doorbell close-up. Every
spatial threshold is expressed in **multiples of the person's own bounding-box
height**, and every speed in **body heights per second** (a brisk walk ≈ 1.2, a
run ≈ 3+). That is what lets one config work on any camera.

---

## Output

```json
{
  "track_id": "person-4",
  "timestamp": "2026-08-05T14:22:07+00:00",
  "location_zone_id": "demo_camera_01",
  "behavioural_risk_score": 0.71,
  "triggered_heuristics": [
    {
      "type": "crouched_near_vehicle",
      "confidence": 0.82,
      "explanation": "Track held a crouched posture — 62% of their own standing height, threshold 72% — beside a car for 9.4s, rather than passing at walking height."
    }
  ],
  "face_match_confidence": null,
  "composite_risk_score": 0.58,
  "requires_human_review": true,
  "reasoning": ["...how the score was reached, step by step..."]
}
```

`push_to_flask_api(event_json)` in `api_output.py` marks where this would POST
to the Flask backend. **It is a stub** — it logs what it *would* send and posts
nothing. The docstring describes what the Flask side would do: persist to
PostGIS against the camera's zone, and route review-flagged events to a Crime
Prevention Unit **review queue** — not to the member alert feed, which requires
an `offender` label this module deliberately cannot produce.

---

## Extension point: a temporal model

The heuristics are rule-based on purpose — a hackathon has no labelled dataset
of people casing houses, and a rule that can be read and argued about beats a
black box that cannot.

The seam for a learned model is `heuristics.py`. `TrackHistory` already holds
exactly what a sequence model needs: pose keypoints and positions over a rolling
window, per track, in scale-invariant units. An LSTM or temporal transformer
over `track.window(seconds)` would drop in as a ninth entry in the `HEURISTICS`
registry, returning the same `(triggered, confidence, explanation)` tuple and
flowing through the same fusion and audit path.

Two constraints it would have to meet, both non-negotiable: it must still
produce a human-readable explanation, and it must not be trained on data that
lets it learn appearance instead of movement.
