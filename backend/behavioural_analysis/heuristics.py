"""Stage 5 — the eight behavioural heuristics.

===============================================================================
KNOWN BIAS RISK — READ BEFORE CHANGING ANY THRESHOLD
===============================================================================
This file decides when a person's movement is worth a human's attention. Every
kind of system that has ever done that has, left unchecked, ended up paying
disproportionate attention to people who were doing nothing wrong. The specific
risks here, and what has been done about each:

1. LOITERING PENALISES STANDING STILL, AND PEOPLE STAND STILL FOR ORDINARY
   REASONS. Waiting for a lift, a taxi or a delivery. Resting because walking is
   painful or tiring. A chronic condition, a pregnancy, a heavy bag, a phone
   call, a cigarette. A wheelchair user parked on a pavement is, to a pixel,
   indistinguishable from someone "dwelling near a property boundary".
   Mitigations: `exempt` zones (taxi rank, bus stop, bench) where loitering can
   never fire; deliberately long dwell thresholds; repeated passes weighted
   above a single long stay; and the lowest fusion weight of any heuristic.

2. GAIT AND POSTURE DIFFER BETWEEN BODIES. A limp, a crutch, a prosthesis, a
   stoop, a short stature or a wheelchair all change what "normal" posture and
   speed look like. A population-average threshold would flag disabled people
   constantly.
   Mitigation: every posture measure is SELF-REFERENCED. "Crouched" means lower
   than THIS person's own standing baseline, learned from their own first
   upright frames. Speeds are in that person's own body heights per second.
   There is no population norm anywhere in this file, and none may be added.

3. CONCEALMENT IS THE MOST DANGEROUS HEURISTIC IN THIS FILE. A covered or
   turned-away face is ordinary: winter clothing, sun, dust, illness, religious
   dress, or simply not knowing a camera was there. In South Africa a balaclava
   in a Highveld winter is a cold person. Read carelessly, this heuristic is a
   machine for flagging people for how they dress.
   Mitigations: it requires sustained approach toward a property as well, so a
   covered face alone can never trigger it; its explanation states what was
   actually observed (the camera cannot see a face) rather than an inference
   about intent; and it exists to mark "facial recognition cannot help here",
   not to accuse.

4. RUNNING IS NOT A CRIME. People run for exercise, for a taxi, from a dog,
   from danger — including danger this system knows nothing about. Someone
   fleeing an attacker moves exactly like someone fleeing a scene.
   Mitigation: low fusion weight, and the explanation always says what it is —
   a sudden change in pace — never "suspect fleeing".

5. HOT-SPOT CONTEXT CAN BECOME REDLINING. Lifting everyone's score because
   their suburb has claims history means people are treated as more suspicious
   for where they live. In the South African context that maps onto apartheid
   spatial geography almost exactly.
   Mitigation: `fusion.zone_weight` is capped low and is MULTIPLICATIVE on
   existing behavioural evidence — at zero behaviour the lift is zero, so
   context can amplify evidence but never manufacture it. Same reasoning as
   ROUTE_MIN_RISK_REDUCTION in backend/config.py.

STRUCTURAL SAFEGUARDS:
  * No threshold is hardcoded here. Every number comes from config.yaml, so it
    can be reviewed, tuned and argued about by someone who does not read Python.
  * Every trigger returns a human-readable explanation stating the observation
    and the threshold it crossed. No opaque scores.
  * Nothing in this module triggers an action. The only output verb is
    `requires_human_review`.
  * These are heuristics about MOVEMENT, not judgements about PEOPLE. The
    explanations are written in that register on purpose and should stay that
    way: "stood near the gate for 74s", never "behaved suspiciously".
===============================================================================
"""

from __future__ import annotations

import math
import statistics
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from pose_extractor import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from settings import Settings
from trajectory_tracker import SceneContext, TrackHistory


class HeuristicResult(NamedTuple):
    """(triggered, confidence, explanation) — plus the audit trail.

    The first three fields are the contract the module was specified against and
    unpack in that order. `inputs` carries the numbers the decision was made
    from, so an auditor can reconstruct it later without the video.
    """

    triggered: bool
    confidence: float
    explanation: str
    name: str = ""
    inputs: Dict[str, object] = {}

    def as_tuple(self) -> Tuple[bool, float, str]:
        return (self.triggered, self.confidence, self.explanation)


def _no(name: str, reason: str, **inputs) -> HeuristicResult:
    """A non-trigger. The reason is recorded — knowing why something did NOT
    fire is half of tuning a threshold."""
    return HeuristicResult(False, 0.0, reason, name, inputs)


def ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 between two thresholds. Confidence, not probability."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _slow_seconds(track: TrackHistory, max_speed: float, window: float) -> float:
    """Seconds in the recent window spent below `max_speed` (body heights/sec)."""
    observations = track.window(window)
    if len(observations) < 2:
        return 0.0
    scale = track.body_height
    if scale <= 0:
        return 0.0

    total = 0.0
    for previous, current in zip(observations, observations[1:]):
        elapsed = current.timestamp - previous.timestamp
        if elapsed <= 0:
            continue
        speed = (math.dist(previous.foot, current.foot) / scale) / elapsed
        if speed <= max_speed:
            total += elapsed
    return total


def _sustained_seconds(samples: List[Tuple[float, bool]]) -> float:
    """Longest run of consecutive True samples, in seconds.

    Used wherever a condition must be SUSTAINED. A momentary reading — one
    frame's glance over the shoulder, one frame of a crouch while stepping over
    something — should never look the same as ten seconds of it.
    """
    best = 0.0
    run_start: Optional[float] = None
    previous_time: Optional[float] = None

    for timestamp, holds in samples:
        if holds:
            if run_start is None:
                run_start = timestamp
            best = max(best, timestamp - run_start)
        else:
            run_start = None
        previous_time = timestamp
    return best


def _oscillation(series: List[Tuple[float, Tuple[float, float]]]) -> Tuple[int, float, float]:
    """(direction reversals, peak-to-peak amplitude, frequency Hz) for a point series.

    Amplitude and position are already in body heights, so the numbers mean the
    same thing on any camera.
    """
    if len(series) < 4:
        return (0, 0.0, 0.0)

    times = [t for t, _ in series]
    points = [p for _, p in series]

    # Project onto the axis of greatest variation — prying is one-dimensional,
    # and which dimension depends entirely on the camera angle.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axis = xs if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else ys

    reversals = 0
    previous_sign = 0
    for a, b in zip(axis, axis[1:]):
        delta = b - a
        if abs(delta) < 1e-4:
            continue
        sign = 1 if delta > 0 else -1
        if previous_sign and sign != previous_sign:
            reversals += 1
        previous_sign = sign

    amplitude = max(axis) - min(axis)
    duration = times[-1] - times[0]
    frequency = (reversals / 2.0) / duration if duration > 0 else 0.0
    return (reversals, amplitude, frequency)


# ---------------------------------------------------------------------------
# 1. Loitering / casing
# ---------------------------------------------------------------------------
def loitering(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Dwelling or repeatedly passing near a property boundary.

    BIAS-SENSITIVE — see notice 1 at the top of this file.
    """
    name = "loitering"
    cfg = settings.heuristics.loitering
    if not cfg.enabled:
        return _no(name, "Loitering detection is disabled in config.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    # Bias guard, first and unconditional: somewhere it is normal to stand still.
    exempt = ctx.zones.exempt_containing(latest.foot)
    if exempt is not None:
        return _no(
            name,
            f"Standing in '{exempt.id}', a zone marked as somewhere people "
            f"legitimately wait. Loitering is not evaluated here.",
            exempt_zone=exempt.id,
        )

    watch = ctx.zones.of_type("property_boundary", "street_frontage")
    if not watch:
        return _no(name, "No property boundary or street frontage zones configured.")

    zone_ids = [z.id for z in watch]
    window = float(cfg.window_seconds)
    dwell = track.dwell_seconds(zone_ids, window, max_speed=float(cfg.dwell_speed_max))
    passes = track.zone_entries(zone_ids, window)

    dwell_threshold = float(cfg.dwell_seconds)
    min_passes = int(cfg.min_passes)
    saturation = dwell_threshold * float(cfg.dwell_saturation_multiplier)

    inputs = {
        "dwell_seconds": round(dwell, 2),
        "dwell_threshold_seconds": dwell_threshold,
        "zone_entries": passes,
        "min_passes": min_passes,
        "window_seconds": window,
        "watched_zones": zone_ids,
    }

    dwell_hit = dwell > dwell_threshold
    passes_hit = passes >= min_passes
    if not (dwell_hit or passes_hit):
        return _no(
            name,
            f"Present near the boundary for {dwell:.1f}s over {passes} pass(es) — "
            f"below the {dwell_threshold:.0f}s / {min_passes}-pass thresholds.",
            **inputs,
        )

    confidence = (
        0.6 * ramp(dwell, dwell_threshold, saturation)
        + 0.4 * ramp(float(passes), float(min_passes), float(min_passes + 2))
    )

    parts = []
    if dwell_hit:
        parts.append(
            f"stayed near the property boundary for {dwell:.0f}s "
            f"(threshold {dwell_threshold:.0f}s)"
        )
    if passes_hit:
        parts.append(f"passed the same boundary {passes} times (threshold {min_passes})")

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        "Track " + " and ".join(parts) + ". Note: standing still has many ordinary "
        "explanations — this is a prompt to look, not a finding.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 2. Perimeter probing
# ---------------------------------------------------------------------------
def perimeter_probing(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Approaching a gate/door/window and pausing or reaching toward it,
    rather than continuing past."""
    name = "perimeter_probing"
    cfg = settings.heuristics.probing
    if not cfg.enabled:
        return _no(name, "Perimeter probing detection is disabled in config.")

    gates = ctx.zones.of_type("gate")
    if not gates:
        return _no(name, "No gate/door/window zones configured.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    zone, _ = ctx.zones.nearest(latest.foot, "gate")
    if zone is None:
        return _no(name, "No gate zone near this track.")

    window = float(cfg.window_seconds)
    distance = track.distance_to(zone)
    closing = track.closing_speed(zone, window)
    pause = _slow_seconds(track, float(cfg.pause_speed_max), window)

    proximity = float(cfg.proximity)
    approach_min = float(cfg.approach_speed_min)
    pause_min = float(cfg.pause_seconds)

    inputs = {
        "zone_id": zone.id,
        "distance_body_heights": round(distance, 2) if distance is not None else None,
        "proximity_threshold": proximity,
        "closing_speed": round(closing, 3) if closing is not None else None,
        "approach_speed_min": approach_min,
        "paused_seconds": round(pause, 2),
        "pause_threshold_seconds": pause_min,
    }

    if distance is None or distance > proximity:
        return _no(
            name,
            f"Nearest gate zone '{zone.id}' is "
            f"{distance if distance is None else round(distance, 2)} body-heights away — "
            f"beyond the {proximity} threshold.",
            **inputs,
        )
    # Approach is required so that walking PAST a gate can never trigger this.
    if closing is None or closing < approach_min:
        return _no(name, f"Near gate '{zone.id}' but not approaching it.", **inputs)
    if pause < pause_min:
        return _no(
            name,
            f"Approached gate '{zone.id}' but did not pause "
            f"({pause:.1f}s < {pause_min:.1f}s).",
            **inputs,
        )

    # Reaching toward the boundary: is a wrist extended past the shoulders in
    # the direction of the zone?
    reach_detected = False
    reach_ratio_seen = 0.0
    pose = track.latest_pose
    if pose is not None:
        shoulder_width = pose.shoulder_width
        shoulders = pose.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
        if shoulder_width and shoulders:
            zone_x = zone.centroid[0]
            direction = 1.0 if zone_x >= shoulders[0] else -1.0
            for wrist_index in (LEFT_WRIST, RIGHT_WRIST):
                wrist = pose.point(wrist_index)
                if wrist is None:
                    continue
                extension = (wrist.x - shoulders[0]) * direction / shoulder_width
                reach_ratio_seen = max(reach_ratio_seen, extension)
            reach_detected = reach_ratio_seen >= float(cfg.reach_ratio)

    inputs["reach_ratio_observed"] = round(reach_ratio_seen, 2)
    inputs["reach_ratio_threshold"] = float(cfg.reach_ratio)
    inputs["reach_detected"] = reach_detected

    if bool(cfg.require_reach) and not reach_detected:
        return _no(
            name,
            f"Paused at gate '{zone.id}' but no reaching motion observed "
            f"(config requires it).",
            **inputs,
        )

    confidence = 0.55 * ramp(pause, pause_min, pause_min * 2.5) + 0.20 * ramp(
        proximity - distance, 0.0, proximity
    )
    detail = ""
    if reach_detected:
        confidence += 0.25
        detail = (
            f", with an arm extended {reach_ratio_seen:.1f} shoulder-widths toward it"
        )

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Track approached '{zone.id}' (closing at {closing:.2f} body-heights/sec) and "
        f"stopped within {distance:.2f} body-heights of it for {pause:.1f}s"
        f"{detail}, rather than continuing past.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 3. Climbing / scaling posture
# ---------------------------------------------------------------------------
def climbing_posture(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Pose consistent with scaling a wall or fence."""
    name = "climbing_posture"
    cfg = settings.heuristics.climbing
    if not cfg.enabled:
        return _no(name, "Climbing detection is disabled in config.")

    walls = ctx.zones.of_type("vertical_structure")
    if not walls:
        return _no(name, "No wall/fence zones configured.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    zone, _ = ctx.zones.nearest(latest.foot, "vertical_structure")
    distance = track.distance_to(zone) if zone else None
    proximity = float(cfg.proximity)

    inputs: Dict[str, object] = {
        "zone_id": zone.id if zone else None,
        "distance_body_heights": round(distance, 2) if distance is not None else None,
        "proximity_threshold": proximity,
    }

    # Near something climbable, or it is just someone stretching.
    if distance is None or distance > proximity:
        return _no(name, "Not near a wall or fence zone.", **inputs)

    window = float(cfg.window_seconds)
    observations = [o for o in track.window(window) if o.pose is not None]
    if not observations:
        return _no(name, "No pose data in the window.", **inputs)

    hands_high_frames = 0
    knee_lift_frames = 0
    max_asymmetry = 0.0
    for observation in observations:
        pose = observation.pose
        left_high, right_high = pose.hands_above_shoulders()
        if left_high or right_high:
            hands_high_frames += 1

        body_height = observation.body_height or track.body_height
        if body_height > 0:
            for hip, ankle in ((LEFT_HIP, LEFT_ANKLE), (RIGHT_HIP, RIGHT_ANKLE)):
                hip_point, ankle_point = pose.point(hip), pose.point(ankle)
                if hip_point and ankle_point:
                    gap = abs(ankle_point.y - hip_point.y) / body_height
                    if gap < float(cfg.knee_lift_ratio):
                        knee_lift_frames += 1
                        break

            left_extension = pose.limb_extension(LEFT_SHOULDER, LEFT_WRIST)
            right_extension = pose.limb_extension(RIGHT_SHOULDER, RIGHT_WRIST)
            if left_extension is not None and right_extension is not None:
                max_asymmetry = max(
                    max_asymmetry, abs(left_extension - right_extension) / body_height
                )

    rise = track.vertical_rise(window) or 0.0
    min_frames = int(cfg.min_frames_hands_high)
    asymmetric = max_asymmetry > float(cfg.limb_asymmetry)
    knee_lifted = knee_lift_frames > 0
    risen = rise > float(cfg.climb_rise_ratio)

    inputs.update({
        "hands_above_shoulder_frames": hands_high_frames,
        "min_frames_hands_high": min_frames,
        "knee_lift_frames": knee_lift_frames,
        "limb_asymmetry": round(max_asymmetry, 3),
        "limb_asymmetry_threshold": float(cfg.limb_asymmetry),
        "vertical_rise_body_heights": round(rise, 3),
        "climb_rise_threshold": float(cfg.climb_rise_ratio),
        "pose_frames_examined": len(observations),
    })

    if hands_high_frames < min_frames:
        return _no(
            name,
            f"Hands above shoulder height in only {hands_high_frames} frame(s) "
            f"(needs {min_frames}).",
            **inputs,
        )
    if not (knee_lifted or asymmetric or risen):
        return _no(
            name,
            "Hands raised near the wall, but with no raised knee, asymmetric reach "
            "or vertical rise — consistent with reaching or stretching, not climbing.",
            **inputs,
        )

    confidence = 0.4 * ramp(float(hands_high_frames), float(min_frames), float(min_frames * 2))
    supporting = []
    if knee_lifted:
        confidence += 0.2
        supporting.append("a raised knee")
    if asymmetric:
        confidence += 0.2
        supporting.append(f"asymmetric arm extension ({max_asymmetry:.2f} body-heights)")
    if risen:
        confidence += 0.2
        supporting.append(f"ground contact rising {rise:.2f} body-heights")

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Track held hands above shoulder height against '{zone.id}' across "
        f"{hands_high_frames} frames, with " + " and ".join(supporting) + ".",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 4. Concealment approach
# ---------------------------------------------------------------------------
def concealment_approach(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Face not visible to the camera while closing on a property.

    BIAS-SENSITIVE, ACUTELY — see notice 3 at the top of this file. This is the
    branch where facial recognition CANNOT contribute, so behaviour has to carry
    the signal on its own. What it observes is a limitation of the camera, not a
    property of the person.
    """
    name = "concealment_approach"
    cfg = settings.heuristics.concealment
    if not cfg.enabled:
        return _no(name, "Concealment detection is disabled in config.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    property_zones = ctx.zones.of_type("property_boundary", "gate")
    if not property_zones:
        return _no(name, "No property zones configured.")

    zone, _ = ctx.zones.nearest(latest.foot, "property_boundary", "gate")
    window = float(cfg.window_seconds)
    closing = track.closing_speed(zone, window) if zone else None
    approach_min = float(cfg.approach_speed_min)

    observations = [o for o in track.window(window) if o.pose is not None]
    if not observations:
        return _no(name, "No pose data in the window.")

    visibility_min = float(cfg.face_visibility_min)
    yaw_threshold = float(cfg.yaw_away_degrees)

    samples: List[Tuple[float, bool]] = []
    visibilities: List[float] = []
    for observation in observations:
        visibility = observation.pose.face_visibility
        yaw = observation.pose.head_yaw_degrees
        visibilities.append(visibility)
        hidden = visibility < visibility_min or (yaw is not None and abs(yaw) > yaw_threshold)
        samples.append((observation.timestamp, hidden))

    hidden_seconds = _sustained_seconds(samples)
    mean_visibility = statistics.mean(visibilities) if visibilities else 0.0
    conceal_min = float(cfg.conceal_seconds)

    inputs = {
        "zone_id": zone.id if zone else None,
        "closing_speed": round(closing, 3) if closing is not None else None,
        "approach_speed_min": approach_min,
        "face_visibility_mean": round(mean_visibility, 3),
        "face_visibility_threshold": visibility_min,
        "yaw_threshold_degrees": yaw_threshold,
        "face_hidden_seconds": round(hidden_seconds, 2),
        "conceal_threshold_seconds": conceal_min,
    }

    # Approach is REQUIRED. Without it, this heuristic would fire on anyone
    # whose face the camera cannot see — which is most people, most of the time.
    if closing is None or closing < approach_min:
        return _no(
            name,
            "Face not clearly visible, but the track is not approaching a property — "
            "not evaluated as concealment.",
            **inputs,
        )
    if hidden_seconds < conceal_min:
        return _no(
            name,
            f"Face obscured for only {hidden_seconds:.1f}s while approaching "
            f"(needs {conceal_min:.1f}s).",
            **inputs,
        )

    confidence = 0.6 * ramp(hidden_seconds, conceal_min, conceal_min * 2.5) + 0.4 * (
        1.0 - min(1.0, mean_visibility / max(visibility_min, 1e-6))
    )

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"The camera could not see this person's face (mean visibility "
        f"{mean_visibility:.2f}, threshold {visibility_min:.2f}) for {hidden_seconds:.1f}s "
        f"while they closed on '{zone.id}' at {closing:.2f} body-heights/sec. "
        f"Facial recognition cannot contribute here, so this event rests on movement "
        f"alone. A covered or turned-away face is ordinary in itself.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 5. Crouched proximity to a vehicle
# ---------------------------------------------------------------------------
def crouched_near_vehicle(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Sustained low posture beside a vehicle, rather than walking past upright."""
    name = "crouched_near_vehicle"
    cfg = settings.heuristics.crouched_vehicle
    if not cfg.enabled:
        return _no(name, "Crouched-near-vehicle detection is disabled in config.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    scale = track.body_height
    if scale <= 0:
        return _no(name, "No usable scale reference for this track.")

    # Prefer a detected vehicle; fall back to a configured vehicle zone, which
    # covers the case where the car is parked out of the camera's view but the
    # bay it occupies is known.
    vehicle, gap_px = ctx.nearest_vehicle(latest.bbox)
    proximity = float(cfg.vehicle_proximity)
    gap = gap_px / scale if vehicle is not None else None
    near_vehicle = gap is not None and gap <= proximity
    vehicle_zone = next(
        (z for z in ctx.zones.of_type("vehicle_zone") if z.contains(latest.foot)), None
    )

    inputs: Dict[str, object] = {
        "nearest_vehicle": vehicle.track_id if vehicle else None,
        "vehicle_gap_body_heights": round(gap, 2) if gap is not None else None,
        "vehicle_proximity_threshold": proximity,
        "vehicle_zone": vehicle_zone.id if vehicle_zone else None,
    }

    if not near_vehicle and vehicle_zone is None:
        return _no(name, "Not near a detected vehicle or a vehicle zone.", **inputs)

    baseline = track.standing_torso
    if baseline is None:
        return _no(
            name,
            "No standing baseline learned for this track yet, so no crouch can be "
            "measured. Posture is only ever compared against this person's own "
            "upright posture, never a population average.",
            **inputs,
        )

    window = float(cfg.window_seconds)
    crouch_threshold = float(cfg.crouch_ratio)
    knee_threshold = float(cfg.knee_angle_max)

    samples: List[Tuple[float, bool]] = []
    lowest_ratio = 1.0
    lowest_knee: Optional[float] = None
    for observation in track.window(window):
        if observation.pose is None:
            continue
        ratio = track.crouch_ratio(observation.pose)
        knee = observation.pose.min_knee_angle
        is_crouched = False
        if ratio is not None:
            lowest_ratio = min(lowest_ratio, ratio)
            is_crouched = ratio <= crouch_threshold
        if knee is not None:
            lowest_knee = knee if lowest_knee is None else min(lowest_knee, knee)
            is_crouched = is_crouched or knee <= knee_threshold
        samples.append((observation.timestamp, is_crouched))

    crouch_seconds = _sustained_seconds(samples)
    required = float(cfg.crouch_seconds)

    inputs.update({
        "standing_baseline_px": round(baseline, 1),
        "lowest_crouch_ratio": round(lowest_ratio, 3),
        "crouch_ratio_threshold": crouch_threshold,
        "lowest_knee_angle": round(lowest_knee, 1) if lowest_knee is not None else None,
        "knee_angle_threshold": knee_threshold,
        "crouch_seconds": round(crouch_seconds, 2),
        "crouch_seconds_threshold": required,
    })

    if crouch_seconds < required:
        return _no(
            name,
            f"Low posture held for {crouch_seconds:.1f}s beside the vehicle "
            f"(needs {required:.1f}s) — consistent with bending down briefly.",
            **inputs,
        )

    confidence = 0.6 * ramp(crouch_seconds, required, required * 2.0) + 0.4 * ramp(
        crouch_threshold - lowest_ratio, 0.0, crouch_threshold * 0.4
    )
    where = f"vehicle {vehicle.class_name}" if vehicle else f"zone '{vehicle_zone.id}'"

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Track held a crouched posture — {lowest_ratio * 100:.0f}% of their own "
        f"standing height, threshold {crouch_threshold * 100:.0f}% — beside a {where} "
        f"for {crouch_seconds:.1f}s, rather than passing at walking height.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 6. Tampering motion
# ---------------------------------------------------------------------------
def tampering_motion(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """Repetitive short-range arm movement at a vehicle, body otherwise still."""
    name = "tampering_motion"
    cfg = settings.heuristics.tampering
    if not cfg.enabled:
        return _no(name, "Tampering detection is disabled in config.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    scale = track.body_height
    if scale <= 0:
        return _no(name, "No usable scale reference for this track.")

    vehicle, gap_px = ctx.nearest_vehicle(latest.bbox)
    proximity = float(cfg.vehicle_proximity)
    gap = gap_px / scale if vehicle is not None else None
    vehicle_zone = next(
        (z for z in ctx.zones.of_type("vehicle_zone") if z.contains(latest.foot)), None
    )

    inputs: Dict[str, object] = {
        "nearest_vehicle": vehicle.track_id if vehicle else None,
        "vehicle_gap_body_heights": round(gap, 2) if gap is not None else None,
        "vehicle_proximity_threshold": proximity,
        "vehicle_zone": vehicle_zone.id if vehicle_zone else None,
    }

    if not ((gap is not None and gap <= proximity) or vehicle_zone is not None):
        return _no(name, "Not near a detected vehicle or a vehicle zone.", **inputs)

    window = float(cfg.window_seconds)
    body_speed = track.speed(window) or 0.0
    body_speed_max = float(cfg.body_speed_max)
    inputs["body_speed"] = round(body_speed, 3)
    inputs["body_speed_max"] = body_speed_max

    # Arms working while the body stays put is the signal. Someone walking past
    # swinging their arms produces the same wrist oscillation and is not this.
    if body_speed > body_speed_max:
        return _no(
            name,
            f"Arm movement present but the whole body is moving at "
            f"{body_speed:.2f} body-heights/sec — consistent with walking, not tampering.",
            **inputs,
        )

    series = track.wrist_series(window)
    best: Optional[Tuple[str, int, float, float]] = None
    for side, points in series.items():
        reversals, amplitude, frequency = _oscillation(points)
        if best is None or reversals > best[1]:
            best = (side, reversals, amplitude, frequency)

    if best is None or best[1] == 0:
        return _no(name, "No wrist movement observed near the vehicle.", **inputs)

    side, reversals, amplitude, frequency = best
    inputs.update({
        "side": side,
        "reversals": reversals,
        "min_reversals": int(cfg.min_reversals),
        "amplitude_body_heights": round(amplitude, 3),
        "amplitude_range": [float(cfg.amplitude_min), float(cfg.amplitude_max)],
        "frequency_hz": round(frequency, 2),
        "frequency_range": [float(cfg.frequency_min), float(cfg.frequency_max)],
    })

    if reversals < int(cfg.min_reversals):
        return _no(
            name,
            f"Only {reversals} direction changes in the {side} wrist "
            f"(needs {int(cfg.min_reversals)}) — not repetitive.",
            **inputs,
        )
    if not (float(cfg.amplitude_min) <= amplitude <= float(cfg.amplitude_max)):
        return _no(
            name,
            f"Wrist movement amplitude {amplitude:.2f} body-heights is outside the "
            f"{cfg.amplitude_min}–{cfg.amplitude_max} band expected of short-range "
            f"prying (too small is tracker jitter, too large is a wave or a throw).",
            **inputs,
        )
    if not (float(cfg.frequency_min) <= frequency <= float(cfg.frequency_max)):
        return _no(
            name,
            f"Wrist oscillation at {frequency:.1f}Hz is outside the "
            f"{cfg.frequency_min}–{cfg.frequency_max}Hz band.",
            **inputs,
        )

    confidence = 0.5 * ramp(float(reversals), float(cfg.min_reversals), float(cfg.min_reversals) * 2) + 0.5 * (
        1.0 - abs(body_speed / max(body_speed_max, 1e-6))
    )

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Repetitive short-range {side}-hand movement at a vehicle: {reversals} direction "
        f"changes at {frequency:.1f}Hz over {amplitude:.2f} body-heights, while the body "
        f"stayed still ({body_speed:.2f} body-heights/sec). Consistent with working at a "
        f"door or window.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 7. Group coordination (lookout + actor)
# ---------------------------------------------------------------------------
def group_coordination(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """One person static at a distance while this one is active at the target.

    Evaluated with `track` as the candidate ACTOR; the lookout is searched for
    among the other tracks. The event is recorded against the actor, and the
    explanation names the other track so a reviewer can find them.
    """
    name = "group_coordination"
    cfg = settings.heuristics.group
    if not cfg.enabled:
        return _no(name, "Group coordination detection is disabled in config.")

    latest = track.latest
    if latest is None:
        return _no(name, "No observations for this track.")

    scale = track.body_height
    if scale <= 0:
        return _no(name, "No usable scale reference for this track.")

    # The actor must be doing something: inside a target zone, or already
    # triggering another heuristic this frame.
    actor_zones = [
        z for z in ctx.zones.containing(latest.foot)
        if z.type in {"gate", "property_boundary", "vehicle_zone"}
    ]
    actor_triggers = [
        t for t in ctx.triggered_this_frame.get(track.track_id, []) if t != name
    ]
    if not actor_zones and not actor_triggers:
        return _no(
            name,
            "This track is not active at a target zone, so there is no actor role to pair.",
        )

    static_max = float(cfg.static_speed_max)
    lookout_seconds = float(cfg.lookout_seconds)
    min_separation = float(cfg.min_separation)
    overlap_min = float(cfg.min_overlap_seconds)

    best: Optional[Dict[str, object]] = None
    for other in ctx.person_tracks():
        if other.track_id == track.track_id:
            continue

        overlap = min(other.last_seen, track.last_seen) - max(other.first_seen, track.first_seen)
        if overlap < overlap_min:
            continue

        other_latest = other.latest
        if other_latest is None:
            continue

        separation = math.dist(latest.foot, other_latest.foot) / scale
        if separation < min_separation:
            continue

        still_seconds = _slow_seconds(other, static_max, lookout_seconds * 1.5)
        if still_seconds < lookout_seconds:
            continue

        yaws = other.head_yaw_series(lookout_seconds * 1.5)
        yaw_variance = statistics.pvariance(yaws) if len(yaws) >= 3 else 0.0

        candidate = {
            "lookout_track_id": other.track_id,
            "separation_body_heights": round(separation, 2),
            "lookout_still_seconds": round(still_seconds, 2),
            "lookout_yaw_variance": round(yaw_variance, 1),
            "overlap_seconds": round(overlap, 2),
        }
        if best is None or still_seconds > float(best["lookout_still_seconds"]):
            best = candidate

    inputs: Dict[str, object] = {
        "actor_zones": [z.id for z in actor_zones],
        "actor_other_triggers": actor_triggers,
        "static_speed_max": static_max,
        "lookout_seconds_threshold": lookout_seconds,
        "min_separation": min_separation,
    }

    if best is None:
        return _no(name, "No second track matching a lookout pattern.", **inputs)

    inputs.update(best)
    scanning = float(best["lookout_yaw_variance"]) > float(cfg.scan_yaw_variance)
    inputs["scanning"] = scanning

    confidence = 0.45 * ramp(
        float(best["lookout_still_seconds"]), lookout_seconds, lookout_seconds * 2
    ) + 0.25 * ramp(float(best["separation_body_heights"]), min_separation, min_separation * 2)
    if scanning:
        confidence += 0.10
    if actor_triggers:
        confidence *= float(cfg.corroboration_bonus)

    where = (
        f"'{actor_zones[0].id}'" if actor_zones else f"({', '.join(actor_triggers)})"
    )
    scanning_note = ", head turning as if scanning" if scanning else ""

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Track {best['lookout_track_id']} stayed still for "
        f"{best['lookout_still_seconds']}s at {best['separation_body_heights']} body-heights' "
        f"distance{scanning_note}, while this track was active at {where}. Two people can "
        f"be together for every ordinary reason — this pairs the observation, it does not "
        f"interpret it.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# 8. Fleeing
# ---------------------------------------------------------------------------
def fleeing(track: TrackHistory, ctx: SceneContext, settings: Settings) -> HeuristicResult:
    """A sudden transition from walking to running, away from a target.

    BIAS-SENSITIVE — see notice 4. Running is not a crime, and someone escaping
    danger moves exactly like someone leaving a scene. This reports a change in
    pace and nothing more.
    """
    name = "fleeing"
    cfg = settings.heuristics.fleeing
    if not cfg.enabled:
        return _no(name, "Fleeing detection is disabled in config.")

    window = float(cfg.window_seconds)
    accel_window = float(cfg.accel_window_seconds)
    run_min = float(cfg.run_speed_min)
    walk_max = float(cfg.walk_speed_max)

    current = track.peak_speed(accel_window) or track.speed(accel_window)
    earlier = track.speed(window)

    inputs: Dict[str, object] = {
        "recent_peak_speed": round(current, 3) if current is not None else None,
        "run_speed_min": run_min,
        "window_speed": round(earlier, 3) if earlier is not None else None,
        "walk_speed_max": walk_max,
        "accel_window_seconds": accel_window,
    }

    if current is None or current < run_min:
        return _no(
            name,
            f"Peak speed {current if current is None else round(current, 2)} "
            f"body-heights/sec is below the running threshold of {run_min}.",
            **inputs,
        )

    # Sustained running (a jogger) is not a transition. Fleeing is the CHANGE.
    observations = track.window(window)
    if len(observations) < 3:
        return _no(name, "Not enough history to judge a change of pace.", **inputs)

    early = track.window(window)[: max(2, len(observations) // 2)]
    early_speed = None
    if len(early) >= 2:
        scale = track.body_height
        elapsed = early[-1].timestamp - early[0].timestamp
        if scale > 0 and elapsed > 1e-6:
            early_speed = (math.dist(early[0].foot, early[-1].foot) / scale) / elapsed

    inputs["earlier_speed"] = round(early_speed, 3) if early_speed is not None else None
    if early_speed is None or early_speed > walk_max:
        return _no(
            name,
            f"Moving fast, but was already moving at "
            f"{early_speed if early_speed is None else round(early_speed, 2)} "
            f"body-heights/sec — sustained pace, not a sudden transition.",
            **inputs,
        )

    latest = track.latest
    zone, _ = ctx.zones.nearest(
        latest.foot, "property_boundary", "gate", "vehicle_zone"
    ) if latest else (None, math.inf)
    closing = track.closing_speed(zone, window) if zone else None
    moving_away = closing is not None and closing < 0
    inputs["zone_id"] = zone.id if zone else None
    inputs["closing_speed"] = round(closing, 3) if closing is not None else None
    inputs["moving_away"] = moving_away

    if bool(cfg.require_away_from_zone) and not moving_away:
        return _no(
            name,
            "Sudden acceleration, but not away from a property or vehicle zone.",
            **inputs,
        )

    confidence = 0.7 * ramp(current, run_min, run_min * 1.6) + 0.3 * ramp(
        current - early_speed, 1.0, 3.0
    )

    # Cross-reference with Discovery's claims data: an abrupt departure close in
    # time to a logged incident at this location is worth more to a reviewer.
    incident_note = ""
    incident_minutes = getattr(ctx, "nearby_incident_minutes", None)
    if incident_minutes is not None and incident_minutes <= float(cfg.incident_link_minutes):
        confidence *= float(cfg.incident_link_bonus)
        incident_note = (
            f" This is within {incident_minutes:.0f} minutes of an incident logged in "
            f"the claims data at this location."
        )
        inputs["incident_minutes_ago"] = incident_minutes

    return HeuristicResult(
        True,
        round(min(1.0, confidence), 3),
        f"Pace changed abruptly from {early_speed:.1f} to {current:.1f} body-heights/sec "
        f"(running threshold {run_min}) moving away from '{zone.id if zone else 'the area'}'."
        f"{incident_note} Running has many ordinary and urgent explanations, including "
        f"escaping danger.",
        name,
        inputs,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
HeuristicFn = Callable[[TrackHistory, SceneContext, Settings], HeuristicResult]

# Order matters only for readability of the output. The names here are the keys
# used by fusion.weights in config.yaml — keep them in step.
HEURISTICS: Tuple[Tuple[str, HeuristicFn], ...] = (
    ("loitering", loitering),
    ("perimeter_probing", perimeter_probing),
    ("climbing_posture", climbing_posture),
    ("concealment_approach", concealment_approach),
    ("crouched_near_vehicle", crouched_near_vehicle),
    ("tampering_motion", tampering_motion),
    ("group_coordination", group_coordination),
    ("fleeing", fleeing),
)

HEURISTIC_NAMES: Tuple[str, ...] = tuple(name for name, _ in HEURISTICS)


def evaluate_all(
    track: TrackHistory,
    ctx: SceneContext,
    settings: Settings,
    *,
    include_misses: bool = False,
) -> List[HeuristicResult]:
    """Run every heuristic against one track.

    `include_misses` returns the non-triggers too, each carrying the reason it
    did not fire. That is what makes threshold tuning possible — and it is the
    difference between "the system saw nothing" and "the system saw it and
    decided it was 3 seconds short".
    """
    # Each heuristic's confidence ramps UP FROM its threshold, so a behaviour
    # that only just qualifies scores near zero. Crossing a configured threshold
    # is itself evidence, though, so a trigger is floored here: the ramp then
    # expresses how far PAST the line it went, not whether it crossed it.
    # Applied in one place so the confidence in the JSON is the same number the
    # fusion used — a payload that disagrees with its own score is not auditable.
    floor = float(settings.fusion.get("min_trigger_confidence", 0.0))

    results: List[HeuristicResult] = []
    for _, function in HEURISTICS:
        try:
            result = function(track, ctx, settings)
        except Exception as exc:  # one broken heuristic must not kill the frame
            result = _no(
                getattr(function, "__name__", "unknown"),
                f"Heuristic raised {type(exc).__name__}: {exc}",
            )
        if result.triggered and result.confidence < floor:
            result = result._replace(confidence=floor)
        if result.triggered:
            ctx.triggered_this_frame.setdefault(track.track_id, []).append(result.name)
        if result.triggered or include_misses:
            results.append(result)
    return results
