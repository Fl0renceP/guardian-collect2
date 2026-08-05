"""Tests for the eight heuristics and the fusion rules, on synthetic fixtures.

No video needed. Each test builds a track by hand — a person standing at a gate,
crouching by a car, running away — and asserts what the heuristic concludes.

That matters for two reasons. Sample footage rarely contains all eight
behaviours (nobody scales a wall on cue for a hackathon clip), and a heuristic
that is only ever exercised through a video cannot be shown to be correct — only
to have fired once.

Run either way:
    python tests/test_heuristics.py          # no pytest needed
    pytest tests/test_heuristics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import heuristics as H  # noqa: E402
from detector import Detection  # noqa: E402
from pose_extractor import (  # noqa: E402
    LEFT_ANKLE, LEFT_EAR, LEFT_EYE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    NOSE, RIGHT_ANKLE, RIGHT_EAR, RIGHT_EYE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
    RIGHT_WRIST, Landmark, PoseKeypoints,
)
from risk_fusion import FaceSignal, face_signal_from_recognition, score_event  # noqa: E402
from settings import Settings, Zone, load_settings  # noqa: E402
from trajectory_tracker import Observation, SceneContext, TrackHistory  # noqa: E402
from zones import ZoneIndex  # noqa: E402

FRAME_W, FRAME_H = 1000, 1000
BODY = 100.0          # every fixture person is 100px tall: 1 body height = 100px

# A purpose-built zone layout in normalised coordinates. Deliberately not the
# shipped config's zones — tests should not break when a demo zone is retuned.
TEST_ZONES = [
    Zone("boundary", "property_boundary", 0.5, ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))),
    Zone("gate", "gate", 0.5, ((0.20, 0.20), (0.35, 0.20), (0.35, 0.40), (0.20, 0.40))),
    Zone("wall", "vertical_structure", 0.5, ((0.55, 0.0), (0.75, 0.0), (0.75, 0.4), (0.55, 0.4))),
    Zone("driveway", "vehicle_zone", 0.5, ((0.55, 0.55), (0.95, 0.55), (0.95, 0.95), (0.55, 0.95))),
    Zone("bus_stop", "exempt", 0.0, ((0.0, 0.60), (0.20, 0.60), (0.20, 0.85), (0.0, 0.85))),
]


def settings() -> Settings:
    return load_settings(MODULE_DIR / "config.yaml")


def zone_index() -> ZoneIndex:
    return ZoneIndex(TEST_ZONES, FRAME_W, FRAME_H)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def make_pose(
    foot: tuple[float, float],
    *,
    body_height: float = BODY,
    torso: float = 30.0,
    hands_up: bool = False,
    face_visible: bool = True,
    wrist_dx: float = 0.0,
    knee_forward: float = 0.0,
    arm_asymmetry: float = 0.0,
    visibility_floor: float = 0.35,
) -> PoseKeypoints:
    """A synthetic skeleton standing at `foot`, built from body-height fractions."""
    fx, fy = foot
    top = fy - body_height
    shoulder_y = top + body_height * 0.20
    hip_y = shoulder_y + torso
    ankle_y = fy
    knee_y = (hip_y + ankle_y) / 2.0
    half_shoulder = body_height * 0.12

    points: dict[int, tuple[float, float, float]] = {}

    def put(index: int, x: float, y: float, visibility: float = 0.9) -> None:
        points[index] = (x, y, visibility)

    face_vis = 0.9 if face_visible else 0.05
    put(NOSE, fx, top + body_height * 0.05, face_vis)
    put(LEFT_EYE, fx - 4, top + body_height * 0.04, face_vis)
    put(RIGHT_EYE, fx + 4, top + body_height * 0.04, face_vis)
    put(LEFT_EAR, fx - 8, top + body_height * 0.05, face_vis)
    put(RIGHT_EAR, fx + 8, top + body_height * 0.05, face_vis)

    put(LEFT_SHOULDER, fx - half_shoulder, shoulder_y)
    put(RIGHT_SHOULDER, fx + half_shoulder, shoulder_y)
    put(LEFT_HIP, fx - half_shoulder * 0.7, hip_y)
    put(RIGHT_HIP, fx + half_shoulder * 0.7, hip_y)

    wrist_y = shoulder_y - body_height * 0.18 if hands_up else shoulder_y + body_height * 0.28
    put(LEFT_WRIST, fx - half_shoulder - wrist_dx, wrist_y)
    put(RIGHT_WRIST, fx + half_shoulder + wrist_dx, wrist_y - arm_asymmetry)

    put(LEFT_KNEE, fx - half_shoulder * 0.7 + knee_forward, knee_y)
    put(RIGHT_KNEE, fx + half_shoulder * 0.7 + knee_forward, knee_y)
    put(LEFT_ANKLE, fx - half_shoulder * 0.7, ankle_y)
    put(RIGHT_ANKLE, fx + half_shoulder * 0.7, ankle_y)

    landmarks = []
    for index in range(33):
        if index in points:
            x, y, visibility = points[index]
            landmarks.append(Landmark(x, y, 0.0, visibility))
        else:
            # Not placed = not seen. `point()` returns None for these, which is
            # what the real extractor does for occluded joints.
            landmarks.append(Landmark(fx, fy, 0.0, 0.0))

    return PoseKeypoints(
        landmarks=landmarks,
        visibility_floor=visibility_floor,
        bbox_height=body_height,
    )


def make_track(
    samples: list[tuple[float, tuple[float, float]]],
    *,
    track_id: str = "person-1",
    body_height: float = BODY,
    pose_factory=None,
    index: ZoneIndex | None = None,
) -> TrackHistory:
    """A TrackHistory from (timestamp, foot_position) samples."""
    index = index or zone_index()
    track = TrackHistory(track_id=track_id, kind="person")
    for timestamp, foot in samples:
        pose = pose_factory(timestamp, foot) if pose_factory else None
        fx, fy = foot
        track.append(
            Observation(
                timestamp=timestamp,
                bbox=(fx - body_height * 0.2, fy - body_height, fx + body_height * 0.2, fy),
                foot=foot,
                centroid=(fx, fy - body_height / 2),
                body_height=body_height,
                pose=pose,
                zone_ids=tuple(z.id for z in index.containing(foot)),
            )
        )
    return track


def make_context(
    tracks: list[TrackHistory],
    *,
    vehicles: list[Detection] | None = None,
    index: ZoneIndex | None = None,
    incident_minutes: float | None = None,
) -> SceneContext:
    index = index or zone_index()
    latest = max((t.last_seen for t in tracks), default=0.0)
    return SceneContext(
        timestamp=latest,
        frame_index=0,
        frame_size=(FRAME_W, FRAME_H),
        zones=index,
        tracks={t.track_id: t for t in tracks},
        vehicles=vehicles or [],
        nearby_incident_minutes=incident_minutes,
    )


def vehicle_at(x: float, y: float, width: float = 200, height: float = 120) -> Detection:
    return Detection(
        track_id="vehicle-1",
        kind="vehicle",
        class_name="car",
        confidence=0.9,
        bbox=(x, y, x + width, y + height),
    )


# ---------------------------------------------------------------------------
# 1. Loitering
# ---------------------------------------------------------------------------
def test_loitering_fires_on_sustained_dwell():
    cfg = settings()
    # Standing inside the property boundary zone for 60s (threshold 45s).
    spot = (200.0, 300.0)
    track = make_track([(t * 2.0, spot) for t in range(31)])
    result = H.loitering(track, make_context([track]), cfg)

    assert result.triggered, result.explanation
    assert result.confidence > 0
    assert "boundary" in result.explanation.lower()
    assert result.inputs["dwell_seconds"] >= 45


def test_loitering_ignores_someone_walking_through():
    cfg = settings()
    # Crossing the same zone at a normal walking pace: 1 body height/sec.
    track = make_track([(t * 0.5, (50.0 + t * 50.0, 300.0)) for t in range(20)])
    result = H.loitering(track, make_context([track]), cfg)
    assert not result.triggered


def test_loitering_never_fires_in_an_exempt_zone():
    """The bias guard: somewhere it is normal to stand still."""
    cfg = settings()
    bus_stop = (100.0, 700.0)     # inside the `bus_stop` exempt zone
    track = make_track([(t * 2.0, bus_stop) for t in range(60)])   # two full minutes
    result = H.loitering(track, make_context([track]), cfg)

    assert not result.triggered
    assert "legitimately wait" in result.explanation
    assert result.inputs["exempt_zone"] == "bus_stop"


# ---------------------------------------------------------------------------
# 2. Perimeter probing
# ---------------------------------------------------------------------------
def test_probing_fires_on_approach_then_pause_at_the_gate():
    cfg = settings()
    gate_front = (300.0, 420.0)   # just below the gate zone

    samples = []
    # Approach from 3 body-heights away over 6 seconds...
    for i in range(13):
        samples.append((i * 0.5, (300.0, 720.0 - i * 25.0)))
    # ...then stand still for 6 seconds.
    for i in range(1, 13):
        samples.append((6.0 + i * 0.5, gate_front))

    track = make_track(samples, pose_factory=lambda t, foot: make_pose(foot))
    result = H.perimeter_probing(track, make_context([track]), cfg)

    assert result.triggered, result.explanation
    assert result.inputs["paused_seconds"] >= cfg.heuristics.probing.pause_seconds


def test_probing_does_not_fire_when_walking_past():
    """Approach is required, so passing a gate can never trigger this."""
    cfg = settings()
    samples = [(i * 0.4, (100.0 + i * 40.0, 420.0)) for i in range(25)]
    track = make_track(samples, pose_factory=lambda t, foot: make_pose(foot))
    result = H.perimeter_probing(track, make_context([track]), cfg)
    assert not result.triggered


# ---------------------------------------------------------------------------
# 3. Climbing posture
# ---------------------------------------------------------------------------
def test_climbing_fires_with_raised_hands_and_asymmetric_reach_at_a_wall():
    cfg = settings()
    at_wall = (660.0, 450.0)      # just below the `wall` zone
    track = make_track(
        [(i * 0.3, at_wall) for i in range(20)],
        pose_factory=lambda t, foot: make_pose(foot, hands_up=True, arm_asymmetry=30.0),
    )
    result = H.climbing_posture(track, make_context([track]), cfg)

    assert result.triggered, result.explanation
    assert result.inputs["hands_above_shoulder_frames"] >= cfg.heuristics.climbing.min_frames_hands_high


def test_climbing_does_not_fire_for_symmetric_reaching_with_no_lift():
    """Hands up near a wall, but nothing else: a stretch, not a climb."""
    cfg = settings()
    at_wall = (660.0, 450.0)
    track = make_track(
        [(i * 0.3, at_wall) for i in range(20)],
        pose_factory=lambda t, foot: make_pose(foot, hands_up=True, arm_asymmetry=0.0),
    )
    result = H.climbing_posture(track, make_context([track]), cfg)
    assert not result.triggered
    assert "stretching" in result.explanation


def test_climbing_does_not_fire_away_from_a_wall():
    cfg = settings()
    open_ground = (500.0, 800.0)
    track = make_track(
        [(i * 0.3, open_ground) for i in range(20)],
        pose_factory=lambda t, foot: make_pose(foot, hands_up=True, arm_asymmetry=30.0),
    )
    result = H.climbing_posture(track, make_context([track]), cfg)
    assert not result.triggered


# ---------------------------------------------------------------------------
# 4. Concealment approach
# ---------------------------------------------------------------------------
def test_concealment_fires_when_face_hidden_while_approaching():
    cfg = settings()
    samples = [(i * 0.5, (300.0, 800.0 - i * 20.0)) for i in range(25)]
    track = make_track(
        samples,
        pose_factory=lambda t, foot: make_pose(foot, face_visible=False),
    )
    result = H.concealment_approach(track, make_context([track]), cfg)

    assert result.triggered, result.explanation
    assert result.inputs["face_hidden_seconds"] >= cfg.heuristics.concealment.conceal_seconds
    # The explanation must describe the camera's limitation, not the person.
    assert "could not see" in result.explanation
    assert "ordinary" in result.explanation


def test_concealment_does_not_fire_without_approach():
    """A covered face alone can never trigger this. That is the whole safeguard."""
    cfg = settings()
    standing = (300.0, 800.0)
    track = make_track(
        [(i * 0.5, standing) for i in range(25)],
        pose_factory=lambda t, foot: make_pose(foot, face_visible=False),
    )
    result = H.concealment_approach(track, make_context([track]), cfg)
    assert not result.triggered
    assert "not approaching" in result.explanation


# ---------------------------------------------------------------------------
# 5. Crouched near a vehicle
# ---------------------------------------------------------------------------
def _crouch_track(torso_when_crouched: float) -> TrackHistory:
    """Walk in upright (learning a baseline), then crouch by the car for 8s."""
    index = zone_index()
    track = TrackHistory(track_id="person-1", kind="person")

    # Approach at walking pace so the standing baseline can be learned.
    for i in range(10):
        timestamp = i * 0.4
        foot = (500.0 + i * 30.0, 800.0)
        pose = make_pose(foot, torso=30.0)
        track.append(Observation(timestamp, (foot[0] - 20, foot[1] - BODY, foot[0] + 20, foot[1]),
                                 foot, (foot[0], foot[1] - 50), BODY, pose,
                                 tuple(z.id for z in index.containing(foot))))
        track.observe_baseline(pose, track.speed(1.0))

    # Then crouch beside the car, stationary, for 8 seconds.
    crouch_spot = (790.0, 800.0)
    for i in range(21):
        timestamp = 4.0 + i * 0.4
        pose = make_pose(crouch_spot, torso=torso_when_crouched)
        track.append(Observation(timestamp,
                                 (crouch_spot[0] - 20, crouch_spot[1] - BODY,
                                  crouch_spot[0] + 20, crouch_spot[1]),
                                 crouch_spot, (crouch_spot[0], crouch_spot[1] - 50),
                                 BODY, pose, tuple(z.id for z in index.containing(crouch_spot))))
    return track


def test_crouch_fires_when_person_drops_below_their_own_standing_height():
    cfg = settings()
    track = _crouch_track(torso_when_crouched=18.0)     # 60% of their own 30px baseline
    context = make_context([track], vehicles=[vehicle_at(820, 700)])
    result = H.crouched_near_vehicle(track, context, cfg)

    assert result.triggered, result.explanation
    assert result.inputs["lowest_crouch_ratio"] < cfg.heuristics.crouched_vehicle.crouch_ratio
    assert "their own" in result.explanation


def test_crouch_is_self_referenced_not_population_averaged():
    """A person with a naturally short torso, standing normally, is NOT crouching.

    The bias test that matters: the measure is always against this person's own
    upright posture, so a different body is never mistaken for a crouch.
    """
    cfg = settings()
    index = zone_index()
    track = TrackHistory(track_id="person-short", kind="person")
    short_torso = 16.0            # much shorter than the 30px "typical" fixture

    for i in range(10):
        foot = (500.0 + i * 30.0, 800.0)
        pose = make_pose(foot, torso=short_torso)
        track.append(Observation(i * 0.4, (foot[0] - 20, foot[1] - BODY, foot[0] + 20, foot[1]),
                                 foot, (foot[0], foot[1] - 50), BODY, pose,
                                 tuple(z.id for z in index.containing(foot))))
        track.observe_baseline(pose, track.speed(1.0))

    spot = (790.0, 800.0)
    for i in range(21):
        pose = make_pose(spot, torso=short_torso)      # same posture, standing still
        track.append(Observation(4.0 + i * 0.4,
                                 (spot[0] - 20, spot[1] - BODY, spot[0] + 20, spot[1]),
                                 spot, (spot[0], spot[1] - 50), BODY, pose,
                                 tuple(z.id for z in index.containing(spot))))

    context = make_context([track], vehicles=[vehicle_at(820, 700)])
    result = H.crouched_near_vehicle(track, context, cfg)
    assert not result.triggered, "A short torso must not read as a crouch."


def test_crouch_needs_a_baseline_before_it_will_judge_anything():
    cfg = settings()
    spot = (790.0, 800.0)
    track = make_track(
        [(i * 0.4, spot) for i in range(21)],
        pose_factory=lambda t, foot: make_pose(foot, torso=15.0),
    )
    context = make_context([track], vehicles=[vehicle_at(820, 700)])
    result = H.crouched_near_vehicle(track, context, cfg)

    assert not result.triggered
    assert "baseline" in result.explanation


# ---------------------------------------------------------------------------
# 6. Tampering motion
# ---------------------------------------------------------------------------
def _tamper_track(*, body_moves: bool) -> TrackHistory:
    index = zone_index()
    track = TrackHistory(track_id="person-1", kind="person")
    for i in range(40):
        timestamp = i * 0.1
        # 2 Hz wrist oscillation: 0.5s period, 5px each way = 0.1 body heights.
        wrist_dx = 5.0 * math.sin(2 * math.pi * 2.0 * timestamp)
        # The walker keeps moving but stays within the vehicle's vicinity, so
        # the heuristic has to reject them on BODY SPEED rather than on
        # proximity — which is the distinction this fixture exists to test.
        x = 790.0 + (i * 4.0 if body_moves else 0.0)
        foot = (x, 800.0)
        pose = make_pose(foot, wrist_dx=wrist_dx)
        track.append(Observation(timestamp, (x - 20, 700.0, x + 20, 800.0), foot,
                                 (x, 750.0), BODY, pose,
                                 tuple(z.id for z in index.containing(foot))))
    return track


def test_tampering_fires_on_repetitive_hand_movement_at_a_still_body():
    cfg = settings()
    track = _tamper_track(body_moves=False)
    context = make_context([track], vehicles=[vehicle_at(820, 700)])
    result = H.tampering_motion(track, context, cfg)

    assert result.triggered, result.explanation
    assert result.inputs["reversals"] >= cfg.heuristics.tampering.min_reversals
    assert cfg.heuristics.tampering.frequency_min <= result.inputs["frequency_hz"] <= cfg.heuristics.tampering.frequency_max


def test_tampering_does_not_fire_for_someone_walking_past_swinging_their_arms():
    cfg = settings()
    track = _tamper_track(body_moves=True)
    context = make_context([track], vehicles=[vehicle_at(820, 700)])
    result = H.tampering_motion(track, context, cfg)

    assert not result.triggered
    assert "walking" in result.explanation


# ---------------------------------------------------------------------------
# 7. Group coordination
# ---------------------------------------------------------------------------
def test_group_coordination_pairs_a_static_lookout_with_an_active_actor():
    cfg = settings()
    index = zone_index()

    # Actor stands inside the gate zone for 20s.
    actor = make_track([(i * 0.5, (300.0, 300.0)) for i in range(41)],
                       track_id="person-1", index=index)
    # Lookout stands still 4 body-heights away for the same 20s.
    lookout = make_track([(i * 0.5, (700.0, 300.0)) for i in range(41)],
                         track_id="person-2", index=index)

    context = make_context([actor, lookout], index=index)
    result = H.group_coordination(actor, context, cfg)

    assert result.triggered, result.explanation
    assert result.inputs["lookout_track_id"] == "person-2"
    assert result.inputs["separation_body_heights"] >= cfg.heuristics.group.min_separation


def test_group_coordination_does_not_fire_when_the_second_person_is_also_moving():
    cfg = settings()
    index = zone_index()
    actor = make_track([(i * 0.5, (300.0, 300.0)) for i in range(41)],
                       track_id="person-1", index=index)
    walker = make_track([(i * 0.5, (700.0 + i * 20.0, 300.0)) for i in range(41)],
                        track_id="person-2", index=index)

    context = make_context([actor, walker], index=index)
    result = H.group_coordination(actor, context, cfg)
    assert not result.triggered


# ---------------------------------------------------------------------------
# 8. Fleeing
# ---------------------------------------------------------------------------
def test_fleeing_fires_on_an_abrupt_transition_away_from_the_property():
    cfg = settings()
    samples = []
    # 3 seconds of slow movement near the gate...
    for i in range(13):
        samples.append((i * 0.25, (300.0, 420.0 + i * 5.0)))
    # ...then a sprint away: 5 body-heights/sec for 2 seconds.
    for i in range(1, 21):
        samples.append((3.0 + i * 0.1, (300.0, 480.0 + i * 50.0)))

    track = make_track(samples)
    result = H.fleeing(track, make_context([track]), cfg)

    assert result.triggered, result.explanation
    assert result.inputs["recent_peak_speed"] >= cfg.heuristics.fleeing.run_speed_min
    # The wording must stay descriptive, not accusatory.
    assert "escaping danger" in result.explanation


def test_fleeing_does_not_fire_for_a_steady_runner():
    """Sustained pace is a jogger. Fleeing is the CHANGE in pace."""
    cfg = settings()
    samples = [(i * 0.1, (300.0, 200.0 + i * 40.0)) for i in range(60)]
    track = make_track(samples)
    result = H.fleeing(track, make_context([track]), cfg)

    assert not result.triggered
    assert "sustained pace" in result.explanation


def test_fleeing_confidence_rises_near_a_logged_claims_incident():
    cfg = settings()
    samples = []
    for i in range(13):
        samples.append((i * 0.25, (300.0, 420.0 + i * 5.0)))
    # A run just over the threshold (3.2 body-heights/sec) rather than a sprint.
    # An all-out sprint already scores 1.0, and the claims bonus is multiplicative
    # on a clamped score — so the bonus is only observable below saturation.
    for i in range(1, 21):
        samples.append((3.0 + i * 0.1, (300.0, 480.0 + i * 32.0)))
    track = make_track(samples)

    without = H.fleeing(track, make_context([track]), cfg)
    with_incident = H.fleeing(track, make_context([track], incident_minutes=3.0), cfg)

    assert with_incident.confidence > without.confidence
    assert "claims data" in with_incident.explanation


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
def _fake_trigger(name: str, confidence: float) -> H.HeuristicResult:
    return H.HeuristicResult(True, confidence, f"synthetic {name}", name, {})


def test_noisy_or_accumulates_without_exceeding_one():
    cfg = settings()
    single = score_event([_fake_trigger("loitering", 1.0)], cfg)
    double = score_event(
        [_fake_trigger("loitering", 1.0), _fake_trigger("tampering_motion", 1.0)], cfg
    )

    assert single.behavioural_risk_score < 1.0, "one heuristic must never reach 1.0 alone"
    assert double.behavioural_risk_score > single.behavioural_risk_score
    assert double.behavioural_risk_score <= 1.0


def test_face_match_with_ordinary_behaviour_is_downweighted():
    """The false-positive brake: a confident face match plus normal movement."""
    cfg = settings()
    weak = [_fake_trigger("loitering", 0.25)]     # 0.25 x weight 0.45 = 0.11

    without_face = score_event(weak, cfg)
    with_face = score_event(weak, cfg, face=FaceSignal(confidence=0.9, label="suspect"))

    assert with_face.composite_risk_score > without_face.composite_risk_score
    assert any("DOWNWEIGHTED" in step for step in with_face.reasoning)
    # And the damping actually bit: without it the composite would be higher.
    undamped = 0.5 * 0.11 + 0.35 * 0.9 + 0.15 * (0.11 * 0.9)
    assert with_face.composite_risk_score < undamped


def test_verified_resident_is_damped_hardest():
    cfg = settings()
    triggers = [_fake_trigger("crouched_near_vehicle", 0.9)]
    resident = score_event(triggers, cfg, face=FaceSignal(confidence=0.9, label="verified"))
    stranger = score_event(triggers, cfg, face=FaceSignal(confidence=0.9, label="suspect"))

    assert resident.composite_risk_score < stranger.composite_risk_score
    assert not resident.requires_human_review
    assert any("verified resident" in step for step in resident.reasoning)


def test_strong_behaviour_with_no_face_match_still_reaches_review():
    """The false-negative catch — the reason this module exists."""
    cfg = settings()
    triggers = [
        _fake_trigger("concealment_approach", 0.8),
        _fake_trigger("perimeter_probing", 0.8),
    ]
    result = score_event(triggers, cfg, face=None)

    assert result.face_match_confidence is None
    assert result.requires_human_review
    assert any("no facial match" in step for step in result.reasoning)


def test_hotspot_context_cannot_create_risk_on_its_own():
    """Zone risk is multiplicative, so zero behaviour stays zero however bad
    the suburb's claims history is. The anti-redlining safeguard."""
    cfg = settings()
    nothing = score_event([], cfg, zone_risk=1.0)
    assert nothing.behavioural_risk_score == 0.0
    assert nothing.composite_risk_score == 0.0
    assert not nothing.requires_human_review


def test_hotspot_context_amplifies_existing_behaviour():
    cfg = settings()
    triggers = [_fake_trigger("perimeter_probing", 0.7)]
    flat = score_event(triggers, cfg, zone_risk=0.0)
    hotspot = score_event(triggers, cfg, zone_risk=1.0)
    assert hotspot.composite_risk_score > flat.composite_risk_score


def test_face_distance_is_converted_to_confidence_not_used_raw():
    """recognition.py returns a cosine DISTANCE. Feeding it in raw would invert
    the fusion — a perfect match would read as zero confidence."""
    near = face_signal_from_recognition(
        {"is_known_user": True, "match_distance": 0.02, "status": "offender"},
        match_threshold=0.30,
    )
    far = face_signal_from_recognition(
        {"is_known_user": True, "match_distance": 0.29, "status": "suspect"},
        match_threshold=0.30,
    )

    assert near is not None and far is not None
    assert near.confidence > far.confidence
    assert near.confidence > 0.9
    assert far.confidence < 0.1
    assert face_signal_from_recognition({"is_known_user": False}) is None
    # Identity must not cross the boundary — only confidence and coarse label.
    assert not hasattr(near, "full_name")


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_event_json_matches_the_agreed_schema():
    import api_output

    cfg = settings()
    triggers = [_fake_trigger("loitering", 0.9), _fake_trigger("fleeing", 0.8)]
    fusion = score_event(triggers, cfg)
    event = api_output.build_event(
        track_id="person-7",
        timestamp="2026-08-05T12:00:00+00:00",
        location_zone_id="demo_camera_01",
        fusion=fusion,
        triggered=triggers,
    )

    required = {
        "track_id", "timestamp", "location_zone_id", "behavioural_risk_score",
        "triggered_heuristics", "face_match_confidence", "composite_risk_score",
        "requires_human_review",
    }
    assert required.issubset(event.keys())
    assert isinstance(event["requires_human_review"], bool)
    assert event["face_match_confidence"] is None

    for trigger in event["triggered_heuristics"]:
        assert set(trigger) >= {"type", "confidence", "explanation"}
        # Explanations are mandatory — no opaque scores anywhere.
        assert trigger["explanation"].strip()

    # No identity data may appear in the payload.
    forbidden = {"person", "full_name", "face_id", "member_id", "embedding", "image_url", "status"}
    assert not (forbidden & set(event.keys()))


def test_push_to_flask_api_is_a_stub_and_does_not_post():
    import api_output

    outcome = api_output.push_to_flask_api({"event_id": "x", "track_id": "person-1"})
    assert outcome["pushed"] is False
    assert outcome["reason"] == "stub"


def test_every_heuristic_returns_an_explanation_even_when_it_does_not_fire():
    cfg = settings()
    track = make_track([(i * 0.5, (300.0, 300.0)) for i in range(20)],
                       pose_factory=lambda t, foot: make_pose(foot))
    context = make_context([track])

    results = H.evaluate_all(track, context, cfg, include_misses=True)
    assert len(results) == len(H.HEURISTICS)
    for result in results:
        assert result.explanation.strip(), f"{result.name} returned an empty explanation"


def test_config_thresholds_are_all_loadable():
    """Every heuristic reads its numbers from config.yaml — no magic numbers."""
    cfg = settings()
    for name in H.HEURISTIC_NAMES:
        assert cfg.heuristic_weight(name) > 0, f"no fusion weight configured for {name}"


# ---------------------------------------------------------------------------
# Minimal runner, so pytest is not required
# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    passed, failed = 0, []

    for name, function in tests:
        try:
            function()
            passed += 1
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failed.append((name, f"assertion: {exc}"))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("\nFailures:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
