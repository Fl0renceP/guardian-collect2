"""Stage 4 — per-track history: where something has been, and how fast.

Keeps a rolling window of observations for each track ID and answers the
temporal questions the heuristics ask: how long has this person been near the
gate, how fast are they moving, are they closing on the property or leaving it,
how does their posture now compare to how they normally stand.

Everything spatial is returned in BODY HEIGHTS (multiples of the person's own
bounding-box height) and everything temporal in SECONDS. No heuristic ever sees
a pixel, which is what lets one config work across cameras with no calibration.

Position is taken from the FOOT POINT (bottom-centre of the box), not the
centroid. A centroid moves half a metre when someone crouches; feet do not.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from detector import Detection, is_confirmed
from pose_extractor import (
    LEFT_WRIST,
    RIGHT_WRIST,
    PoseKeypoints,
)
from zones import PixelZone, ZoneIndex

Point = Tuple[float, float]


@dataclass
class Observation:
    """One track, one frame."""

    timestamp: float
    bbox: Tuple[float, float, float, float]
    foot: Point
    centroid: Point
    body_height: float                    # pixels — the scale reference
    pose: Optional[PoseKeypoints] = None
    zone_ids: Tuple[str, ...] = ()        # zones containing the foot point


@dataclass
class TrackHistory:
    """Rolling history for one anonymous track ID."""

    track_id: str
    kind: str
    observations: Deque[Observation] = field(default_factory=deque)
    first_seen: float = 0.0
    last_seen: float = 0.0
    # Learned from this person's own first upright frames. The self-referenced
    # baseline that makes "crouched" mean "lower than they normally stand"
    # rather than "shorter than average" — see the bias notice in heuristics.py.
    _standing_torso: Optional[float] = None
    _baseline_samples: List[float] = field(default_factory=list)

    # -- basics ------------------------------------------------------------
    @property
    def latest(self) -> Optional[Observation]:
        return self.observations[-1] if self.observations else None

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def body_height(self) -> float:
        """Median recent box height in pixels — the unit everything divides by.

        Median rather than latest: a single frame's box can jitter by 20%, and
        a jittering scale reference makes every distance and speed jitter with it.
        """
        recent = [o.body_height for o in list(self.observations)[-9:] if o.body_height > 0]
        if not recent:
            return 0.0
        return statistics.median(recent)

    @property
    def poses(self) -> List[PoseKeypoints]:
        return [o.pose for o in self.observations if o.pose is not None]

    @property
    def latest_pose(self) -> Optional[PoseKeypoints]:
        for observation in reversed(self.observations):
            if observation.pose is not None:
                return observation.pose
        return None

    def window(self, seconds: float, now: Optional[float] = None) -> List[Observation]:
        """Observations from the last `seconds`."""
        if not self.observations:
            return []
        end = self.last_seen if now is None else now
        cutoff = end - seconds
        return [o for o in self.observations if o.timestamp >= cutoff]

    # -- motion ------------------------------------------------------------
    def speed(self, seconds: float = 1.0, now: Optional[float] = None) -> Optional[float]:
        """Body heights per second, from net displacement over the window.

        Net displacement rather than accumulated path length, because tracker
        jitter accumulates: a stationary person's box wobbles a few pixels a
        frame, and summing those wobbles turns standing still into a slow walk.
        Net displacement over a window is immune to that.

        For reference: standing ~0, strolling ~0.6, brisk walk ~1.2, run 3+.
        """
        window = self.window(seconds, now)
        if len(window) < 2:
            return None
        scale = self.body_height
        if scale <= 0:
            return None

        elapsed = window[-1].timestamp - window[0].timestamp
        if elapsed <= 1e-6:
            return None

        distance = math.dist(window[0].foot, window[-1].foot)
        return (distance / scale) / elapsed

    def peak_speed(self, seconds: float, sample: float = 0.5, now: Optional[float] = None) -> Optional[float]:
        """Highest short-window speed inside a longer window."""
        window = self.window(seconds, now)
        if len(window) < 3:
            return None
        scale = self.body_height
        if scale <= 0:
            return None

        peak = 0.0
        for i, start in enumerate(window):
            for end in window[i + 1:]:
                elapsed = end.timestamp - start.timestamp
                if elapsed < sample:
                    continue
                speed = (math.dist(start.foot, end.foot) / scale) / elapsed
                peak = max(peak, speed)
                break
        return peak or None

    def displacement(self, seconds: float, now: Optional[float] = None) -> Optional[Point]:
        """(dx, dy) in body heights over the window."""
        window = self.window(seconds, now)
        if len(window) < 2:
            return None
        scale = self.body_height
        if scale <= 0:
            return None
        return (
            (window[-1].foot[0] - window[0].foot[0]) / scale,
            (window[-1].foot[1] - window[0].foot[1]) / scale,
        )

    def vertical_rise(self, seconds: float, now: Optional[float] = None) -> Optional[float]:
        """How far the feet have risen, in body heights. Positive = off the ground.

        Image y grows downward, so a rise is a DECREASE in y. Someone scaling a
        wall lifts their ground contact point; someone walking never does.
        """
        window = self.window(seconds, now)
        if len(window) < 2:
            return None
        scale = self.body_height
        if scale <= 0:
            return None
        return (max(o.foot[1] for o in window) - window[-1].foot[1]) / scale

    # -- zones -------------------------------------------------------------
    def dwell_seconds(
        self,
        zone_ids: Iterable[str],
        seconds: float,
        *,
        max_speed: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """Time spent inside any of these zones during the window.

        With `max_speed`, only time spent *slowly* counts — walking through a
        zone is not dwelling in it.
        """
        wanted = set(zone_ids)
        window = self.window(seconds, now)
        if len(window) < 2:
            return 0.0

        scale = self.body_height
        total = 0.0
        for previous, current in zip(window, window[1:]):
            if not (set(previous.zone_ids) & wanted):
                continue
            elapsed = current.timestamp - previous.timestamp
            if elapsed <= 0:
                continue
            if max_speed is not None and scale > 0:
                step_speed = (math.dist(previous.foot, current.foot) / scale) / elapsed
                if step_speed > max_speed:
                    continue
            total += elapsed
        return total

    def zone_entries(
        self,
        zone_ids: Iterable[str],
        seconds: float,
        now: Optional[float] = None,
    ) -> int:
        """How many times the track entered these zones during the window.

        Repeated entries are the "casing" signal — going back and forth past a
        property says more than standing near it once.
        """
        wanted = set(zone_ids)
        window = self.window(seconds, now)
        entries = 0
        was_inside = False
        for observation in window:
            inside = bool(set(observation.zone_ids) & wanted)
            if inside and not was_inside:
                entries += 1
            was_inside = inside
        return entries

    def distance_to(self, zone: PixelZone, now: Optional[float] = None) -> Optional[float]:
        """Current distance to a zone, in body heights."""
        observation = self.latest if now is None else self._at(now)
        if observation is None:
            return None
        scale = self.body_height
        if scale <= 0:
            return None
        return zone.distance(observation.foot) / scale

    def distance_series(self, zone: PixelZone, seconds: float, now: Optional[float] = None):
        """[(timestamp, distance_in_body_heights)] over the window."""
        scale = self.body_height
        if scale <= 0:
            return []
        return [(o.timestamp, zone.distance(o.foot) / scale) for o in self.window(seconds, now)]

    def closing_speed(
        self, zone: PixelZone, seconds: float, now: Optional[float] = None
    ) -> Optional[float]:
        """Body heights per second the track is CLOSING on a zone.

        Positive = approaching, negative = leaving. This is the signal that
        separates "walking toward the gate" from "walking past it", and it is
        required by both the concealment and probing heuristics precisely so
        that merely passing a property can never trigger them.
        """
        series = self.distance_series(zone, seconds, now)
        if len(series) < 2:
            return None
        elapsed = series[-1][0] - series[0][0]
        if elapsed <= 1e-6:
            return None
        return (series[0][1] - series[-1][1]) / elapsed

    # -- posture -----------------------------------------------------------
    def observe_baseline(self, pose: PoseKeypoints, speed: Optional[float]) -> None:
        """Learn this person's own standing torso height while they walk upright."""
        torso = pose.torso_height
        if torso is None or torso <= 0:
            return
        # Only sample while actually moving: sampling a crouch as the baseline
        # would make the crouch look normal and hide it forever after.
        if speed is None or speed < 0.35:
            return
        knee = pose.min_knee_angle
        if knee is not None and knee < 150:
            return
        self._baseline_samples.append(torso)
        if len(self._baseline_samples) >= 3:
            self._standing_torso = statistics.median(self._baseline_samples[-15:])

    @property
    def standing_torso(self) -> Optional[float]:
        """This person's own upright torso height in pixels, if learned yet."""
        return self._standing_torso

    def crouch_ratio(self, pose: Optional[PoseKeypoints] = None) -> Optional[float]:
        """Current torso height as a fraction of this person's own standing height.

        1.0 = standing as they normally do. 0.6 = markedly lower than their own
        norm. Returns None until a baseline exists — better to say nothing than
        to compare someone against an assumed average body.
        """
        pose = pose or self.latest_pose
        if pose is None or not self._standing_torso:
            return None
        torso = pose.torso_height
        if torso is None or self._standing_torso <= 0:
            return None
        return torso / self._standing_torso

    def wrist_series(
        self, seconds: float, now: Optional[float] = None
    ) -> Dict[str, List[Tuple[float, Point]]]:
        """Wrist positions over the window, in body heights, per side."""
        scale = self.body_height
        out: Dict[str, List[Tuple[float, Point]]] = {"left": [], "right": []}
        if scale <= 0:
            return out

        for observation in self.window(seconds, now):
            if observation.pose is None:
                continue
            for side, index in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
                point = observation.pose.point(index)
                if point is not None:
                    out[side].append(
                        (observation.timestamp, (point.x / scale, point.y / scale))
                    )
        return out

    def head_yaw_series(self, seconds: float, now: Optional[float] = None) -> List[float]:
        return [
            o.pose.head_yaw_degrees
            for o in self.window(seconds, now)
            if o.pose is not None and o.pose.head_yaw_degrees is not None
        ]

    def face_visibility_series(self, seconds: float, now: Optional[float] = None) -> List[Tuple[float, float]]:
        return [
            (o.timestamp, o.pose.face_visibility)
            for o in self.window(seconds, now)
            if o.pose is not None
        ]

    # -- internals ---------------------------------------------------------
    def _at(self, timestamp: float) -> Optional[Observation]:
        if not self.observations:
            return None
        return min(self.observations, key=lambda o: abs(o.timestamp - timestamp))

    def append(self, observation: Observation) -> None:
        if not self.observations:
            self.first_seen = observation.timestamp
        self.observations.append(observation)
        self.last_seen = observation.timestamp

    def prune(self, history_seconds: float) -> None:
        cutoff = self.last_seen - history_seconds
        while self.observations and self.observations[0].timestamp < cutoff:
            self.observations.popleft()


@dataclass
class SceneContext:
    """Everything a heuristic can see about the current moment."""

    timestamp: float
    frame_index: int
    frame_size: Tuple[int, int]
    zones: ZoneIndex
    tracks: Dict[str, TrackHistory]
    vehicles: List[Detection]
    # Heuristics already fired this frame, per track. Lets group coordination
    # ask "is the other person actually doing something" without re-running
    # every heuristic.
    triggered_this_frame: Dict[str, List[str]] = field(default_factory=dict)
    # Minutes between this moment and the nearest incident logged in the claims
    # data for this location, when known. This is the hook into Discovery's
    # claims dataset — the `fleeing` heuristic weights an abrupt departure more
    # heavily when it lands close to a real logged incident. None means "no
    # claims context available", which is the normal standalone case.
    nearby_incident_minutes: Optional[float] = None

    def person_tracks(self) -> List[TrackHistory]:
        return [t for t in self.tracks.values() if t.kind == "person"]

    def nearest_vehicle(self, bbox: Sequence[float]) -> Tuple[Optional[Detection], float]:
        """Closest vehicle to a box, and the horizontal pixel gap to it."""
        from zones import bbox_horizontal_gap

        best: Optional[Detection] = None
        best_gap = math.inf
        for vehicle in self.vehicles:
            gap = bbox_horizontal_gap(bbox, vehicle.bbox)
            if gap < best_gap:
                best, best_gap = vehicle, gap
        return best, best_gap

    def zone_risk_for(self, track: TrackHistory) -> float:
        observation = track.latest
        return self.zones.risk_at(observation.foot) if observation else 0.0


class TrajectoryTracker:
    """Owns every track's history and updates it once per frame."""

    def __init__(self, *, history_seconds: float = 30.0, drop_after_seconds: float = 5.0):
        self.history_seconds = history_seconds
        self.drop_after_seconds = drop_after_seconds
        self.tracks: Dict[str, TrackHistory] = {}

    def update(
        self,
        *,
        timestamp: float,
        detections: Sequence[Detection],
        poses: Dict[str, PoseKeypoints],
        zone_index: ZoneIndex,
    ) -> List[TrackHistory]:
        """Fold this frame's detections into the histories. Returns updated tracks."""
        updated: List[TrackHistory] = []

        for detection in detections:
            # An object without confirmed tracking continuity has no trajectory,
            # and every heuristic here is about trajectory. Draw it, ignore it.
            if not is_confirmed(detection.track_id):
                continue

            track = self.tracks.get(detection.track_id)
            if track is None:
                track = TrackHistory(track_id=detection.track_id, kind=detection.kind)
                self.tracks[detection.track_id] = track

            pose = poses.get(detection.track_id)
            foot = detection.foot_point
            observation = Observation(
                timestamp=timestamp,
                bbox=detection.bbox,
                foot=foot,
                centroid=detection.centroid,
                body_height=detection.height,
                pose=pose,
                zone_ids=tuple(z.id for z in zone_index.containing(foot)),
            )
            track.append(observation)
            track.prune(self.history_seconds)

            if pose is not None:
                track.observe_baseline(pose, track.speed(1.0))

            updated.append(track)

        self._drop_stale(timestamp)
        return updated

    def _drop_stale(self, timestamp: float) -> None:
        """Forget tracks that left the frame.

        Also the retention policy: nothing about a person who has walked away is
        kept once they are gone. Only emitted events persist, in the audit log.
        """
        stale = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen > self.drop_after_seconds
        ]
        for track_id in stale:
            del self.tracks[track_id]
