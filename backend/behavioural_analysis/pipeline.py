"""The pipeline, assembled.

Wires the seven stages together:

    frames -> YOLO+ByteTrack -> MediaPipe Pose -> trajectory history
           -> 8 heuristics -> fusion -> JSON event -> audit log

Kept free of CLI concerns so `main.py`, `run_sample.py` and (later) a Flask
route can all drive the same code.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import api_output
from audit_log import AuditLog, from_settings as audit_from_settings
from detector import Detection, ObjectDetector, is_confirmed
from frame_ingest import Frame, SourceInfo, iter_frames, open_source
from heuristics import HeuristicResult, evaluate_all
from pose_extractor import PoseExtractor, PoseKeypoints
from risk_fusion import FaceSignal, score_event
from settings import Settings, load_settings
from trajectory_tracker import SceneContext, TrackHistory, TrajectoryTracker
from zones import ZoneIndex

logger = logging.getLogger(__name__)

# Given a track id, return what the SEPARATE face module concluded about it, or
# None. Standalone runs have no provider, which is the case this module is built
# to handle: behaviour has to stand on its own.
FaceProvider = Callable[[str], Optional[FaceSignal]]


@dataclass
class FrameResult:
    """One processed frame — everything the debug view and the caller need."""

    frame: Frame
    detections: List[Detection]
    poses: Dict[str, PoseKeypoints]
    scores: Dict[str, float] = field(default_factory=dict)
    triggered: Dict[str, List[str]] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


class BehaviouralPipeline:
    """Runs the behavioural analysis over a video source."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        face_provider: Optional[FaceProvider] = None,
        audit: Optional[AuditLog] = None,
        wall_clock_start: Optional[datetime] = None,
        explain_misses: bool = False,
    ):
        self.settings = settings or load_settings()
        self.face_provider = face_provider
        self.audit = audit if audit is not None else audit_from_settings(self.settings)
        # Records why heuristics did NOT fire. Knowing that a trigger missed by
        # three seconds is what makes a threshold tunable — and it is the
        # difference between "the system saw nothing" and "the system saw it
        # and decided it did not qualify".
        self.explain_misses = explain_misses
        self.miss_reasons: Counter = Counter()

        pipeline_cfg = self.settings.pipeline
        self.detector = ObjectDetector(
            model_name=pipeline_cfg.yolo_model,
            confidence=float(pipeline_cfg.yolo_confidence),
            tracker=pipeline_cfg.tracker,
            imgsz=int(pipeline_cfg.get("yolo_imgsz", 640)),
        )
        self.pose_extractor = PoseExtractor(
            model_complexity=int(pipeline_cfg.pose_model_complexity),
            min_detection_confidence=float(pipeline_cfg.pose_min_detection_confidence),
            min_tracking_confidence=float(pipeline_cfg.pose_min_tracking_confidence),
            visibility_floor=float(pipeline_cfg.landmark_visibility_floor),
            min_crop_height=int(pipeline_cfg.get("pose_min_crop_height", 64)),
            min_crop_width=int(pipeline_cfg.get("pose_min_crop_width", 24)),
        )
        self.tracker = TrajectoryTracker(history_seconds=float(pipeline_cfg.history_seconds))

        self._zone_index: Optional[ZoneIndex] = None
        # (track_id, heuristic) -> stream time it last produced an event.
        self._cooldowns: Dict[Tuple[str, str], float] = {}
        self._events: List[Dict[str, Any]] = []
        # Video time is relative to the start of the clip; events need a real
        # date. For a file, anchor to now unless the caller supplies otherwise.
        self._wall_clock_start = wall_clock_start or datetime.now(timezone.utc)

    # -- helpers -----------------------------------------------------------
    @property
    def events(self) -> List[Dict[str, Any]]:
        return self._events

    def _zones_for(self, width: int, height: int) -> ZoneIndex:
        if self._zone_index is None or not self._zone_index.matches(width, height):
            self._zone_index = ZoneIndex(self.settings.zones, width, height)
            logger.info(
                "Projected %d zone(s) onto a %dx%d frame.",
                len(self._zone_index.zones), width, height,
            )
        return self._zone_index

    def event_timestamp(self, stream_seconds: float) -> datetime:
        """Stream time -> wall clock.

        Public because the body-position snapshots need it too: those are
        correlated against face scans taken by a browser, and the two only line
        up on real time.
        """
        return self._wall_clock_start + timedelta(seconds=stream_seconds)

    # Kept for internal callers written before this was public.
    _event_timestamp = event_timestamp

    def _in_cooldown(self, track_id: str, heuristic: str, now: float) -> bool:
        last = self._cooldowns.get((track_id, heuristic))
        if last is None:
            return False
        return (now - last) < float(self.settings.output.event_cooldown_seconds)

    def warm_up(self) -> None:
        """Load both models before the clock starts."""
        self.detector.warm_up()
        self.pose_extractor.warm_up()

    def reset(self) -> None:
        """Clear all state. Call between videos."""
        self.detector.reset()
        self.tracker = TrajectoryTracker(
            history_seconds=float(self.settings.pipeline.history_seconds)
        )
        self._cooldowns.clear()
        self._events.clear()

    # -- per-frame ---------------------------------------------------------
    def process_frame(self, frame: Frame) -> FrameResult:
        """Detect, pose, track, evaluate and score one frame."""
        zone_index = self._zones_for(frame.width, frame.height)
        detections = self.detector.detect(frame.image)

        poses: Dict[str, PoseKeypoints] = {}
        for detection in detections:
            if not detection.is_person() or not is_confirmed(detection.track_id):
                continue
            pose = self.pose_extractor.extract(frame.image, detection.bbox)
            if pose is not None:
                poses[detection.track_id] = pose

        self.tracker.update(
            timestamp=frame.timestamp,
            detections=detections,
            poses=poses,
            zone_index=zone_index,
        )

        context = SceneContext(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            frame_size=frame.size,
            zones=zone_index,
            tracks=self.tracker.tracks,
            vehicles=[d for d in detections if not d.is_person()],
        )

        result = FrameResult(frame=frame, detections=detections, poses=poses)
        min_frames = int(self.settings.pipeline.min_track_frames)

        for track in context.person_tracks():
            # One noisy frame should never become an event.
            if track.frame_count < min_frames:
                continue

            evaluated = evaluate_all(
                track, context, self.settings, include_misses=self.explain_misses
            )
            if self.explain_misses:
                for evaluation in evaluated:
                    if not evaluation.triggered:
                        self.miss_reasons[(evaluation.name, evaluation.explanation)] += 1

            triggered = [r for r in evaluated if r.triggered]
            if not triggered:
                continue

            result.triggered[track.track_id] = [r.name for r in triggered]

            face = self.face_provider(track.track_id) if self.face_provider else None
            fusion = score_event(
                triggered,
                self.settings,
                face=face,
                zone_risk=context.zone_risk_for(track),
            )
            result.scores[track.track_id] = fusion.composite_risk_score

            # Cooldown: a heuristic that has just produced an event for this
            # track stays quiet for a while. Without it a 30s clip emits
            # hundreds of copies of the same finding. The event still carries
            # every currently-triggered heuristic, so the score is never
            # partial — only the re-emission is suppressed.
            fresh = [r for r in triggered if not self._in_cooldown(track.track_id, r.name, frame.timestamp)]
            if not fresh:
                continue
            for r in triggered:
                self._cooldowns[(track.track_id, r.name)] = frame.timestamp

            event = api_output.build_event(
                track_id=track.track_id,
                timestamp=self._event_timestamp(frame.timestamp),
                location_zone_id=self.settings.output.location_zone_id,
                fusion=fusion,
                triggered=triggered,
            )
            event["stream_time_seconds"] = round(frame.timestamp, 2)
            event["frame_index"] = frame.index

            self._events.append(event)
            result.events.append(event)

            if self.audit is not None:
                self.audit.record_event(
                    event,
                    reasoning=fusion.reasoning,
                    source=str(getattr(self, "_source_label", "")),
                    config_source=str(self.settings.source_path),
                    heuristic_inputs=api_output.heuristic_inputs(triggered),
                )

        return result

    # -- whole run ---------------------------------------------------------
    def run(
        self,
        source: str | int | Path,
        *,
        max_seconds: Optional[float] = None,
        start_seconds: float = 0.0,
        stride: Optional[int] = None,
        on_frame: Optional[Callable[[FrameResult], bool]] = None,
    ) -> Iterator[FrameResult]:
        """Process a whole source, yielding each frame's result.

        `on_frame` may return False to stop early — that is how the debug window
        handles someone pressing q.
        """
        capture, info = open_source(source)
        self._source_label = info.source
        logger.info("Source: %s", info.describe())

        stride = stride if stride is not None else int(self.settings.pipeline.frame_stride)
        started = time.monotonic()
        processed = 0

        try:
            for frame in iter_frames(
                source,
                stride=stride,
                start_seconds=start_seconds,
                max_seconds=max_seconds,
                capture=capture,
                info=info,
            ):
                result = self.process_frame(frame)
                processed += 1
                yield result

                if on_frame is not None and on_frame(result) is False:
                    break
        finally:
            capture.release()
            elapsed = time.monotonic() - started
            logger.info(
                "Processed %d frames in %.1fs (%.1f frames/sec), producing %d event(s).",
                processed, elapsed, processed / elapsed if elapsed else 0.0, len(self._events),
            )

    def close(self) -> None:
        self.pose_extractor.close()
        if self.audit is not None:
            self.audit.close()

    def __enter__(self) -> "BehaviouralPipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def static_face_provider(confidence: Optional[float], label: Optional[str]) -> Optional[FaceProvider]:
    """A fixed face signal for every track — a DEMO AID, not a data source.

    It exists so the fusion rules can be shown working (a face match with normal
    behaviour being damped; a verified resident being damped harder) without
    standing up the facial recognition module. The value is whatever the
    operator typed on the command line, and the run says so on screen.

    In production, `face_provider` is backed by the real face module and returns
    None whenever there is no match — which is most of the time, and is exactly
    the case this whole module exists to cover.
    """
    if confidence is None and label is None:
        return None

    signal = FaceSignal(confidence=confidence, label=label)

    def provider(_track_id: str) -> Optional[FaceSignal]:
        return signal

    return provider
