"""Stage 2 — object detection and tracking (YOLOv8 + ByteTrack).

Detects people and vehicles per frame and gives each a persistent track ID, so
"the same person" can be followed across frames. Ultralytics' `.track()` runs
ByteTrack internally and keeps that state between calls when `persist=True`.

The track ID is the ONLY identifier this module ever produces. It is an
anonymous per-run counter — track 3 in this video has no relationship to track 3
in any other video, and it maps to no person, no member, no claim, and nothing
in the face registry. That is deliberate: the behavioural module reasons about
movement, not identity (PROJECT_CONTEXT §9, POPIA).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO class ids as emitted by the stock YOLOv8 weights.
PERSON_CLASS = 0
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACKED_CLASSES = [PERSON_CLASS, *sorted(VEHICLE_CLASSES)]


@dataclass
class Detection:
    """One tracked object in one frame. Pixel coordinates."""

    track_id: str          # e.g. "person-4" / "vehicle-1" — anonymous, per-run
    kind: str              # "person" | "vehicle"
    class_name: str        # "person" | "car" | "truck" | ...
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 in pixels

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding-box height in pixels.

        For a person this is the scale reference the whole module runs on: every
        spatial threshold is expressed in multiples of this number, which is
        what makes one config work on a doorbell close-up and a driveway wide
        shot without camera calibration.
        """
        return self.bbox[3] - self.bbox[1]

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def foot_point(self) -> Tuple[float, float]:
        """Bottom-centre of the box — where the object meets the ground.

        Used instead of the centroid for anything about position on the ground
        (zone membership, dwell, distance to a gate). A centroid rises and falls
        with posture: crouch, and your centroid moves half a metre without you
        going anywhere. Feet stay put.
        """
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def is_person(self) -> bool:
        return self.kind == "person"


class ObjectDetector:
    """YOLOv8 + ByteTrack, wrapped so the rest of the pipeline sees dataclasses."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        *,
        confidence: float = 0.35,
        tracker: str = "bytetrack.yaml",
        device: Optional[str] = None,
        imgsz: int = 640,
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.tracker = tracker
        self.device = device
        self.imgsz = imgsz
        self._model = None
        # Ultralytics only assigns a track id once a detection has been confirmed
        # over several frames; unconfirmed ones come back with id=None. We keep
        # our own counter for those so they can still be drawn, but they are
        # never handed to the heuristics — an object without continuity has no
        # trajectory, and every heuristic here is about trajectory.
        self._unconfirmed = 0

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # imported lazily: it pulls in torch

            logger.info("Loading YOLO weights: %s", self.model_name)
            self._model = YOLO(self.model_name)
            if self.device:
                self._model.to(self.device)
        return self._model

    def warm_up(self) -> None:
        """Load weights up front so the first real frame is not the slow one."""
        model = self._load()
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        model.predict(blank, verbose=False, conf=self.confidence)

    def reset(self) -> None:
        """Drop tracker state. Call between videos, or track ids leak across."""
        if self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                for tracker in predictor.trackers:
                    tracker.reset()
        self._unconfirmed = 0

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect and track people and vehicles in one frame."""
        model = self._load()
        results = model.track(
            image,
            persist=True,          # keep ByteTrack state across calls
            verbose=False,
            conf=self.confidence,
            classes=TRACKED_CLASSES,
            tracker=self.tracker,
            imgsz=self.imgsz,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        raw_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        detections: List[Detection] = []

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i].tolist())

            if class_id == PERSON_CLASS:
                kind, class_name = "person", "person"
            elif class_id in VEHICLE_CLASSES:
                kind, class_name = "vehicle", VEHICLE_CLASSES[class_id]
            else:
                continue

            raw_id = raw_ids[i]
            if raw_id is None:
                self._unconfirmed += 1
                track_id = f"{kind}-pending-{self._unconfirmed}"
            else:
                track_id = f"{kind}-{int(raw_id)}"

            detections.append(
                Detection(
                    track_id=track_id,
                    kind=kind,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )

        return detections


def split_by_kind(detections: Sequence[Detection]) -> Dict[str, List[Detection]]:
    """Convenience split into {"person": [...], "vehicle": [...]}."""
    grouped: Dict[str, List[Detection]] = {"person": [], "vehicle": []}
    for detection in detections:
        grouped.setdefault(detection.kind, []).append(detection)
    return grouped


def is_confirmed(track_id: str) -> bool:
    """False for the placeholder ids given to not-yet-confirmed detections."""
    return "-pending-" not in track_id
