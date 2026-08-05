"""OpenCV debug overlay — see what the pipeline sees.

Draws zone polygons, tracked bounding boxes, pose skeletons and live scores onto
a frame so detection can be verified visually during the demo, and so a
threshold that is obviously wrong looks obviously wrong.

Colours are BGR (OpenCV's order, not RGB).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from detector import Detection, is_confirmed
from pose_extractor import SKELETON_EDGES, PoseKeypoints
from zones import ZoneIndex

# Zone colours by type.
ZONE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "property_boundary": (200, 120, 60),
    "gate": (60, 180, 240),
    "vertical_structure": (200, 60, 200),
    "vehicle_zone": (80, 200, 120),
    "street_frontage": (150, 150, 150),
    "exempt": (60, 220, 60),          # green: the bias guard, visible on purpose
}

PERSON_COLOUR = (255, 200, 80)
VEHICLE_COLOUR = (120, 220, 120)
PENDING_COLOUR = (120, 120, 120)
SKELETON_COLOUR = (240, 240, 240)
JOINT_COLOUR = (80, 160, 255)

# Risk banding for the per-track score readout.
LOW_COLOUR = (140, 220, 140)
MEDIUM_COLOUR = (80, 200, 255)
HIGH_COLOUR = (80, 80, 255)


def risk_colour(score: float, review_threshold: float) -> Tuple[int, int, int]:
    if score >= review_threshold:
        return HIGH_COLOUR
    if score >= review_threshold * 0.6:
        return MEDIUM_COLOUR
    return LOW_COLOUR


def draw_zones(frame: np.ndarray, zone_index: ZoneIndex, *, alpha: float = 0.18) -> np.ndarray:
    """Translucent zone polygons with labels."""
    overlay = frame.copy()
    for zone in zone_index.zones:
        colour = ZONE_COLOURS.get(zone.type, (180, 180, 180))
        points = np.array([[int(x), int(y)] for x, y in zone.polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [points], colour)
        cv2.polylines(frame, [points], isClosed=True, color=colour, thickness=2)

        label_x, label_y = points[0]
        label = f"{zone.id} [{zone.type}]"
        if zone.is_exempt:
            label += " — loitering never fires here"
        cv2.putText(
            frame, label, (int(label_x) + 4, int(label_y) + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    *,
    score: Optional[float] = None,
    review_threshold: float = 0.5,
    triggered: Sequence[str] = (),
) -> None:
    """One bounding box, its track id, and its current composite score."""
    x1, y1, x2, y2 = (int(v) for v in detection.bbox)

    if not is_confirmed(detection.track_id):
        colour = PENDING_COLOUR
    elif detection.is_person():
        colour = risk_colour(score, review_threshold) if score is not None else PERSON_COLOUR
    else:
        colour = VEHICLE_COLOUR

    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

    label = detection.track_id
    if score is not None:
        label += f"  risk {score:.2f}"

    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 8, y1), colour, -1)
    cv2.putText(
        frame, label, (x1 + 4, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
    )

    for i, name in enumerate(triggered):
        cv2.putText(
            frame, f"! {name}", (x1 + 2, y2 + 16 + i * 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, HIGH_COLOUR, 1, cv2.LINE_AA,
        )

    # The foot point — the anchor every position measurement actually uses.
    foot_x, foot_y = (int(v) for v in detection.foot_point)
    cv2.circle(frame, (foot_x, foot_y), 4, colour, -1)


def draw_pose(frame: np.ndarray, pose: PoseKeypoints) -> None:
    """Skeleton for one person. Only landmarks MediaPipe actually saw are drawn,
    so a sparse skeleton on screen means sparse data in the heuristics too."""
    for start, end in SKELETON_EDGES:
        a, b = pose.point(start), pose.point(end)
        if a is None or b is None:
            continue
        cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), SKELETON_COLOUR, 2, cv2.LINE_AA)

    for index in range(len(pose.landmarks)):
        point = pose.point(index)
        if point is not None:
            cv2.circle(frame, (int(point.x), int(point.y)), 3, JOINT_COLOUR, -1)


def draw_hud(
    frame: np.ndarray,
    *,
    timestamp: float,
    frame_index: int,
    tracked: int,
    events: int,
    fps: Optional[float] = None,
    note: str = "",
) -> None:
    """Run status, top-left."""
    height, width = frame.shape[:2]
    lines = [
        f"t={timestamp:6.2f}s  frame {frame_index}",
        f"tracks: {tracked}   events: {events}" + (f"   {fps:.1f} fps" if fps else ""),
    ]
    if note:
        lines.append(note)

    box_height = 12 + 20 * len(lines)
    cv2.rectangle(frame, (0, 0), (min(width, 430), box_height), (25, 25, 25), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (10, 22 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA,
        )


def draw_event_banner(frame: np.ndarray, event: Dict[str, object], *, lines: int = 3) -> None:
    """The most recent event, along the bottom — what a judge reads."""
    height, width = frame.shape[:2]
    triggers = event.get("triggered_heuristics", []) or []
    text_lines = [
        f"{event.get('track_id')}  composite {float(event.get('composite_risk_score', 0)):.2f}"
        + ("   REVIEW REQUIRED" if event.get("requires_human_review") else "")
    ]
    for trigger in triggers[:lines]:
        explanation = str(trigger.get("explanation", ""))
        text_lines.append(f"- {trigger.get('type')}: {explanation[:96]}")

    box_top = height - (14 + 20 * len(text_lines))
    cv2.rectangle(frame, (0, box_top), (width, height), (25, 25, 25), -1)
    colour = HIGH_COLOUR if event.get("requires_human_review") else MEDIUM_COLOUR
    for i, line in enumerate(text_lines):
        cv2.putText(
            frame, line, (10, box_top + 20 + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            colour if i == 0 else (225, 225, 225), 1, cv2.LINE_AA,
        )


def render_frame(
    frame: np.ndarray,
    *,
    zone_index: ZoneIndex,
    detections: Sequence[Detection],
    poses: Dict[str, PoseKeypoints],
    scores: Dict[str, float],
    triggered: Dict[str, List[str]],
    timestamp: float,
    frame_index: int,
    events_so_far: int,
    review_threshold: float,
    latest_event: Optional[Dict[str, object]] = None,
    fps: Optional[float] = None,
    note: str = "",
) -> np.ndarray:
    """Everything, on a copy of the frame."""
    canvas = frame.copy()
    draw_zones(canvas, zone_index)

    for detection in detections:
        draw_detection(
            canvas,
            detection,
            score=scores.get(detection.track_id),
            review_threshold=review_threshold,
            triggered=triggered.get(detection.track_id, ()),
        )

    for pose in poses.values():
        draw_pose(canvas, pose)

    draw_hud(
        canvas,
        timestamp=timestamp,
        frame_index=frame_index,
        tracked=len(detections),
        events=events_so_far,
        fps=fps,
        note=note,
    )

    if latest_event is not None:
        draw_event_banner(canvas, latest_event)

    return canvas
