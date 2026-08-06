"""Stage 7 — JSON output, and the interface to the Flask backend.

The event schema is fixed:

    {
      "track_id": str,
      "timestamp": iso8601,
      "location_zone_id": str,
      "behavioural_risk_score": float,
      "triggered_heuristics": [{"type": str, "confidence": float, "explanation": str}],
      "face_match_confidence": float | null,
      "composite_risk_score": float,
      "requires_human_review": bool
    }

Plus two additive fields the schema does not forbid and a reviewer needs:
`event_id` (so the same event can be referenced twice without duplicating it)
and `reasoning` (the plain-English trail of how the composite was arrived at).

NOTE ON WHAT IS ABSENT. There is no `alert`, no `status`, no `offender`, no
`suspect`, and no member or person identifier. Those belong to the facial
recognition module and to `services/alerts_service.py`, which has its own
audience rules (members see offenders only). A behavioural flag is not an
identity claim and must never enter that path as one. The only decision this
payload carries is `requires_human_review`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from heuristics import HeuristicResult
from risk_fusion import FusionResult

logger = logging.getLogger(__name__)


def build_event(
    *,
    track_id: str,
    timestamp: datetime | str,
    location_zone_id: str,
    fusion: FusionResult,
    triggered: Sequence[HeuristicResult],
    event_id: Optional[str] = None,
    include_reasoning: bool = True,
) -> Dict[str, Any]:
    """Assemble one event in the agreed schema."""
    if isinstance(timestamp, datetime):
        stamp = timestamp.astimezone(timezone.utc).isoformat()
    else:
        stamp = str(timestamp)

    event: Dict[str, Any] = {
        "event_id": event_id or f"{track_id}@{stamp}",
        "track_id": track_id,
        "timestamp": stamp,
        "location_zone_id": location_zone_id,
        "behavioural_risk_score": round(float(fusion.behavioural_risk_score), 3),
        "triggered_heuristics": [
            {
                "type": result.name,
                "confidence": round(float(result.confidence), 3),
                # Required, always. A score without a sentence a human can read
                # is not reviewable, and this module's whole justification is
                # that its decisions are reviewable.
                "explanation": result.explanation,
            }
            for result in triggered
        ],
        "face_match_confidence": fusion.face_match_confidence,
        "composite_risk_score": round(float(fusion.composite_risk_score), 3),
        "requires_human_review": bool(fusion.requires_human_review),
    }

    if include_reasoning:
        event["reasoning"] = list(fusion.reasoning)

    return event


def heuristic_inputs(triggered: Sequence[HeuristicResult]) -> Dict[str, Dict[str, Any]]:
    """{heuristic_name: inputs} — the numbers behind each trigger, for the audit."""
    return {result.name: dict(result.inputs or {}) for result in triggered}


def format_for_console(event: Dict[str, Any], *, show_reasoning: bool = True) -> str:
    """A readable block for the CLI and the demo."""
    review = "REVIEW REQUIRED" if event["requires_human_review"] else "no review needed"
    face = event["face_match_confidence"]
    face_text = f"{face:.2f}" if face is not None else "none (no face match)"

    lines = [
        "",
        "=" * 78,
        f"  {event['track_id']}  @  {event['timestamp']}   [{event['location_zone_id']}]",
        "=" * 78,
        f"  behavioural risk : {event['behavioural_risk_score']:.2f}",
        f"  face confidence  : {face_text}",
        f"  composite risk   : {event['composite_risk_score']:.2f}   -->  {review}",
        "",
        "  What was observed:",
    ]

    for trigger in event["triggered_heuristics"]:
        lines.append(f"    * {trigger['type']}  (confidence {trigger['confidence']:.2f})")
        for wrapped in _wrap(trigger["explanation"], width=70):
            lines.append(f"      {wrapped}")

    if show_reasoning and event.get("reasoning"):
        lines.append("")
        lines.append("  How the score was reached:")
        for step in event["reasoning"]:
            for i, wrapped in enumerate(_wrap(step, width=70)):
                lines.append(f"    {'-' if i == 0 else ' '} {wrapped}")

    lines.append("")
    lines.append("  This is a prompt for a human to look. No automatic action is taken.")
    lines.append("=" * 78)
    return "\n".join(lines)


def _wrap(text: str, width: int = 70) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > width and current:
            lines.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += len(word) + 1
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def write_events(events: Sequence[Dict[str, Any]], path: str) -> None:
    """Save all events from a run as one JSON array."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(events), handle, indent=2, default=str)


# ---------------------------------------------------------------------------
# Interface to the Flask backend — STUB
# ---------------------------------------------------------------------------
def push_to_flask_api(
    event_json: Dict[str, Any],
    *,
    url: Optional[str] = None,
    timeout: float = 5.0,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Where this event WOULD be posted to the Guardian Collective backend.

    NOT WIRED UP. It logs by default and only performs a real POST when given a
    URL and `dry_run=False`. That is deliberate — PROJECT_CONTEXT §9 is explicit
    that unwired feeds must not be faked, and a review queue that silently fills
    with events nobody agreed to send is worse than an empty one.

    In the full system this would become:

        POST {backend}/api/v1/behavioural-events
        Content-Type: application/json
        body: this event, unchanged

    and the Flask side would:
      1. Persist it to PostGIS against the camera's `location_zone_id`, so
         behavioural events can be aggregated onto the same hot-spot map as
         claims (`services/claims_service.py`, `routes/hotspot_routes.py`).
      2. Put events with `requires_human_review: true` into a REVIEW QUEUE for
         a Crime Prevention Unit operator — the same pattern as the employee
         claims review queue, NOT the member alert feed.
      3. Leave `services/alerts_service.py` alone. Member alerts require an
         `offender` label from facial recognition, and this module produces no
         labels. A behavioural flag is a request to look, not an identification,
         and routing it to members would be exactly the false-alarm fatigue that
         audience rule exists to prevent.

    Returns a small dict describing what happened, so the caller can log it.
    """
    if dry_run or not url:
        logger.info(
            "push_to_flask_api (STUB): would POST event %s for %s "
            "(composite %.2f, review=%s) to %s",
            event_json.get("event_id"),
            event_json.get("track_id"),
            event_json.get("composite_risk_score", 0.0),
            event_json.get("requires_human_review"),
            url or "<no url configured>",
        )
        return {
            "pushed": False,
            "reason": "stub" if not url else "dry_run",
            "would_post_to": url,
            "event_id": event_json.get("event_id"),
        }

    try:
        import requests  # already a backend dependency

        response = requests.post(url, json=event_json, timeout=timeout)
        response.raise_for_status()

        # The backend assigns the review id, and the module needs it to attach
        # a clip once the trailing seconds have been captured.
        body = {}
        try:
            body = response.json() or {}
        except ValueError:
            pass

        return {
            "pushed": True,
            "status_code": response.status_code,
            "event_id": event_json.get("event_id"),
            "review_id": body.get("review_id"),
            "queued_for_review": body.get("queued_for_review"),
        }
    except Exception as exc:
        # A backend that is down must never stop the analysis pipeline. The
        # event is already in the audit log by this point, so nothing is lost.
        logger.warning("push_to_flask_api failed for %s: %s", event_json.get("event_id"), exc)
        return {"pushed": False, "reason": str(exc), "event_id": event_json.get("event_id")}


def push_tracks_to_flask_api(
    snapshots: List[Dict[str, Any]],
    *,
    camera_id: str,
    url: Optional[str] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Publish where the tracked BODIES were, so a face match can be attached.

    This is the half of the join only this module can supply. A face scan knows
    where the face was; nothing but the behavioural pipeline knows where the
    bodies were, and without both there is no way to tell whose face it is once
    two people are in shot.

    Boxes are normalised 0..1 against the frame, so they stay meaningful across
    the browser's video, the uploaded still and this pipeline's own resolution.

    WHAT THIS SENDS IS BODY POSITION AND NOTHING ELSE — no pose, no keypoints,
    no identity, no imagery. The backend keeps it for minutes, not indefinitely
    (PRESENCE_RETENTION_MINUTES), because a continuous record of where every
    body stood is more intrusive than the sparse events it exists to support.
    """
    if not url or not snapshots:
        return {"pushed": False, "reason": "not configured", "snapshots": len(snapshots)}

    payload = {"camera_id": camera_id, "snapshots": snapshots}
    try:
        import requests

        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return {"pushed": True, "snapshots": len(snapshots)}
    except Exception as exc:
        logger.warning("push_tracks_to_flask_api failed: %s", exc)
        return {"pushed": False, "reason": str(exc), "snapshots": len(snapshots)}


_LIVE_SESSION = None


def _live_session():
    """One keep-alive connection for the live relay.

    Every other push here is occasional — an event, a batch of positions — and a
    fresh connection costs nothing noticeable. The relay posts several times a
    second forever, and a TCP handshake per frame is pure overhead against a
    server on the same machine.
    """
    global _LIVE_SESSION
    if _LIVE_SESSION is None:
        import requests

        _LIVE_SESSION = requests.Session()
    return _LIVE_SESSION


def push_live_frame(
    jpeg: bytes,
    *,
    camera_id: str,
    url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Relay one annotated frame to the backend for the live view.

    This is the debug window's own image — boxes, skeletons, zones, scores —
    sent so a reviewer can watch a camera from the app instead of standing over
    the laptop running this module.

    The timeout is deliberately short. A live view is worth nothing if keeping
    it fed slows the analysis down; a frame that cannot be delivered promptly is
    better dropped, because the next one is already more current than it is.

    The backend holds this in memory and overwrites it with the next frame. It
    is never stored — unlike a clip, which is footage of a moment a human has
    been asked to review, this is footage of whoever is simply in shot.
    """
    if not url or not jpeg:
        return {"pushed": False, "reason": "not configured"}

    try:
        response = _live_session().post(
            url,
            params={"camera_id": camera_id},
            data=jpeg,
            headers={"Content-Type": "image/jpeg"},
            timeout=timeout,
        )
        response.raise_for_status()
        return {"pushed": True, "bytes": len(jpeg)}
    except Exception as exc:
        # Debug, not warning: a dropped live frame is a cosmetic loss, and at
        # several frames a second a warning per failure would bury the log.
        logger.debug("push_live_frame failed: %s", exc)
        return {"pushed": False, "reason": str(exc)}


def track_snapshot(frame_result, timestamp: datetime | str) -> Optional[Dict[str, Any]]:
    """Build one body-position snapshot from a processed frame.

    `timestamp` must be WALL-CLOCK, not the frame's stream time. The correlator
    matches these against face scans taken by a browser, and the two only line
    up on real time — stream seconds are relative to the start of a clip.

    Only confirmed person tracks are included. An unconfirmed detection has no
    stable id, so an identity attached to it would be attached to nothing.
    """
    from detector import is_confirmed

    tracks = []
    for detection in frame_result.detections:
        if not detection.is_person() or not is_confirmed(detection.track_id):
            continue
        width, height = frame_result.frame.width, frame_result.frame.height
        if not width or not height:
            continue
        x1, y1, x2, y2 = detection.bbox
        tracks.append({
            "track_id": detection.track_id,
            "bbox": {
                "x": round(max(0.0, x1 / width), 5),
                "y": round(max(0.0, y1 / height), 5),
                "w": round(min(1.0, (x2 - x1) / width), 5),
                "h": round(min(1.0, (y2 - y1) / height), 5),
            },
        })

    if not tracks:
        return None

    stamp = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
    return {"timestamp": stamp, "tracks": tracks}
