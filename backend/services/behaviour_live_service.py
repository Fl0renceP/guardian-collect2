"""The live annotated camera feed, held in memory only.

The behavioural module already draws an annotated frame for its debug window —
boxes, pose skeletons, zone overlays, the running scores. This relays that same
frame into the app so a reviewer can watch a camera without standing over the
laptop running the pipeline.

WHY MEMORY AND NOTHING ELSE. A live feed is video of whoever happens to be in
front of a camera right now, almost always someone who has done nothing and will
never know they were recorded. `behaviour_clip_service` writes clips to blob
storage because a flagged moment is evidence a human has to review; there is no
equivalent justification for the continuous stream. So:

  * frames live in a single slot per camera and are overwritten by the next one
  * nothing is written to disk, blob storage or the database
  * a feed with no frame for LIVE_STALE_SECONDS is simply not live any more —
    the last frame is dropped rather than served as if it were current

That last rule is the one that matters. A stale frame served without comment is
a lie about the present, and the whole point of separating "live" from "the
recorded clip" on the review card is that the two must never be confused.

THIS FEED IS UNAUTHENTICATED, like every other endpoint in this codebase
(PROJECT_CONTEXT §9). For claims data that is a demo shortcut; for a live camera
it is a genuinely different level of exposure, and it must not ship this way.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# A feed that has not produced a frame for this long is no longer live. Sized
# generously: this pipeline manages 0.3–2 frames per second on a laptop with no
# GPU, so a threshold in single-digit seconds would flicker constantly.
LIVE_STALE_SECONDS = 12.0

# One annotated 640x480 JPEG is ~40-60KB. The cap is a guard against a caller
# posting something that is not a frame at all, not a tuning parameter.
MAX_FRAME_BYTES = 4 * 1024 * 1024

# JPEG magic bytes. A frame endpoint that accepts arbitrary bytes and streams
# them back to a browser under an image mimetype is an open redirect for
# content, so the payload is checked to be what it claims.
_JPEG_MAGIC = b"\xff\xd8\xff"

_lock = threading.Lock()
# camera_id -> {"jpeg": bytes, "at": monotonic seconds, "seq": int}
_frames: dict[str, dict] = {}


class LiveFrameError(ValueError):
    """The posted frame was missing, too large, or not a JPEG."""


def store_frame(camera_id: str, jpeg: bytes) -> dict:
    """Replace the current frame for `camera_id`."""
    if not camera_id:
        raise LiveFrameError("A camera_id is required.")
    if not jpeg:
        raise LiveFrameError("Empty frame.")
    if len(jpeg) > MAX_FRAME_BYTES:
        raise LiveFrameError(
            f"Frame is {len(jpeg)} bytes, over the {MAX_FRAME_BYTES} byte limit."
        )
    if not jpeg.startswith(_JPEG_MAGIC):
        raise LiveFrameError("Frame is not a JPEG.")

    with _lock:
        previous = _frames.get(camera_id)
        _frames[camera_id] = {
            "jpeg": jpeg,
            "at": time.monotonic(),
            "seq": (previous["seq"] + 1) if previous else 1,
        }
        return {"camera_id": camera_id, "bytes": len(jpeg), "seq": _frames[camera_id]["seq"]}


def latest(camera_id: str) -> tuple[bytes, int] | None:
    """The current frame and its sequence number, or None if not live."""
    with _lock:
        entry = _frames.get(camera_id)
        if entry is None or (time.monotonic() - entry["at"]) > LIVE_STALE_SECONDS:
            return None
        return entry["jpeg"], entry["seq"]


def status(camera_id: str | None = None) -> dict:
    """Which cameras are streaming, and how stale each one is."""
    now = time.monotonic()
    with _lock:
        entries = {
            cid: {
                "camera_id": cid,
                "age_seconds": round(now - entry["at"], 2),
                "live": (now - entry["at"]) <= LIVE_STALE_SECONDS,
                "frames": entry["seq"],
            }
            for cid, entry in _frames.items()
        }

    if camera_id is not None:
        return entries.get(
            camera_id,
            {"camera_id": camera_id, "live": False, "age_seconds": None, "frames": 0},
        )
    return {
        "cameras": sorted(entries.values(), key=lambda e: e["camera_id"]),
        "stale_after_seconds": LIVE_STALE_SECONDS,
    }


def drop(camera_id: str) -> None:
    """Forget a camera's frame. Called when a pipeline shuts down cleanly."""
    with _lock:
        _frames.pop(camera_id, None)
