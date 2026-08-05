"""Optional per-frame context sent alongside a face scan.

The face module matches a FACE. The behavioural module tracks whole BODIES with
anonymous track IDs. To fuse the two you have to know they describe the same
person, and the only thing that can establish that is where in the frame the
face was — a face match with no coordinates cannot be attached to any particular
body once more than one person is in shot.

`LiveScanDemo.jsx` already computes that box to draw its reticle and then throws
it away. This module accepts it, validates it, and normalises it so the answer
survives the two resolution changes between the browser and the analysis.

WHY NORMALISED COORDINATES ARE THE INTEROPERABLE PART
-----------------------------------------------------
Three different pixel spaces are in play for one scan:

  1. the source video          e.g. 960x720  <- the face box is measured here
  2. the uploaded JPEG         max width 720, downscaled to save bandwidth
  3. the behavioural pipeline  whatever the camera feeding YOLO produces

Pixel coordinates from (1) mean nothing in (2) or (3). Normalising to 0..1
against the source dimensions makes the box comparable in any of them: multiply
by that space's width and height and you have the box. Same reasoning as the
zone polygons in behavioural_analysis/config.yaml.

NOTHING HERE TOUCHES IDENTITY. A box is four numbers describing a region of a
picture. The name behind the face stays in the recognition module.
"""

import json
import logging

logger = logging.getLogger(__name__)

# A camera id is used as a grouping key and echoed into events, so keep it to
# something safe to put in a log line, a filename or a URL.
MAX_CAMERA_ID_LENGTH = 64
DEFAULT_CAMERA_ID = "demo_upload"

# Sanity ceiling on frame dimensions. Anything past this is a malformed or
# hostile value rather than a camera.
MAX_FRAME_DIMENSION = 20000


def _to_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN and infinities survive float() and would poison every downstream
    # comparison silently.
    if result != result or result in (float("inf"), float("-inf")):
        return None
    return result


def _to_dimension(value):
    number = _to_float(value)
    if number is None or number <= 0 or number > MAX_FRAME_DIMENSION:
        return None
    return int(number)


def parse_face_box(raw):
    """Parse the `face_box` form field into {x, y, w, h}, or None.

    Accepts a JSON string (what the browser sends) or an already-decoded dict.
    Returns None for anything malformed — a bad box must never fail the scan
    itself, because the face match is still perfectly valid without it.
    """
    if raw is None or raw == "":
        return None

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("scan-face: face_box was not valid JSON; ignoring it.")
            return None

    if not isinstance(raw, dict):
        logger.warning("scan-face: face_box was not an object; ignoring it.")
        return None

    box = {key: _to_float(raw.get(key)) for key in ("x", "y", "w", "h")}
    if any(value is None for value in box.values()):
        logger.warning("scan-face: face_box is missing x/y/w/h; ignoring it.")
        return None

    # A zero-or-negative-area box is not a region of an image.
    if box["w"] <= 0 or box["h"] <= 0:
        logger.warning("scan-face: face_box has no area; ignoring it.")
        return None

    return box


def normalise_face_box(box, frame_width, frame_height):
    """Convert a pixel box to 0..1 against the frame it was measured in.

    Returns None when the frame size is unknown — a normalised box computed
    against a guessed denominator is worse than no box at all, because it looks
    usable and points at the wrong part of the picture.
    """
    if not box or not frame_width or not frame_height:
        return None

    normalised = {
        "x": box["x"] / frame_width,
        "y": box["y"] / frame_height,
        "w": box["w"] / frame_width,
        "h": box["h"] / frame_height,
    }

    # Detectors routinely return boxes that hang slightly off the edge of the
    # frame. Clamp rather than reject: the overhang is a pixel or two of padding
    # around a real face, not a bad detection.
    x = min(max(normalised["x"], 0.0), 1.0)
    y = min(max(normalised["y"], 0.0), 1.0)
    return {
        "x": round(x, 5),
        "y": round(y, 5),
        "w": round(min(normalised["w"], 1.0 - x), 5),
        "h": round(min(normalised["h"], 1.0 - y), 5),
    }


def face_box_centre(normalised_box):
    """Centre of a normalised box — the point the body-track join tests.

    A face belongs to the person track whose body box contains this point
    (BEHAVIOUR_REVIEW_API.md §1). The centre is used rather than a corner
    because a corner of a head box can easily sit outside the body box on a
    turned head, while the centre stays on the person.
    """
    if not normalised_box:
        return None
    return {
        "x": round(normalised_box["x"] + normalised_box["w"] / 2.0, 5),
        "y": round(normalised_box["y"] + normalised_box["h"] / 2.0, 5),
    }


def _clean_camera_id(raw):
    camera_id = (raw or "").strip()
    if not camera_id:
        return DEFAULT_CAMERA_ID
    return camera_id[:MAX_CAMERA_ID_LENGTH]


def _clean_timestamp(raw):
    """Pass an ISO8601 timestamp through, or None.

    Deliberately not parsed into a datetime: this value is echoed back for the
    caller to correlate against, and re-serialising it risks changing the
    timezone representation of a string that is already correct.
    """
    stamp = (raw or "").strip()
    return stamp[:64] if stamp else None


def build_frame_context(form):
    """Read the optional frame fields off a multipart scan request.

    Every field is optional. With none of them supplied this returns the same
    shape with empty values, so the endpoint behaves exactly as it did before
    and nothing that calls it today breaks.
    """
    getter = form.get if hasattr(form, "get") else (lambda key, default=None: None)

    frame_width = _to_dimension(getter("frame_width"))
    frame_height = _to_dimension(getter("frame_height"))
    box = parse_face_box(getter("face_box"))
    normalised = normalise_face_box(box, frame_width, frame_height)

    if box and not normalised:
        logger.info(
            "scan-face: a face_box was supplied without usable frame_width/frame_height, "
            "so it cannot be normalised and will not be usable for the behavioural join."
        )

    return {
        "face_box": box,
        "face_box_normalised": normalised,
        "face_box_centre": face_box_centre(normalised),
        "frame_size": (
            {"w": frame_width, "h": frame_height} if frame_width and frame_height else None
        ),
        "camera_id": _clean_camera_id(getter("camera_id")),
        "captured_at": _clean_timestamp(getter("captured_at")),
    }
