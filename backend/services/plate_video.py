"""Live licence-plate reading from a video stream, frame by frame.

Three things separate this from plate_ocr.py, which reads a single still image
through Azure Vision:

1. OCR runs locally through EasyOCR. A live stream is hundreds of frames, and
   billing an API call per frame is the wrong shape of cost — as well as putting
   a network round trip inside the frame loop.

2. The plate is located in the frame BEFORE it is read. Handing a whole 720p
   dashboard view to OCR wastes most of the time on tarmac and sky, and invites
   every shop sign in the background to become a candidate. Cropping to the
   plate is what makes per-frame cost survivable.

3. A reading must repeat across frames before it is trusted. Single-frame OCR on
   video is genuinely unreliable — a plate is misread for one frame at a bad
   angle, then read correctly for the next ten. On a watchlist that matters: a
   one-frame misread that names a passing car as stolen is the failure mode with
   real-world consequences, so a plate is reported provisionally until several
   frames agree.

The candidate extraction and registry lookup are deliberately NOT reimplemented
here. plate_ocr already solved the domain problems — plate-shaped filtering, and
the O/0 I/1 S/5 glyph confusions that plate fonts provoke — and a second copy
would drift out of step with it.
"""

import logging
import os
import time
from collections import deque

import cv2
import numpy as np

from services import plate_ocr
from services.plate_recognition import get_reader

logger = logging.getLogger(__name__)

# --- Plate geometry -------------------------------------------------------
# A plate is a wide, short rectangle. South African plates are about 4.5:1;
# the window is opened up because perspective squashes the ratio when a car is
# approaching at an angle rather than square-on to the camera.
PLATE_MIN_ASPECT = float(os.getenv("PLATE_MIN_ASPECT", "1.8"))
PLATE_MAX_ASPECT = float(os.getenv("PLATE_MAX_ASPECT", "6.5"))
# Below this the characters are too few pixels tall for OCR to resolve, and
# reading it anyway produces confident nonsense.
PLATE_MIN_WIDTH_PX = int(os.getenv("PLATE_MIN_WIDTH_PX", "90"))
# How many candidate rectangles to OCR per frame. Each costs real time, and the
# plate is almost always in the largest few contours.
PLATE_MAX_REGIONS = int(os.getenv("PLATE_MAX_REGIONS", "3"))

# --- Multi-frame agreement ------------------------------------------------
# Of the last WINDOW frames that produced a reading, MIN must agree before the
# plate is treated as confirmed and allowed to raise an alert.
PLATE_VOTE_WINDOW = int(os.getenv("PLATE_VOTE_WINDOW", "6"))
PLATE_VOTE_MIN = int(os.getenv("PLATE_VOTE_MIN", "3"))
# A stream that goes quiet for this long is a different vehicle when it resumes.
PLATE_VOTE_TTL_SECONDS = float(os.getenv("PLATE_VOTE_TTL_SECONDS", "8.0"))

# Plates carry no lower case. Constraining the character set stops EasyOCR
# "correcting" a blocky 0 into an O, which is the same confusion plate_ocr then
# has to undo downstream.
#
# The space and hyphen matter as much as the alphabet: SA plates are separated
# ("CA 123-456"), and an allowlist without them leaves OCR no legal character
# for the gap, so it emits the nearest permitted glyph instead. Measured on the
# test rig, that turned CA 123-456 into CA1236456 and ZZ 999-111 into ZZ999H111
# — a hallucinated character in the middle of every plate. clean() strips both
# downstream, so allowing them here costs nothing.
PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"


def _decode(image_bytes):
    """Frame bytes (JPEG from the browser) to a BGR array."""
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _overlap_ratio(a, b):
    """Intersection over the smaller box.

    Not the usual IoU: the duplicates being suppressed here are a plate's inner
    and outer border, where one box sits almost wholly inside the other. IoU
    scores that pair low precisely because the union is large, which is the one
    case that needs catching.
    """
    ix0, iy0 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix1 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy1 = min(a["y"] + a["h"], b["y"] + b["h"])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    intersection = (ix1 - ix0) * (iy1 - iy0)
    smaller = min(a["w"] * a["h"], b["w"] * b["h"])
    return intersection / float(smaller) if smaller else 0.0


def locate_plate_regions(image, max_regions=PLATE_MAX_REGIONS):
    """Candidate plate rectangles, largest first.

    The pipeline is the classic OpenCV one: blur away texture, find edges, keep
    rectangles. Two departures from the usual recipe, both learned from the fact
    that a video frame is not a posed photograph:

    - The usual version keeps only contours that approximate to exactly four
      points. A plate photographed at an angle rounds off at the corners and
      approximates to five or six, so the four-point test throws away precisely
      the frames a moving car produces. Aspect ratio survives the angle, so the
      shape test is applied to the bounding box instead.

    - Every surviving rectangle is returned rather than just the best one. On a
      real frame the top contour is often a window or a bumper shadow, and the
      plate sits second or third.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bilateral rather than Gaussian: it flattens paintwork and tarmac texture
    # while leaving the plate's border sharp, which is the edge we need to keep.
    filtered = cv2.bilateralFilter(gray, 13, 15, 15)
    edges = cv2.Canny(filtered, 30, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    frame_h, frame_w = image.shape[:2]
    regions = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0 or w < PLATE_MIN_WIDTH_PX:
            continue
        aspect = w / float(h)
        if not (PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT):
            continue

        # A few pixels of margin: the contour traces the plate's inner border,
        # and characters sit close enough to it that a tight crop shaves the
        # first and last glyph.
        pad_x, pad_y = int(w * 0.04) + 4, int(h * 0.12) + 4
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(frame_w, x + w + pad_x), min(frame_h, y + h + pad_y)

        box = {
            "x": int(x0), "y": int(y0),
            "w": int(x1 - x0), "h": int(y1 - y0),
            "aspect": round(aspect, 2),
        }

        # Canny traces both sides of the plate's painted border, so one plate
        # arrives as two or three nested rectangles. Left in, each is OCR'd
        # separately: measured on the test rig that tripled the OCR bill per
        # frame (2907ms against 970ms) and returned the same plate three times.
        if any(_overlap_ratio(box, kept) > 0.6 for kept in regions):
            continue

        box["crop"] = image[y0:y1, x0:x1]
        regions.append(box)
        if len(regions) >= max_regions:
            break

    return regions


def _ocr_lines(image):
    """Run EasyOCR and return results in the shape plate_ocr.extract_candidates wants.

    That function was written against Azure's line/word structure, but it only
    reads `text` and `confidence`, so translating here keeps one implementation
    of the plate-shape and glyph-confusion rules rather than two.
    """
    if image is None or image.size == 0:
        return []
    results = get_reader().readtext(image, allowlist=PLATE_ALLOWLIST)
    lines = []
    for item in results:
        # EasyOCR yields (bbox, text, confidence) with detail=1, the default.
        if len(item) < 3:
            continue
        _, text, confidence = item[0], item[1], item[2]
        if not text:
            continue
        lines.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "words": [text],
        })
    return lines


class PlateVoteTracker:
    """Remembers recent readings per stream so a plate must repeat to be trusted.

    Keyed by a caller-supplied stream id. The browser sends one per camera
    session, so two operators scanning at once do not vote in each other's
    ballot. Entries expire, because a stream that pauses and resumes is looking
    at a different vehicle.
    """

    def __init__(self, window=PLATE_VOTE_WINDOW, required=PLATE_VOTE_MIN,
                 ttl=PLATE_VOTE_TTL_SECONDS):
        self.window = window
        self.required = required
        self.ttl = ttl
        self._streams = {}

    def _bucket(self, stream_id):
        now = time.monotonic()
        entry = self._streams.get(stream_id)
        if entry is None or now - entry["last_seen"] > self.ttl:
            entry = {"reads": deque(maxlen=self.window), "last_seen": now,
                     "announced": None}
            self._streams[stream_id] = entry
        entry["last_seen"] = now
        return entry

    def record(self, stream_id, normalised):
        """Add this frame's reading and report whether the plate is confirmed.

        `newly_confirmed` fires on the single frame where a plate crosses the
        threshold, and not again while the same car stays in view. Callers raise
        the alarm on that edge: a confirmed plate held in frame for ten seconds
        is one event to an operator, not three hundred.
        """
        entry = self._bucket(stream_id)
        reads = entry["reads"]
        reads.append(normalised)
        agreeing = sum(1 for r in reads if r == normalised and r is not None)
        confirmed = normalised is not None and agreeing >= self.required

        newly_confirmed = confirmed and entry["announced"] != normalised
        if newly_confirmed:
            entry["announced"] = normalised

        return {
            "agreeing_frames": agreeing,
            "required_frames": self.required,
            "window": self.window,
            "confirmed": confirmed,
            "newly_confirmed": newly_confirmed,
        }

    def reset(self, stream_id):
        self._streams.pop(stream_id, None)


_tracker = PlateVoteTracker()


def read_frame(image_bytes, db_conn=None, stream_id="default"):
    """Read one video frame and report any registered plate in it.

    The payload mirrors plate_ocr.process_plate_image so the same UI code can
    render either, with live-only additions: `regions` for drawing the box,
    and `stability` for the multi-frame vote.

    `alert` is deliberately held back until the vote confirms. An unconfirmed
    reading is still reported — the operator sees it working — but it does not
    raise the alarm.
    """
    started = time.perf_counter()
    frame = _decode(image_bytes)
    if frame is None:
        return {"success": False, "provider": "easyocr-local",
                "error": "Frame could not be decoded."}

    timings = {}
    locate_started = time.perf_counter()
    regions = locate_plate_regions(frame)
    timings["locate"] = round((time.perf_counter() - locate_started) * 1000, 2)

    ocr_started = time.perf_counter()
    lines = []
    for region in regions:
        lines.extend(_ocr_lines(region["crop"]))

    # Nothing plate-shaped in the frame, or the crops read as nothing. Fall back
    # to the whole frame: contour detection fails on plates that are dirty, at a
    # sharp angle, or lit from behind, and OCR alone still finds those. It costs
    # more time, which is why it is the fallback and not the default.
    localised = bool(lines)
    if not lines:
        lines = _ocr_lines(frame)
    timings["ocr"] = round((time.perf_counter() - ocr_started) * 1000, 2)

    candidates = plate_ocr.extract_candidates(lines)
    payload = {
        "success": True,
        "provider": "easyocr-local",
        "localised": localised,
        "regions": [{k: r[k] for k in ("x", "y", "w", "h", "aspect")} for r in regions],
        "frame": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "raw_text": " ".join(line["text"] for line in lines),
        "candidates": candidates,
    }

    if not candidates:
        stability = _tracker.record(stream_id, None)
        payload.update({
            "match_found": False, "alert": False, "plate": None,
            "stability": stability,
            "message": ("Text read, but nothing plate-shaped in frame."
                        if lines else "No plate found in frame."),
        })
        payload["timings_ms"] = timings
        return payload

    best = candidates[0]
    stability = _tracker.record(stream_id, best["normalised"])
    payload["stability"] = stability
    payload["extracted_text"] = best["plate"]

    if db_conn is None:
        payload.update({"match_found": False, "alert": False, "plate": None,
                        "message": "Read only — no database connection supplied."})
        payload["timings_ms"] = timings
        return payload

    match_started = time.perf_counter()
    cursor = db_conn.cursor()
    try:
        match = plate_ocr.match_plate(cursor, candidates)
    finally:
        cursor.close()
    timings["match"] = round((time.perf_counter() - match_started) * 1000, 2)
    timings["total"] = round((time.perf_counter() - started) * 1000, 2)
    payload["timings_ms"] = timings

    if not match:
        payload.update({
            "match_found": False, "alert": False, "plate": None,
            "message": f"Plate '{best['plate']}' read but not in the registry.",
        })
        return payload

    plate_id, plate_number, status, owner_name, image_url = match["row"]
    is_flagged = status in ("offender", "suspect")
    confirmed = stability["confirmed"]

    payload.update({
        "match_found": True,
        # Held back until the vote lands: one bad frame must not raise an alarm.
        "alert": bool(is_flagged and confirmed),
        "provisional": not confirmed,
        "status": status,
        "match_type": match["match_type"],
        "confidence": match["candidate"]["confidence"],
        "plate": {
            "id": str(plate_id),
            "plate_number": plate_number,
            "status": status,
            "owner_name": owner_name,
            # Signed: the plate container is private, so a raw URL would not load.
            "image_url": plate_ocr.sign_image_url(image_url),
        },
    })

    if not confirmed:
        payload["message"] = (
            f"Reading {plate_number} — {stability['agreeing_frames']} of "
            f"{stability['required_frames']} frames agree."
        )
    elif is_flagged:
        payload["message"] = f"ALERT: {status.upper()} VEHICLE — {plate_number} ({owner_name})"
    else:
        payload["message"] = f"Vehicle {plate_number} is registered to {owner_name}."

    if match["match_type"] == "normalised":
        payload["message"] += (f" Read as '{match['candidate']['plate']}' and resolved to "
                               f"'{plate_number}' through OCR character confusion.")
    return payload


def reset_stream(stream_id="default"):
    """Forget a stream's vote history — used when the camera restarts."""
    _tracker.reset(stream_id)


def warm_plate_reader():
    """Load the EasyOCR model at boot so the first frame is not a 30s stall."""
    started = time.perf_counter()
    get_reader()
    return round((time.perf_counter() - started) * 1000, 2)
