"""Read licence plates from images using Azure AI Vision (Read / OCR).

Chosen over local OCR because it needs no model download, runs in a few hundred
milliseconds, and is the tool PROJECT_CONTEXT.md already nominated for plates.
Unlike the Azure Face API, Read is generally available — no Limited Access
approval to wait on.

Two pieces of domain handling sit between "text in the picture" and "a plate we
can match":

1. A photo of a car contains plenty of text that is not a plate — dealer frames,
   bumper stickers, road signs, shop fronts behind it. Candidates are filtered by
   shape before they ever reach the database.

2. OCR reliably confuses certain glyphs on plates, because plate fonts are
   deliberately blocky and plates get dirty: O/0, I/1, S/5, B/8, Z/2, G/6.
   An exact string match would miss a plate that was read almost perfectly, so
   near-misses are resolved through a normalised form.
"""

import logging
import os
import re
import threading

from config import Config

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()

# Glyph pairs OCR mixes up on plates. Mapped to a single canonical character so
# CA123456 and CAI23456 collapse to the same normalised key.
_CONFUSIONS = str.maketrans({
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
})

# South African plates run roughly 6-9 characters once separators are stripped
# (CA123456, ABC123GP, CY 987-654). Below 5 is noise; above 10 is a sentence.
MIN_PLATE_LENGTH = int(os.getenv("PLATE_MIN_LENGTH", "5"))
MAX_PLATE_LENGTH = int(os.getenv("PLATE_MAX_LENGTH", "10"))

# South African registration formats, matched against the CLEANED value —
# separators stripped, upper-cased, but BEFORE the confusion table runs.
# Confusion mapping turns O->0 and S->5, which would destroy the letter/digit
# shape these patterns depend on.
#
# Province suffixes for the letter-suffix formats.
_SA_PROVINCE_SUFFIX = "(?:GP|MP|NW|EC|FS|NC|WC|ZN|L)"
_SA_PLATE_PATTERNS = (
    # Western/Eastern/Northern Cape and Free State: letters then digits.
    # Covers the seeded demo plates (CA123456, CF41043, CAA227793).
    re.compile(r"^[A-Z]{2,3}\d{3,6}$"),
    # Gauteng and the other suffix provinces: ABC 123 GP.
    re.compile(r"^[A-Z]{3}\d{3}" + _SA_PROVINCE_SUFFIX + r"$"),
    # Older / shorter suffix-province issues: AB 1234 GP.
    re.compile(r"^[A-Z]{1,3}\d{2,4}" + _SA_PROVINCE_SUFFIX + r"$"),
    # Personalised plates, which are free-form but still province-suffixed.
    re.compile(r"^[A-Z0-9]{2,8}" + _SA_PROVINCE_SUFFIX + r"$"),
)

# Whether a candidate that passes the generic shape test but matches no SA
# format is dropped outright. Off by default: the shape test already removes
# most signage, and a hard reject would silently discard a legitimate plate
# this list does not anticipate (trailers, diplomatic, cross-border). Turn it
# on where precision matters more than recall.
STRICT_SA_FORMAT = os.getenv("PLATE_STRICT_SA_FORMAT", "false").strip().lower() in ("1", "true", "yes")


def configured():
    return bool(getattr(Config, "AZURE_VISION_ENDPOINT", None)
                and getattr(Config, "AZURE_VISION_KEY", None))


def get_client():
    """Build the Azure Vision client once and reuse it."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                if not configured():
                    raise RuntimeError(
                        "Azure Vision is not configured. Add AZURE_VISION_ENDPOINT and "
                        "AZURE_VISION_KEY to backend/.env — both are on the 'Keys and "
                        "Endpoint' page of your Azure AI Services resource."
                    )
                from azure.ai.vision.imageanalysis import ImageAnalysisClient
                from azure.core.credentials import AzureKeyCredential

                _client = ImageAnalysisClient(
                    endpoint=Config.AZURE_VISION_ENDPOINT,
                    credential=AzureKeyCredential(Config.AZURE_VISION_KEY),
                )
                logger.info("Azure Vision client ready (%s)", Config.AZURE_VISION_ENDPOINT)
    return _client


def read_text(image_bytes):
    """Return every text line Azure finds, with its confidence.

    Confidence is per word, so a line's score is the weakest word in it — a plate
    is only as trustworthy as its least certain character.
    """
    from azure.ai.vision.imageanalysis.models import VisualFeatures

    result = get_client().analyze(
        image_data=image_bytes,
        visual_features=[VisualFeatures.READ],
    )

    lines = []
    if result.read is not None:
        for block in result.read.blocks:
            for line in block.lines:
                confidences = [w.confidence for w in (line.words or []) if w.confidence is not None]
                lines.append({
                    "text": line.text,
                    "confidence": round(min(confidences), 4) if confidences else None,
                    "words": [w.text for w in (line.words or [])],
                })
    return lines


def clean(text):
    """Strip everything that is not a letter or digit, and upper-case."""
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def normalise(plate):
    """Collapse OCR-confusable glyphs so near-misses compare equal."""
    return clean(plate).translate(_CONFUSIONS)


def sign_image_url(url):
    """Short-lived read link for a stored plate image.

    Imported lazily and failure-tolerant: a storage misconfiguration should
    cost a thumbnail, not the plate match that the alert depends on.
    """
    if not url:
        return url
    try:
        from services.blob_storage import sign_url

        return sign_url(url)
    except Exception:
        return url


def matches_sa_format(candidate):
    """True when the candidate matches a known South African plate layout.

    Reported alongside every candidate rather than used as a silent filter, so
    a read that is plate-shaped but not SA-shaped is visible as such instead of
    disappearing. See STRICT_SA_FORMAT for the filtering behaviour.
    """
    value = clean(candidate)
    return any(p.match(value) for p in _SA_PLATE_PATTERNS)


def looks_like_plate(candidate):
    """Shape test, so shop signage and bumper stickers do not reach the database."""
    if not (MIN_PLATE_LENGTH <= len(candidate) <= MAX_PLATE_LENGTH):
        return False
    has_digit = any(c.isdigit() for c in candidate)
    has_alpha = any(c.isalpha() for c in candidate)
    # Almost every plate mixes letters and digits. A pure-digit string of the
    # right length is usually a phone number or a price.
    if not (has_digit and has_alpha):
        return False
    if STRICT_SA_FORMAT and not matches_sa_format(candidate):
        return False
    return True


def extract_candidates(lines):
    """Plate-shaped strings from OCR output, best-confidence first.

    Also tries adjacent lines joined together: plates are often split across two
    text blocks, either by the separator dash or by a two-row layout.
    """
    candidates = {}

    def offer(text, confidence, source):
        value = clean(text)
        if not looks_like_plate(value):
            return
        existing = candidates.get(value)
        if existing is None or (confidence or 0) > (existing["confidence"] or 0):
            candidates[value] = {
                "plate": value,
                "normalised": normalise(value),
                "confidence": confidence,
                "source": source,
                "sa_format": matches_sa_format(value),
            }

    for index, line in enumerate(lines):
        offer(line["text"], line["confidence"], "line")
        if index + 1 < len(lines):
            nxt = lines[index + 1]
            joined_confidence = min(
                [c for c in (line["confidence"], nxt["confidence"]) if c is not None],
                default=None,
            )
            offer(line["text"] + nxt["text"], joined_confidence, "joined-lines")

    # A read that matches a real SA layout outranks a higher-confidence read
    # that does not. OCR confidence measures glyph legibility, not whether the
    # string is a registration — a crisply-read shop sign scores well on the
    # first and fails the second.
    return sorted(
        candidates.values(),
        key=lambda c: (not c["sa_format"], -(c["confidence"] or 0)),
    )


def match_plate(cursor, candidates):
    """Look candidates up in vehicle_plates, exact first then confusion-tolerant."""
    if not candidates:
        return None

    exact_values = [c["plate"] for c in candidates]
    cursor.execute(
        """
        SELECT p.id, p.plate_number, p.status, p.owner_name,
               (SELECT image_url FROM vehicle_plate_images i
                 WHERE i.plate_id = p.id ORDER BY created_at LIMIT 1)
        FROM vehicle_plates p
        WHERE p.plate_number = ANY(%s);
        """,
        (exact_values,),
    )
    row = cursor.fetchone()
    if row:
        matched = next(c for c in candidates if c["plate"] == row[1])
        return {"row": row, "candidate": matched, "match_type": "exact"}

    # Nothing matched exactly. Compare on the normalised form, which folds the
    # glyph pairs OCR confuses — a plate read as CAI23456 still finds CA123456.
    cursor.execute(
        "SELECT id, plate_number, status, owner_name FROM vehicle_plates;"
    )
    registry = cursor.fetchall()
    lookup = {normalise(r[1]): r for r in registry}
    for candidate in candidates:
        row = lookup.get(candidate["normalised"])
        if row:
            cursor.execute(
                "SELECT image_url FROM vehicle_plate_images WHERE plate_id = %s "
                "ORDER BY created_at LIMIT 1;",
                (row[0],),
            )
            image = cursor.fetchone()
            return {
                "row": (row[0], row[1], row[2], row[3], image[0] if image else None),
                "candidate": candidate,
                "match_type": "normalised",
            }
    return None


def process_plate_image(image_bytes, db_conn=None):
    """Read an image and report any registered plate found in it."""
    if not configured():
        return {
            "success": False,
            "provider": "azure-vision-read",
            "error": "Azure Vision is not configured.",
            "hint": "Add AZURE_VISION_ENDPOINT and AZURE_VISION_KEY to backend/.env",
        }

    try:
        lines = read_text(image_bytes)
    except Exception as exc:
        logger.error("Azure OCR failed: %s", exc)
        return {"success": False, "provider": "azure-vision-read", "error": str(exc)}

    candidates = extract_candidates(lines)
    payload = {
        "success": True,
        "provider": "azure-vision-read",
        "raw_text": " ".join(line["text"] for line in lines),
        "lines": lines,
        "candidates": candidates,
    }

    if not candidates:
        payload.update({
            "match_found": False,
            "plate": None,
            "message": ("Text was read, but nothing plate-shaped was found."
                        if lines else "No text found in the image."),
        })
        return payload

    if db_conn is None:
        payload.update({"match_found": False, "plate": None,
                        "message": "Read only — no database connection supplied."})
        return payload

    cursor = db_conn.cursor()
    try:
        match = match_plate(cursor, candidates)
    finally:
        cursor.close()

    if not match:
        best = candidates[0]
        payload.update({
            "match_found": False,
            "plate": None,
            "extracted_text": best["plate"],
            "alert": False,
            "message": f"Plate '{best['plate']}' read but not in the registry.",
        })
        return payload

    plate_id, plate_number, status, owner_name, image_url = match["row"]
    is_flagged = status in ("offender", "suspect")
    payload.update({
        "match_found": True,
        "alert": is_flagged,
        "status": status,
        "extracted_text": match["candidate"]["plate"],
        "match_type": match["match_type"],
        "confidence": match["candidate"]["confidence"],
        "plate": {
            "id": str(plate_id),
            "plate_number": plate_number,
            "status": status,
            "owner_name": owner_name,
            # Signed: the plate container is private, so a raw URL would not load.
            "image_url": sign_image_url(image_url),
        },
        "message": (f"ALERT: {status.upper()} VEHICLE — {plate_number} ({owner_name})"
                    if is_flagged else f"Vehicle {plate_number} is registered to {owner_name}."),
    })
    if match["match_type"] == "normalised":
        payload["message"] += (f" Read as '{match['candidate']['plate']}' and resolved to "
                               f"'{plate_number}' through OCR character confusion.")
    return payload
