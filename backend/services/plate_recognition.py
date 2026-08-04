"""Licence-plate scanning: condition the frame, read it, resolve it.

This module is the orchestration layer. The grammar that decides what counts as
a plate lives in `plate_text`; the image conditioning and OCR transport live in
`plate_vision`. Both OCR engines feed the same grammar, so a plate read by the
local fallback is filtered and matched exactly as a plate read by Azure is.

Engine order is Azure Vision first, EasyOCR only when Azure is unconfigured,
out of free-tier budget, or erroring — the fallback keeps the demo alive on a
dead network, it is not a second opinion.
"""

import logging
import threading
import time
from typing import List, Optional, Sequence

from config import Config
from services import plate_text, plate_vision
from services.plate_text import OcrLine, PlateCandidate

logger = logging.getLogger(__name__)

# The registry is a handful of rows that change rarely, and fuzzy matching needs
# all of them in memory. Re-reading it per frame would put a database round trip
# in the middle of a 1-in-4-second live loop for no benefit.
REGISTRY_TTL_SECONDS = Config.PLATE_REGISTRY_TTL_SECONDS
_registry_cache = {"at": 0.0, "rows": None}
_registry_lock = threading.Lock()


def get_registry(db_conn, force_refresh: bool = False) -> List[dict]:
    now = time.monotonic()
    with _registry_lock:
        rows = _registry_cache["rows"]
        if not force_refresh and rows is not None and now - _registry_cache["at"] < REGISTRY_TTL_SECONDS:
            return rows

    rows = plate_text.load_registry(db_conn)
    with _registry_lock:
        _registry_cache["rows"] = rows
        _registry_cache["at"] = now
    return rows


def invalidate_registry_cache() -> None:
    with _registry_lock:
        _registry_cache["rows"] = None
        _registry_cache["at"] = 0.0


def _passes_allowed(requested: Optional[int]) -> int:
    """How many OCR calls this scan may spend.

    A second pass only happens with clear headroom in the minute's budget. The
    live loop calls this constantly; one scan is not allowed to eat the quota
    that the next ten need.
    """
    if requested is not None:
        return max(1, min(2, requested))
    return 2 if plate_vision.quota_gate.remaining() > Config.PLATE_SECOND_PASS_MIN_BUDGET else 1


def _read_variants(variants, frame_size, reader, engine_label) -> tuple:
    """Run an OCR engine over the prepared variants until a plate turns up.

    Both engines share this loop so the stopping rule is the same: a full plate
    reading ends the scan, because a second pass can only cost budget it cannot
    improve on.
    """
    best: Optional[PlateCandidate] = None
    best_lines: Sequence[OcrLine] = []
    best_variant = None
    passes = 0
    quota_hit = False
    last_error = None

    for variant in variants:
        try:
            lines = reader(variant.payload)
        except plate_vision.QuotaExceeded:
            quota_hit = True
            break
        except Exception as exc:
            last_error = exc
            logger.warning("%s pass '%s' failed: %s", engine_label, variant.label, exc)
            continue

        passes += 1
        candidate = plate_text.best_candidate(lines, frame_size)
        if candidate and (best is None or candidate.score > best.score):
            best, best_lines, best_variant = candidate, lines, variant
        elif best is None and lines:
            best_lines, best_variant = lines, variant

        if best is not None and best.kind == "plate":
            break

    return best, best_lines, best_variant, passes, quota_hit, last_error


def scan_plate_image(
    image_bytes: bytes,
    db_conn,
    *,
    roi: Optional[dict] = None,
    engine: str = "auto",
    max_passes: Optional[int] = None,
    allow_fallback: bool = True,
):
    """Read one frame and resolve it against the plate registry.

    `roi` is the browser's guess at where the plate is, in normalised
    coordinates. It is a hint that saves the server from searching the whole
    frame, not an instruction — see `plate_vision.crop_roi`.

    Returns the result dict, or a (payload, status) tuple on a hard failure.
    """
    started = time.perf_counter()

    if not image_bytes:
        return {"error": "Empty image payload."}, 400

    frame_size = plate_vision.frame_size_of(image_bytes)
    allowed = _passes_allowed(max_passes)
    variants, diagnostics = plate_vision.build_scan_variants(image_bytes, roi=roi, max_variants=allowed)

    prepared_ms = round((time.perf_counter() - started) * 1000, 2)

    candidate = None
    lines: Sequence[OcrLine] = []
    variant = None
    used_engine = None
    passes = 0
    quota_hit = False
    azure_error = None

    want_azure = engine in {"auto", "azure"} and plate_vision.azure_configured()
    if want_azure:
        candidate, lines, variant, passes, quota_hit, azure_error = _read_variants(
            variants, frame_size, plate_vision.azure_read, "Azure OCR"
        )
        if passes:
            used_engine = "azure_vision_read"

    need_fallback = (
        engine == "easyocr"
        or (engine != "azure" and used_engine is None and allow_fallback)
    )
    if need_fallback and plate_vision.easyocr_available():
        fb_candidate, fb_lines, fb_variant, _, _, _ = _read_variants(
            variants, frame_size, plate_vision.easyocr_read, "EasyOCR"
        )
        if fb_candidate or not candidate:
            candidate, lines, variant = fb_candidate, fb_lines, fb_variant
        used_engine = "easyocr"

    if used_engine is None:
        if quota_hit:
            # Not an error: the limiter did its job. The live UI paces on this
            # rather than showing a failed scan.
            return {
                "throttled": True,
                "match_found": False,
                "plate_detected": False,
                "engine": "azure_vision_read",
                "retry_after_seconds": round(plate_vision.quota_gate.retry_after_seconds(), 1),
                "azure_calls_remaining": plate_vision.quota_gate.remaining(),
                "message": "Azure Vision free-tier budget reached for this minute. Pacing scans.",
            }
        if engine == "easyocr" or not plate_vision.azure_configured():
            return {
                "error": "No OCR engine available. Configure AZURE_VISION_KEY and "
                         "AZURE_VISION_ENDPOINT, or install easyocr for the offline fallback."
            }, 503
        return {"error": f"OCR request failed: {azure_error}"}, 502

    registry = get_registry(db_conn)
    result = plate_text.build_result(
        candidate,
        registry,
        engine=used_engine,
        lines=lines,
        extra={
            "throttled": False,
            "passes_used": max(passes, 1),
            "azure_calls_remaining": plate_vision.quota_gate.remaining(),
            "preprocessing": diagnostics,
            "frame_size": {"w": frame_size[0], "h": frame_size[1]} if frame_size else None,
        },
    )
    # The box OCR reported is in the coordinates of the conditioned image — a
    # deskewed crop, or one tile of a stacked sheet. The caller drew the frame,
    # so translate it back before handing it over, and normalise it so the
    # browser can overlay it at whatever size it happens to be rendering.
    if candidate and variant is not None:
        mapped = plate_vision.map_box_to_source(variant, candidate.box, frame_size)
        if mapped:
            x, y, w, h = mapped
            result["plate_box"] = {
                "x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1),
            }
            if frame_size and frame_size[0] and frame_size[1]:
                result["plate_box_norm"] = {
                    "x": round(x / frame_size[0], 5),
                    "y": round(y / frame_size[1], 5),
                    "w": round(w / frame_size[0], 5),
                    "h": round(h / frame_size[1], 5),
                }
        result["source_variant"] = variant.label

    result["timings_ms"] = {
        "preprocess": prepared_ms,
        "total": round((time.perf_counter() - started) * 1000, 2),
    }
    return result


def process_incoming_plate_image(image_bytes: bytes, db_conn):
    """Legacy entry point for /api/v1/scan-plate.

    Kept so the existing upload page keeps working. It no longer runs EasyOCR
    unconditionally — the engine choice is now Azure-first with EasyOCR behind
    it, the same as every other path — and it runs the same grammar, so it too
    returns the plate rather than everything written on the car.
    """
    return scan_plate_image(image_bytes, db_conn, engine="auto", allow_fallback=True)


# Retained: other modules import this helper.
clean_plate_text = plate_text.normalise
