"""Image conditioning and OCR transport for licence-plate scanning.

A doorbell frame is the hard case for OCR. The plate may be twenty metres down
a driveway (tens of pixels tall), turned thirty degrees away from the lens, and
lit by a porch bulb or a low sun. Handing that frame straight to a cloud OCR
service returns nothing useful.

So before a frame reaches Azure it is localised, deskewed, upscaled and
contrast-normalised here. Azure then sees an image where the plate fills the
frame and reads horizontally, which is the condition it is good at.

The other constraint shaping this module is the Azure Vision free tier: 20
calls per minute, total, for the whole app. Every call is therefore metered
through a sliding-window gate, and the expensive work above exists partly so
that one call is usually enough.
"""

import importlib
import logging
import threading
import time
from collections import deque
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import Config
from services.plate_text import OcrLine, OcrWord, polygon_to_box

logger = logging.getLogger(__name__)

# Azure rejects anything under 50x50, and a plate needs far more than that to
# be legible, so crops are upscaled to this width before being sent.
TARGET_PLATE_WIDTH = Config.PLATE_TARGET_WIDTH
MIN_AZURE_DIMENSION = 60
MAX_SEND_WIDTH = Config.PLATE_MAX_SEND_WIDTH
JPEG_QUALITY = Config.PLATE_JPEG_QUALITY


# --------------------------------------------------------------------------
# Free-tier quota gate
# --------------------------------------------------------------------------


class AzureQuotaGate:
    """Sliding-window limiter over the Azure Vision call rate.

    The free F0 tier allows 20 transactions per minute and answers 429 past
    that. A live camera can trivially outrun this, and a burst of 429s reads to
    the user as "the scanner is broken", so the ceiling is enforced locally and
    a refused scan is reported as pacing rather than failure.
    """

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0):
        self.limit = max(1, limit_per_minute)
        self.window = window_seconds
        self._calls = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._calls and now - self._calls[0] > self.window:
            self._calls.popleft()

    def remaining(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return max(0, self.limit - len(self._calls))

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._calls) >= self.limit:
                return False
            self._calls.append(now)
            return True

    def retry_after_seconds(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._calls) < self.limit or not self._calls:
                return 0.0
            return max(0.0, self.window - (now - self._calls[0]))


quota_gate = AzureQuotaGate(Config.AZURE_VISION_RATE_LIMIT_PER_MIN)


class QuotaExceeded(RuntimeError):
    """Raised when a scan is refused locally to stay inside the free tier."""

    def __init__(self, retry_after: float):
        super().__init__("Azure Vision call budget reached for this minute.")
        self.retry_after = retry_after


# --------------------------------------------------------------------------
# Plate localisation and conditioning
# --------------------------------------------------------------------------


def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def crop_roi_with_offset(
    image: np.ndarray, roi: Optional[dict], pad: float = 0.25
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Crop to a normalised region of interest, generously padded.

    The browser sends the box its own cheap detector found. Trusting it exactly
    would be a mistake — it tracks the character block, not the plate housing —
    so the crop is padded outwards before anything downstream looks at it. The
    offset comes back too, so a box found inside the crop can be reported in
    the coordinates of the frame the caller sent.
    """
    if not roi:
        return image, (0, 0)
    h, w = image.shape[:2]
    try:
        rx = float(roi.get("x", 0.0))
        ry = float(roi.get("y", 0.0))
        rw = float(roi.get("w", 0.0))
        rh = float(roi.get("h", 0.0))
    except (AttributeError, TypeError, ValueError):
        return image, (0, 0)
    if rw <= 0 or rh <= 0:
        return image, (0, 0)

    # Accept either normalised (0-1) or pixel coordinates.
    if max(rx + rw, ry + rh) <= 1.5:
        rx, ry, rw, rh = rx * w, ry * h, rw * w, rh * h

    px, py = rw * pad, rh * pad
    x0 = int(max(0, rx - px))
    y0 = int(max(0, ry - py))
    x1 = int(min(w, rx + rw + px))
    y1 = int(min(h, ry + rh + py))
    if x1 - x0 < 12 or y1 - y0 < 8:
        return image, (0, 0)
    return image[y0:y1, x0:x1], (x0, y0)


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    points = points.reshape(4, 2).astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    total = points.sum(axis=1)
    ordered[0] = points[np.argmin(total)]
    ordered[2] = points[np.argmax(total)]
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]
    return ordered


def locate_plate_candidates(image: np.ndarray, max_regions: int = 3) -> List[np.ndarray]:
    """Find regions that could be a plate, best first.

    Plates are found by their texture rather than their colour: a row of
    stamped characters produces a dense band of vertical edges. Closing that
    edge response with a wide horizontal kernel turns the characters into one
    solid blob whose bounding rectangle is the plate. Colour would be the wrong
    signal — SA plates are white, yellow or (on older stock) black, and porch
    lighting shifts all of them.

    Several regions are returned rather than one because this test cannot
    separate a plate from a model badge: "HILUX" stamped on a tailgate is also
    a horizontal band of high-contrast characters, and on a rear view it is
    frequently the *stronger* response of the two. Picking a single winner here
    would mean betting the whole scan on a heuristic that has no idea what a
    plate says. Instead the shortlist is handed to the OCR, and the plate
    grammar decides — which is the one test that actually knows.
    """
    if image is None or image.size == 0:
        return []

    h, w = image.shape[:2]
    scale = 640.0 / max(w, 1)
    if scale < 1.0:
        working = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        working = image
        scale = 1.0

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)

    gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient = np.absolute(gradient)
    span = gradient.max() - gradient.min()
    if span <= 0:
        return []
    gradient = (255 * (gradient - gradient.min()) / span).astype("uint8")

    gradient = cv2.GaussianBlur(gradient, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, None, iterations=2)
    binary = cv2.erode(binary, None, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    frame_area = float(working.shape[0] * working.shape[1])
    scored: List[Tuple[float, np.ndarray]] = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), angle = rect
        if rw < 1 or rh < 1:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        ratio = long_side / short_side
        # Perspective squeezes a 4.7:1 plate; 1.8 is about as square as a real
        # plate gets, 8.0 about as stretched as an oblique view leaves it.
        if not 1.8 <= ratio <= 8.0:
            continue
        area = rw * rh
        coverage = area / frame_area
        if coverage < 0.0015 or coverage > 0.75:
            continue
        if long_side < 40:
            continue
        # A plate's blob is close to solid; tree branches and railings are not.
        fill = cv2.contourArea(contour) / max(area, 1.0)
        if fill < 0.35:
            continue

        score = fill * 2.0 + (1.0 - min(abs(ratio - 4.0) / 4.0, 1.0)) + min(coverage * 4.0, 1.0)

        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            quad = approx.astype("float32")
            score += 0.6  # a genuine quad beats a bounding box for deskewing
        else:
            quad = cv2.boxPoints(rect).astype("float32")

        scored.append((score, _order_quad(quad) / scale))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [quad for _, quad in scored[:max_regions]]


def warp_plate(image: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """Flatten an angled plate into a front-on rectangle.

    This is the step that makes "weird angles" tractable. OCR models are
    trained on horizontal text; a plate photographed from the side is not a
    rotation problem but a projective one, and only a four-point transform
    undoes it.
    """
    try:
        tl, tr, br, bl = quad
        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
        if width < 20 or height < 8:
            return None

        target_w = int(max(TARGET_PLATE_WIDTH, width))
        target_w = min(target_w, MAX_SEND_WIDTH)
        target_h = int(target_w * (height / width))
        if target_h < MIN_AZURE_DIMENSION:
            target_h = MIN_AZURE_DIMENSION

        destination = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(quad.astype("float32"), destination)
        return cv2.warpPerspective(image, matrix, (target_w, target_h), flags=cv2.INTER_CUBIC)
    except Exception as exc:  # a degenerate quad should not kill the scan
        logger.debug("Plate warp failed: %s", exc)
        return None


def enhance(image: np.ndarray) -> np.ndarray:
    """Normalise lighting and recover character edges.

    CLAHE rather than a global equalisation because the failure mode is local:
    a plate half in porch light and half in shadow has a fine histogram
    overall. The unsharp mask afterwards puts back the edge definition that
    upscaling a small crop costs.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    balanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)

    blurred = cv2.GaussianBlur(balanced, (0, 0), 2.0)
    return cv2.addWeighted(balanced, 1.6, blurred, -0.6, 0)


def upscale_for_ocr(image: np.ndarray, target_width: int = TARGET_PLATE_WIDTH) -> np.ndarray:
    """Bring a distant plate up to a size the OCR service can resolve."""
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    factor = max(target_width / float(w), MIN_AZURE_DIMENSION / float(h), 1.0)
    factor = min(factor, MAX_SEND_WIDTH / float(w)) if w * factor > MAX_SEND_WIDTH else factor
    if factor <= 1.01:
        # Still enforce Azure's floor on tiny crops.
        if w >= MIN_AZURE_DIMENSION and h >= MIN_AZURE_DIMENSION:
            return image
        factor = max(MIN_AZURE_DIMENSION / float(w), MIN_AZURE_DIMENSION / float(h))
    return cv2.resize(image, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_CUBIC)


def encode_jpeg(image: np.ndarray, quality: int = JPEG_QUALITY) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None


class ScanVariant:
    """One image to send to OCR, plus how to read its coordinates back.

    `tiles` records where each stacked region came from in the caller's frame,
    so a plate found at y=340 of a contact sheet can still be drawn as a box on
    the original doorbell image.
    """

    __slots__ = ("label", "payload", "tiles", "size")

    def __init__(self, label: str, payload: bytes, tiles: Optional[List[dict]] = None, size=None):
        self.label = label
        self.payload = payload
        self.tiles = tiles or []
        self.size = size

    def __iter__(self):
        # Older call sites unpack these as (label, payload).
        return iter((self.label, self.payload))


TILE_SEPARATOR_PX = 14


def _contact_sheet(tiles: Sequence[Tuple[np.ndarray, Tuple[float, float, float, float]]]):
    """Stack plate candidates into one image, remembering where each came from.

    This is the move that makes the free tier workable. Three candidate regions
    would be three OCR calls at 20 calls a minute; stacked into one image they
    are one call, and the plate grammar downstream reads whichever tile
    actually holds a plate. The separator band stops OCR from running a line
    across the boundary between two tiles.
    """
    if not tiles:
        return None, []

    width = min(MAX_SEND_WIDTH, max(int(t[0].shape[1]) for t in tiles))
    rendered = []
    for image, source_box in tiles:
        h, w = image.shape[:2]
        if w != width:
            scale = width / float(w)
            image = cv2.resize(image, (width, max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
        rendered.append((image, source_box))

    separator = np.full((TILE_SEPARATOR_PX, width, 3), 128, np.uint8)
    parts, mapping, y = [], [], 0
    for index, (image, source_box) in enumerate(rendered):
        if index:
            parts.append(separator)
            y += TILE_SEPARATOR_PX
        parts.append(image)
        mapping.append({"y0": y, "y1": y + image.shape[0], "source": source_box})
        y += image.shape[0]

    sheet = np.vstack(parts)
    if sheet.shape[0] > MAX_SEND_WIDTH * 2:  # keep the payload sane
        return None, []
    return sheet, mapping


def map_box_to_source(variant: ScanVariant, box, fallback_size=None):
    """Translate a box found in a prepared image back to the caller's frame."""
    if not box:
        return None
    x, y, w, h = box

    for tile in variant.tiles:
        if tile["y0"] <= y + h / 2.0 <= tile["y1"]:
            sx, sy, sw, sh = tile["source"]
            tile_h = max(1.0, tile["y1"] - tile["y0"])
            tile_w = float(variant.size[0]) if variant.size else max(w, 1.0)
            return (
                sx + (x / max(tile_w, 1.0)) * sw,
                sy + ((y - tile["y0"]) / tile_h) * sh,
                (w / max(tile_w, 1.0)) * sw,
                (h / tile_h) * sh,
            )

    if variant.size and fallback_size:
        scale_x = fallback_size[0] / float(variant.size[0])
        scale_y = fallback_size[1] / float(variant.size[1])
        return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
    return box


def build_scan_variants(
    image_bytes: bytes,
    roi: Optional[dict] = None,
    max_variants: int = 2,
    max_regions: int = 3,
) -> Tuple[List[ScanVariant], dict]:
    """Produce the image(s) to send to OCR, best first.

    Variant one is a contact sheet of the plate-like regions, each deskewed and
    upscaled so its characters are legible — one OCR call covering several
    guesses. Variant two is the whole enhanced frame, deliberately skipping
    localisation, because the case it exists to catch is localisation missing
    the plate altogether. On the free tier variant two usually does not run.
    """
    diagnostics = {"localised": False, "warped": False, "roi_applied": bool(roi), "regions": 0}

    image = decode_image(image_bytes)
    if image is None:
        return [ScanVariant("original", image_bytes)], diagnostics

    height, width = image.shape[:2]
    diagnostics["source_size"] = {"w": int(width), "h": int(height)}

    region, (offset_x, offset_y) = crop_roi_with_offset(image, roi)
    variants: List[ScanVariant] = []

    quads = locate_plate_candidates(region, max_regions=max_regions)
    diagnostics["regions"] = len(quads)
    tiles = []
    for quad in quads:
        warped = warp_plate(region, quad)
        if warped is None:
            continue
        xs, ys = quad[:, 0], quad[:, 1]
        source_box = (
            float(xs.min()) + offset_x,
            float(ys.min()) + offset_y,
            float(xs.max() - xs.min()),
            float(ys.max() - ys.min()),
        )
        tiles.append((enhance(warped), source_box))

    if tiles:
        diagnostics["localised"] = True
        diagnostics["warped"] = True
        sheet, mapping = _contact_sheet(tiles)
        if sheet is not None:
            encoded = encode_jpeg(sheet)
            if encoded:
                variants.append(
                    ScanVariant(
                        "plate_regions" if len(tiles) > 1 else "plate_warp",
                        encoded,
                        tiles=mapping,
                        size=(sheet.shape[1], sheet.shape[0]),
                    )
                )

    if not variants:
        enhanced = enhance(upscale_for_ocr(region))
        encoded = encode_jpeg(enhanced)
        if encoded:
            variants.append(
                ScanVariant(
                    "region_enhanced",
                    encoded,
                    tiles=[{
                        "y0": 0,
                        "y1": enhanced.shape[0],
                        "source": (float(offset_x), float(offset_y),
                                   float(region.shape[1]), float(region.shape[0])),
                    }],
                    size=(enhanced.shape[1], enhanced.shape[0]),
                )
            )

    if len(variants) < max_variants:
        enhanced = enhance(upscale_for_ocr(image, target_width=1280))
        encoded = encode_jpeg(enhanced)
        if encoded:
            variants.append(
                ScanVariant(
                    "frame_enhanced",
                    encoded,
                    tiles=[{
                        "y0": 0,
                        "y1": enhanced.shape[0],
                        "source": (0.0, 0.0, float(width), float(height)),
                    }],
                    size=(enhanced.shape[1], enhanced.shape[0]),
                )
            )

    if not variants:
        variants.append(ScanVariant("original", image_bytes))

    return variants[:max_variants], diagnostics


# --------------------------------------------------------------------------
# OCR engines
# --------------------------------------------------------------------------

_azure_client = None
_azure_lock = threading.Lock()


def azure_configured() -> bool:
    return bool(Config.AZURE_VISION_ENDPOINT and Config.AZURE_VISION_KEY)


def get_azure_client():
    """One Image Analysis client for the process — it holds a connection pool."""
    global _azure_client
    if _azure_client is not None:
        return _azure_client

    with _azure_lock:
        if _azure_client is not None:
            return _azure_client

        endpoint = Config.AZURE_VISION_ENDPOINT
        key = Config.AZURE_VISION_KEY
        if not endpoint or not key:
            raise RuntimeError(
                "Azure Vision is not configured. Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY."
            )

        try:
            imageanalysis = importlib.import_module("azure.ai.vision.imageanalysis")
            credentials = importlib.import_module("azure.core.credentials")
        except ImportError as exc:
            raise RuntimeError(
                "Azure Vision SDK is not installed. Install requirements and retry."
            ) from exc

        _azure_client = imageanalysis.ImageAnalysisClient(
            endpoint=endpoint,
            credential=credentials.AzureKeyCredential(key),
        )
    return _azure_client


def azure_read(image_bytes: bytes) -> List[OcrLine]:
    """Run Azure Vision READ and return lines with their geometry.

    The geometry is the point: the old caller took only `.text` and joined it,
    which is exactly what made it impossible to tell a plate from a badge.
    """
    if not quota_gate.try_acquire():
        raise QuotaExceeded(quota_gate.retry_after_seconds())

    models = importlib.import_module("azure.ai.vision.imageanalysis.models")
    result = get_azure_client().analyze(
        image_data=image_bytes,
        visual_features=[models.VisualFeatures.READ],
    )

    lines: List[OcrLine] = []
    if result.read is None:
        return lines

    for block in result.read.blocks:
        for line in block.lines:
            words = []
            for word in getattr(line, "words", None) or ():
                words.append(
                    OcrWord(
                        text=word.text,
                        box=polygon_to_box(getattr(word, "bounding_polygon", None)),
                        confidence=float(getattr(word, "confidence", 0.0) or 0.0),
                    )
                )
            confidences = [w.confidence for w in words if w.confidence]
            lines.append(
                OcrLine(
                    text=line.text,
                    box=polygon_to_box(getattr(line, "bounding_polygon", None)),
                    confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                    words=words,
                )
            )
    return lines


_easyocr_reader = None
_easyocr_lock = threading.Lock()
_easyocr_failed = False


def easyocr_available() -> bool:
    if _easyocr_failed:
        return False
    try:
        importlib.import_module("easyocr")
        return True
    except ImportError:
        return False


def get_easyocr_reader():
    """Lazily build the EasyOCR reader.

    Lazy because the model load is slow and the dependency is optional — this
    path only runs when Azure is unreachable or unconfigured, and the app must
    still start on a machine where easyocr was never installed.
    """
    global _easyocr_reader, _easyocr_failed
    if _easyocr_reader is not None:
        return _easyocr_reader

    with _easyocr_lock:
        if _easyocr_reader is not None:
            return _easyocr_reader
        try:
            easyocr = importlib.import_module("easyocr")
            _easyocr_reader = easyocr.Reader(["en"], gpu=False)
        except Exception as exc:
            _easyocr_failed = True
            raise RuntimeError(f"EasyOCR fallback unavailable: {exc}") from exc
    return _easyocr_reader


def easyocr_read(image_bytes: bytes) -> List[OcrLine]:
    """Local OCR fallback, normalised into the same shape as the Azure path."""
    image = decode_image(image_bytes)
    if image is None:
        return []

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = get_easyocr_reader().readtext(rgb)

    lines: List[OcrLine] = []
    for detection in detections:
        if len(detection) < 3:
            continue
        polygon, text, confidence = detection[0], detection[1], float(detection[2] or 0.0)
        box = polygon_to_box(polygon)
        # EasyOCR has no word split, so the line is also its own single word.
        lines.append(
            OcrLine(
                text=text,
                box=box,
                confidence=confidence,
                words=[OcrWord(text=text, box=box, confidence=confidence)],
            )
        )
    return lines


def frame_size_of(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    image = decode_image(image_bytes)
    if image is None:
        return None
    h, w = image.shape[:2]
    return (w, h)
