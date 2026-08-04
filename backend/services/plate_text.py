"""South African licence-plate text extraction and registry matching.

The problem this module exists to solve: OCR over a photo of a car returns
*every* string in the frame — the model badge ("HILUX"), the dealer frame
("www.somemotors.co.za"), the province name printed on the plate itself, the
"ZA" oval — and the previous code concatenated all of it into one string. The
registry lookup was then matching against noise.

So we don't ask "what text is in this image", we ask "which run of characters
in this image is *shaped like* a South African plate". The grammar below is a
whitelist: anything that fails it is discarded no matter how confidently the
OCR engine read it. That single change is what stops model and dealer text
reaching the registry query.

The second job is matching a read that is nearly right. At doorbell distances
0/O, 1/I, 5/S and 8/B swap constantly, so an exact string compare against the
registry throws away most true hits. Matching happens in tiers, and anything
below an exact compare is returned as "probable" rather than a hard alert.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Grammar
# --------------------------------------------------------------------------

# Province suffixes used by the "ABC 123 GP" style provinces. Western Cape and
# KwaZulu-Natal instead encode the town in a 2-3 letter *prefix* (CA = Cape
# Town, CAA/CAW = later Cape Town series, CF = Bellville, CY = Paarl,
# CL = Vredenburg, ND = Durban), which is what the seeded registry uses.
_PROVINCE_SUFFIX = r"(?:GP|MP|NW|FS|NC|EC|ZN|WP|L)"

# A full, unambiguous plate. Matching one of these is what earns a registry
# lookup and, if it hits, an alert.
STRONG_PATTERNS: Tuple[re.Pattern, ...] = (
    # Prefix style — CA123456, CAA227793, CF41043, CY987654, ND123456.
    re.compile(r"^[A-Z]{2,3}\d{4,6}$"),
    # Suffix style — ABC123GP.
    re.compile(rf"^[A-Z]{{3}}\d{{3}}{_PROVINCE_SUFFIX}$"),
    # Newer Gauteng series — BB12CDGP.
    re.compile(rf"^[A-Z]{{2}}\d{{2}}[A-Z]{{2}}{_PROVINCE_SUFFIX}$"),
)

# Fragments. A plate 30 m from the lens frequently comes back with the prefix
# and the digits as two separate OCR lines, or with one end cropped. These are
# never alerted on directly — they only count if they resolve to exactly one
# registry entry (see match_registry).
PARTIAL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"^[A-Z]{2,3}\d{2,3}$"),   # leading fragment: CA123
    re.compile(r"^[A-Z]{3}\d{3}$"),       # suffix style minus the province
    re.compile(r"^\d{4,6}$"),             # digits only: 227793
)

MIN_PLATE_LEN = 5
MAX_PLATE_LEN = 10

# Secondary safety net behind the grammar. Checked against whole OCR *words*,
# never as a substring of the assembled candidate — "CO" as a substring would
# reject legitimate prefix codes.
BLOCKED_WORDS = frozenset(
    {
        # Makes
        "TOYOTA", "VW", "VOLKSWAGEN", "FORD", "BMW", "AUDI", "NISSAN", "ISUZU",
        "MAZDA", "HYUNDAI", "KIA", "RENAULT", "SUZUKI", "HONDA", "MERCEDES",
        "BENZ", "MITSUBISHI", "CHERY", "HAVAL", "GWM", "JEEP", "LANDROVER",
        "PEUGEOT", "OPEL", "VOLVO", "SUBARU", "DATSUN", "MAHINDRA", "TATA",
        # Models / badges
        "HILUX", "RANGER", "AMAROK", "NAVARA", "TRITON", "COROLLA", "FORTUNER",
        "ETIOS", "QUANTUM", "POLO", "GOLF", "VIVO", "STARLET", "URBAN", "CROSS",
        "GTI", "TDI", "TSI", "AMG", "TURBO", "DIESEL", "HYBRID", "SPORT",
        "LTD", "LIMITED", "AUTO", "MANUAL", "4X4", "4X2", "AWD", "XDRIVE",
        # Dealer frames / stickers / plate furniture
        "WWW", "COM", "CO", "ZA", "NET", "ORG", "MOTORS", "MOTOR", "DEALER",
        "DEALERS", "GROUP", "SALES", "SERVICE", "TEL", "CELL", "CALL",
        "SOUTH", "AFRICA", "WESTERN", "CAPE", "GAUTENG", "NATAL", "KWAZULU",
        "LIMPOPO", "MPUMALANGA", "FREE", "STATE", "NORTHERN", "EASTERN",
        "NORTH", "WEST", "PROVINCE", "REPUBLIC",
    }
)

# Characters OCR interchanges at range. Folding both the read and the registry
# entry through this map turns "C4123456" and "CA123456" into the same key.
_CONFUSABLES = str.maketrans(
    {
        "O": "0", "Q": "0", "D": "0",
        "I": "1", "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "A": "4",
    }
)


def normalise(text: Optional[str]) -> str:
    """Strip everything that isn't A-Z0-9 and upper-case. 'CA 123-456' -> 'CA123456'."""
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


def canonical(text: Optional[str]) -> str:
    """Fold OCR-confusable characters so near-reads compare equal."""
    return normalise(text).translate(_CONFUSABLES)


# Position-aware repair maps. An SA plate has letters in known places and
# digits in the others, so a character can be corrected by where it sits: a "2"
# in the prefix is almost certainly a Z, and a "Z" in the number block is
# almost certainly a 2. A blind fold cannot do this — it would rewrite the "A"
# of CA into a 4 and destroy the prefix.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
             "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}

# How many characters may be corrected before a reading is invention rather
# than a repair. Two is already generous on a six-character number block.
MAX_REPAIRS = 2


def _coerce(chunk: str, mapping: dict, want) -> Optional[Tuple[str, int]]:
    out, changes = [], 0
    for char in chunk:
        if want(char):
            out.append(char)
        elif char in mapping:
            out.append(mapping[char])
            changes += 1
        else:
            return None
    return "".join(out), changes


def repair(text: str) -> Optional[Tuple[str, int]]:
    """Coerce a near-miss reading onto a plate pattern, or None.

    This is what makes tolerant matching reachable. OCR at doorbell range
    returns "CA1Z3456" for CA123456 far more often than it returns nothing, and
    the grammar rejects that string outright — so without a repair step the
    fuzzy registry tiers below could never be consulted for the single most
    common class of error.

    Returns the repaired plate and how many characters had to be changed, so
    the caller can weigh how much of the reading it invented.
    """
    if not text or not (MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN):
        return None

    best: Optional[Tuple[str, int]] = None

    # Prefix style: 2-3 letters then 4-6 digits (CA123456, CAA227793, CF41043).
    for prefix_len in (2, 3):
        digits_len = len(text) - prefix_len
        if not 4 <= digits_len <= 6:
            continue
        head = _coerce(text[:prefix_len], _TO_LETTER, str.isalpha)
        tail = _coerce(text[prefix_len:], _TO_DIGIT, str.isdigit)
        if not head or not tail:
            continue
        changes = head[1] + tail[1]
        if changes == 0 or changes > MAX_REPAIRS:
            continue
        if best is None or changes < best[1]:
            best = (head[0] + tail[0], changes)

    # Suffix-province style: 3 letters, 3 digits, then the province code.
    for suffix in ("GP", "MP", "NW", "FS", "NC", "EC", "ZN", "WP", "L"):
        if len(text) != 6 + len(suffix):
            continue
        if text[-len(suffix):] != suffix:
            continue
        head = _coerce(text[:3], _TO_LETTER, str.isalpha)
        mid = _coerce(text[3:6], _TO_DIGIT, str.isdigit)
        if not head or not mid:
            continue
        changes = head[1] + mid[1]
        if changes == 0 or changes > MAX_REPAIRS:
            continue
        if best is None or changes < best[1]:
            best = (head[0] + mid[0] + suffix, changes)

    return best


def classify(text: str) -> Tuple[Optional[str], float, str]:
    """Categorise a normalised string.

    Returns (kind, base score, resolved text). `kind` is 'plate' for a clean
    read, 'repaired' when characters had to be corrected by position,
    'partial' for a fragment, or None. `resolved text` is what should be
    matched against the registry — the repaired form where one was needed.
    """
    if not text or not (MIN_PLATE_LEN <= len(text) <= MAX_PLATE_LEN):
        # Digit-only fragments are allowed to be shorter than a whole plate.
        if not (text and text.isdigit() and 4 <= len(text) <= 6):
            return None, 0.0, text

    if len(set(text)) == 1:  # "IIIIII" — an OCR artefact, not a plate
        return None, 0.0, text

    for pattern in STRONG_PATTERNS:
        if pattern.match(text):
            return "plate", 100.0, text

    repaired = repair(text)
    if repaired:
        # Always ranked below a clean read, and further discounted per
        # character invented, so a legible plate elsewhere in frame wins.
        return "repaired", 72.0 - 8.0 * repaired[1], repaired[0]

    for pattern in PARTIAL_PATTERNS:
        if pattern.match(text):
            return "partial", 45.0, text

    return None, 0.0, text


def edit_distance_within(a: str, b: str, limit: int = 1) -> Optional[int]:
    """Levenshtein distance, or None once it is known to exceed `limit`."""
    if abs(len(a) - len(b)) > limit:
        return None
    if a == b:
        return 0

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # deletion
                    current[j - 1] + 1,     # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        if min(current) > limit:
            return None
        previous = current

    distance = previous[-1]
    return distance if distance <= limit else None


# --------------------------------------------------------------------------
# OCR geometry
# --------------------------------------------------------------------------


@dataclass
class OcrWord:
    """One word from an OCR engine, with its axis-aligned box in pixels."""

    text: str
    box: Tuple[float, float, float, float]  # x, y, w, h
    confidence: float = 0.0


@dataclass
class OcrLine:
    text: str
    box: Tuple[float, float, float, float]
    confidence: float = 0.0
    words: List[OcrWord] = field(default_factory=list)


@dataclass
class PlateCandidate:
    text: str                 # what we match on, e.g. "CA123456" (repaired if needed)
    raw_text: str             # as the engine read it, e.g. "CA 123-456"
    kind: str                 # "plate" | "repaired" | "partial"
    score: float
    confidence: float
    box: Tuple[float, float, float, float]
    ocr_text: str = ""        # normalised as read, before any repair
    blocked_words: List[str] = field(default_factory=list)

    @property
    def was_repaired(self) -> bool:
        return bool(self.ocr_text) and self.ocr_text != self.text

    def as_dict(self) -> dict:
        x, y, w, h = self.box
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "kind": self.kind,
            "score": round(self.score, 2),
            "ocr_confidence": round(self.confidence, 4),
            "box": {"x": round(x, 1), "y": round(y, 1), "w": round(w, 1), "h": round(h, 1)},
        }


def polygon_to_box(points: Sequence) -> Tuple[float, float, float, float]:
    """Azure and EasyOCR both return quads; flatten to x, y, w, h."""
    xs, ys = [], []
    for point in points or ():
        if hasattr(point, "x"):
            xs.append(float(point.x))
            ys.append(float(point.y))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


def _merge_boxes(boxes: Iterable[Tuple[float, float, float, float]]):
    boxes = [b for b in boxes if b]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _shape_bonus(box: Tuple[float, float, float, float]) -> float:
    """Reward boxes shaped like a plate.

    An SA long plate is roughly 4.7:1; the square two-row plates fitted to
    imports and some bakkies run about 2:1. Anything far from either is more
    likely a badge or a sticker.
    """
    _, _, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    ratio = w / h
    best = min(abs(ratio - 4.7) / 4.7, abs(ratio - 2.0) / 2.0)
    return max(0.0, 15.0 * (1.0 - min(best, 1.0)))


def _size_bonus(box: Tuple[float, float, float, float], frame_size) -> float:
    """Mild preference for the larger reading when two candidates tie.

    Deliberately mild: a plate at the end of a driveway is legitimately tiny,
    so size must never be able to outvote the grammar.
    """
    if not frame_size:
        return 0.0
    fw, fh = frame_size
    if not fw or not fh:
        return 0.0
    _, _, w, h = box
    coverage = (w * h) / float(fw * fh)
    return min(8.0, coverage * 80.0)


def _same_row(a: OcrLine, b: OcrLine) -> bool:
    """True when two lines sit on one horizontal band, e.g. 'CA' then '123-456'."""
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    if not ah or not bh:
        return False
    height = (ah + bh) / 2.0
    if abs((ay + ah / 2.0) - (by + bh / 2.0)) > height * 0.6:
        return False
    if min(ah, bh) / max(ah, bh) < 0.55:
        return False
    gap = bx - (ax + aw) if bx >= ax else ax - (bx + bw)
    return gap < height * 2.0


def _stacked(a: OcrLine, b: OcrLine) -> bool:
    """True when two lines form a two-row plate."""
    ax, ay, aw, ah = a.box
    bx, by, bw, bh = b.box
    if not ah or not bh or by < ay:
        return False
    if min(aw, bw) / max(aw, bw) < 0.5:
        return False
    vertical_gap = by - (ay + ah)
    if vertical_gap > ah * 0.9 or vertical_gap < -ah * 0.3:
        return False
    # Horizontal overlap
    overlap = min(ax + aw, bx + bw) - max(ax, bx)
    return overlap > 0.4 * min(aw, bw)


def _blocked_in(texts: Iterable[str]) -> List[str]:
    return [t for t in texts if normalise(t) in BLOCKED_WORDS]


def _make_candidate(
    raw_parts: Sequence[str],
    box,
    confidence: float,
    frame_size,
) -> Optional[PlateCandidate]:
    raw_text = " ".join(p for p in raw_parts if p).strip()
    ocr_text = normalise(raw_text)
    kind, base, resolved = classify(ocr_text)
    if not kind:
        return None

    blocked = _blocked_in(raw_parts)
    score = (
        base
        + confidence * 20.0
        + _shape_bonus(box)
        + _size_bonus(box, frame_size)
        - (40.0 * len(blocked))
    )
    return PlateCandidate(
        text=resolved,
        raw_text=raw_text,
        kind=kind,
        score=score,
        confidence=confidence,
        box=box,
        ocr_text=ocr_text,
        blocked_words=blocked,
    )


def extract_candidates(lines: Sequence[OcrLine], frame_size=None) -> List[PlateCandidate]:
    """Assemble every plate-shaped reading the OCR output can support.

    Beyond the obvious "each line is a candidate", two assemblies matter in
    practice. Within a line, the plate is often split across words by the dash
    or the space ("CA" "123-456"), so contiguous word runs are also tried. And
    across lines, a plate viewed at an angle frequently comes back as two
    separate lines that a human reads as one — either side by side or stacked
    as a two-row plate.
    """
    candidates: List[PlateCandidate] = []

    for line in lines:
        candidate = _make_candidate([line.text], line.box, line.confidence, frame_size)
        if candidate:
            candidates.append(candidate)

        words = line.words or []
        for start in range(len(words)):
            for end in range(start + 1, min(start + 4, len(words)) + 1):
                run = words[start:end]
                if len(run) == 1 and run[0].text == line.text:
                    continue  # already covered by the whole-line candidate
                confidences = [w.confidence for w in run if w.confidence]
                candidate = _make_candidate(
                    [w.text for w in run],
                    _merge_boxes([w.box for w in run]),
                    sum(confidences) / len(confidences) if confidences else line.confidence,
                    frame_size,
                )
                if candidate:
                    candidates.append(candidate)

    for i, first in enumerate(lines):
        for second in lines[i + 1:]:
            if not (_same_row(first, second) or _stacked(first, second)):
                continue
            confidence = (first.confidence + second.confidence) / 2.0
            candidate = _make_candidate(
                [first.text, second.text],
                _merge_boxes([first.box, second.box]),
                confidence,
                frame_size,
            )
            if candidate:
                candidates.append(candidate)

    # A clean read always outranks a repaired one, which always outranks a
    # fragment — whatever the geometry and OCR confidence say.
    rank = {"plate": 3, "repaired": 2, "partial": 1}
    candidates.sort(key=lambda c: (rank.get(c.kind, 0), c.score), reverse=True)

    deduped: List[PlateCandidate] = []
    seen = set()
    for candidate in candidates:
        if candidate.text in seen:
            continue
        seen.add(candidate.text)
        deduped.append(candidate)
    return deduped


def best_candidate(lines: Sequence[OcrLine], frame_size=None) -> Optional[PlateCandidate]:
    candidates = extract_candidates(lines, frame_size)
    return candidates[0] if candidates else None


def discarded_text(lines: Sequence[OcrLine], kept: Optional[PlateCandidate]) -> List[str]:
    """Everything the grammar threw away — surfaced in the API so the demo can
    show *why* the badge and the dealer sticker were ignored."""
    # Compare against what OCR actually returned, not the repaired form, or the
    # line the plate came from would be reported as "ignored".
    kept_text = (kept.ocr_text or kept.text) if kept else None
    out = []
    for line in lines:
        text = (line.text or "").strip()
        if text and normalise(text) != kept_text:
            out.append(text)
    return out


# --------------------------------------------------------------------------
# Registry matching
# --------------------------------------------------------------------------

REGISTRY_SQL = """
    SELECT p.id,
           p.plate_number,
           p.status,
           p.owner_name,
           (SELECT i.image_url
              FROM vehicle_plate_images i
             WHERE i.plate_id = p.id
             ORDER BY i.created_at DESC
             LIMIT 1) AS image_url
      FROM vehicle_plates p;
"""


def load_registry(db_conn) -> List[dict]:
    """Read the whole plate registry. It is a handful of rows, and holding it in
    memory is what makes fuzzy and fragment matching affordable per frame."""
    cursor = db_conn.cursor()
    try:
        cursor.execute(REGISTRY_SQL)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    registry = []
    for row in rows:
        plate_number = row[1]
        registry.append(
            {
                "id": str(row[0]),
                "plate_number": plate_number,
                "status": row[2],
                "owner_name": row[3],
                "image_url": row[4],
                "_norm": normalise(plate_number),
                "_canon": canonical(plate_number),
            }
        )
    return registry


def _entry_payload(entry: dict) -> dict:
    return {
        "id": entry["id"],
        "plate_number": entry["plate_number"],
        "status": entry["status"],
        "owner_name": entry["owner_name"],
        "image_url": entry["image_url"],
    }


def match_registry(text: str, registry: Sequence[dict], *, kind: str = "plate") -> Optional[dict]:
    """Resolve a read against the registry, cheapest and strictest tier first.

    Returns None when nothing matches. `match_confidence` is "confirmed" only
    for an exact compare on a clean read — every tolerant tier, including a
    reading that needed character repair to become a plate at all, is reported
    as "probable" so the UI can badge it rather than fire an alert on a guess.
    """
    target = normalise(text)
    if not target:
        return None

    # Tier 1 — exact.
    for entry in registry:
        if entry["_norm"] == target:
            repaired = kind == "repaired"
            return {
                "entry": entry,
                "plate": _entry_payload(entry),
                "match_confidence": "probable" if repaired else "confirmed",
                "match_reason": "character_repair" if repaired else "exact",
            }

    # Tier 2 — confusable-folded (0/O, 1/I, 5/S, 8/B).
    folded = canonical(target)
    folded_hits = [e for e in registry if e["_canon"] == folded]
    if len(folded_hits) == 1:
        return {
            "entry": folded_hits[0],
            "plate": _entry_payload(folded_hits[0]),
            "match_confidence": "probable",
            "match_reason": "confusable_characters",
        }

    # Tier 3 — one character out.
    near = []
    for entry in registry:
        distance = edit_distance_within(target, entry["_norm"], limit=1)
        if distance is not None:
            near.append((distance, entry))
    if len(near) == 1:
        entry = near[0][1]
        return {
            "entry": entry,
            "plate": _entry_payload(entry),
            "match_confidence": "probable",
            "match_reason": "edit_distance_1",
        }

    # Tier 4 — a fragment, but one that can only be a single registry entry.
    if kind == "partial" and len(target) >= 4:
        fragment_hits = [
            e for e in registry
            if e["_canon"].startswith(folded) or e["_canon"].endswith(folded)
        ]
        if len(fragment_hits) == 1:
            entry = fragment_hits[0]
            return {
                "entry": entry,
                "plate": _entry_payload(entry),
                "match_confidence": "probable",
                "match_reason": "unique_partial_read",
            }

    return None


def build_result(
    candidate: Optional[PlateCandidate],
    registry: Sequence[dict],
    *,
    engine: str,
    lines: Sequence[OcrLine] = (),
    extra: Optional[dict] = None,
) -> dict:
    """Shape the response every plate endpoint returns.

    `extracted_text` is now the plate and only the plate — never the model
    badge or the dealer frame. What was rejected is still reported under
    `ignored_text` so the behaviour is visible rather than mysterious.
    """
    result = {
        "engine": engine,
        "match_found": False,
        "plate_detected": candidate is not None,
        "extracted_text": candidate.text if candidate else None,
        "raw_text": candidate.raw_text if candidate else None,
        # What OCR literally returned, before position-aware repair. Kept
        # separate so a corrected reading is never passed off as a clean one.
        "ocr_text": candidate.ocr_text if candidate else None,
        "repaired": bool(candidate and candidate.was_repaired),
        "detection_kind": candidate.kind if candidate else None,
        "ocr_confidence": round(candidate.confidence, 4) if candidate else None,
        "plate_box": candidate.as_dict()["box"] if candidate else None,
        "ignored_text": discarded_text(lines, candidate),
    }
    if extra:
        result.update(extra)

    if not candidate:
        result["message"] = (
            "No South African plate pattern found in frame. Vehicle text such as "
            "make, model and dealer branding is ignored by design."
        )
        return result

    match = match_registry(candidate.text, registry, kind=candidate.kind)
    if not match:
        result["message"] = f"Plate '{candidate.text}' read but not flagged in the registry."
        result["match_confidence"] = "unmatched"
        return result

    result["match_found"] = True
    result["plate"] = match["plate"]
    result["match_confidence"] = match["match_confidence"]
    result["match_reason"] = match["match_reason"]
    result["registry_plate"] = match["plate"]["plate_number"]
    if match["match_confidence"] == "probable":
        read_as = candidate.ocr_text or candidate.text
        result["message"] = (
            f"Read '{read_as}' — probable match to registry plate "
            f"{match['plate']['plate_number']} ({match['match_reason'].replace('_', ' ')})."
        )
    else:
        result["message"] = (
            f"Plate {match['plate']['plate_number']} matched in registry as "
            f"{(match['plate']['status'] or 'unknown').upper()}."
        )
    return result
