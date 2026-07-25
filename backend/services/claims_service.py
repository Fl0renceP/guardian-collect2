"""Reads and aggregates Discovery Insure claims, then rolls them into hot-spots.

**Source of truth is Azure Cosmos DB** (`guardian-db/insurance-data`). It's
writable, so a claim submitted by a member and approved by a Discovery employee
appears on the hot-spot map without a redeploy — that's the whole reason the map
doesn't read the CSV any more. The CSV survives only as an offline fallback
(`CLAIMS_SOURCE=csv`, or automatically when Cosmos is unreachable), so a dead
network on demo day doesn't take the map down with it.

Claims are snapshotted into memory for `CLAIMS_CACHE_TTL_SECONDS`. The filter UI
re-queries on every chip click; serving those from a snapshot keeps the map
responsive and the RU spend flat instead of fanning out across partitions each
time. Writes made through this app should call `invalidate_cache()` so approvals
show up instantly rather than waiting out the TTL.

Field shapes differ between the two sources and `_normalize` absorbs it:
Cosmos gives real numbers, ISO-T datetimes and nulls; the CSV gives a UTF-8 BOM,
semicolons, comma decimals ("391572,28") and empty strings.
"""

import csv
import logging
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime

from config import Config
from services.cosmos_client import CosmosUnavailable, get_container, is_configured

logger = logging.getLogger(__name__)

# Suburb value used by the source data when the location wasn't captured.
UNKNOWN_SUBURB = "UNKNOWN"

_DATE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)

# Claims carrying a workflow status only count once approved — a member's
# pending submission must not move the hot-spot map before an employee verifies
# it (PROJECT_CONTEXT §3.5). Historical records have no status field at all, so
# absence means "already part of the dataset" and is counted.
APPROVED_STATUSES = {"approved", "verified", "accepted"}

_snapshot = None
_snapshot_at = 0.0
_snapshot_source = None
_lock = threading.Lock()
_refreshing = False


def _parse_amount(raw):
    """Cosmos gives a number; the CSV gives '391572,28'. Both land as float."""
    if isinstance(raw, (int, float)):
        return float(raw)
    if not raw:
        return 0.0
    try:
        return float(str(raw).strip().replace(" ", "").replace(",", "."))
    except ValueError:
        return 0.0


def _parse_datetime(raw):
    """Parse INCIDENT_DATE_TIME, tolerating every shape present in either source."""
    if isinstance(raw, datetime):
        return raw
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    # Last resort: ISO strings carrying a timezone or fractional seconds.
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _text(value, default=""):
    """Cosmos nulls and CSV empty strings both collapse to the default."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize(record):
    """One claim record from either source -> the shape the rest of the app uses."""
    return {
        "incident": _text(record.get("Incident")),
        "peril": _text(record.get("PERIL"), "Unspecified"),
        "suburb": _text(record.get("SUBURB"), UNKNOWN_SUBURB).upper(),
        "item_type": _text(record.get("ITEM_TYPE"), "Unspecified"),
        "item_category": _text(record.get("ITEM_CATEGORY")),
        "vehicle_make": _text(record.get("VEHICLE_MAKE")),
        "incident_at": _parse_datetime(record.get("INCIDENT_DATE_TIME")),
        "amount": _parse_amount(record.get("CLAIM_AMOUNT")),
        "status": _text(record.get("status")).lower(),
    }


def _is_countable(claim):
    return not claim["status"] or claim["status"] in APPROVED_STATUSES


def _load_from_cosmos():
    container = get_container()
    # Cosmos reserves the _-prefixed system fields; selecting only what we use
    # keeps the payload (and the RU charge) down.
    query = (
        "SELECT c.Incident, c.PERIL, c.SUBURB, c.ITEM_TYPE, c.ITEM_CATEGORY, "
        "c.VEHICLE_MAKE, c.INCIDENT_DATE_TIME, c.CLAIM_AMOUNT, c.status FROM c"
    )
    try:
        records = list(container.query_items(query, enable_cross_partition_query=True))
    except Exception as exc:
        raise CosmosUnavailable(f"claims query failed: {exc}") from exc
    return [_normalize(r) for r in records]


def _load_from_csv():
    path = Config.CLAIMS_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Claims CSV not found at {path}. Set CLAIMS_CSV_PATH in .env to point at it."
        )
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [
            _normalize(row)
            for row in csv.DictReader(handle, delimiter=Config.CLAIMS_CSV_DELIMITER)
        ]


def _fetch():
    """Pull a fresh claim list, honouring CLAIMS_SOURCE. Returns (claims, source)."""
    source = Config.CLAIMS_SOURCE

    if source == "csv":
        return _load_from_csv(), "csv"

    if source == "cosmos":
        return _load_from_cosmos(), "cosmos"

    # "auto": prefer Cosmos, fall back to the CSV so a network failure degrades
    # the demo to stale-but-working rather than broken.
    if is_configured():
        try:
            return _load_from_cosmos(), "cosmos"
        except CosmosUnavailable as exc:
            logger.warning("Cosmos unavailable (%s) — falling back to the CSV.", exc)
    else:
        logger.warning("Cosmos not configured — using the CSV fallback.")
    return _load_from_csv(), "csv-fallback"


def _is_fresh():
    return (
        _snapshot is not None
        and (time.monotonic() - _snapshot_at) < Config.CLAIMS_CACHE_TTL_SECONDS
    )


def _refresh_now():
    """Re-fetch and replace the snapshot. Holds the lock for the duration."""
    global _snapshot, _snapshot_at, _snapshot_source

    with _lock:
        started = time.monotonic()
        try:
            claims, source = _fetch()
        except Exception:
            if _snapshot is not None:
                # Serving a stale snapshot beats serving an error page.
                logger.exception("Claims refresh failed — keeping the previous snapshot.")
                # Back off a full TTL before retrying, so a hard outage doesn't
                # spawn a refresh thread per request.
                _snapshot_at = time.monotonic()
                return _snapshot
            raise

        countable = [c for c in claims if _is_countable(c)]
        _snapshot = countable
        _snapshot_at = time.monotonic()
        _snapshot_source = source
        logger.info(
            "Loaded %d claims (%d countable) from %s in %.2fs",
            len(claims), len(countable), source, time.monotonic() - started,
        )
        return _snapshot


def _refresh_in_background():
    """Refresh off the request thread, at most one at a time."""
    global _refreshing

    def run():
        global _refreshing
        try:
            _refresh_now()
        except Exception:
            logger.exception("Background claims refresh failed.")
        finally:
            _refreshing = False

    _refreshing = True
    threading.Thread(target=run, name="claims-refresh", daemon=True).start()


def load_claims(force_refresh=False):
    """Every countable claim.

    Stale-while-revalidate: an expired snapshot is returned immediately and
    refreshed on a background thread. A full pull of the collection takes a few
    seconds, so blocking on it would stall the filter UI every time the TTL
    lapsed. Only the very first call (no snapshot yet) waits.
    """
    if force_refresh:
        return _refresh_now()

    if _snapshot is None:
        return _refresh_now()

    if not _is_fresh() and not _refreshing:
        _refresh_in_background()

    return _snapshot


def warm_cache():
    """Populate the snapshot at startup so no user request pays the cold-load cost."""
    if _snapshot is None and not _refreshing:
        _refresh_in_background()


def invalidate_cache():
    """Mark the snapshot stale so the next read re-queries.

    Call this after writing a claim (submission or approval) so the map reflects
    it promptly rather than waiting out the TTL. Pair it with
    `load_claims(force_refresh=True)` when the caller needs the write reflected
    in the very next response.
    """
    global _snapshot_at
    _snapshot_at = 0.0


def source_status():
    """Where the current data came from and how stale it is — surfaced by /api/health."""
    if _snapshot is None:
        return {
            "loaded": False,
            "configured_source": Config.CLAIMS_SOURCE,
            "cosmos_configured": is_configured(),
        }
    return {
        "loaded": True,
        "configured_source": Config.CLAIMS_SOURCE,
        "active_source": _snapshot_source,
        "cosmos_configured": is_configured(),
        "claims": len(_snapshot),
        "age_seconds": round(time.monotonic() - _snapshot_at, 1),
        "ttl_seconds": Config.CLAIMS_CACHE_TTL_SECONDS,
        "refreshing": _refreshing,
    }


def distinct_suburbs(include_unknown=False):
    """Every distinct suburb name, sorted by claim volume (busiest first).

    Volume order matters for the geocoder: if a run is interrupted, the suburbs
    covering the most claims are already cached.
    """
    counts = Counter(
        c["suburb"]
        for c in load_claims()
        if include_unknown or c["suburb"] != UNKNOWN_SUBURB
    )
    return [suburb for suburb, _ in counts.most_common()]


def filter_options():
    """Filter values the frontend can offer, derived from the data rather than hardcoded."""
    claims = load_claims()
    dates = [c["incident_at"] for c in claims if c["incident_at"]]
    peril_counts = Counter(c["peril"] for c in claims)
    item_counts = Counter(c["item_type"] for c in claims)

    return {
        "perils": [{"value": p, "count": n} for p, n in peril_counts.most_common()],
        "item_types": [{"value": t, "count": n} for t, n in item_counts.most_common()],
        "date_min": min(dates).date().isoformat() if dates else None,
        "date_max": max(dates).date().isoformat() if dates else None,
        "total_claims": len(claims),
    }


def _matches(claim, perils, item_types, date_from, date_to):
    if perils and claim["peril"] not in perils:
        return False
    if item_types and claim["item_type"] not in item_types:
        return False
    if date_from or date_to:
        incident_at = claim["incident_at"]
        if incident_at is None:
            return False
        if date_from and incident_at.date() < date_from:
            return False
        if date_to and incident_at.date() > date_to:
            return False
    return True


def aggregate_hotspots(geocache, perils=None, item_types=None, date_from=None, date_to=None):
    """Aggregate matching claims into per-suburb hot-spots ready for the heatmap.

    `geocache` maps SUBURB -> {"lat": float, "lng": float} (see geocode_service).
    Claims whose suburb is UNKNOWN or missing from the cache can't be placed on
    the map; they're counted in `unplaced` rather than silently dropped, so the
    UI can be honest about coverage.
    """
    by_suburb = defaultdict(
        lambda: {"count": 0, "total_amount": 0.0, "perils": Counter(), "item_types": Counter()}
    )
    matched = 0
    unknown_suburb = 0
    not_geocoded = Counter()

    for claim in load_claims():
        if not _matches(claim, perils, item_types, date_from, date_to):
            continue
        matched += 1

        suburb = claim["suburb"]
        if suburb == UNKNOWN_SUBURB:
            unknown_suburb += 1
            continue
        if suburb not in geocache:
            not_geocoded[suburb] += 1
            continue

        bucket = by_suburb[suburb]
        bucket["count"] += 1
        # Clamp negatives: a handful of rows carry credit/reversal amounts that
        # would otherwise pull a suburb's total below zero and misstate severity.
        bucket["total_amount"] += max(claim["amount"], 0.0)
        bucket["perils"][claim["peril"]] += 1
        bucket["item_types"][claim["item_type"]] += 1

    hotspots = []
    for suburb, bucket in by_suburb.items():
        point = geocache[suburb]
        hotspots.append(
            {
                "suburb": suburb,
                "lat": point["lat"],
                "lng": point["lng"],
                "count": bucket["count"],
                "total_amount": round(bucket["total_amount"], 2),
                "top_peril": bucket["perils"].most_common(1)[0][0],
                "perils": dict(bucket["perils"].most_common()),
                "item_types": dict(bucket["item_types"].most_common()),
            }
        )
    hotspots.sort(key=lambda h: h["count"], reverse=True)

    placed = sum(h["count"] for h in hotspots)
    return {
        "hotspots": hotspots,
        "max_count": hotspots[0]["count"] if hotspots else 0,
        "matched_claims": matched,
        "placed_claims": placed,
        "unplaced_claims": matched - placed,
        "unplaced_breakdown": {
            "unknown_suburb": unknown_suburb,
            "not_geocoded": sum(not_geocoded.values()),
            "not_geocoded_suburbs": len(not_geocoded),
        },
    }
