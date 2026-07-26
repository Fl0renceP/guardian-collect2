"""Read access to the suburb -> lat/lng cache built by scripts/geocode_suburbs.py.

Nothing here calls Nominatim at request time. Geocoding is a build step, not a
runtime dependency: the API must stay fast and must not hammer a free service on
every page load.
"""

import json
import logging

from config import Config

logger = logging.getLogger(__name__)

_cache = None
_cache_mtime = None


def load_geocache():
    """Return {SUBURB: {"lat", "lng", ...}} for successfully located suburbs only.

    Reloaded automatically when the file changes on disk, so the map picks up a
    still-running geocode job without a server restart.
    """
    global _cache, _cache_mtime

    path = Config.GEOCACHE_PATH
    if not path.exists():
        if _cache is None:
            logger.warning(
                "Geocache missing at %s — the map will have no points. "
                "Run: uv run backend/scripts/geocode_suburbs.py",
                path,
            )
            _cache = {}
        return _cache

    mtime = path.stat().st_mtime
    if _cache is None or mtime != _cache_mtime:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        # Entries are None for suburbs Nominatim couldn't resolve; drop them so
        # callers can treat cache membership as "has coordinates".
        _cache = {k: v for k, v in raw.items() if v}
        _cache_mtime = mtime
        logger.info("Loaded %d geocoded suburbs from %s", len(_cache), path)

    return _cache


def ensure_suburb(suburb):
    """Geocode one suburb on demand and persist it to the cache.

    Called only when a claim is *approved*, never per map load — a newly
    submitted suburb otherwise counts toward hot-spots but has nowhere to sit on
    the map. Returns the cache entry, or None if it couldn't be resolved.

    Failures are the caller's to swallow: an approval must not fail because
    OpenStreetMap was slow.
    """
    suburb = (suburb or "").strip().upper()
    if not suburb or suburb == "UNKNOWN":
        return None

    cache = load_geocache()
    if suburb in cache:
        return cache[suburb]

    path = Config.GEOCACHE_PATH
    raw = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    if suburb in raw:
        # Already attempted and cached as a miss — don't re-ask on every approval.
        return raw[suburb]

    # Imported lazily: the geocoder is a script-side concern and pulls in requests.
    import sys

    sys.path.insert(0, str(Config.GEOCACHE_PATH.parent.parent / "scripts"))
    import requests
    from geocode_suburbs import geocode

    session = requests.Session()
    session.headers.update({"User-Agent": Config.NOMINATIM_USER_AGENT})
    entry = geocode(session, suburb, use_variants=True)

    raw[suburb] = entry
    tmp = path.with_suffix(".json.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=1, sort_keys=True, ensure_ascii=False)
    tmp.replace(path)

    logger.info("Geocoded new suburb %s on approval: %s", suburb, "hit" if entry else "miss")
    return entry


def geocache_status():
    """Coverage summary, surfaced by /api/health so the team can see if geocoding is done."""
    path = Config.GEOCACHE_PATH
    if not path.exists():
        return {"ready": False, "located": 0, "attempted": 0, "path": str(path)}

    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    located = sum(1 for v in raw.values() if v)
    return {
        "ready": located > 0,
        "located": located,
        "attempted": len(raw),
        "not_found": len(raw) - located,
        "path": str(path),
    }
