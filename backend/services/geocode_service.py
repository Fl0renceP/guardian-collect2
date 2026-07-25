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
