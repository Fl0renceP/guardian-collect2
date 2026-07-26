"""One-time (resumable) geocoder: suburb name -> lat/lng, via OpenStreetMap Nominatim.

The claims CSV has no coordinates, so every distinct SUBURB has to be resolved
once and cached before anything spatial can happen. Results land in
`backend/data/suburb_geocache.json` and are committed, so the rest of the team
never has to re-run this.

Usage (from the repo root):
    uv run --with requests --with python-dotenv backend/scripts/geocode_suburbs.py
    uv run ... backend/scripts/geocode_suburbs.py --retry-misses   # re-try failures only
    uv run ... backend/scripts/geocode_suburbs.py --limit 50       # quick smoke test

Nominatim's usage policy is not optional: max 1 request/second, and a real
User-Agent identifying the application. Both are enforced below. Do not
parallelise this — you'll get the whole team's IP blocked.

Safe to interrupt: the cache is flushed to disk every 25 lookups and on exit,
and suburbs are processed busiest-first, so an interrupted run still leaves you
with coverage of the highest-volume areas.
"""

import argparse
import json
import re
import signal
import sys
import time
from pathlib import Path

import requests

# Make `config` and `services` importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from services.claims_service import distinct_suburbs  # noqa: E402

FLUSH_EVERY = 25
_interrupted = False


def _handle_interrupt(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupt received — flushing cache and exiting cleanly...", flush=True)


def load_cache(path):
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def save_cache(path, cache):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then replace, so an interrupt mid-write can't corrupt the cache.
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=1, sort_keys=True, ensure_ascii=False)
    tmp.replace(path)


def in_south_africa(lat, lng):
    b = Config.SA_BOUNDS
    return b["min_lat"] <= lat <= b["max_lat"] and b["min_lng"] <= lng <= b["max_lng"]


# Tokens that describe a *development* rather than a place OSM is likely to know:
# private estates, agricultural holdings, cadastral portions, township extensions.
_NOISE_WORDS = re.compile(
    r"\b(ESTATES?|GOLF|SECURITY|LIFESTYLE|RETIREMENT|COUNTRY|MANOR|AH|SH|"
    r"EXT|EXTENSION|PORTION|PTN|SMALLHOLDINGS?|AGRICULTURAL\s+HOLDINGS?)\b",
    re.I,
)
# "BEDFORD 68-IR", "EXT 12", trailing bare numbers.
_CADASTRAL = re.compile(r"\b\d+\s*-?\s*[A-Z]{0,2}\b")
_DIRECTIONS = re.compile(r"\b(NORTH|SOUTH|EAST|WEST|CENTRAL|UPPER|LOWER)\b", re.I)


def _clean(text):
    return re.sub(r"\s{2,}", " ", text).strip(" ,-&").strip()


def query_variants(suburb):
    """Progressively broader queries for one suburb, most precise first.

    Only the first is an exact match; every later one trades precision for a
    pin, so callers mark those `approximate` rather than passing them off as
    exact hits. This is what recovers names like "BANKENVELD GOLF ESTATE" or
    "ALBERTON NORTH" that OSM doesn't carry as distinct places.
    """
    seen = set()

    def offer(candidate):
        candidate = _clean(candidate)
        # Two characters isn't a place name — it's the remains of over-stripping.
        if len(candidate) < 3 or candidate.upper() in seen:
            return None
        seen.add(candidate.upper())
        return candidate

    first = offer(suburb)
    if first:
        yield first, False

    # Drop cadastral/extension numbering: "BEDFORD 68-IR" -> "BEDFORD".
    stripped = offer(_CADASTRAL.sub(" ", suburb))
    if stripped:
        yield stripped, True

    # Drop development nouns: "BANKENVELD GOLF ESTATE" -> "BANKENVELD".
    denoised = offer(_NOISE_WORDS.sub(" ", _CADASTRAL.sub(" ", suburb)))
    if denoised:
        yield denoised, True

    # Drop a directional qualifier: "ALBERTON NORTH" -> "ALBERTON". The parent
    # suburb's centre is a few km off at worst — fine at the zoom this map uses.
    undirected = offer(_DIRECTIONS.sub(" ", denoised or suburb))
    if undirected:
        yield undirected, True

    # Last resort for long names: keep the leading two words. Single words are
    # deliberately not tried — "AMBER" would match anything.
    words = (denoised or suburb).split()
    if len(words) > 2:
        head = offer(" ".join(words[:2]))
        if head:
            yield head, True


def _lookup(session, query):
    response = session.get(
        Config.NOMINATIM_URL,
        params={
            "q": f"{query}, South Africa",
            "countrycodes": "za",
            "format": "jsonv2",
            "limit": 1,
            "addressdetails": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    hit = results[0]
    lat, lng = float(hit["lat"]), float(hit["lon"])
    if not in_south_africa(lat, lng):
        # Name collision with somewhere outside SA — better no pin than a wrong one.
        return None

    return {
        "lat": lat,
        "lng": lng,
        "display_name": hit.get("display_name", ""),
        "osm_type": hit.get("addresstype") or hit.get("type", ""),
        "importance": hit.get("importance"),
    }


def geocode(session, suburb, use_variants=False, on_request=None, skip_exact=False):
    """Resolve one suburb. Returns a cache entry dict, or None if nothing matched.

    With `use_variants`, falls back through progressively broader queries and
    records `approximate: True` plus the query that actually hit, so an
    approximate pin is never mistaken for an exact one downstream.

    `skip_exact` drops the verbatim query — on a retry pass that one is already
    known to return nothing, and re-asking wastes a second of the rate limit.
    """
    variants = list(query_variants(suburb)) if use_variants else [(suburb, False)]
    if skip_exact:
        variants = [(q, a) for q, a in variants if a]

    for query, approximate in variants:
        if on_request:
            on_request()
        entry = _lookup(session, query)
        if entry:
            entry["approximate"] = approximate
            if approximate:
                entry["matched_query"] = query
            return entry
    return None


def main():
    parser = argparse.ArgumentParser(description="Geocode claim suburbs via Nominatim.")
    parser.add_argument("--limit", type=int, help="only process the N busiest uncached suburbs")
    parser.add_argument(
        "--retry-misses",
        action="store_true",
        help="re-attempt suburbs previously cached as not-found",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_interrupt)
    signal.signal(signal.SIGTERM, _handle_interrupt)

    cache_path = Config.GEOCACHE_PATH
    cache = load_cache(cache_path)
    suburbs = distinct_suburbs()

    def needs_lookup(suburb):
        if suburb not in cache:
            return True
        return args.retry_misses and cache[suburb] is None

    pending = [s for s in suburbs if needs_lookup(s)]
    if args.limit:
        pending = pending[: args.limit]

    hits = sum(1 for v in cache.values() if v)
    print(
        f"{len(suburbs)} distinct suburbs | {len(cache)} cached ({hits} located) "
        f"| {len(pending)} to look up",
        flush=True,
    )
    if not pending:
        print("Nothing to do — cache is complete.", flush=True)
        return

    # A retry pass tries several query shapes per suburb, so budget for more
    # than one request each.
    per_suburb = 3 if args.retry_misses else 1
    eta_minutes = len(pending) * per_suburb * Config.NOMINATIM_DELAY_SECONDS / 60
    print(
        f"Estimated runtime: up to {eta_minutes:.0f} min at 1 req/sec. Safe to interrupt.\n",
        flush=True,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": Config.NOMINATIM_USER_AGENT})

    # The 1 req/sec limit is per *request*, not per suburb — with fallback
    # variants one suburb can make several, so the pacing lives here.
    def pace():
        time.sleep(Config.NOMINATIM_DELAY_SECONDS)

    located = approximate = failed = 0
    for index, suburb in enumerate(pending, start=1):
        if _interrupted:
            break

        try:
            entry = geocode(
                session,
                suburb,
                use_variants=args.retry_misses,
                on_request=pace,
                # Already cached as a miss, so the verbatim query is a known dead end.
                skip_exact=args.retry_misses and cache.get(suburb, "absent") is None,
            )
        except requests.RequestException as exc:
            # Transient network/HTTP problem: leave it uncached so a re-run retries it.
            print(f"  [{index}/{len(pending)}] {suburb}: request failed ({exc})", flush=True)
            time.sleep(Config.NOMINATIM_DELAY_SECONDS * 3)
            continue

        cache[suburb] = entry
        if entry:
            located += 1
            if entry.get("approximate"):
                approximate += 1
        else:
            failed += 1

        if index % FLUSH_EVERY == 0:
            save_cache(cache_path, cache)
            print(
                f"  [{index}/{len(pending)}] {located} located "
                f"({approximate} approximate), {failed} not found (last: {suburb})",
                flush=True,
            )

    save_cache(cache_path, cache)
    total_hits = sum(1 for v in cache.values() if v)
    print(
        f"\nDone. This run: {located} located ({approximate} approximate), "
        f"{failed} not found. "
        f"Cache now holds {total_hits}/{len(cache)} located suburbs -> {cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
