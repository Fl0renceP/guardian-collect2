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


def geocode(session, suburb):
    """Resolve one suburb. Returns a cache entry dict, or None if nothing usable came back."""
    response = session.get(
        Config.NOMINATIM_URL,
        params={
            "q": f"{suburb}, South Africa",
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

    eta_minutes = len(pending) * Config.NOMINATIM_DELAY_SECONDS / 60
    print(f"Estimated runtime: {eta_minutes:.0f} min at 1 req/sec. Safe to interrupt.\n", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": Config.NOMINATIM_USER_AGENT})

    located = failed = 0
    for index, suburb in enumerate(pending, start=1):
        if _interrupted:
            break

        try:
            entry = geocode(session, suburb)
        except requests.RequestException as exc:
            # Transient network/HTTP problem: leave it uncached so a re-run retries it.
            print(f"  [{index}/{len(pending)}] {suburb}: request failed ({exc})", flush=True)
            time.sleep(Config.NOMINATIM_DELAY_SECONDS * 3)
            continue

        cache[suburb] = entry
        if entry:
            located += 1
        else:
            failed += 1

        if index % FLUSH_EVERY == 0:
            save_cache(cache_path, cache)
            print(
                f"  [{index}/{len(pending)}] {located} located, {failed} not found "
                f"(last: {suburb})",
                flush=True,
            )

        time.sleep(Config.NOMINATIM_DELAY_SECONDS)

    save_cache(cache_path, cache)
    total_hits = sum(1 for v in cache.values() if v)
    print(
        f"\nDone. This run: {located} located, {failed} not found. "
        f"Cache now holds {total_hits}/{len(cache)} located suburbs -> {cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
