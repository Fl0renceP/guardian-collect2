"""Travel-risk surface built from historical claims.

The design is dictated by what the claims data can actually carry:

  * Every claim is located to a **suburb centroid**, not a street. So the
    surface is binned to H3 cells at resolution 7 (~5 km across) — finer than
    that would invent precision the data doesn't have.
  * Claims are extremely sparse per suburb (median 2, and 41% of suburbs have
    exactly one). A per-cell-per-hour estimate would be noise, so the surface
    separates *where* from *when*: spatial density is smoothed across
    neighbouring cells, and a single **pooled** hour/day profile scales it.
    That means ~30 temporal parameters estimated off thousands of claims,
    instead of tens of thousands estimated off one or two each.
  * Only travel-relevant perils count. 9,046 of the claims are home contents —
    routing a driver around household burglary would be nonsense.

The output is a normalised 0..1 score per cell. It is a *relative* measure of
where travel-related claims concentrate — not a probability, and not adjusted
for exposure (see `README` → Known gaps: more claims partly means more Discovery
members driving there).
"""

import logging
import math
import threading
from collections import Counter, defaultdict

from config import Config
from services.claims_service import load_claims
from services.geocode_service import load_geocache

logger = logging.getLogger(__name__)

# How much each peril counts toward *travel* risk, which is not the same as how
# common it is. The threat to someone in motion is being stopped and robbed;
# vehicle theft overwhelmingly happens to parked cars, so it still signals a bad
# area but must not dominate. Without this weighting the 6,666 vehicle-theft
# claims drown out the 941 violent ones and the surface peaks at lunchtime —
# which is when cars get stolen, not when people get hijacked.
VIOLENT_PERILS = {"Hijack", "Attempted Hijack", "Armed Robbery"}
PERIL_WEIGHTS = {
    "Hijack": 1.0,
    "Attempted Hijack": 1.0,
    "Armed Robbery": 1.0,
    "Remote jamming": 0.6,       # targets vehicles, usually while stopped
    "Stolen and recovered": 0.3,
    "Theft": 0.3,                # vehicle theft only, per _claim_weight
    "Attempted Theft": 0.3,
}

_cache = {}
_lock = threading.Lock()


def _claim_weight(claim):
    """Travel-risk weight for one claim, or 0 if a traveller wouldn't care.

    Home contents claims score 0 — routing a driver around household burglary
    would be nonsense.
    """
    peril = claim["peril"]
    if peril in VIOLENT_PERILS:
        return PERIL_WEIGHTS[peril]
    if claim["item_type"].lower() == "vehicle":
        return PERIL_WEIGHTS.get(peril, 0.0)
    return 0.0


def _temporal_profile(weighted_claims):
    """Pooled hour-of-day and day-of-week multipliers, mean-normalised to 1.0.

    Weighted by the same travel-risk weights as the spatial surface, so the
    evening hijack peak isn't flattened by daytime vehicle theft.

    Records stored at exactly 00:00 are date-only (6.6% of the data) and would
    otherwise manufacture a midnight spike — they're excluded from the timing
    profile but still count toward spatial density.
    """
    hours = Counter()
    dows = Counter()
    total = 0.0
    timed = 0

    for claim, weight in weighted_claims:
        when = claim["incident_at"]
        if when is None or (when.hour == 0 and when.minute == 0):
            continue
        hours[when.hour] += weight
        dows[when.weekday()] += weight
        total += weight
        timed += 1

    if timed < 100:
        # Too little signal to justify time-varying risk; stay flat and say so.
        logger.warning("Only %d timed claims — temporal profile disabled.", timed)
        return {h: 1.0 for h in range(24)}, {d: 1.0 for d in range(7)}, timed

    hour_mean = total / 24
    dow_mean = total / 7
    # Clamped so a thin hour can't swing a route wildly.
    hour_mult = {h: min(max(hours.get(h, 0) / hour_mean, 0.25), 3.0) for h in range(24)}
    dow_mult = {d: min(max(dows.get(d, 0) / dow_mean, 0.6), 1.6) for d in range(7)}
    return hour_mult, dow_mult, timed


def _spatial_density(weighted_claims, geocache, resolution, rings):
    """Claims binned to H3 cells, then smoothed outward so single-claim suburbs
    contribute a gradient rather than a pinpoint.

    Weight decays by half per ring, which turns a centroid into a plausible
    neighbourhood-sized footprint instead of pretending the incident happened
    at one exact coordinate.
    """
    import h3

    raw = Counter()
    placed = 0
    for claim, weight in weighted_claims:
        point = geocache.get(claim["suburb"])
        if not point:
            continue
        # Approximate pins are a parent suburb's centre, so they're already a
        # rough location — down-weight rather than treat them as exact.
        if point.get("approximate"):
            weight *= 0.5
        raw[h3.latlng_to_cell(point["lat"], point["lng"], resolution)] += weight
        placed += 1

    smoothed = defaultdict(float)
    for cell, weight in raw.items():
        for ring in range(rings + 1):
            decay = 0.5**ring
            # grid_ring gives just that ring; fall back for the centre cell.
            neighbours = h3.grid_disk(cell, ring) if ring == 0 else _ring(h3, cell, ring)
            for neighbour in neighbours:
                smoothed[neighbour] += weight * decay

    return smoothed, raw, placed


def _ring(h3, cell, k):
    """Cells exactly k steps out (grid_disk minus the smaller disk)."""
    try:
        return set(h3.grid_disk(cell, k)) - set(h3.grid_disk(cell, k - 1))
    except Exception:
        return set()


def build_risk_surface(resolution=None, rings=None):
    """Compute (and cache) the base surface. Time scaling is applied per query."""
    resolution = resolution or Config.RISK_H3_RESOLUTION
    rings = rings if rings is not None else Config.RISK_SMOOTHING_RINGS
    key = (resolution, rings)

    cached = _cache.get(key)
    if cached:
        return cached

    with _lock:
        if key in _cache:
            return _cache[key]

        weighted = [(c, _claim_weight(c)) for c in load_claims()]
        weighted = [(c, w) for c, w in weighted if w > 0]
        geocache = load_geocache()
        smoothed, raw, placed = _spatial_density(weighted, geocache, resolution, rings)
        hour_mult, dow_mult, timed = _temporal_profile(weighted)
        claims = weighted

        peak = max(smoothed.values()) if smoothed else 1.0
        # Fixed reference: the worst cell, at the worst hour, on the worst day.
        # Scores are normalised against this rather than against the current
        # moment's peak — otherwise the time multiplier scales every cell
        # equally and cancels out in the division, leaving 05:00 Tuesday
        # indistinguishable from 20:00 Saturday.
        reference_peak = max(hour_mult.values()) * max(dow_mult.values())
        surface = {
            "resolution": resolution,
            "rings": rings,
            "reference_peak": reference_peak,
            "cells": {cell: value / peak for cell, value in smoothed.items()},
            "raw_counts": dict(raw),
            "hour_multiplier": hour_mult,
            "dow_multiplier": dow_mult,
            "claims_used": len(claims),
            "claims_placed": placed,
            "claims_timed": timed,
        }
        _cache[key] = surface
        logger.info(
            "Risk surface: %d travel-relevant claims -> %d cells at res %d (%d timed)",
            len(claims), len(smoothed), resolution, timed,
        )
        return surface


def invalidate():
    """Drop the cached surface — call after the claims snapshot changes."""
    _cache.clear()


def _scaled_surface(hour=None, weekday=None):
    """Cells scaled for a moment in time, plus the fixed reference denominator.

    Cell scores and route scores both come through here so they share one scale —
    otherwise a route's "mean risk" wouldn't be comparable to the numbers on the
    map. The denominator is the *fixed* reference peak, not this moment's peak,
    so 1.0 always means "as bad as this country gets, at the worst hour" and a
    quiet 05:00 genuinely scores lower than a busy 20:00.

    No clamping before normalising: that would flatten every busy cell to
    exactly 1.0 and destroy discrimination precisely where it matters — the top
    of the range is what routing decisions turn on.
    """
    surface = build_risk_surface()
    hour_mult = surface["hour_multiplier"].get(hour, 1.0) if hour is not None else 1.0
    dow_mult = surface["dow_multiplier"].get(weekday, 1.0) if weekday is not None else 1.0
    scale = hour_mult * dow_mult

    key = ("scaled", surface["resolution"], surface["rings"], hour, weekday)
    cached = _cache.get(key)
    if cached:
        return cached

    scaled = {cell: value * scale for cell, value in surface["cells"].items()}
    result = (scaled, surface["reference_peak"], surface)
    _cache[key] = result
    return result


def scored_cells(hour=None, weekday=None, min_score=0.0, limit=None):
    """Risk cells for a moment in time, as a list of dicts sorted high to low."""
    import h3

    scaled, peak, surface = _scaled_surface(hour, weekday)

    out = []
    for cell, value in scaled.items():
        score = value / peak
        if score < min_score:
            continue
        lat, lng = h3.cell_to_latlng(cell)
        out.append(
            {
                "cell": cell,
                "score": round(score, 4),
                "lat": lat,
                "lng": lng,
                "claims": round(surface["raw_counts"].get(cell, 0), 1),
            }
        )

    out.sort(key=lambda c: c["score"], reverse=True)
    return out[:limit] if limit else out


def cell_boundary(cell):
    """[[lng, lat], ...] ring for one cell, in the order Valhalla wants."""
    import h3

    return [[lng, lat] for lat, lng in h3.cell_to_boundary(cell)]


def score_at(lat, lng, hour=None, weekday=None):
    """Risk score at a single point, on the same 0..1 scale as `scored_cells`."""
    import h3

    scaled, peak, surface = _scaled_surface(hour, weekday)
    cell = h3.latlng_to_cell(lat, lng, surface["resolution"])
    return scaled.get(cell, 0.0) / peak


def cell_at(lat, lng):
    """The H3 cell containing a point, at the surface's resolution."""
    import h3

    return h3.latlng_to_cell(lat, lng, build_risk_surface()["resolution"])


def timing_profile():
    """The pooled hour/day multipliers, for charting and for honest disclosure."""
    surface = build_risk_surface()
    return {
        "hour_multiplier": surface["hour_multiplier"],
        "dow_multiplier": surface["dow_multiplier"],
        "claims_used": surface["claims_used"],
        "claims_timed": surface["claims_timed"],
        "resolution": surface["resolution"],
        "cell_km2": round(_hex_area(surface["resolution"]), 2),
    }


def _hex_area(resolution):
    import h3

    return h3.average_hexagon_area(resolution, unit="km^2")


def haversine_km(a, b):
    """Great-circle distance between two (lat, lng) pairs."""
    lat1, lng1 = a
    lat2, lng2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))
