"""Route planning via Valhalla (OpenStreetMap), with a risk-aware alternative.

Valhalla is used rather than Azure Maps for the same reason the hot-spot map is
Leaflet: no subscription key exists, and the public FOSSGIS instance needs none.
It also has `exclude_polygons`, which is what makes "route around these areas"
a one-parameter change rather than a custom graph build.

Two routes are always produced for comparison — the fastest one, and one that
avoids the highest-risk cells for the departure time. Showing both, with the
time cost stated, is deliberate: the member chooses, the app doesn't quietly
send someone the long way round.
"""

import logging

import requests

from config import Config
from services import risk_service

logger = logging.getLogger(__name__)

# Valhalla's own naming; the UI offers the first three.
COSTING_MODES = {"auto", "pedestrian", "bicycle", "motorcycle", "motor_scooter"}


class RoutingError(RuntimeError):
    pass


def decode_polyline(encoded, precision=6):
    """Decode a Google-format encoded polyline. Valhalla uses precision 6.

    Written out rather than pulled from a library — it's 20 lines and avoids a
    dependency whose only job would be this.
    """
    coords = []
    index = lat = lng = 0
    factor = 10**precision

    while index < len(encoded):
        for axis in range(2):
            shift = result = 0
            while True:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else (result >> 1)
            if axis == 0:
                lat += delta
            else:
                lng += delta
        coords.append((lat / factor, lng / factor))
    return coords


def _call_valhalla(locations, costing, exclude_polygons=None):
    body = {
        "locations": [{"lat": lat, "lon": lng} for lat, lng in locations],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }
    if exclude_polygons:
        body["exclude_polygons"] = exclude_polygons

    try:
        response = requests.post(
            Config.VALHALLA_URL,
            json=body,
            headers={"User-Agent": Config.NOMINATIM_USER_AGENT},
            timeout=Config.VALHALLA_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RoutingError(f"Routing service unreachable: {exc}") from exc

    if response.status_code != 200:
        detail = ""
        try:
            detail = response.json().get("error", "")
        except ValueError:
            detail = response.text[:200]
        raise RoutingError(f"Routing failed ({response.status_code}): {detail}")

    return response.json()


def _shape_of(trip):
    points = []
    for leg in trip.get("legs", []):
        points.extend(decode_polyline(leg["shape"]))
    return points


def _sample(points, step_km=0.4):
    """Thin a route's shape to roughly one point per step_km.

    Scoring every vertex would over-weight dense urban geometry, where a
    kilometre of road carries far more points than a kilometre of highway.
    """
    if not points:
        return []
    kept = [points[0]]
    travelled = 0.0
    for previous, current in zip(points, points[1:]):
        travelled += risk_service.haversine_km(previous, current)
        if travelled >= step_km:
            kept.append(current)
            travelled = 0.0
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def score_route(points, hour=None, weekday=None):
    """Summarise a route's exposure to the risk surface.

    `mean` is what to compare between two routes (exposure per unit of path);
    `peak` and `high_share` describe how bad the worst of it gets.
    """
    samples = _sample(points)
    if not samples:
        return {"mean": 0.0, "peak": 0.0, "high_share": 0.0, "samples": 0}

    scores = [risk_service.score_at(lat, lng, hour, weekday) for lat, lng in samples]
    high = sum(1 for s in scores if s >= Config.ROUTE_HIGH_RISK_THRESHOLD)
    return {
        "mean": round(sum(scores) / len(scores), 4),
        "peak": round(max(scores), 4),
        "high_share": round(high / len(scores), 4),
        "samples": len(samples),
    }


def _summarise(trip, hour, weekday, label, avoided_cells=None):
    points = _shape_of(trip)
    summary = trip.get("summary", {})
    return {
        "label": label,
        "distance_km": round(summary.get("length", 0.0), 2),
        "duration_min": round(summary.get("time", 0.0) / 60.0, 1),
        "coordinates": [[lat, lng] for lat, lng in points],
        "risk": score_route(points, hour, weekday),
        "avoided_cells": avoided_cells or 0,
    }


def avoid_cells_on_route(route_points, origin, destination, hour, weekday):
    """Pick the risk cells worth routing around — only ones the route goes through.

    This must be driven by the actual path, not by the worst cells nationally.
    Excluding the country's top-risk cells would hand Valhalla polygons in Cape
    Town for a Johannesburg trip: useless, and it burns the polygon budget that
    the cells actually in the way needed.

    Two further guards, both load-bearing:
      * the origin and destination cells are never excluded — Valhalla can't
        start or end inside an excluded polygon;
      * only cells at or above the threshold qualify, so a quiet trip proposes
        no detour at all rather than inventing one.
    """
    endpoints = {
        risk_service.cell_at(origin[0], origin[1]),
        risk_service.cell_at(destination[0], destination[1]),
    }

    seen = {}
    for lat, lng in _sample(route_points, step_km=0.3):
        cell = risk_service.cell_at(lat, lng)
        if cell in endpoints or cell in seen:
            continue
        score = risk_service.score_at(lat, lng, hour, weekday)
        if score >= Config.ROUTE_AVOID_THRESHOLD:
            seen[cell] = score

    chosen = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    return [
        {"cell": cell, "score": round(score, 4)}
        for cell, score in chosen[: Config.ROUTE_MAX_AVOID_POLYGONS]
    ]


class UnsupportedMode(ValueError):
    """A travel mode the router doesn't offer — the caller's mistake, not the router's."""


def _attempt_sizes(count):
    """How many of the worst cells to try excluding, most ambitious first."""
    sizes, size = [], count
    while size >= 1:
        sizes.append(size)
        if size == 1:
            break
        size = max(1, size // 2)
    return sizes


def _try_avoiding(locations, costing, cells, fastest_minutes):
    """Find the most risk avoidance that still fits the detour budget.

    Excluding every high-risk cell on a route often severs the only viable
    corridor — you get a technically-safer route twice as long, which nobody
    takes. So step down through smaller exclusion sets and keep the first that
    lands inside the budget. Avoiding the worst two areas for eight extra
    minutes is a real option; avoiding six for an extra hour is not.

    Returns (trip, cells_used, over_budget_fallback) — the fallback is the best
    over-budget attempt, still worth showing for comparison.
    """
    fallback = None

    for size in _attempt_sizes(len(cells)):
        attempt = cells[:size]
        polygons = [risk_service.cell_boundary(c["cell"]) for c in attempt]
        try:
            trip = _call_valhalla(locations, costing, exclude_polygons=polygons)["trip"]
        except RoutingError as exc:
            # Walled off the destination — try excluding less.
            logger.info("Avoiding route failed with %d polygons: %s", size, exc)
            continue

        minutes = trip.get("summary", {}).get("time", 0.0) / 60.0
        ratio = minutes / fastest_minutes if fastest_minutes else 1.0
        if ratio <= Config.ROUTE_MAX_DETOUR_RATIO:
            return trip, attempt, None
        if fallback is None:
            fallback = (trip, attempt)

    if fallback:
        return None, [], fallback
    return None, [], None


def compare_routes(origin, destination, costing="auto", hour=None, weekday=None):
    """Fastest route vs risk-avoiding route, with the trade-off made explicit."""
    if costing not in COSTING_MODES:
        raise UnsupportedMode(
            f"Unsupported travel mode '{costing}'. Use one of: {', '.join(sorted(COSTING_MODES))}."
        )

    locations = [origin, destination]
    fastest_trip = _call_valhalla(locations, costing)["trip"]
    fastest = _summarise(fastest_trip, hour, weekday, "fastest")

    cells = avoid_cells_on_route(
        [tuple(p) for p in fastest["coordinates"]], origin, destination, hour, weekday
    )
    result = {
        "fastest": fastest,
        "safer": None,
        "recommendation": "fastest",
        "reason": None,
        "avoid_cells": cells,
        "mode": costing,
        "hour": hour,
        "weekday": weekday,
    }

    if not cells:
        result["reason"] = "No elevated-risk areas on this route at this time."
        return result

    safer_trip, used_cells, fallback = _try_avoiding(
        locations, costing, cells, fastest["duration_min"]
    )

    over_budget = False
    if safer_trip is None and fallback is not None:
        safer_trip, used_cells = fallback
        over_budget = True
    elif safer_trip is None:
        # Excluding areas can make a route impossible. That's not an error the
        # member needs to see — it just means there's no safer option here.
        result["reason"] = "No practical alternative avoids those areas."
        return result

    result["avoid_cells"] = used_cells
    cells = used_cells
    safer = _summarise(safer_trip, hour, weekday, "safer", avoided_cells=len(cells))
    result["safer"] = safer

    # Decide what to actually recommend.
    detour_ratio = (
        safer["duration_min"] / fastest["duration_min"] if fastest["duration_min"] else 1.0
    )
    risk_drop = fastest["risk"]["mean"] - safer["risk"]["mean"]
    relative_drop = risk_drop / fastest["risk"]["mean"] if fastest["risk"]["mean"] else 0.0

    result["detour_ratio"] = round(detour_ratio, 3)
    result["risk_reduction"] = round(relative_drop, 3)
    extra_minutes = round(safer["duration_min"] - fastest["duration_min"])

    if over_budget:
        result["reason"] = (
            f"Every alternative that avoids those areas takes about "
            f"{round((detour_ratio - 1) * 100)}% longer — shown for comparison, "
            "but not recommended."
        )
    elif relative_drop < Config.ROUTE_MIN_RISK_REDUCTION:
        # Never nudge someone off a route for a rounding error. Detours have a
        # real cost, and suggesting one on noise is how a safety feature turns
        # into an app that just tells people to avoid certain neighbourhoods.
        result["reason"] = "Both routes carry similar risk — the fastest route is fine."
    else:
        result["recommendation"] = "safer"
        result["reason"] = (
            f"Avoids {len(cells)} elevated-risk area{'s' if len(cells) != 1 else ''} "
            f"for about {extra_minutes} more minute{'s' if extra_minutes != 1 else ''}."
        )
    return result
