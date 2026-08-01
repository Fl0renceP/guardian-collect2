"""Patrol planning for Crime Prevention Units.

This is deliberately **not** the member routing problem. A member asks "what's
the safest way from A to B" — one origin, one destination, shortest path with a
penalty. A unit asks "given N vehicles and a risk surface that shifts by hour,
where should we be to cover the most risk per kilometre driven?" That's a
coverage/allocation problem, and solving it as a shortest path would answer the
wrong question.

The approach here:

  1. Take the highest-risk cells inside the unit's operating area for the hour
     they're planning for.
  2. Split them across the available vehicles by geography (k-means on the
     cell centres), so two vehicles don't shadow each other.
  3. Order each vehicle's stops into a loop from the base and back
     (nearest-neighbour, then 2-opt), and ask Valhalla for the real road path
     so the distance and time are drivable numbers rather than straight lines.

Each vehicle then gets **two** loops through the same stops, for comparison:

  * the plain fastest road loop — what any consumer navigation app would give
    you for that list of stops, and
  * the patrol loop, which is deliberately nudged onto streets inside
    high-risk cells between the stops (see `_via_points_for_legs`).

That is the exact mirror of the member feature: a member routes *around* risk
cells with Valhalla's `exclude_polygons`, a patrol is routed *through* them with
extra via points. Showing both, with the extra kilometres stated, keeps the
trade-off in the controller's hands rather than in the algorithm's.

**This is a heuristic, not an optimal solve.** Nearest-neighbour with 2-opt gets
close on a dozen stops and needs no extra dependency; a true VRP solver (VROOM
or OR-Tools, both open source) is the upgrade path when time windows, shift
lengths or vehicle capabilities enter the picture.

Two different "risk" numbers appear below and they are not interchangeable:
`risk_covered` is the summed score of the *stops assigned* to a vehicle (what
the coverage figures have always meant), while `risk_seen` is the summed score
of every distinct cell the vehicle's *path* actually passes through. Only the
second one can compare two routes through the same stops, because by definition
they share the first.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from config import Config
from services import risk_service
from services.routing_service import RoutingError, _call_valhalla, _sample, _shape_of

logger = logging.getLogger(__name__)

MAX_VEHICLES = 8
MAX_STOPS_PER_VEHICLE = 10


def _cells_in_area(unit, hour, weekday, radius_km, max_cells):
    """Highest-risk cells within the unit's operating radius."""
    base = (unit["base_lat"], unit["base_lng"])
    candidates = risk_service.scored_cells(hour=hour, weekday=weekday, min_score=0.25)

    inside = []
    for cell in candidates:
        distance = risk_service.haversine_km(base, (cell["lat"], cell["lng"]))
        if distance <= radius_km:
            inside.append({**cell, "distance_km": round(distance, 1)})

    inside.sort(key=lambda c: c["score"], reverse=True)
    return inside[:max_cells]


def _kmeans(points, k, iterations=25):
    """Tiny k-means over (lat, lng). Enough to split stops geographically.

    Seeded deterministically by spreading the initial centroids through the
    score-ordered list, so the same inputs always produce the same plan — a
    patrol plan that reshuffles on refresh is not something a controller can use.
    """
    if k >= len(points):
        return [[p] for p in points]

    step = max(1, len(points) // k)
    centroids = [(points[min(i * step, len(points) - 1)]["lat"],
                  points[min(i * step, len(points) - 1)]["lng"]) for i in range(k)]

    clusters = [[] for _ in range(k)]
    for _ in range(iterations):
        clusters = [[] for _ in range(k)]
        for point in points:
            distances = [
                risk_service.haversine_km((point["lat"], point["lng"]), c) for c in centroids
            ]
            clusters[distances.index(min(distances))].append(point)

        moved = False
        for index, cluster in enumerate(clusters):
            if not cluster:
                continue
            lat = sum(p["lat"] for p in cluster) / len(cluster)
            lng = sum(p["lng"] for p in cluster) / len(cluster)
            if (lat, lng) != centroids[index]:
                centroids[index] = (lat, lng)
                moved = True
        if not moved:
            break

    return [c for c in clusters if c]


def _order_stops(base, stops):
    """Nearest-neighbour tour from the base, improved with 2-opt."""
    remaining = list(stops)
    tour = []
    current = base

    while remaining:
        nearest = min(
            remaining,
            key=lambda s: risk_service.haversine_km(current, (s["lat"], s["lng"])),
        )
        tour.append(nearest)
        remaining.remove(nearest)
        current = (nearest["lat"], nearest["lng"])

    def leg_length(order):
        total = 0.0
        point = base
        for stop in order:
            total += risk_service.haversine_km(point, (stop["lat"], stop["lng"]))
            point = (stop["lat"], stop["lng"])
        return total + risk_service.haversine_km(point, base)  # return to base

    # 2-opt: repeatedly reverse a segment if it shortens the loop. On ten stops
    # this converges in milliseconds and removes the obvious crossings that
    # plain nearest-neighbour leaves behind.
    improved = True
    while improved:
        improved = False
        for i in range(len(tour) - 1):
            for j in range(i + 1, len(tour)):
                candidate = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                if leg_length(candidate) < leg_length(tour) - 1e-9:
                    tour = candidate
                    improved = True
    return tour


def _road_loop(locations):
    """Ask Valhalla for the drivable loop through a list of (lat, lng) points."""
    try:
        trip = _call_valhalla(locations, "auto")["trip"]
    except RoutingError as exc:
        logger.info("Patrol leg routing failed: %s", exc)
        return None

    summary = trip.get("summary", {})
    return {
        "distance_km": round(summary.get("length", 0.0), 2),
        "duration_min": round(summary.get("time", 0.0) / 60.0, 1),
        "coordinates": [[lat, lng] for lat, lng in _shape_of(trip)],
    }


def _via_points_for_legs(base, stops, pool, used_cells):
    """High-risk cells worth driving through on the way between stops.

    For each leg of the loop, take the best-value cell that sits on the way —
    the extra distance it adds must fit the allowance (see
    PATROL_VIA_MAX_DETOUR_KM), and among those, the winner is the one with the
    best score per kilometre of detour. Ranking on score alone would always pick
    the cell at the very edge of the allowance; a slightly quieter area half a
    kilometre off the route is the better use of a shift.

    `used_cells` must include every cell the fastest route already drives
    through, not just the stops. Choosing via points blind to that produces the
    worst possible result — real extra kilometres bought with no extra coverage,
    because the vehicle was passing through that cell anyway. It is the same
    reason `avoid_cells_on_route` scores the fastest path before deciding what
    to avoid: you cannot improve on a route you haven't looked at.
    """
    chosen = []
    taken = set(used_cells)
    legs = list(zip([base] + [(s["lat"], s["lng"]) for s in stops],
                    [(s["lat"], s["lng"]) for s in stops] + [base]))

    for leg_index, (start, end) in enumerate(legs):
        if len(chosen) >= Config.PATROL_MAX_VIA_POINTS:
            break
        direct = risk_service.haversine_km(start, end)
        allowance = max(
            Config.PATROL_VIA_MAX_DETOUR_KM,
            direct * (Config.PATROL_VIA_CORRIDOR_RATIO - 1),
        )

        best = best_value = None
        for cell in pool:
            if cell["cell"] in taken or cell["score"] < Config.PATROL_VIA_MIN_SCORE:
                continue
            point = (cell["lat"], cell["lng"])
            extra = (
                risk_service.haversine_km(start, point)
                + risk_service.haversine_km(point, end)
                - direct
            )
            if extra > allowance:
                continue
            value = cell["score"] / (1 + extra / Config.PATROL_VIA_MAX_DETOUR_KM)
            if best_value is None or value > best_value:
                best, best_value = cell, value

        if best is not None:
            taken.add(best["cell"])
            chosen.append({"leg_index": leg_index, "cell": best})

    return chosen


def _interleave(base, stops, vias):
    """Build the Valhalla location list, dropping each via point into its leg."""
    by_leg = {v["leg_index"]: v for v in vias}
    locations = [base]
    # Leg i runs from location i to stop i, so a via for leg i goes in first.
    for index, stop in enumerate(stops):
        via = by_leg.get(index)
        if via:
            locations.append((via["cell"]["lat"], via["cell"]["lng"]))
        locations.append((stop["lat"], stop["lng"]))
    final = by_leg.get(len(stops))
    if final:
        locations.append((final["cell"]["lat"], final["cell"]["lng"]))
    locations.append(base)
    return locations


def _cells_on_path(coordinates, hour, weekday):
    """Distinct risk cells a drawn path passes through, keyed to their score.

    Sampled rather than scored vertex by vertex for the same reason the member
    routes are: dense urban geometry carries far more vertices per kilometre
    than a highway, and counting them raw would flatter city routes.
    """
    seen = {}
    for lat, lng in _sample([tuple(p) for p in coordinates or []], step_km=0.4):
        cell = risk_service.cell_at(lat, lng)
        if cell not in seen:
            seen[cell] = risk_service.score_at(lat, lng, hour, weekday)
    return seen


def _path_risk(cells):
    """Headline figures for a path's coverage, from `_cells_on_path`."""
    return {"cells": len(cells), "risk_seen": round(sum(cells.values()), 3)}


def plan_patrols(unit, vehicles=None, hour=None, weekday=None, radius_km=None, max_cells=None):
    """Build one patrol loop per vehicle. Returns the plan and its coverage."""
    vehicles = max(1, min(int(vehicles or unit["vehicles"]), MAX_VEHICLES))
    radius_km = radius_km or unit.get("radius_km", 25)
    max_cells = max_cells or (vehicles * MAX_STOPS_PER_VEHICLE)

    base = (unit["base_lat"], unit["base_lng"])
    cells = _cells_in_area(unit, hour, weekday, radius_km, max_cells)

    if not cells:
        return {
            "unit": unit,
            "vehicles": vehicles,
            "hour": hour,
            "weekday": weekday,
            "radius_km": radius_km,
            "routes": [],
            "coverage": {"cells": 0, "risk_covered": 0.0, "risk_available": 0.0, "share": 0.0},
            "comparison": None,
            "reason": "No elevated-risk areas in this unit's operating area at this time.",
        }

    # Everything inside the operating area: the denominator for an honest
    # coverage percentage, and the pool via points are drawn from. A vehicle
    # must not be detoured outside the area it is responsible for.
    in_area = [
        c
        for c in risk_service.scored_cells(hour=hour, weekday=weekday, min_score=0.25)
        if risk_service.haversine_km(base, (c["lat"], c["lng"])) <= radius_km
    ]
    risk_available = sum(c["score"] for c in in_area)

    clusters = _kmeans(cells, vehicles)

    # Assigning stops is pure arithmetic, so it stays off the worker threads —
    # nothing then touches the risk surface's cache concurrently.
    plans = [
        {"vehicle": index, "stops": _order_stops(base, cluster[:MAX_STOPS_PER_VEHICLE])}
        for index, cluster in enumerate(clusters, start=1)
    ]

    def route_all(location_lists):
        workers = max(1, min(Config.PATROL_ROUTE_WORKERS, len(location_lists) or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_road_loop, location_lists))

    # Round one: the plain fastest loop through each vehicle's stops. This is
    # both the comparison route and the input to choosing via points.
    fastest_legs = route_all(
        [[base] + [(s["lat"], s["lng"]) for s in p["stops"]] + [base] for p in plans]
    )

    # Round two: nudge each loop through high-risk cells it wasn't already
    # covering. `taken` is fleet-wide, so two vehicles don't detour to the same
    # area and report the coverage twice.
    taken = {c["cell"] for c in cells}
    for plan, fastest in zip(plans, fastest_legs):
        plan["fastest"] = fastest
        plan["fastest_cells"] = _cells_on_path(
            fastest["coordinates"] if fastest else [], hour, weekday
        )
        plan["vias"] = (
            _via_points_for_legs(base, plan["stops"], in_area, taken | set(plan["fastest_cells"]))
            if fastest
            else []
        )
        taken.update(v["cell"]["cell"] for v in plan["vias"])

    detour_plans = [p for p in plans if p["vias"]]
    patrol_legs = route_all([_interleave(base, p["stops"], p["vias"]) for p in detour_plans])
    for plan, patrol in zip(detour_plans, patrol_legs):
        plan["patrol"] = patrol

    # Round three, only for the vehicles that overshot: halve the via list,
    # keeping the ones picked first (they scored best per kilometre), and route
    # once more. Same idea as `_try_avoiding` on the member side — take the
    # improvement that fits the budget rather than the biggest one available.
    def over_budget(plan):
        patrol, fastest = plan.get("patrol"), plan["fastest"]
        if not patrol or not fastest or not fastest["duration_min"]:
            return False
        return patrol["duration_min"] / fastest["duration_min"] > Config.PATROL_MAX_DETOUR_RATIO

    trimmed = [p for p in detour_plans if over_budget(p) and len(p["vias"]) > 1]
    for plan in trimmed:
        plan["vias"] = plan["vias"][: len(plan["vias"]) // 2]
    for plan, patrol in zip(
        trimmed, route_all([_interleave(base, p["stops"], p["vias"]) for p in trimmed])
    ):
        plan["patrol"] = patrol

    # Still over after trimming, or down to a single unaffordable detour: drive
    # the fastest loop. A plan a unit can't actually run is worse than no
    # suggestion, and the coverage it claimed would never have happened.
    for plan in detour_plans:
        if over_budget(plan):
            plan["patrol"] = None
            plan["vias"] = []

    routes = []
    for plan in plans:
        index = plan["vehicle"]
        stops = plan["stops"]
        fastest = plan["fastest"]
        # No via points, or the risk-seeking loop failed to route: the patrol
        # route simply *is* the fastest one. Said plainly rather than hidden, so
        # a controller doesn't read a missing detour as a covered area.
        patrol = plan.get("patrol") or fastest
        detoured = patrol is not None and patrol is not fastest
        patrol_cells = (
            _cells_on_path(patrol["coordinates"], hour, weekday)
            if detoured
            else plan["fastest_cells"]
        )

        covered = sum(s["score"] for s in stops)
        routes.append(
            {
                "vehicle": index,
                "stops": [
                    {
                        "cell": s["cell"],
                        "lat": s["lat"],
                        "lng": s["lng"],
                        "score": s["score"],
                        "claims": s["claims"],
                    }
                    for s in stops
                ],
                "via_points": [
                    {
                        "cell": v["cell"]["cell"],
                        "lat": v["cell"]["lat"],
                        "lng": v["cell"]["lng"],
                        "score": v["cell"]["score"],
                    }
                    for v in plan["vias"]
                ]
                if detoured
                else [],
                "risk_covered": round(covered, 3),
                # "Ours" — the route the vehicle is being asked to drive.
                "distance_km": patrol["distance_km"] if patrol else None,
                "duration_min": patrol["duration_min"] if patrol else None,
                "coordinates": patrol["coordinates"] if patrol else [],
                "path": _path_risk(patrol_cells) if patrol else None,
                # The plain fastest loop through the same stops, for comparison.
                "fastest": {
                    "distance_km": fastest["distance_km"],
                    "duration_min": fastest["duration_min"],
                    "coordinates": fastest["coordinates"],
                    "path": _path_risk(plan["fastest_cells"]),
                }
                if fastest
                else None,
                "detoured": detoured,
                "routable": patrol is not None,
            }
        )

    risk_covered = sum(r["risk_covered"] for r in routes)
    total_km = sum(r["distance_km"] or 0 for r in routes)

    # Fleet-wide trade-off: what the detours buy, and what they cost.
    fastest_km = sum(r["fastest"]["distance_km"] for r in routes if r["fastest"])
    fastest_min = sum(r["fastest"]["duration_min"] for r in routes if r["fastest"])
    patrol_min = sum(r["duration_min"] or 0 for r in routes)
    seen_patrol = sum(r["path"]["risk_seen"] for r in routes if r["path"])
    seen_fastest = sum(r["fastest"]["path"]["risk_seen"] for r in routes if r["fastest"])
    cells_patrol = sum(r["path"]["cells"] for r in routes if r["path"])
    cells_fastest = sum(r["fastest"]["path"]["cells"] for r in routes if r["fastest"])

    comparison = {
        "detoured_vehicles": sum(1 for r in routes if r["detoured"]),
        "patrol": {
            "distance_km": round(total_km, 2),
            "duration_min": round(patrol_min, 1),
            "risk_seen": round(seen_patrol, 3),
            "cells": cells_patrol,
        },
        "fastest": {
            "distance_km": round(fastest_km, 2),
            "duration_min": round(fastest_min, 1),
            "risk_seen": round(seen_fastest, 3),
            "cells": cells_fastest,
        },
        "extra_km": round(total_km - fastest_km, 2),
        "extra_min": round(patrol_min - fastest_min, 1),
        "extra_risk_seen": round(seen_patrol - seen_fastest, 3),
        "extra_cells": cells_patrol - cells_fastest,
        # How much more risk each extra kilometre of patrolling actually buys.
        # None rather than 0 when there is no detour, so the UI can say "no
        # detour was worth taking" instead of printing a meaningless ratio.
        "risk_per_extra_km": (
            round((seen_patrol - seen_fastest) / (total_km - fastest_km), 3)
            if total_km - fastest_km > 0.05
            else None
        ),
    }

    return {
        "unit": unit,
        "vehicles": vehicles,
        "hour": hour,
        "weekday": weekday,
        "radius_km": radius_km,
        "routes": routes,
        "coverage": {
            "cells": sum(len(r["stops"]) for r in routes),
            "risk_covered": round(risk_covered, 3),
            "risk_available": round(risk_available, 3),
            "share": round(risk_covered / risk_available, 4) if risk_available else 0.0,
            "total_km": round(total_km, 2),
            # The number a unit actually manages against: how much risk each
            # kilometre of fuel buys.
            "risk_per_km": round(risk_covered / total_km, 4) if total_km else 0.0,
        },
        "comparison": comparison,
        "reason": None,
    }
