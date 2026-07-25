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

**This is a heuristic, not an optimal solve.** Nearest-neighbour with 2-opt gets
close on a dozen stops and needs no extra dependency; a true VRP solver (VROOM
or OR-Tools, both open source) is the upgrade path when time windows, shift
lengths or vehicle capabilities enter the picture. The reported "risk covered"
is honest either way — it's the summed score of the cells actually visited.
"""

import logging

from config import Config
from services import risk_service
from services.routing_service import RoutingError, _call_valhalla, _shape_of

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


def _road_loop(base, stops):
    """Ask Valhalla for the drivable loop through the stops and back to base."""
    locations = [base] + [(s["lat"], s["lng"]) for s in stops] + [base]
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
            "reason": "No elevated-risk areas in this unit's operating area at this time.",
        }

    # What's on the table, for an honest coverage percentage.
    available = risk_service.scored_cells(hour=hour, weekday=weekday, min_score=0.25)
    risk_available = sum(
        c["score"]
        for c in available
        if risk_service.haversine_km(base, (c["lat"], c["lng"])) <= radius_km
    )

    clusters = _kmeans(cells, vehicles)

    routes = []
    for index, cluster in enumerate(clusters, start=1):
        stops = _order_stops(base, cluster[:MAX_STOPS_PER_VEHICLE])
        road = _road_loop(base, stops)
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
                "risk_covered": round(covered, 3),
                "distance_km": road["distance_km"] if road else None,
                "duration_min": road["duration_min"] if road else None,
                "coordinates": road["coordinates"] if road else [],
                "routable": road is not None,
            }
        )

    risk_covered = sum(r["risk_covered"] for r in routes)
    total_km = sum(r["distance_km"] or 0 for r in routes)

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
        "reason": None,
    }
