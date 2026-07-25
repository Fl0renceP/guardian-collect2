"""Travel-risk surface and safer-route comparison endpoints.

Thin handlers — the surface lives in `services/risk_service.py` and the routing
in `services/routing_service.py`.
"""

import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from services import risk_service
from services.routing_service import RoutingError, UnsupportedMode, compare_routes

logger = logging.getLogger(__name__)

route_bp = Blueprint("routes", __name__, url_prefix="/api")


def _int_arg(name, default=None, low=None, high=None):
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}")
    if low is not None and value < low:
        raise ValueError(f"{name} must be at least {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be at most {high}")
    return value


@route_bp.get("/risk")
def get_risk():
    """Risk cells for a moment in time.

    Query params:
      hour      0-23, defaults to now
      weekday   0-6 (Mon=0), defaults to today
      min_score only return cells at or above this (0-1)
      limit     cap the number of cells returned
    """
    now = datetime.now()
    try:
        hour = _int_arg("hour", default=now.hour, low=0, high=23)
        weekday = _int_arg("weekday", default=now.weekday(), low=0, high=6)
        limit = _int_arg("limit", default=None, low=1, high=5000)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        min_score = float(request.args.get("min_score", 0.15))
    except ValueError:
        return jsonify({"error": "min_score must be a number between 0 and 1"}), 400

    cells = risk_service.scored_cells(
        hour=hour, weekday=weekday, min_score=min_score, limit=limit
    )
    profile = risk_service.timing_profile()

    return jsonify(
        {
            "hour": hour,
            "weekday": weekday,
            "cells": cells,
            "count": len(cells),
            "hour_multiplier": round(profile["hour_multiplier"].get(hour, 1.0), 3),
            "dow_multiplier": round(profile["dow_multiplier"].get(weekday, 1.0), 3),
            "profile": profile,
        }
    )


@route_bp.get("/risk/profile")
def get_risk_profile():
    """The pooled hour/day multipliers behind the surface, for charting."""
    return jsonify(risk_service.timing_profile())


@route_bp.post("/routes/compare")
def compare():
    """Fastest route vs a route avoiding elevated-risk areas.

    Body: {origin: [lat, lng], destination: [lat, lng],
           mode: auto|pedestrian|bicycle, hour?, weekday?}
    """
    payload = request.get_json(silent=True) or {}

    def point(name):
        raw = payload.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"{name} must be [lat, lng]")
        try:
            lat, lng = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be two numbers")
        if not -90 <= lat <= 90 or not -180 <= lng <= 180:
            raise ValueError(f"{name} is not a valid coordinate")
        return (lat, lng)

    try:
        origin = point("origin")
        destination = point("destination")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    now = datetime.now()
    hour = payload.get("hour")
    weekday = payload.get("weekday")
    hour = now.hour if hour is None else int(hour)
    weekday = now.weekday() if weekday is None else int(weekday)
    if not 0 <= hour <= 23 or not 0 <= weekday <= 6:
        return jsonify({"error": "hour must be 0-23 and weekday 0-6"}), 400

    mode = (payload.get("mode") or "auto").strip().lower()

    try:
        result = compare_routes(origin, destination, costing=mode, hour=hour, weekday=weekday)
    except UnsupportedMode as exc:
        # The caller's mistake, not the routing service's — 400, not 502.
        return jsonify({"error": "validation_failed", "fields": {"mode": str(exc)}}), 400
    except RoutingError as exc:
        return jsonify({"error": "routing_failed", "message": str(exc)}), 502

    return jsonify(result)
