"""Crime Prevention Unit endpoints: alerts and patrol planning."""

import logging
from datetime import datetime

from flask import Blueprint, jsonify, request

from services import alerts_service, users_service
from services.members_service import get_unit, list_units
from services.patrol_service import plan_patrols

logger = logging.getLogger(__name__)

cpu_bp = Blueprint("cpu", __name__, url_prefix="/api")


def _int_arg(name, default=None, low=None, high=None, source=None):
    raw = (source or request.args).get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a whole number, got {raw!r}")
    if low is not None and value < low:
        raise ValueError(f"{name} must be at least {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be at most {high}")
    return value


@cpu_bp.get("/units")
def get_units():
    """Crime Prevention Unit directory. Demo identities — see members_service."""
    return jsonify({"units": list_units()})


@cpu_bp.get("/alerts")
def get_alerts():
    """Alerts for an audience.

    Query params:
      audience   member | cpu  (default cpu). Members never see suspect matches.
      unit_id    restrict to a unit's operating area
      since_days lookback window (default 30)
      limit
    """
    audience = (request.args.get("audience") or "cpu").strip().lower()
    if audience not in ("member", "cpu"):
        return jsonify({"error": "audience must be 'member' or 'cpu'"}), 400

    try:
        since_days = _int_arg(
            "since_days", default=alerts_service.DEFAULT_SINCE_DAYS, low=1, high=1825
        )
        limit = _int_arg("limit", default=60, low=1, high=500)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    near = radius_km = None
    unit = None
    located = False

    unit_id = (request.args.get("unit_id") or "").strip()
    if unit_id:
        unit = get_unit(unit_id)
        if not unit:
            return jsonify({"error": f"Unknown unit '{unit_id}'"}), 400
        near = (unit["base_lat"], unit["base_lng"])
        radius_km = unit["radius_km"]
        located = True

    # A member can scope alerts to their home, but only if they opted in.
    # member_home() is the single place that check lives — never read the
    # coordinates off the profile directly.
    member_id = (request.args.get("member_id") or "").strip()
    if member_id and not unit_id:
        home = users_service.member_home(member_id)
        if home:
            near = (home[0], home[1])
            radius_km = home[2]
            located = True

    alerts = alerts_service.list_alerts(
        audience=audience, since_days=since_days, near=near, radius_km=radius_km, limit=limit
    )
    return jsonify(
        {
            "alerts": alerts,
            "summary": alerts_service.alert_summary(alerts),
            "sources": alerts_service.source_status(),
            "unit": unit,
            "audience": audience,
            "since_days": since_days,
            # Lets the UI say "showing alerts near your home" vs "showing all
            # alerts — add a home location to narrow this down".
            "scoped_to_location": located,
            "radius_km": radius_km,
        }
    )


@cpu_bp.post("/patrol/plan")
def post_patrol_plan():
    """Build patrol loops for a unit.

    Body: {unit_id, vehicles?, hour?, weekday?, radius_km?}
    """
    payload = request.get_json(silent=True) or {}

    unit = get_unit((payload.get("unit_id") or "").strip())
    if not unit:
        return jsonify(
            {"error": "validation_failed", "fields": {"unit_id": "Unknown unit."}}
        ), 400

    now = datetime.now()
    try:
        vehicles = _int_arg("vehicles", default=unit["vehicles"], low=1, high=8, source=payload)
        hour = _int_arg("hour", default=now.hour, low=0, high=23, source=payload)
        weekday = _int_arg("weekday", default=now.weekday(), low=0, high=6, source=payload)
        radius_km = _int_arg(
            "radius_km", default=unit["radius_km"], low=1, high=200, source=payload
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    plan = plan_patrols(
        unit, vehicles=vehicles, hour=hour, weekday=weekday, radius_km=radius_km
    )
    return jsonify(plan)
