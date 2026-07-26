"""Hot-spot map endpoints. Thin handlers — aggregation lives in services/."""

from datetime import date

from flask import Blueprint, jsonify, request

from services.claims_service import (
    aggregate_hotspots,
    filter_options,
    invalidate_cache,
    load_claims,
    source_status,
)
from services.geocode_service import load_geocache

hotspot_bp = Blueprint("hotspots", __name__, url_prefix="/api")


def _csv_param(name):
    """Read a repeatable or comma-separated query param into a list of values.

    Accepts both ?peril=Theft&peril=Hijack and ?peril=Theft,Hijack.
    """
    values = []
    for raw in request.args.getlist(name):
        values.extend(part.strip() for part in raw.split(",") if part.strip())
    return values


def _date_param(name):
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD), got {raw!r}")


@hotspot_bp.get("/hotspots")
def get_hotspots():
    """Per-suburb claim hot-spots for the heatmap.

    Query params (all optional, all AND-ed together):
      peril      - e.g. Theft, Hijack, Burglary (repeatable or comma-separated)
      item_type  - Contents | Vehicle
      date_from  - ISO date, inclusive
      date_to    - ISO date, inclusive
    """
    try:
        date_from = _date_param("date_from")
        date_to = _date_param("date_to")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if date_from and date_to and date_from > date_to:
        return jsonify({"error": "date_from must not be after date_to"}), 400

    perils = _csv_param("peril")
    item_types = _csv_param("item_type")

    result = aggregate_hotspots(
        geocache=load_geocache(),
        perils=perils or None,
        item_types=item_types or None,
        date_from=date_from,
        date_to=date_to,
    )
    result["filters_applied"] = {
        "peril": perils,
        "item_type": item_types,
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
    }
    return jsonify(result)


@hotspot_bp.get("/filters")
def get_filters():
    """Filter values available in the dataset, so the UI never hardcodes peril names."""
    return jsonify(filter_options())


@hotspot_bp.post("/claims/refresh")
def refresh_claims():
    """Force an immediate re-read of the claims collection, bypassing the TTL.

    Claims written through this app should call `invalidate_cache()` directly;
    this endpoint covers claims added out-of-band (a direct Cosmos write, a bulk
    import) that you don't want to wait out the TTL for.
    """
    invalidate_cache()
    load_claims(force_refresh=True)
    return jsonify({"refreshed": True, "claims": source_status()})

