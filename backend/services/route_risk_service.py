"""Predictive route-risk scoring from stored data only.

The output is an advisory risk forecast, not real-time incident confirmation.
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import atan2, cos, radians, sin, sqrt

import psycopg2

from config import Config
from services.claims_service import load_claims
from services.geocode_service import load_geocache

PERIL_WEIGHTS = {
    "THEFT": 1.00,
    "HIJACK": 1.15,
    "ARMED ROBBERY": 1.20,
    "BURGLARY": 1.05,
    "ATTEMPTED THEFT": 0.90,
    "REMOTE JAMMING": 1.10,
}


def _haversine_km(lat1, lng1, lat2, lng2):
    radius = 6371.0
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)
    a_value = (
        sin(delta_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(delta_lng / 2) ** 2
    )
    c_value = 2 * atan2(sqrt(a_value), sqrt(1 - a_value))
    return radius * c_value


def _normalize(value, max_value):
    if max_value <= 0:
        return 0.0
    return min(1.0, max(0.0, value / max_value))


def _classify(score):
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.30:
        return "low"
    return "none"


def _recency_weight(hours_ago):
    if hours_ago <= 2:
        return 1.0
    if hours_ago <= 24:
        return 0.65
    if hours_ago <= 72:
        return 0.35
    return 0.10


def _parse_iso_datetime(raw_value):
    if not raw_value:
        return None
    parsed = datetime.fromisoformat(str(raw_value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_iso_date(raw_value, field_name):
    if raw_value is None:
        return None
    try:
        return datetime.fromisoformat(str(raw_value)).date()
    except Exception as exc:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD)") from exc


def _build_hotspots(perils=None, item_types=None, date_from=None, date_to=None):
    geocache = load_geocache()
    grouped = defaultdict(
        lambda: {
            "count": 0,
            "total_amount": 0.0,
            "perils": Counter(),
            "hours": Counter(),
        }
    )

    for claim in load_claims():
        if perils and claim["peril"] not in perils:
            continue
        if item_types and claim["item_type"] not in item_types:
            continue
        if date_from or date_to:
            incident_at = claim["incident_at"]
            if incident_at is None:
                continue
            incident_date = incident_at.date()
            if date_from and incident_date < date_from:
                continue
            if date_to and incident_date > date_to:
                continue

        suburb = claim["suburb"]
        if suburb not in geocache:
            continue

        bucket = grouped[suburb]
        bucket["count"] += 1
        bucket["total_amount"] += max(float(claim["amount"]), 0.0)
        bucket["perils"][claim["peril"]] += 1
        if claim["incident_at"] is not None:
            bucket["hours"][claim["incident_at"].hour] += 1

    if not grouped:
        return [], 0, 0.0

    max_count = max(v["count"] for v in grouped.values())
    max_amount = max(v["total_amount"] for v in grouped.values())

    hotspots = []
    for suburb, bucket in grouped.items():
        top_peril = bucket["perils"].most_common(1)[0][0] if bucket["perils"] else "Unspecified"
        active_hours = [h for h, _ in bucket["hours"].most_common(4)]
        point = geocache[suburb]
        hotspots.append(
            {
                "suburb": suburb,
                "lat": point["lat"],
                "lng": point["lng"],
                "count": bucket["count"],
                "total_amount": round(bucket["total_amount"], 2),
                "top_peril": top_peril,
                "active_hours": active_hours,
            }
        )

    return hotspots, max_count, max_amount


def _score_hotspot(point, departure_hour, hotspots, max_count, max_amount):
    best_score = 0.0
    best_context = None
    radius_km = Config.ROUTE_RISK_HOTSPOT_RADIUS_KM

    for hotspot in hotspots:
        distance_km = _haversine_km(point["lat"], point["lng"], hotspot["lat"], hotspot["lng"])
        if distance_km > radius_km:
            continue

        distance_weight = max(0.0, 1.0 - (distance_km / radius_km))
        count_weight = _normalize(hotspot["count"], max_count)
        amount_weight = _normalize(hotspot["total_amount"], max_amount)
        severity = (0.60 * count_weight) + (0.40 * amount_weight)
        peril_weight = PERIL_WEIGHTS.get(str(hotspot["top_peril"]).upper(), 1.0)
        time_weight = 1.0 if departure_hour in hotspot["active_hours"] else 0.60

        score = min(1.0, severity * distance_weight * peril_weight * time_weight)
        if score > best_score:
            best_score = score
            best_context = {
                **hotspot,
                "distance_km": round(distance_km, 3),
            }

    return best_score, best_context


def _load_recent_sightings(now_utc):
    """Load stored detections if the detections table is present."""
    if not Config.DATABASE_URL:
        return []

    try:
        conn = psycopg2.connect(Config.DATABASE_URL)
    except Exception:
        return []

    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = 'detections'
                )
                """
            )
            if not cur.fetchone()[0]:
                return []

            cur.execute(
                """
                SELECT match_label, location_lat, location_lng, detected_at, COALESCE(match_score, 0.70)
                FROM detections
                WHERE match_label IN ('offender', 'suspect')
                  AND location_lat IS NOT NULL
                  AND location_lng IS NOT NULL
                  AND detected_at >= (NOW() - (%s || ' hours')::interval)
                ORDER BY detected_at DESC
                """,
                (str(Config.ROUTE_RISK_RECENT_SIGHTING_HOURS),),
            )

            sightings = []
            for label, lat, lng, detected_at, score in cur.fetchall():
                detected_utc = detected_at.astimezone(timezone.utc)
                hours_ago = max(0.0, (now_utc - detected_utc).total_seconds() / 3600.0)
                sightings.append(
                    {
                        "label": str(label),
                        "lat": float(lat),
                        "lng": float(lng),
                        "seen_at": detected_utc.isoformat(),
                        "hours_ago": round(hours_ago, 2),
                        "confidence": float(score),
                    }
                )
            return sightings
        finally:
            cur.close()
    except Exception:
        return []
    finally:
        conn.close()


def _score_sighting(point, sightings):
    best_score = 0.0
    best_context = None
    radius_km = Config.ROUTE_RISK_SIGHTING_RADIUS_KM

    for sighting in sightings:
        distance_km = _haversine_km(point["lat"], point["lng"], sighting["lat"], sighting["lng"])
        if distance_km > radius_km:
            continue

        distance_weight = max(0.0, 1.0 - (distance_km / radius_km))
        recency = _recency_weight(sighting["hours_ago"])
        confidence = min(1.0, max(0.0, sighting["confidence"]))
        score = confidence * recency * distance_weight

        if score > best_score:
            best_score = score
            best_context = {
                **sighting,
                "distance_km": round(distance_km, 3),
            }

    return best_score, best_context


def score_route(payload):
    route_points = payload.get("route_points")
    if not isinstance(route_points, list) or not route_points:
        raise ValueError("route_points must be a non-empty array")

    departure = _parse_iso_datetime(payload.get("departure_time_utc")) or datetime.now(
        timezone.utc
    )
    departure_hour = departure.hour

    perils = payload.get("perils") or None
    item_types = payload.get("item_types") or None
    if perils is not None and not isinstance(perils, list):
        raise ValueError("perils must be an array")
    if item_types is not None and not isinstance(item_types, list):
        raise ValueError("item_types must be an array")

    date_from = _parse_iso_date(payload.get("date_from"), "date_from")
    date_to = _parse_iso_date(payload.get("date_to"), "date_to")
    if date_from and date_to and date_from > date_to:
        raise ValueError("date_from must not be after date_to")

    hotspots, max_count, max_amount = _build_hotspots(
        perils=perils,
        item_types=item_types,
        date_from=date_from,
        date_to=date_to,
    )
    now_utc = datetime.now(timezone.utc)
    sightings = _load_recent_sightings(now_utc)

    alerts = []
    max_route_score = 0.0
    cooldown = max(0, int(Config.ROUTE_RISK_COOLDOWN_POINTS))
    next_allowed_idx = 0

    for idx, point in enumerate(route_points):
        try:
            lat = float(point["lat"])
            lng = float(point["lng"])
        except Exception as exc:
            raise ValueError(
                f"route point at index {idx} must include numeric lat and lng"
            ) from exc

        route_point = {"lat": lat, "lng": lng}
        hotspot_score, hotspot_context = _score_hotspot(
            route_point, departure_hour, hotspots, max_count, max_amount
        )
        sighting_score, sighting_context = _score_sighting(route_point, sightings)

        total_score = min(1.0, (0.65 * hotspot_score) + (0.35 * sighting_score))
        max_route_score = max(max_route_score, total_score)
        level = _classify(total_score)

        if level == "none":
            continue
        if idx < next_allowed_idx:
            continue
        next_allowed_idx = idx + cooldown + 1

        message = "Historically high-risk area ahead."
        if sighting_context:
            message = "Suspect person or vehicle was reported near this route segment recently."

        alerts.append(
            {
                "route_index": idx,
                "lat": lat,
                "lng": lng,
                "alert_level": level,
                "score": round(total_score, 3),
                "message": message,
                "hotspot_context": hotspot_context,
                "recent_sighting_context": sighting_context,
            }
        )

    return {
        "mode": "predictive_stored_data_only",
        "summary_alert_level": _classify(max_route_score),
        "max_route_score": round(max_route_score, 3),
        "alerts": alerts,
        "metadata": {
            "hotspots_considered": len(hotspots),
            "stored_sightings_considered": len(sightings),
            "cooldown_points": cooldown,
            "filters_applied": {
                "perils": perils or [],
                "item_types": item_types or [],
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
                "departure_time_utc": departure.isoformat(),
            },
        },
        "disclaimer": (
            "Predictive advisories are based on historical hot-spots and stored sightings, "
            "not real-time incident confirmation."
        ),
    }
