"""Alert feed, composed from whatever sources currently exist.

**Audience routing is the load-bearing rule here** (PROJECT_CONTEXT §2):
Discovery members only ever see `offender` alerts, because an unverified suspect
match would scare people over a maybe. Crime Prevention Units see `offender` and
`suspect` both, since they're the ones who act on the ground and need the fuller
picture. `audience_for` implements that, and every alert goes through it —
including the ones from sources that don't exist yet, so the rule is already
correct the day Phase 1 starts emitting detections.

Sources, and their honest status today:

  * `incident`  — recent incidents from the claims dataset. REAL.
  * `submission`— claims members have just filed, awaiting review. REAL, live.
  * `detection` — face/plate matches from `/api/detect`. **NOT WIRED YET**:
                  Phase 1 doesn't exist, so `_detection_alerts` returns nothing.
                  It is not stubbed with fake matches — an alerts panel that
                  invents offender sightings is worse than an empty one.
  * `predicted` — Azure Functions risk forecasts. NOT WIRED YET, same reasoning.

To plug a source in, return a list of alert dicts from its `_*_alerts` function.
The shape is documented on `_alert`.
"""

import logging
from datetime import datetime, timedelta

from services.claims_service import STATUS_PENDING, list_claims, load_claims
from services.geocode_service import load_geocache
from services.risk_service import haversine_km

logger = logging.getLogger(__name__)

# Which detection labels each audience is allowed to see.
AUDIENCE_LABELS = {
    "member": {"offender"},
    "cpu": {"offender", "suspect"},
}

# Perils serious enough to raise as an alert rather than leave to the map.
ALERTING_PERILS = {"Hijack", "Attempted Hijack", "Armed Robbery", "Burglary"}

# Serious incidents are rare — roughly 19 a month nationally across the whole
# dataset — so a 30-day window around one city is reliably empty. 90 days is the
# smallest window that gives a unit something to act on; the UI states it.
DEFAULT_SINCE_DAYS = 90

SEVERITY_BY_PERIL = {
    "Hijack": "critical",
    "Attempted Hijack": "serious",
    "Armed Robbery": "critical",
    "Burglary": "serious",
}


def audience_for(label):
    """Who may see an alert about a person with this detection label."""
    audiences = [a for a, allowed in AUDIENCE_LABELS.items() if label in allowed]
    return audiences or ["cpu"]


def _alert(
    alert_id,
    kind,
    severity,
    title,
    detail,
    at,
    audience,
    suburb=None,
    lat=None,
    lng=None,
    source="claims",
    meta=None,
):
    """One alert. `audience` is a list of "member" / "cpu"."""
    return {
        "id": alert_id,
        "kind": kind,
        "severity": severity,
        "title": title,
        "detail": detail,
        "at": at.isoformat() if isinstance(at, datetime) else at,
        "audience": audience,
        "suburb": suburb,
        "lat": lat,
        "lng": lng,
        "source": source,
        "meta": meta or {},
    }


def _incident_alerts(since_days, geocache):
    """Serious incidents from the claims dataset, newest first.

    Anchored to the newest incident in the data rather than to today: the
    dataset ends before the current date, so a window measured from `now` would
    be empty and the panel would look broken.
    """
    claims = load_claims()
    dated = [c for c in claims if c["incident_at"]]
    if not dated:
        return []

    latest = max(c["incident_at"] for c in dated)
    cutoff = latest - timedelta(days=since_days)

    out = []
    for claim in dated:
        if claim["incident_at"] < cutoff or claim["peril"] not in ALERTING_PERILS:
            continue
        point = geocache.get(claim["suburb"])
        # An alert nobody can locate isn't actionable for a patrol unit, and
        # "incident in UNKNOWN" reads like a bug. Leave those to the map.
        if not point:
            continue
        out.append(
            _alert(
                alert_id=f"inc-{claim['incident']}",
                kind="incident",
                severity=SEVERITY_BY_PERIL.get(claim["peril"], "warning"),
                title=f"{claim['peril']} reported",
                detail=f"{claim['item_type']} incident in {claim['suburb'].title()}.",
                at=claim["incident_at"],
                # Both audiences: this is a confirmed incident, not a person match.
                audience=["member", "cpu"],
                suburb=claim["suburb"],
                lat=point["lat"] if point else None,
                lng=point["lng"] if point else None,
                source="claims",
                meta={"peril": claim["peril"], "item_type": claim["item_type"]},
            )
        )

    out.sort(key=lambda a: a["at"], reverse=True)
    return out


def _submission_alerts(geocache):
    """Claims members have just filed. Live, and useful to a unit immediately —
    a pending report is still a report of something that happened."""
    try:
        pending = list_claims(status=STATUS_PENDING, limit=50)
    except Exception:
        logger.exception("Could not read pending submissions for the alert feed.")
        return []

    out = []
    for claim in pending:
        suburb = (claim.get("SUBURB") or "").upper()
        point = geocache.get(suburb)
        out.append(
            _alert(
                alert_id=f"sub-{claim.get('Incident')}",
                kind="submission",
                severity=SEVERITY_BY_PERIL.get(claim.get("PERIL"), "warning"),
                title=f"{claim.get('PERIL')} reported by a member",
                detail=(claim.get("description") or "")[:180],
                at=claim.get("INCIDENT_DATE_TIME") or claim.get("submitted_at"),
                # Unverified member report — operationally useful to a unit, but
                # not something to push at other members before an assessor sees it.
                audience=["cpu"],
                suburb=suburb,
                lat=point["lat"] if point else None,
                lng=point["lng"] if point else None,
                source="member_submission",
                meta={"status": claim.get("status"), "member": claim.get("member_name")},
            )
        )
    return out


def _detection_alerts():
    """Face/plate matches from Phase 1.

    Returns nothing because `/api/detect` and the `detections` store don't exist
    yet. When they do, emit one alert per match with
    `audience=audience_for(match_label)` and the rest of this module needs no
    changes — members will automatically see offender matches only.
    """
    return []


def _predicted_alerts():
    """Forecast risk from Azure Functions. Not wired yet — see module docstring."""
    return []


def list_alerts(audience="cpu", since_days=DEFAULT_SINCE_DAYS, near=None, radius_km=None, limit=60):
    """Alerts this audience may see, newest first.

    `near` is an optional (lat, lng) — with `radius_km` it restricts the feed to
    a unit's operating area, which is what makes the panel useful rather than a
    national firehose.
    """
    geocache = load_geocache()

    alerts = []
    alerts.extend(_incident_alerts(since_days, geocache))
    alerts.extend(_submission_alerts(geocache))
    alerts.extend(_detection_alerts())
    alerts.extend(_predicted_alerts())

    alerts = [a for a in alerts if audience in a["audience"]]

    if near and radius_km:
        kept = []
        for alert in alerts:
            if alert["lat"] is None:
                continue
            distance = haversine_km(near, (alert["lat"], alert["lng"]))
            if distance <= radius_km:
                alert["distance_km"] = round(distance, 1)
                kept.append(alert)
        alerts = kept

    alerts.sort(key=lambda a: (a["at"] or ""), reverse=True)
    return alerts[:limit]


def alert_summary(alerts):
    """Counts by severity and kind, for the header tiles."""
    severity = {}
    kind = {}
    for alert in alerts:
        severity[alert["severity"]] = severity.get(alert["severity"], 0) + 1
        kind[alert["kind"]] = kind.get(alert["kind"], 0) + 1
    return {"total": len(alerts), "by_severity": severity, "by_kind": kind}


def source_status():
    """Which alert sources are actually live — surfaced in the UI so nobody
    mistakes an empty detections feed for a broken one."""
    return [
        {"source": "claims", "label": "Recent incidents", "live": True},
        {"source": "member_submission", "label": "Member reports", "live": True},
        {
            "source": "detections",
            "label": "Face / plate matches",
            "live": False,
            "note": "Awaiting Phase 1 (/api/detect)",
        },
        {
            "source": "predicted",
            "label": "Predicted risk",
            "live": False,
            "note": "Awaiting Azure Functions",
        },
    ]
