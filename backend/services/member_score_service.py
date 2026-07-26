"""Guardian Safety Score — the member-facing reward score.

**This is not Victoria's `safety_score.py`.** That one rates how risky a single
*claim* is, for an assessor. This one rates how much a *member* has engaged with
the safety features, to reward them — the Discovery Miles model. Same phrase,
opposite subject, so they deliberately live in separate modules.

Four categories, weighted to total 100, matching the product mockup:

    App activity                  30
    Route optimisation            25
    Camera & facial recognition   25
    Claims response               20

Every point is traceable: each contribution records what earned it and how many
points it was worth, and the API returns that breakdown. Nothing is a magic
number the member can't account for — that's the whole premise of an incentive
score, and it's also what stops it feeling arbitrary when someone's score drops.

**Where the inputs come from.** Two categories are computed from real data the
app already stores, so the score genuinely moves when a member does something:

  * Camera & facial recognition — reads `camera_consent` on their claims.
  * Claims response — reads their approved claims and attached evidence.

The other two need behaviour the app doesn't record yet (there's no session log
and no "member took the safer route" event). Those read seeded counters on
`member_profile.activity`. Replacing them later means writing real counters to
that same dict — no change here.
"""

import logging

from services.claims_service import STATUS_APPROVED, list_claims
from services.users_service import get_user_raw

logger = logging.getLogger(__name__)

# Miles awarded per safety point. Stated in the UI so the number is never a
# mystery — 76 points earns 228 miles.
MILES_PER_POINT = 3

TIERS = [
    {"name": "Bronze", "min": 0},
    {"name": "Silver", "min": 40},
    {"name": "Gold", "min": 60},
    {"name": "Platinum", "min": 80},
]

CATEGORY_MAX = {
    "app_activity": 30,
    "route_optimisation": 25,
    "camera": 25,
    "claims_response": 20,
}

CATEGORY_LABEL = {
    "app_activity": "App activity",
    "route_optimisation": "Route optimisation",
    "camera": "Camera & facial recognition",
    "claims_response": "Claims response",
}

CATEGORY_HINT = {
    "app_activity": "Opening the app, checking hot-spots, and acting on alerts.",
    "route_optimisation": "Planning journeys and choosing the safer route when one is offered.",
    "camera": "Linking a door camera and allowing footage to help confirm your claims.",
    "claims_response": "Reporting incidents promptly, with evidence, that pass review.",
}


def _award(entries, label, earned, cap, detail):
    """Record one contribution, capped, and return the points actually given."""
    points = max(0, min(int(earned), cap))
    entries.append({"label": label, "points": points, "max": cap, "detail": detail})
    return points


def _tier_for(score):
    current = TIERS[0]
    for tier in TIERS:
        if score >= tier["min"]:
            current = tier
    nxt = next((t for t in TIERS if t["min"] > score), None)
    return current, nxt


def _member_claims(member_id):
    try:
        return list_claims(member_id=member_id, limit=200)
    except Exception:
        logger.exception("Could not read claims for %s — scoring without them.", member_id)
        return []


def _app_activity(activity):
    entries = []
    total = 0
    opens = int(activity.get("app_opens", 0))
    views = int(activity.get("hotspot_views", 0))
    acks = int(activity.get("alerts_acknowledged", 0))

    total += _award(entries, "Opening the app", opens // 2, 12, f"{opens} sessions")
    total += _award(entries, "Checking crime hot-spots", views // 3, 8, f"{views} map views")
    total += _award(entries, "Acting on alerts", acks, 10, f"{acks} alerts acknowledged")
    return total, entries


def _route_optimisation(activity):
    entries = []
    total = 0
    planned = int(activity.get("routes_planned", 0))
    safer = int(activity.get("safer_routes_taken", 0))

    total += _award(entries, "Planning a route before travelling", planned // 2, 7,
                    f"{planned} routes planned")
    total += _award(entries, "Taking the safer route when offered", safer * 3, 18,
                    f"{safer} safer routes taken")
    return total, entries


def _camera(activity, claims):
    entries = []
    total = 0
    consented = [c for c in claims if c.get("camera_consent")]

    linked = bool(activity.get("camera_linked"))
    total += _award(entries, "Door camera linked", 9 if linked else 0, 9,
                    "Linked" if linked else "Not linked yet")
    total += _award(entries, "Footage shared to confirm a claim", len(consented) * 8, 16,
                    f"{len(consented)} claim(s) with permission given")
    return total, entries


def _claims_response(claims):
    entries = []
    total = 0
    approved = [c for c in claims if c.get("status") == STATUS_APPROVED]
    with_media = [c for c in approved if c.get("media")]

    total += _award(entries, "Reports accepted after review", len(approved) * 6, 18,
                    f"{len(approved)} approved claim(s)")
    total += _award(entries, "Reports submitted with photos or video", len(with_media), 2,
                    f"{len(with_media)} with evidence attached")
    return total, entries


def calculate_member_score(member_id):
    """The member's Guardian Safety Score, with a full traceable breakdown."""
    user = get_user_raw(member_id)
    if not user or user.get("role") != "member":
        return None

    profile = user.get("member_profile") or {}
    activity = profile.get("activity") or {}
    claims = _member_claims(member_id)

    computed = {
        "app_activity": _app_activity(activity),
        "route_optimisation": _route_optimisation(activity),
        "camera": _camera(activity, claims),
        "claims_response": _claims_response(claims),
    }

    categories = []
    score = 0
    for key, (points, entries) in computed.items():
        capped = min(points, CATEGORY_MAX[key])
        score += capped
        categories.append(
            {
                "key": key,
                "label": CATEGORY_LABEL[key],
                "hint": CATEGORY_HINT[key],
                "points": capped,
                "max": CATEGORY_MAX[key],
                # Guarded: a category max is never 0, but a future edit could.
                "share": round(capped / CATEGORY_MAX[key], 4) if CATEGORY_MAX[key] else 0.0,
                "contributions": entries,
            }
        )

    score = max(0, min(score, 100))
    tier, next_tier = _tier_for(score)

    if next_tier:
        span = next_tier["min"] - tier["min"]
        progress = (score - tier["min"]) / span if span else 1.0
        to_next = next_tier["min"] - score
    else:
        progress = 1.0
        to_next = 0

    # What the member could still earn, biggest opportunity first — this is the
    # nudge, and it has to point at something they can actually act on.
    gaps = sorted(
        (c for c in categories if c["points"] < c["max"]),
        key=lambda c: c["max"] - c["points"],
        reverse=True,
    )

    return {
        "member_id": member_id,
        "member_name": user.get("full_name"),
        "score": score,
        "max_score": 100,
        "tier": tier["name"],
        "next_tier": next_tier["name"] if next_tier else None,
        "points_to_next_tier": to_next,
        "tier_progress": round(progress, 4),
        "miles": score * MILES_PER_POINT,
        "miles_per_point": MILES_PER_POINT,
        "categories": categories,
        "opportunities": [
            {
                "label": c["label"],
                "available": c["max"] - c["points"],
                "hint": c["hint"],
            }
            for c in gaps[:3]
        ],
        "tiers": TIERS,
    }
