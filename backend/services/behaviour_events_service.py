"""Ingest and read behavioural events.

Events arrive from `behavioural_analysis/api_output.py::push_to_flask_api()`.
See BEHAVIOUR_REVIEW_API.md section 3 for the contract.

TWO RULES ARE ENFORCED HERE RATHER THAN TRUSTED:

1. **Every triggered heuristic must carry a non-empty explanation.** An event
   missing one is rejected outright. The entire justification for flagging
   people on movement is that a human can read why, disagree, and be right —
   a bare score cannot be reviewed, only obeyed.

2. **No identity may be attached to an event.** The payload is rejected if it
   carries a name, person id, embedding or image. The behavioural module works
   on anonymous track ids and must not become a second, quieter route by which
   biometric identity ends up in the event log. Identity enters this schema in
   exactly one place — `behavioural_reviews.matched_person_id` — and only
   because a named human put it there.

Ingest is idempotent on `event_id`, so the pusher can retry after a network
failure without duplicating an event or a review.
"""

import json
import logging

logger = logging.getLogger(__name__)


class EventValidationError(ValueError):
    """Raised when a payload is not a usable behavioural event.

    `fields` maps field name -> message, matching the shape the frontend's
    ApiError already understands for per-field validation.
    """

    def __init__(self, message, fields=None):
        super().__init__(message)
        self.fields = fields or {}


REQUIRED_FIELDS = (
    "event_id",
    "track_id",
    "timestamp",
    "behavioural_risk_score",
    "composite_risk_score",
    "requires_human_review",
    "triggered_heuristics",
)

# Keys that would mean identity has leaked into an event. Rejected, not stripped:
# silently dropping them would hide a caller that is trying to send identity,
# and the caller needs to know it is wrong.
FORBIDDEN_IDENTITY_FIELDS = (
    "person",
    "person_id",
    "full_name",
    "name",
    "face_id",
    "matched_face_id",
    "member_id",
    "embedding",
    "image_url",
    "identity",
)


def _score(value, field, fields):
    try:
        number = float(value)
    except (TypeError, ValueError):
        fields[field] = f"{field} must be a number, got {value!r}"
        return None
    if not 0.0 <= number <= 1.0:
        fields[field] = f"{field} must be between 0 and 1, got {number}"
        return None
    return number


def validate_event(payload):
    """Validate an incoming event. Returns a normalised dict, or raises."""
    if not isinstance(payload, dict):
        raise EventValidationError("Expected a JSON object describing one behavioural event.")

    fields = {}

    leaked = [key for key in FORBIDDEN_IDENTITY_FIELDS if key in payload]
    if leaked:
        raise EventValidationError(
            "A behavioural event must not carry identity data. Remove "
            f"{', '.join(sorted(leaked))}. This module works on anonymous track ids; "
            "identity belongs to the facial recognition module and is linked only by a "
            "human on review.",
            {key: "not permitted on a behavioural event" for key in leaked},
        )

    for field in REQUIRED_FIELDS:
        if payload.get(field) is None:
            fields[field] = f"{field} is required"

    behavioural = _score(payload.get("behavioural_risk_score"), "behavioural_risk_score", fields)
    composite = _score(payload.get("composite_risk_score"), "composite_risk_score", fields)

    face_confidence = payload.get("face_match_confidence")
    if face_confidence is not None:
        face_confidence = _score(face_confidence, "face_match_confidence", fields)

    triggered = payload.get("triggered_heuristics")
    if triggered is not None:
        if not isinstance(triggered, list) or not triggered:
            fields["triggered_heuristics"] = "must be a non-empty list of triggered heuristics"
        else:
            for index, heuristic in enumerate(triggered):
                if not isinstance(heuristic, dict):
                    fields[f"triggered_heuristics[{index}]"] = "must be an object"
                    continue
                if not (heuristic.get("type") or "").strip():
                    fields[f"triggered_heuristics[{index}].type"] = "is required"
                # Rule 1 — the reason this check exists is in the module docstring.
                if not (heuristic.get("explanation") or "").strip():
                    fields[f"triggered_heuristics[{index}].explanation"] = (
                        "is required — an event without a human-readable explanation "
                        "cannot be reviewed"
                    )
                if heuristic.get("confidence") is not None:
                    _score(
                        heuristic.get("confidence"),
                        f"triggered_heuristics[{index}].confidence",
                        fields,
                    )

    if fields:
        raise EventValidationError("The behavioural event is not valid.", fields)

    return {
        "event_id": str(payload["event_id"])[:200],
        "track_id": str(payload["track_id"])[:100],
        "camera_id": (payload.get("camera_id") or payload.get("location_zone_id") or None),
        "location_zone_id": payload.get("location_zone_id"),
        "location_lat": payload.get("location_lat"),
        "location_lng": payload.get("location_lng"),
        "occurred_at": payload["timestamp"],
        "behavioural_risk_score": behavioural,
        "face_match_confidence": face_confidence,
        "composite_risk_score": composite,
        "requires_human_review": bool(payload["requires_human_review"]),
        "triggered_heuristics": triggered,
        "reasoning": payload.get("reasoning") or [],
        "source": payload.get("source"),
    }


def record_event(event, db_conn):
    """Store one validated event, opening a review if it needs a human.

    Idempotent on event_id: a retried push returns the original review instead
    of creating a second one.
    """
    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO behavioural_events (
                event_id, track_id, camera_id, location_zone_id,
                location_lat, location_lng, occurred_at,
                behavioural_risk_score, face_match_confidence, composite_risk_score,
                requires_human_review, triggered_heuristics, reasoning, source
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (event_id) DO NOTHING
            RETURNING event_id
            """,
            (
                event["event_id"],
                event["track_id"],
                event["camera_id"],
                event["location_zone_id"],
                event["location_lat"],
                event["location_lng"],
                event["occurred_at"],
                event["behavioural_risk_score"],
                event["face_match_confidence"],
                event["composite_risk_score"],
                event["requires_human_review"],
                json.dumps(event["triggered_heuristics"]),
                json.dumps(event["reasoning"]),
                event["source"],
            ),
        )
        inserted = cursor.fetchone() is not None

        review_id = None
        escalate, gate_reason, gate_context = escalation_gate(cursor, event)

        if escalate:
            # Reaching a human takes BOTH: behaviour over the threshold, and an
            # already-flagged high-risk identity on that body. Everything else
            # is stored as the denominator — it is what measures how much the
            # filter suppresses, which is the only evidence the filter works.
            cursor.execute(
                """
                INSERT INTO behavioural_reviews (event_id)
                VALUES (%s)
                ON CONFLICT (event_id) DO NOTHING
                RETURNING review_id
                """,
                (event["event_id"],),
            )
            row = cursor.fetchone()
            if row:
                review_id = row[0]
            else:
                cursor.execute(
                    "SELECT review_id FROM behavioural_reviews WHERE event_id = %s",
                    (event["event_id"],),
                )
                existing = cursor.fetchone()
                review_id = existing[0] if existing else None

    db_conn.commit()

    logger.info(
        "behavioural event %s stored (new=%s) track=%s composite=%.2f review=%s gate=%s",
        event["event_id"], inserted, event["track_id"],
        event["composite_risk_score"], review_id or "none", gate_reason,
    )

    return {
        "event_id": event["event_id"],
        "review_id": review_id,
        "queued_for_review": bool(review_id),
        "duplicate": not inserted,
        # Why this event did or did not reach a human. A suppressed event that
        # cannot say why it was suppressed is indistinguishable from a bug.
        "escalation": {
            "escalated": bool(review_id),
            "reason": gate_reason,
            "behaviour_would_flag": bool(event["requires_human_review"]),
            "identity": gate_context,
        },
    }


VALID_REVIEW_STATUSES = ("pending", "confirmed", "denied")


def _clip_url(blob_name):
    """Sign a clip for reading, now. Never raises: a card without footage is
    still a usable card, and the explanations are the actual record."""
    if not blob_name:
        return None
    from services.behaviour_clip_service import safe_read_url

    return safe_read_url(blob_name)

# Effects spelled out in the payload so the card states what a click DOES,
# rather than leaving a reviewer to infer it from a button labelled "Confirm".
DECISION_EFFECTS = {
    "confirm_effect": (
        "Records your identification of this person and alerts Crime Prevention "
        "Units. Members are not alerted by a behavioural flag."
    ),
    "deny_effect": (
        "Records this as a false flag. No alert is sent, and the event becomes "
        "part of how these thresholds get measured."
    ),
}


def _identity_block(row):
    """The identity half of a card, from a row carrying both possible sources.

    Two things can attach a name to a behavioural track, and they do NOT rank
    equally:

      1. a human confirming it   -> behavioural_reviews.matched_person_id
      2. the automatic geometric join -> behavioural_face_links

    A human's decision always wins. The automatic link is a hypothesis produced
    by a face box falling inside a body box, and the card labels it as such —
    `source: "automatic"` with the evidence attached — so a reviewer weighs it
    rather than reads it as established fact.

    `attached: false` is the normal case and is not an error. It means facial
    recognition found no match, or could not be tied to this body. The card
    says so plainly rather than implying an unknown person is a matched one.
    """
    (human_person_id, human_label, human_name, human_status,
     link_person_id, link_label, link_confidence, link_name, link_status,
     link_delta_ms, link_candidates) = row

    if human_person_id:
        return {
            "attached": True,
            "source": "human",
            "label": human_label or human_status,
            "confidence": None,
            "person_id": str(human_person_id),
            "full_name": human_name,
            "first_seen_label": "Confirmed by a reviewer",
        }

    if link_person_id:
        return {
            "attached": True,
            "source": "automatic",
            "label": link_label or link_status,
            "confidence": link_confidence,
            "person_id": str(link_person_id),
            "full_name": link_name,
            "first_seen_label": "Face matched and correlated to this body by position",
            "link_evidence": {
                "time_delta_ms": link_delta_ms,
                "bodies_considered": link_candidates,
            },
        }

    return {
        "attached": False,
        "source": None,
        "label": None,
        "confidence": None,
        "person_id": None,
        "full_name": None,
        "first_seen_label": "No facial match available",
    }


# The identity half, joined from both sources. Shared by the list and the card
# so the two can never disagree about who a review is about.
_IDENTITY_SELECT = """
    r.matched_person_id, r.matched_label, hp.full_name, hp.status,
    l.person_id, l.label, l.confidence, lp.full_name, lp.status,
    l.time_delta_ms, l.candidates
"""

# A track id is only unique WITHIN ONE RUN of the behavioural module. Restart
# it and person-1 is a different human being. So a link is only applied to an
# event that happened around the same time as the scan that produced it —
# without that bound, tomorrow's person-1 would silently inherit today's
# identity, which is exactly the wrong-name-on-wrong-body failure this join is
# built to avoid.
#
# The proper fix is a per-run session id carried on both events and links; this
# time bound is the safe interim, and it is deliberately tight.
LINK_VALIDITY_MINUTES = 30

_IDENTITY_JOIN = f"""
    LEFT JOIN persons hp ON hp.id = r.matched_person_id
    LEFT JOIN behavioural_face_links l
           ON l.camera_id = e.camera_id
          AND l.track_id = e.track_id
          AND l.scan_captured_at IS NOT NULL
          AND ABS(EXTRACT(EPOCH FROM (l.scan_captured_at - e.occurred_at)))
              <= {LINK_VALIDITY_MINUTES} * 60
    LEFT JOIN persons lp ON lp.id = l.person_id
"""

# --- the conditional filter -------------------------------------------------
#
# Behaviour is NOT a second detector running in parallel with face recognition.
# Run that way it produces a flag for every person who stands still too long,
# which on a busy camera is most of them, and a queue of those is a queue nobody
# reads. It works as a CONDITIONAL FILTER: it narrows an already-flagged
# high-risk context, and it escalates nothing on its own.
#
# So an event opens a review only where a face match of a high-risk label is
# already attached to that body. Unusual movement by someone unmatched, or by a
# known resident, is recorded and not escalated.
#
# WHAT THIS GIVES UP, STATED PLAINLY: the false-negative catch. The original
# design escalated strong behaviour with no face match precisely because that is
# the case facial recognition cannot cover — a covered face, or someone not in
# the registry. Under this rule that person reaches nobody. The gain is that a
# face match alone no longer escalates either: ordinary movement suppresses it.
# That trade is the point of the filter framing, and it is a product decision
# rather than a technical one, so it lives here where it can be found and argued
# with rather than buried in a threshold.
HIGH_RISK_LABELS = ("offender", "suspect")


def escalation_gate(cursor, event):
    """Should this event reach a human? Returns (escalate, reason, context).

    `reason` is written into the ingest response so a suppressed event says why
    it was suppressed. A filter that silently drops things cannot be tuned.
    """
    if not event["requires_human_review"]:
        return False, "behaviour_below_threshold", None

    cursor.execute(
        f"""
        SELECT l.person_id, l.label, l.confidence, p.full_name
        FROM behavioural_face_links l
        LEFT JOIN persons p ON p.id = l.person_id
        WHERE l.camera_id = %s
          AND l.track_id = %s
          AND l.scan_captured_at IS NOT NULL
          AND ABS(EXTRACT(EPOCH FROM (l.scan_captured_at - %s::timestamptz)))
              <= {LINK_VALIDITY_MINUTES} * 60
        """,
        (event["camera_id"], event["track_id"], event["occurred_at"]),
    )
    link = cursor.fetchone()

    if link is None:
        return False, "no_identity_attached", None

    person_id, label, confidence, full_name = link
    context = {
        "person_id": str(person_id) if person_id else None,
        "full_name": full_name,
        "label": label,
        "confidence": confidence,
    }

    if label not in HIGH_RISK_LABELS:
        # A known resident behaving oddly at their own house is the textbook
        # false positive this filter exists to swallow.
        return False, f"identity_not_high_risk ({label})", context

    return True, f"high_risk_identity_confirmed ({label})", context


def list_reviews(db_conn, *, status="pending", camera_id=None, limit=50):
    """The review queue: one row per event that reached a human."""
    if status not in VALID_REVIEW_STATUSES:
        raise EventValidationError(
            f"status must be one of {', '.join(VALID_REVIEW_STATUSES)}",
            {"status": f"unknown status {status!r}"},
        )

    clauses = ["r.status = %s"]
    params = [status]
    if camera_id:
        clauses.append("e.camera_id = %s")
        params.append(camera_id)
    params.append(max(1, min(int(limit), 200)))

    with db_conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT r.review_id, r.status, r.opened_at,
                   e.event_id, e.camera_id, e.location_zone_id, e.occurred_at,
                   e.behavioural_risk_score, e.composite_risk_score,
                   e.triggered_heuristics,
                   {_IDENTITY_SELECT}
            FROM behavioural_reviews r
            JOIN behavioural_events e ON e.event_id = r.event_id
            {_IDENTITY_JOIN}
            WHERE {' AND '.join(clauses)}
            ORDER BY e.composite_risk_score DESC, r.opened_at DESC
            LIMIT %s
            """,
            params,
        )
        rows = cursor.fetchall()

        cursor.execute(
            "SELECT status, COUNT(*) FROM behavioural_reviews GROUP BY status"
        )
        counts = {name: count for name, count in cursor.fetchall()}

    reviews = []
    for row in rows:
        triggered = row[9] or []
        top = triggered[0] if triggered else {}
        identity = _identity_block(row[10:21])
        reviews.append({
            "review_id": row[0],
            "status": row[1],
            "opened_at": row[2].isoformat() if row[2] else None,
            "event_id": row[3],
            "camera_id": row[4],
            "location_zone_id": row[5],
            # No camera -> suburb mapping exists yet, so this is honestly null
            # rather than a guess. The hot-spot map joins on suburb, so wiring
            # it is a prerequisite for putting behavioural events on that map.
            "suburb": None,
            "occurred_at": row[6].isoformat() if row[6] else None,
            "composite_risk_score": row[8],
            "face": {
                "label": identity["label"],
                "confidence": identity["confidence"],
                "attached": identity["attached"],
                "source": identity["source"],
            },
            "top_heuristic": top.get("type"),
            "headline": (top.get("explanation") or "").split(". ")[0][:140] or None,
            "trigger_count": len(triggered),
            "still_url": None,   # media is step 5
        })

    return {
        "reviews": reviews,
        "counts": {name: counts.get(name, 0) for name in VALID_REVIEW_STATUSES},
    }


def _reference_photo(cursor, person_id):
    """The registry photo for a matched person, signed for reading now.

    This is the picture a reviewer compares against the footage, so it comes
    from the enrolment record rather than from anything the camera captured —
    the whole question being asked is whether the person on camera is the person
    in the registry, and answering it with two frames of the same camera would
    be circular.

    Signed per read and short-lived, like every other face image here. Never
    stored on the review.
    """
    if not person_id:
        return None

    cursor.execute(
        """
        SELECT image_url FROM person_faces
        WHERE person_id = %s AND image_url IS NOT NULL
        ORDER BY quality_score DESC NULLS LAST, created_at ASC
        LIMIT 1
        """,
        (person_id,),
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return None

    try:
        from services.recognition import _readable_face_url

        return _readable_face_url(row[0])
    except Exception:
        # An unsigned URL against a private container simply will not load, and
        # the card handles a missing photo. Better than failing the whole read.
        logger.warning("Could not sign reference photo for person %s", person_id)
        return None


def get_review(db_conn, review_id):
    """One full review card. Returns None when the id is unknown."""
    with db_conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT r.review_id, r.status, r.opened_at,
                   r.decided_by, r.decided_at, r.decision_note, r.denial_reason, r.clip_url,
                   e.event_id, e.track_id, e.camera_id, e.location_zone_id, e.occurred_at,
                   e.behavioural_risk_score, e.composite_risk_score,
                   e.triggered_heuristics, e.reasoning,
                   {_IDENTITY_SELECT}
            FROM behavioural_reviews r
            JOIN behavioural_events e ON e.event_id = r.event_id
            {_IDENTITY_JOIN}
            WHERE r.review_id = %s
            """,
            (review_id,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        identity = _identity_block(row[17:28])
        identity["reference_image_url"] = _reference_photo(cursor, identity.get("person_id"))

    return {
        "review_id": row[0],
        "status": row[1],
        "opened_at": row[2].isoformat() if row[2] else None,
        "camera_id": row[10],
        "location_zone_id": row[11],
        "suburb": None,
        "occurred_at": row[12].isoformat() if row[12] else None,
        "identity": identity,
        "behaviour": {
            "track_id": row[9],
            "behavioural_risk_score": row[13],
            "composite_risk_score": row[14],
            # A fresh short-lived SAS, signed per read. What is STORED is the
            # blob name — persisting a signed URL would make the container's
            # privacy decorative, since the link outlives the page.
            "clip_url": _clip_url(row[7]),
            "live_stream_url": None,
            "triggered_heuristics": row[15] or [],
            "reasoning": row[16] or [],
        },
        "decision": {
            "options": ["confirm", "deny"],
            "decided_by": row[3],
            "decided_at": row[4].isoformat() if row[4] else None,
            "decision_note": row[5],
            "denial_reason": row[6],
            **DECISION_EFFECTS,
        },
    }


def list_events(db_conn, *, camera_id=None, review_only=False, limit=50):
    """Recent events, newest first. Read side of the ingest, used by the queue."""
    clauses = []
    params = []
    if camera_id:
        clauses.append("camera_id = %s")
        params.append(camera_id)
    if review_only:
        clauses.append("requires_human_review = TRUE")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(max(1, min(int(limit), 200)))

    with db_conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT e.event_id, e.track_id, e.camera_id, e.location_zone_id, e.occurred_at,
                   e.behavioural_risk_score, e.face_match_confidence, e.composite_risk_score,
                   e.requires_human_review, e.triggered_heuristics, e.reasoning,
                   r.review_id, r.status
            FROM behavioural_events e
            LEFT JOIN behavioural_reviews r ON r.event_id = e.event_id
            {where}
            ORDER BY e.occurred_at DESC
            LIMIT %s
            """,
            params,
        )
        rows = cursor.fetchall()

    return [
        {
            "event_id": row[0],
            "track_id": row[1],
            "camera_id": row[2],
            "location_zone_id": row[3],
            "timestamp": row[4].isoformat() if row[4] else None,
            "behavioural_risk_score": row[5],
            "face_match_confidence": row[6],
            "composite_risk_score": row[7],
            "requires_human_review": row[8],
            "triggered_heuristics": row[9],
            "reasoning": row[10],
            "review_id": row[11],
            "review_status": row[12],
        }
        for row in rows
    ]
