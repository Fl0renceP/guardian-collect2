"""Recording a human's decision on a behavioural flag.

This is the only place in the whole feature where a person's judgement is
written down, and it is the point the entire module has been building toward:
everything upstream produces a prompt to look, and nothing acts until someone
here says so.

FOUR RULES, ALL ENFORCED BELOW
------------------------------

1. **A confirmation is the REVIEWER'S identification, not the system's.**
   No score, however high, reaches this code on its own. It is reached by a
   person clicking, and the row records which person.

2. **Confirming does NOT rewrite the identity registry.**
   The decision is recorded against the review. It does not change
   `persons.status`, so a behavioural flag can never quietly promote someone to
   `offender` in the curated registry that facial recognition matches against.
   That registry is deliberately small and deliberate (PROJECT_CONTEXT §5), and
   a one-click path into it from a movement heuristic is precisely the
   auto-labelling pattern this project already rejected once. Promotion stays a
   separate, deliberate act.

3. **A deny needs a reason.**
   A denied flag is a measured false positive. The reason is the most valuable
   data this system produces — it is how thresholds get tuned and how anyone
   can honestly claim the fusion reduces false alarms.

4. **Every decision is reversible, and reversal is itself recorded.**
   A confirmation made in error must be undoable, and the undo must not erase
   the original. Both are facts; an audit trail that only shows the tidy final
   state is not an audit trail.

ALERTS: routed through `alerts_service.audience_for`, never around it. Members
see `offender` only; Crime Prevention Units see `offender` and `suspect`. A
confirmation with no identity attached alerts nobody — there is no one to
alert about — but the review still stands in the queue.
"""

import logging

logger = logging.getLogger(__name__)

# How long a decision can be undone. Long enough to catch a misclick or a
# second opinion the same shift; short enough that the record settles.
REVERSIBLE_HOURS = 24

VALID_LABELS = ("offender", "suspect", "verified")


class DecisionError(ValueError):
    def __init__(self, message, fields=None, status=400):
        super().__init__(message)
        self.fields = fields or {}
        self.status = status


def _current(cursor, review_id):
    cursor.execute(
        """
        SELECT r.review_id, r.status, r.matched_person_id, r.matched_label,
               e.camera_id, e.location_zone_id, e.track_id, e.composite_risk_score,
               l.person_id, l.label
        FROM behavioural_reviews r
        JOIN behavioural_events e ON e.event_id = r.event_id
        LEFT JOIN behavioural_face_links l
               ON l.camera_id = e.camera_id AND l.track_id = e.track_id
              AND l.scan_captured_at IS NOT NULL
              AND ABS(EXTRACT(EPOCH FROM (l.scan_captured_at - e.occurred_at))) <= 30 * 60
        WHERE r.review_id = %s
        """,
        (review_id,),
    )
    return cursor.fetchone()


def _send_alert(person_id, label, camera_id, full_name=None):
    """Raise an alert for a confirmed identification.

    Goes through alerts_service so the audience rule applies unchanged: members
    only ever see `offender`. Confirming a `suspect` reaches Crime Prevention
    Units and nobody else.
    """
    if not person_id or label not in ("offender", "suspect"):
        return []

    try:
        from services import alerts_service

        event = alerts_service.record_detection(
            match_label=label,
            entity_type="person",
            title=f"{label.title()} confirmed by a reviewer",
            detail=(
                f"{full_name or 'A person'} was confirmed as {label} by a reviewer "
                f"after a behavioural flag."
            ),
            meta={"person_id": str(person_id), "source": "behavioural_review"},
        )
        return event.get("audience", []) if event else []
    except Exception:
        # A failed alert must not lose the decision — the decision is the record
        # that matters and it is already committed by the time this runs.
        logger.exception("Alert dispatch failed for a confirmed behavioural review.")
        return []


def decide(db_conn, review_id, *, decision, reviewer_id, label=None, note=None, reason=None):
    """Record a confirm or deny against a review."""
    reviewer_id = (reviewer_id or "").strip()
    if not reviewer_id:
        raise DecisionError(
            "reviewer_id is required — a decision with no one behind it is not an audit trail.",
            {"reviewer_id": "is required"},
        )

    if decision not in ("confirm", "deny"):
        raise DecisionError("decision must be 'confirm' or 'deny'.", {"decision": "unknown"})

    if decision == "deny" and not (reason or "").strip():
        raise DecisionError(
            "A reason is required to deny a flag. A denied flag is a measured false "
            "positive, and the reason is how these thresholds get tuned.",
            {"reason": "is required"},
        )

    if label is not None and label not in VALID_LABELS:
        raise DecisionError(
            f"label must be one of {', '.join(VALID_LABELS)}.", {"label": "unknown label"}
        )

    with db_conn.cursor() as cursor:
        row = _current(cursor, review_id)
        if row is None:
            raise DecisionError(f"No review {review_id}.", status=404)

        status = row[1]
        if status != "pending":
            raise DecisionError(
                f"This review was already {status}. Reopen it before deciding again.",
                {"status": f"already {status}"},
                status=409,
            )

        camera_id = row[4]
        # Identity comes from the human's own earlier confirmation if present,
        # otherwise from the automatic link. It may legitimately be absent: a
        # flag with no facial match is confirmed as "the behaviour is real",
        # which asserts nothing about who the person is.
        person_id = row[2] or row[8]
        resolved_label = label or row[3] or row[9]

        if decision == "confirm":
            new_status, new_person, new_label = "confirmed", person_id, resolved_label
        else:
            # Denying does not assert an identity, so none is stored.
            new_status, new_person, new_label = "denied", None, None

        cursor.execute(
            """
            UPDATE behavioural_reviews
               SET status = %s, matched_person_id = %s, matched_label = %s,
                   decided_by = %s, decided_at = CURRENT_TIMESTAMP,
                   decision_note = %s, denial_reason = %s
             WHERE review_id = %s
            """,
            (new_status, new_person, new_label, reviewer_id, note, reason, review_id),
        )

        cursor.execute(
            """
            INSERT INTO behavioural_review_decisions
                (review_id, decision, reviewer_id, person_id, label, reason, note,
                 identity_written, alerts_sent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING decided_at
            """,
            (review_id, decision, reviewer_id, new_person, new_label, reason, note,
             # Rule 2: the identity registry is never rewritten from here.
             False, None),
        )
        decided_at = cursor.fetchone()[0]

        full_name = None
        if new_person:
            cursor.execute("SELECT full_name FROM persons WHERE id = %s", (new_person,))
            found = cursor.fetchone()
            full_name = found[0] if found else None

    db_conn.commit()

    audiences = []
    if decision == "confirm":
        audiences = _send_alert(new_person, new_label, camera_id, full_name)
        if audiences:
            with db_conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE behavioural_review_decisions SET alerts_sent = %s
                     WHERE id = (SELECT MAX(id) FROM behavioural_review_decisions
                                  WHERE review_id = %s)
                    """,
                    (",".join(audiences), review_id),
                )
            db_conn.commit()

    logger.info(
        "review %s %sed by %s (person=%s label=%s alerts=%s)",
        review_id, decision, reviewer_id, new_person, new_label, audiences or "none",
    )

    return {
        "review_id": review_id,
        "status": new_status,
        "decided_by": reviewer_id,
        "decided_at": decided_at.isoformat() if decided_at else None,
        "reversible_hours": REVERSIBLE_HOURS,
        # Always False, and returned explicitly so nobody has to guess whether
        # confirming changed the identity registry. It does not — see rule 2.
        "identity_written": False,
        "alerts_sent": audiences,
        "matched_person_id": str(new_person) if new_person else None,
        "matched_label": new_label,
    }


def reopen(db_conn, review_id, *, reviewer_id, reason=None):
    """Undo a decision, within the reversal window.

    The original decision is NOT erased — this appends a `reopen` row saying it
    was undone, and clears the current state so the review can be decided again.
    """
    reviewer_id = (reviewer_id or "").strip()
    if not reviewer_id:
        raise DecisionError("reviewer_id is required.", {"reviewer_id": "is required"})

    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT status, decided_at,
                   EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - decided_at)) / 3600.0
            FROM behavioural_reviews WHERE review_id = %s
            """,
            (review_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise DecisionError(f"No review {review_id}.", status=404)

        status, decided_at, hours_since = row
        if status == "pending":
            raise DecisionError(
                "This review has not been decided, so there is nothing to undo.",
                status=409,
            )
        if hours_since is not None and float(hours_since) > REVERSIBLE_HOURS:
            raise DecisionError(
                f"This decision was taken {float(hours_since):.0f} hours ago and can no "
                f"longer be undone here (limit {REVERSIBLE_HOURS}h). The decision history "
                f"remains on the record.",
                status=409,
            )

        cursor.execute(
            """
            UPDATE behavioural_reviews
               SET status = 'pending', matched_person_id = NULL, matched_label = NULL,
                   decided_by = NULL, decided_at = NULL,
                   decision_note = NULL, denial_reason = NULL
             WHERE review_id = %s
            """,
            (review_id,),
        )
        cursor.execute(
            """
            INSERT INTO behavioural_review_decisions
                (review_id, decision, reviewer_id, reason)
            VALUES (%s, 'reopen', %s, %s)
            """,
            (review_id, reviewer_id, reason),
        )

    db_conn.commit()
    logger.info("review %s reopened by %s (was %s)", review_id, reviewer_id, status)

    return {
        "review_id": review_id,
        "status": "pending",
        "reopened_by": reviewer_id,
        "previous_status": status,
    }


def history(db_conn, review_id):
    """Every decision ever taken on this review, oldest first."""
    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT d.decision, d.reviewer_id, d.decided_at, d.label, d.reason, d.note,
                   d.identity_written, d.alerts_sent, p.full_name
            FROM behavioural_review_decisions d
            LEFT JOIN persons p ON p.id = d.person_id
            WHERE d.review_id = %s
            ORDER BY d.decided_at ASC, d.id ASC
            """,
            (review_id,),
        )
        rows = cursor.fetchall()

    return [
        {
            "decision": row[0],
            "reviewer_id": row[1],
            "decided_at": row[2].isoformat() if row[2] else None,
            "label": row[3],
            "reason": row[4],
            "note": row[5],
            "identity_written": row[6],
            "alerts_sent": (row[7].split(",") if row[7] else []),
            "person_name": row[8],
        }
        for row in rows
    ]
