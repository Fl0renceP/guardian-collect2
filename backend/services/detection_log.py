"""Record every face check in the detections log.

Kept out of services.recognition on purpose: recognition answers "who is this",
logging is a separate concern, and mixing them means a logging failure can poison
the transaction that produced the answer.

The guiding rule here is that logging must never break a scan. An audit trail is
valuable, but not more valuable than telling the operator there is an offender at
the door. Every failure is swallowed and reported as a warning.
"""

import logging

logger = logging.getLogger(__name__)

INSERT_SQL = """
INSERT INTO detections (
    camera_id, location_lat, location_lng,
    matched_person_id, matched_name, match_label,
    match_score, match_threshold, margin_to_next,
    alert_sent, faces_detected, capture_quality, quality_passed
) VALUES (
    %(camera_id)s, %(lat)s, %(lng)s,
    %(person_id)s, %(person_name)s, %(label)s,
    %(score)s, %(threshold)s, %(margin)s,
    %(alert)s, %(faces)s, %(quality)s, %(quality_passed)s
)
RETURNING id;
"""


def label_for(result):
    """Collapse a scan result into one of the five logged outcomes."""
    if not result.get("success"):
        return "no_face"
    if result.get("is_known_user"):
        # offender / suspect / verified, straight off the matched person.
        return result.get("status") or "no_match"
    return "no_match"


def record(conn, result, threshold, camera_id="demo_upload", lat=None, lng=None):
    """Write one detection row. Returns its id, or None if logging failed.

    conn is the same connection the scan used. The insert is committed on its own
    so a later failure elsewhere cannot silently discard the audit record.
    """
    person = result.get("person") or {}
    quality = result.get("capture_quality") or {}

    params = {
        "camera_id": camera_id or "demo_upload",
        "lat": lat,
        "lng": lng,
        "person_id": person.get("id"),
        "person_name": person.get("full_name"),
        "label": label_for(result),
        "score": result.get("match_distance"),
        "threshold": threshold,
        "margin": result.get("margin_to_next_person"),
        "alert": bool(result.get("alert")),
        "faces": result.get("faces_detected"),
        "quality": quality.get("quality_score"),
        "quality_passed": quality.get("passes"),
    }

    try:
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)
            detection_id = cur.fetchone()[0]
        conn.commit()
        return str(detection_id)
    except Exception as exc:
        # Never let the audit trail take down the answer.
        logger.warning("Could not record detection: %s", exc)
        try:
            conn.rollback()
        except Exception:
            pass
        return None


def recent(conn, limit=50, alerts_only=False):
    """Most recent detections, newest first."""
    sql = """
        SELECT id, camera_id, matched_name, match_label, match_score,
               match_threshold, margin_to_next, alert_sent, faces_detected,
               capture_quality, quality_passed, detected_at
        FROM detections
        {where}
        ORDER BY detected_at DESC
        LIMIT %(limit)s;
    """.format(where="WHERE alert_sent" if alerts_only else "")

    with conn.cursor() as cur:
        cur.execute(sql, {"limit": limit})
        rows = cur.fetchall()

    return [
        {
            "id": str(r[0]),
            "camera_id": r[1],
            "matched_name": r[2],
            "match_label": r[3],
            "match_score": round(float(r[4]), 4) if r[4] is not None else None,
            "match_threshold": round(float(r[5]), 4) if r[5] is not None else None,
            "margin_to_next": round(float(r[6]), 4) if r[6] is not None else None,
            "alert_sent": r[7],
            "faces_detected": r[8],
            "capture_quality": round(float(r[9]), 3) if r[9] is not None else None,
            "quality_passed": r[10],
            "detected_at": r[11].isoformat() if r[11] else None,
        }
        for r in rows
    ]


def summary(conn):
    """Counts by outcome — the shape of the continuous dataset so far."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT match_label, count(*), count(*) FILTER (WHERE alert_sent)
            FROM detections GROUP BY match_label ORDER BY count(*) DESC;
            """
        )
        by_label = [
            {"match_label": r[0], "count": r[1], "alerts": r[2]} for r in cur.fetchall()
        ]
        cur.execute("SELECT count(*), count(*) FILTER (WHERE alert_sent) FROM detections")
        total, alerts = cur.fetchone()

    return {"total": total, "alerts": alerts, "by_label": by_label}
