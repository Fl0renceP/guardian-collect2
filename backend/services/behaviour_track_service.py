"""The join: attaching a face match to a tracked body.

The two signals describe the same person but know different things. The face
module knows WHO and WHERE THE FACE WAS. The behavioural module knows WHERE THE
BODIES WERE and what they did, under anonymous track ids. Neither can attach a
name to a behaviour on its own.

This module holds the only thing that connects them — geometry:

    a face match belongs to the person track whose body box contained the
    centre of the face box, at the same moment

GETTING THIS WRONG IS THE WORST FAILURE IN THE SYSTEM
----------------------------------------------------
A bad join puts one person's identity on another person's behaviour. That is
worse than either signal failing alone: facial recognition failing gives you an
unknown person, but a wrong join gives you a confident, specific, wrong
accusation — and every explanation downstream will read as though it were about
the named person. So this module is deliberately reluctant:

  * outside a tight time window, no link
  * if the face centre falls in no body box, no link
  * if two bodies are similar candidates, NO LINK — ambiguity is resolved by
    refusing, not by guessing the nearer one
  * every link stores the evidence behind it (time delta, candidate count) so
    a reviewer can second-guess it

"No facial match" is a perfectly good answer and the card says it plainly. A
wrong name is not.
"""

import logging

logger = logging.getLogger(__name__)

# How far apart a scan and a body observation may be and still describe the same
# moment. The live scan runs about every 1.5s and the behavioural pipeline at a
# few frames a second, so a second either side covers normal jitter. Wider than
# this and a person can have walked out of their own box.
MATCH_WINDOW_SECONDS = 1.0

# Body-position rows are deleted once they are older than this. They exist only
# to serve the correlation above and are far more intrusive than the sparse
# events they support — a continuous record of where every body stood.
PRESENCE_RETENTION_MINUTES = 30

# Two candidate bodies count as ambiguous unless the smaller is clearly smaller.
# 0.75 means the best candidate must be at most 75% of the runner-up's area to
# be treated as "obviously the nearer person". Otherwise: no link.
AMBIGUITY_AREA_RATIO = 0.75


class TrackSnapshotError(ValueError):
    def __init__(self, message, fields=None):
        super().__init__(message)
        self.fields = fields or {}


def _num(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def validate_snapshots(payload):
    """Validate a batch of body-position snapshots from the behavioural module."""
    if not isinstance(payload, dict):
        raise TrackSnapshotError("Expected a JSON object.")

    camera_id = (payload.get("camera_id") or "").strip()
    if not camera_id:
        raise TrackSnapshotError("camera_id is required.", {"camera_id": "is required"})

    snapshots = payload.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise TrackSnapshotError(
            "snapshots must be a non-empty list.", {"snapshots": "must be a non-empty list"}
        )

    rows = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            continue
        observed_at = snapshot.get("timestamp")
        if not observed_at:
            continue
        for track in snapshot.get("tracks") or []:
            track_id = (track.get("track_id") or "").strip()
            box = track.get("bbox") or {}
            x, y = _num(box.get("x")), _num(box.get("y"))
            w, h = _num(box.get("w")), _num(box.get("h"))
            if not track_id or None in (x, y, w, h) or w <= 0 or h <= 0:
                continue
            rows.append((camera_id[:64], track_id[:100], observed_at, x, y, w, h))

    if not rows:
        raise TrackSnapshotError(
            "No usable track positions in the batch.", {"snapshots": "contained no valid boxes"}
        )
    return rows


def record_snapshots(rows, db_conn):
    """Store body positions, and expire the old ones."""
    with db_conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO behavioural_tracks
                (camera_id, track_id, observed_at, bbox_x, bbox_y, bbox_w, bbox_h)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            """,
            rows,
        )
        # Retention is enforced here rather than by a scheduled job so it cannot
        # be forgotten: the only code that writes these rows also expires them.
        cursor.execute(
            """
            DELETE FROM behavioural_tracks
            WHERE observed_at < now() - interval '%s minutes'
            """,
            (PRESENCE_RETENTION_MINUTES,),
        )
        expired = cursor.rowcount
    db_conn.commit()
    return {"stored": len(rows), "expired": expired}


def find_track_for_face(db_conn, *, camera_id, captured_at, face_centre):
    """Which tracked body was the face inside? Returns a dict, never raises.

    The result always explains itself — `linked` plus a `reason` — because the
    interesting case is the refusal, and a silent None would leave nobody able
    to tell "no camera data" from "two people, too close to call".
    """
    if not camera_id or not captured_at or not face_centre:
        return {"linked": False, "reason": "no_face_box", "track_id": None}

    centre_x = _num(face_centre.get("x"))
    centre_y = _num(face_centre.get("y"))
    if centre_x is None or centre_y is None:
        return {"linked": False, "reason": "no_face_box", "track_id": None}

    with db_conn.cursor() as cursor:
        # The nearest instant on this camera to when the scan was taken.
        cursor.execute(
            """
            SELECT observed_at,
                   ABS(EXTRACT(EPOCH FROM (observed_at - %s::timestamptz))) AS delta
            FROM behavioural_tracks
            WHERE camera_id = %s
              AND observed_at BETWEEN %s::timestamptz - interval '%s seconds'
                                  AND %s::timestamptz + interval '%s seconds'
            ORDER BY delta ASC
            LIMIT 1
            """,
            (captured_at, camera_id, captured_at, MATCH_WINDOW_SECONDS,
             captured_at, MATCH_WINDOW_SECONDS),
        )
        nearest = cursor.fetchone()
        if nearest is None:
            return {
                "linked": False,
                "reason": "no_body_tracking_at_that_moment",
                "track_id": None,
            }

        observed_at, delta_seconds = nearest[0], float(nearest[1])

        # Every body seen at that instant whose box contains the face centre.
        cursor.execute(
            """
            SELECT track_id, bbox_w * bbox_h AS area
            FROM behavioural_tracks
            WHERE camera_id = %s
              AND observed_at = %s
              AND %s BETWEEN bbox_x AND bbox_x + bbox_w
              AND %s BETWEEN bbox_y AND bbox_y + bbox_h
            ORDER BY area ASC
            """,
            (camera_id, observed_at, centre_x, centre_y),
        )
        candidates = cursor.fetchall()

    if not candidates:
        # A face nobody's body contains. Usually the behavioural module never
        # confirmed a track for this person, or the face is a reflection or a
        # photograph. Either way it cannot be attached.
        return {"linked": False, "reason": "face_outside_every_body_box", "track_id": None}

    best_track, best_area = candidates[0][0], float(candidates[0][1])

    if len(candidates) > 1:
        runner_up_area = float(candidates[1][1])
        # Smallest box = nearest person, but only if it is clearly smaller.
        # Two people at similar distance, one behind the other, produce boxes of
        # similar size and there is no honest way to pick. Refuse.
        if runner_up_area <= 0 or (best_area / runner_up_area) > AMBIGUITY_AREA_RATIO:
            logger.info(
                "face/track join refused as ambiguous on %s: %d overlapping bodies",
                camera_id, len(candidates),
            )
            return {
                "linked": False,
                "reason": "ambiguous_several_bodies_overlap",
                "track_id": None,
                "candidates": len(candidates),
            }

    return {
        "linked": True,
        "reason": "face_centre_inside_body_box",
        "track_id": best_track,
        "time_delta_ms": int(delta_seconds * 1000),
        "candidates": len(candidates),
    }


def link_face_to_track(
    db_conn,
    *,
    camera_id,
    track_id,
    person_id,
    label=None,
    confidence=None,
    match_distance=None,
    scan_captured_at=None,
    time_delta_ms=None,
    candidates=None,
):
    """Record that a track is believed to be a person.

    Keeps the most confident link for a track rather than the most recent: a
    later, weaker match should not overwrite a strong one, since the strong one
    is more likely to be right about who this is.
    """
    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO behavioural_face_links (
                camera_id, track_id, person_id, label, confidence, match_distance,
                scan_captured_at, time_delta_ms, candidates
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (camera_id, track_id) DO UPDATE
                SET person_id        = EXCLUDED.person_id,
                    label            = EXCLUDED.label,
                    confidence       = EXCLUDED.confidence,
                    match_distance   = EXCLUDED.match_distance,
                    scan_captured_at = EXCLUDED.scan_captured_at,
                    linked_at        = CURRENT_TIMESTAMP,
                    time_delta_ms    = EXCLUDED.time_delta_ms,
                    candidates       = EXCLUDED.candidates
                WHERE EXCLUDED.confidence IS NOT NULL
                  AND (behavioural_face_links.confidence IS NULL
                       OR EXCLUDED.confidence > behavioural_face_links.confidence)
            RETURNING track_id
            """,
            (camera_id, track_id, person_id, label, confidence, match_distance,
             scan_captured_at, time_delta_ms, candidates),
        )
        updated = cursor.fetchone() is not None
    db_conn.commit()

    logger.info(
        "face/track link %s: camera=%s track=%s person=%s confidence=%s",
        "written" if updated else "kept existing (stronger)",
        camera_id, track_id, person_id, confidence,
    )
    return updated


def correlate_scan(db_conn, scan_result, frame_context):
    """Attempt the join for a completed face scan. Returns a summary dict.

    Called from /api/v1/scan-face. Never raises: a failed correlation must not
    cost the caller their face match.
    """
    summary = {"linked": False, "reason": "no_face_match", "track_id": None}

    if not scan_result.get("is_known_user"):
        return summary

    person = scan_result.get("person") or {}
    person_id = person.get("id")
    if not person_id:
        return {"linked": False, "reason": "match_without_person_id", "track_id": None}

    try:
        outcome = find_track_for_face(
            db_conn,
            camera_id=frame_context.get("camera_id"),
            captured_at=frame_context.get("captured_at"),
            face_centre=frame_context.get("face_box_centre"),
        )

        if outcome.get("linked"):
            link_face_to_track(
                db_conn,
                camera_id=frame_context["camera_id"],
                track_id=outcome["track_id"],
                person_id=person_id,
                label=scan_result.get("status"),
                confidence=_confidence_from_distance(scan_result.get("match_distance")),
                match_distance=scan_result.get("match_distance"),
                scan_captured_at=frame_context.get("captured_at"),
                time_delta_ms=outcome.get("time_delta_ms"),
                candidates=outcome.get("candidates"),
            )
        return outcome
    except Exception:
        # A correlation failure is not a scan failure.
        logger.exception("face/track correlation failed; the face match stands.")
        db_conn.rollback()
        return {"linked": False, "reason": "correlation_error", "track_id": None}


def _confidence_from_distance(distance, threshold=0.30):
    """Cosine distance -> 0..1 confidence. Mirrors the behavioural module's
    face_signal_from_recognition, so both sides agree on what a match is worth."""
    if distance is None:
        return None
    try:
        return max(0.0, min(1.0, 1.0 - (float(distance) / max(threshold, 1e-6))))
    except (TypeError, ValueError):
        return None
