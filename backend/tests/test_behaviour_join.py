"""Tests for the face-to-body join.

Runs against the local Postgres, namespaced to its own camera ids and cleaned up
afterwards, because the join IS database logic — a version tested against fakes
would prove nothing about the SQL that actually decides who a face belongs to.

The refusals matter more than the successes here. A wrong join puts one person's
identity onto another person's behaviour, which is worse than either signal
failing: facial recognition failing gives you an unknown person, but a wrong
join gives you a confident, specific, wrong accusation.

    python tests/test_behaviour_join.py
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import psycopg2  # noqa: E402

from config import Config  # noqa: E402
from services.behaviour_track_service import (  # noqa: E402
    find_track_for_face,
    record_snapshots,
    validate_snapshots,
)

CAMERA = "test_join_cam"

# Anchored to NOW, not a fixed date. Body positions are expired after
# PRESENCE_RETENTION_MINUTES, and record_snapshots enforces that on every write
# — so a fixture pinned to a past timestamp gets deleted the moment it is
# inserted. That retention is a feature; the test has to live inside it.
T0 = datetime.now(timezone.utc).replace(microsecond=0)


def connect():
    return psycopg2.connect(Config.DATABASE_URL, connect_timeout=6)


def clean(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM behavioural_face_links WHERE camera_id LIKE 'test_join%'")
        cur.execute("DELETE FROM behavioural_tracks WHERE camera_id LIKE 'test_join%'")
    conn.commit()


def put_bodies(conn, bodies, at=T0, camera=CAMERA):
    """bodies: {track_id: (x, y, w, h)} in normalised coordinates."""
    rows = validate_snapshots({
        "camera_id": camera,
        "snapshots": [{
            "timestamp": at.isoformat(),
            "tracks": [
                {"track_id": tid, "bbox": {"x": b[0], "y": b[1], "w": b[2], "h": b[3]}}
                for tid, b in bodies.items()
            ],
        }],
    })
    record_snapshots(rows, conn)


# --- the happy path -------------------------------------------------------
def test_face_inside_a_single_body_links_to_it():
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {"person-1": (0.10, 0.20, 0.20, 0.60)})
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.20, "y": 0.28},   # head area of that body
        )
        assert result["linked"] is True, result
        assert result["track_id"] == "person-1"
        assert result["reason"] == "face_centre_inside_body_box"
    finally:
        clean(conn)
        conn.close()


def test_the_nearer_of_two_clearly_different_bodies_wins():
    """Overlapping boxes: the smaller one is the nearer person, but only when
    it is clearly smaller."""
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {
            "person-far": (0.00, 0.00, 0.90, 0.90),   # area 0.81
            "person-near": (0.15, 0.15, 0.20, 0.30),  # area 0.06 — clearly smaller
        })
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.20, "y": 0.20},
        )
        assert result["linked"] is True, result
        assert result["track_id"] == "person-near"
        assert result["candidates"] == 2
    finally:
        clean(conn)
        conn.close()


# --- the refusals ---------------------------------------------------------
def test_two_similar_bodies_are_refused_not_guessed():
    """THE IMPORTANT ONE. Two people at similar distance, boxes overlapping —
    there is no honest way to pick, so the join refuses."""
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {
            "person-a": (0.10, 0.10, 0.30, 0.50),   # area 0.150
            "person-b": (0.12, 0.10, 0.31, 0.52),   # area 0.161 — near identical
        })
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.25, "y": 0.20},     # inside both
        )
        assert result["linked"] is False, result
        assert result["reason"] == "ambiguous_several_bodies_overlap"
        assert result["track_id"] is None
    finally:
        clean(conn)
        conn.close()


def test_face_outside_every_body_is_not_linked():
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {"person-1": (0.10, 0.20, 0.20, 0.60)})
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.80, "y": 0.80},
        )
        assert result["linked"] is False
        assert result["reason"] == "face_outside_every_body_box"
    finally:
        clean(conn)
        conn.close()


def test_no_body_tracking_at_that_moment_is_not_linked():
    conn = connect()
    try:
        clean(conn)
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.2, "y": 0.2},
        )
        assert result["linked"] is False
        assert result["reason"] == "no_body_tracking_at_that_moment"
    finally:
        clean(conn)
        conn.close()


def test_a_body_seen_too_long_ago_does_not_count():
    """Outside the window a person can have walked out of their own box."""
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {"person-1": (0.10, 0.20, 0.20, 0.60)}, at=T0 - timedelta(seconds=30))
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.20, "y": 0.28},
        )
        assert result["linked"] is False
        assert result["reason"] == "no_body_tracking_at_that_moment"
    finally:
        clean(conn)
        conn.close()


def test_a_body_on_a_different_camera_does_not_count():
    conn = connect()
    try:
        clean(conn)
        put_bodies(conn, {"person-1": (0.10, 0.20, 0.20, 0.60)}, camera="test_join_other")
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(),
            face_centre={"x": 0.20, "y": 0.28},
        )
        assert result["linked"] is False
        assert result["reason"] == "no_body_tracking_at_that_moment"
    finally:
        clean(conn)
        conn.close()


def test_a_missing_face_box_is_not_linked():
    conn = connect()
    try:
        result = find_track_for_face(
            conn, camera_id=CAMERA, captured_at=T0.isoformat(), face_centre=None
        )
        assert result["linked"] is False
        assert result["reason"] == "no_face_box"
    finally:
        conn.close()


# --- writing the link -----------------------------------------------------
def test_a_stronger_match_replaces_a_weaker_one_but_not_the_reverse():
    """Keep the most confident link for a track, not the most recent — the
    stronger match is likelier to be right about who this is."""
    conn = connect()
    try:
        clean(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM persons LIMIT 2")
            people = cur.fetchall()
        if len(people) < 2:
            print("     (skipped: needs 2 seeded persons)")
            return

        from services.behaviour_track_service import link_face_to_track

        link_face_to_track(conn, camera_id=CAMERA, track_id="person-1",
                           person_id=people[0][0], confidence=0.9)
        link_face_to_track(conn, camera_id=CAMERA, track_id="person-1",
                           person_id=people[1][0], confidence=0.4)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_id, confidence FROM behavioural_face_links "
                "WHERE camera_id=%s AND track_id=%s", (CAMERA, "person-1"))
            person_id, confidence = cur.fetchone()

        assert str(person_id) == str(people[0][0]), "the weaker match overwrote the stronger"
        assert abs(confidence - 0.9) < 1e-6
    finally:
        clean(conn)
        conn.close()


def test_snapshot_validation_rejects_junk():
    from services.behaviour_track_service import TrackSnapshotError

    for bad in (
        {},
        {"camera_id": "c"},
        {"camera_id": "c", "snapshots": []},
        {"camera_id": "", "snapshots": [{"timestamp": "x", "tracks": []}]},
        {"camera_id": "c", "snapshots": [{"timestamp": "x", "tracks": [
            {"track_id": "t", "bbox": {"x": 0, "y": 0, "w": 0, "h": 1}}]}]},   # no area
    ):
        try:
            validate_snapshots(bad)
        except TrackSnapshotError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
