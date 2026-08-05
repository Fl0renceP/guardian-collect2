"""Tests for the human decision step.

Runs against the local Postgres, namespaced and cleaned up.

The rules being tested are not implementation details — they are the promises
the whole feature rests on. In particular: confirming must never rewrite the
identity registry, denying must never be possible without a reason, and undoing
must never erase what it undoes.

    python tests/test_behaviour_decisions.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import psycopg2  # noqa: E402

from config import Config  # noqa: E402
from services.behaviour_decision_service import (  # noqa: E402
    DecisionError,
    decide,
    history,
    reopen,
)

CAMERA = "test_decision_cam"
EVENT_ID = "TESTDECISION-1"


def connect():
    return psycopg2.connect(Config.DATABASE_URL, connect_timeout=6)


def clean(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM behavioural_events WHERE event_id LIKE 'TESTDECISION%'")
        cur.execute("DELETE FROM behavioural_face_links WHERE camera_id = %s", (CAMERA,))
    conn.commit()


def make_review(conn, *, with_identity=False):
    """A pending review to decide on. Returns (review_id, person_id|None)."""
    clean(conn)
    now = datetime.now(timezone.utc)
    person_id = None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO behavioural_events (event_id, track_id, camera_id, location_zone_id,
                occurred_at, behavioural_risk_score, composite_risk_score,
                requires_human_review, triggered_heuristics, reasoning)
            VALUES (%s,'person-1',%s,%s,%s,0.7,0.6,TRUE,%s,%s)
            """,
            (EVENT_ID, CAMERA, CAMERA, now,
             json.dumps([{"type": "loitering", "confidence": 0.6, "explanation": "test"}]),
             json.dumps([])),
        )
        cur.execute(
            "INSERT INTO behavioural_reviews (event_id) VALUES (%s) RETURNING review_id",
            (EVENT_ID,),
        )
        review_id = cur.fetchone()[0]

        if with_identity:
            cur.execute("SELECT id FROM persons LIMIT 1")
            found = cur.fetchone()
            if found:
                person_id = found[0]
                cur.execute(
                    """
                    INSERT INTO behavioural_face_links
                        (camera_id, track_id, person_id, label, confidence, scan_captured_at)
                    VALUES (%s,'person-1',%s,'suspect',0.9,%s)
                    ON CONFLICT (camera_id, track_id) DO UPDATE SET person_id = EXCLUDED.person_id
                    """,
                    (CAMERA, person_id, now),
                )
    conn.commit()
    return review_id, person_id


# --- the core promises ----------------------------------------------------
def test_confirming_does_not_rewrite_the_identity_registry():
    """THE ONE THAT MATTERS. A behavioural flag must never be a one-click path
    into the curated registry facial recognition matches against."""
    conn = connect()
    try:
        review_id, person_id = make_review(conn, with_identity=True)
        if not person_id:
            print("     (skipped: no seeded persons)")
            return

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM persons WHERE id = %s", (person_id,))
            status_before = cur.fetchone()[0]

        result = decide(conn, review_id, decision="confirm", reviewer_id="emp-test",
                        label="offender", note="test confirm")

        with conn.cursor() as cur:
            cur.execute("SELECT status FROM persons WHERE id = %s", (person_id,))
            status_after = cur.fetchone()[0]

        assert status_after == status_before, "confirming changed persons.status"
        assert result["identity_written"] is False
        assert result["status"] == "confirmed"
        # The judgement IS recorded — on the review, not the registry.
        assert result["matched_label"] == "offender"
    finally:
        clean(conn)
        conn.close()


def test_denying_without_a_reason_is_refused():
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        for missing in (None, "", "   "):
            try:
                decide(conn, review_id, decision="deny", reviewer_id="emp-test", reason=missing)
            except DecisionError as exc:
                assert "reason" in exc.fields
                continue
            raise AssertionError(f"deny with reason={missing!r} should have been refused")
    finally:
        clean(conn)
        conn.close()


def test_a_decision_without_a_reviewer_is_refused():
    """The reviewer id is the audit trail's only signature."""
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        for who in (None, "", "  "):
            try:
                decide(conn, review_id, decision="confirm", reviewer_id=who)
            except DecisionError as exc:
                assert "reviewer_id" in exc.fields
                continue
            raise AssertionError("a decision with no reviewer should have been refused")
    finally:
        clean(conn)
        conn.close()


def test_undo_restores_pending_without_erasing_the_original():
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        decide(conn, review_id, decision="deny", reviewer_id="emp-a", reason="resident")
        reopen(conn, review_id, reviewer_id="emp-b", reason="second opinion")

        with conn.cursor() as cur:
            cur.execute("SELECT status, decided_by FROM behavioural_reviews WHERE review_id = %s",
                        (review_id,))
            status, decided_by = cur.fetchone()

        assert status == "pending"
        assert decided_by is None

        trail = history(conn, review_id)
        assert [d["decision"] for d in trail] == ["deny", "reopen"], trail
        assert trail[0]["reason"] == "resident", "the undone decision was erased"
        assert trail[0]["reviewer_id"] == "emp-a"
        assert trail[1]["reviewer_id"] == "emp-b"
    finally:
        clean(conn)
        conn.close()


def test_deciding_twice_is_refused_until_reopened():
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        decide(conn, review_id, decision="confirm", reviewer_id="emp-a")
        try:
            decide(conn, review_id, decision="deny", reviewer_id="emp-b", reason="changed mind")
        except DecisionError as exc:
            assert exc.status == 409
        else:
            raise AssertionError("a second decision should have been refused")

        reopen(conn, review_id, reviewer_id="emp-b")
        decide(conn, review_id, decision="deny", reviewer_id="emp-b", reason="changed mind")

        assert [d["decision"] for d in history(conn, review_id)] == ["confirm", "reopen", "deny"]
    finally:
        clean(conn)
        conn.close()


def test_denying_asserts_no_identity():
    """Denying says the flag was wrong. It must not leave a name attached."""
    conn = connect()
    try:
        review_id, person_id = make_review(conn, with_identity=True)
        if not person_id:
            print("     (skipped: no seeded persons)")
            return

        decide(conn, review_id, decision="deny", reviewer_id="emp-test",
               reason="Resident at their own car.")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT matched_person_id, matched_label FROM behavioural_reviews "
                "WHERE review_id = %s", (review_id,))
            person, label = cur.fetchone()

        assert person is None and label is None
    finally:
        clean(conn)
        conn.close()


def test_a_flag_with_no_identity_can_still_be_confirmed():
    """Confirming an unattached flag asserts the BEHAVIOUR was real, and makes
    no claim about who the person is."""
    conn = connect()
    try:
        review_id, _ = make_review(conn, with_identity=False)
        result = decide(conn, review_id, decision="confirm", reviewer_id="emp-test")

        assert result["status"] == "confirmed"
        assert result["matched_person_id"] is None
        assert result["matched_label"] is None
        # Nobody to alert about.
        assert result["alerts_sent"] == []
    finally:
        clean(conn)
        conn.close()


def test_unknown_review_and_bad_label_are_rejected():
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        try:
            decide(conn, "rev-nope", decision="confirm", reviewer_id="e")
        except DecisionError as exc:
            assert exc.status == 404
        else:
            raise AssertionError("unknown review should 404")

        try:
            decide(conn, review_id, decision="confirm", reviewer_id="e", label="villain")
        except DecisionError as exc:
            assert "label" in exc.fields
        else:
            raise AssertionError("a bad label should be rejected")
    finally:
        clean(conn)
        conn.close()


def test_undo_of_an_undecided_review_is_refused():
    conn = connect()
    try:
        review_id, _ = make_review(conn)
        try:
            reopen(conn, review_id, reviewer_id="emp-a")
        except DecisionError as exc:
            assert exc.status == 409
        else:
            raise AssertionError("undoing a pending review should be refused")
    finally:
        clean(conn)
        conn.close()


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
