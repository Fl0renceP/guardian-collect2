"""Behavioural analysis endpoints.

Ingest for events emitted by `backend/behavioural_analysis/`. See
BEHAVIOUR_REVIEW_API.md for the contract.

WHAT THIS DELIBERATELY DOES NOT DO: send an alert. A behavioural event is a
statement about movement, not an identification, and `alerts_service` routes on
identity labels — members see `offender` matches only, which is what stops them
drowning in maybes. A flag reaching a review queue is the whole design; a flag
reaching a member is the failure mode that design exists to prevent.
"""

import logging
import time

from flask import Blueprint, Response, jsonify, request, stream_with_context
from psycopg2 import OperationalError

from services import behaviour_live_service
from services.behaviour_live_service import LiveFrameError
from services.behaviour_events_service import (
    EventValidationError,
    get_review,
    list_events,
    list_reviews,
    record_event,
    validate_event,
)
from services.behaviour_clip_service import ClipStorageUnavailable, upload_clip
from services.behaviour_decision_service import DecisionError, decide, history, reopen
from services.behaviour_track_service import (
    TrackSnapshotError,
    record_snapshots,
    validate_snapshots,
)

logger = logging.getLogger(__name__)

behaviour_bp = Blueprint("behaviour", __name__, url_prefix="/api/v1/behaviour")


def _db():
    """Borrow a connection from the pool app.py owns.

    Imported at call time, not module load: app.py imports this blueprint, so a
    module-level import would be circular.
    """
    from app import get_db_connection, release_db_connection

    return get_db_connection, release_db_connection


@behaviour_bp.route("/events", methods=["POST"])
def ingest_event():
    """Store one behavioural event, opening a review if it needs a human."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected a JSON body."}), 400

    try:
        event = validate_event(payload)
    except EventValidationError as exc:
        logger.warning("Rejected behavioural event: %s %s", exc, exc.fields)
        return jsonify({"error": str(exc), "fields": exc.fields}), 400

    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError as exc:
        logger.warning("behaviour ingest: DB unavailable: %s", exc)
        return jsonify({
            "error": "Behavioural event store temporarily unavailable.",
            "code": "db_unavailable",
        }), 503

    try:
        result = record_event(event, conn)
        # 200 rather than 201 for a replay, so a retrying pusher can tell the
        # difference between "stored" and "already had it".
        return jsonify(result), (200 if result["duplicate"] else 201)
    except OperationalError as exc:
        conn.rollback()
        logger.warning("behaviour ingest: DB operation failed: %s", exc)
        return jsonify({
            "error": "Behavioural event store temporarily unavailable.",
            "code": "db_unavailable",
        }), 503
    except Exception:
        conn.rollback()
        logger.exception("behaviour ingest failed")
        return jsonify({"error": "Internal error storing the behavioural event."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/tracks", methods=["POST"])
def ingest_tracks():
    """Body positions from the behavioural pipeline.

    Only the behavioural module knows where bodies are, and a face match cannot
    be attached to one without that. These rows are short-lived by design — see
    PRESENCE_RETENTION_MINUTES in services/behaviour_track_service.py.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Expected a JSON body."}), 400

    try:
        rows = validate_snapshots(payload)
    except TrackSnapshotError as exc:
        return jsonify({"error": str(exc), "fields": exc.fields}), 400

    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        return jsonify(record_snapshots(rows, conn)), 201
    except Exception:
        conn.rollback()
        logger.exception("track snapshot ingest failed")
        return jsonify({"error": "Internal error storing track positions."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/review-queue", methods=["GET"])
def review_queue():
    """Events that reached a human, newest and highest-scoring first."""
    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Review queue unavailable.", "code": "db_unavailable"}), 503

    try:
        result = list_reviews(
            conn,
            status=request.args.get("status", "pending"),
            camera_id=request.args.get("camera_id"),
            limit=request.args.get("limit", 50),
        )
        return jsonify(result), 200
    except EventValidationError as exc:
        return jsonify({"error": str(exc), "fields": exc.fields}), 400
    except Exception:
        logger.exception("review queue read failed")
        return jsonify({"error": "Internal error reading the review queue."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/review-queue/<review_id>", methods=["GET"])
def review_card(review_id):
    """One review card: the identity half, the behaviour half, and what a
    decision would mean."""
    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Review queue unavailable.", "code": "db_unavailable"}), 503

    try:
        review = get_review(conn, review_id)
        if review is None:
            return jsonify({"error": f"No review {review_id}."}), 404
        return jsonify(review), 200
    except Exception:
        logger.exception("review card read failed")
        return jsonify({"error": "Internal error reading the review."}), 500
    finally:
        release_conn(conn)


def _decision_endpoint(review_id, decision):
    """Shared body for confirm and deny.

    Kept as one function because the two differ only in what they assert, and a
    divergence between them is how audit gaps appear.
    """
    payload = request.get_json(silent=True) or {}

    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        result = decide(
            conn,
            review_id,
            decision=decision,
            reviewer_id=payload.get("reviewer_id"),
            label=payload.get("label"),
            note=payload.get("note"),
            reason=payload.get("reason") or payload.get("denial_reason"),
        )
        return jsonify(result), 200
    except DecisionError as exc:
        conn.rollback()
        return jsonify({"error": str(exc), "fields": exc.fields}), exc.status
    except Exception:
        conn.rollback()
        logger.exception("review decision failed")
        return jsonify({"error": "Internal error recording the decision."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/review-queue/<review_id>/confirm", methods=["POST"])
def confirm_review(review_id):
    """A reviewer confirms the flag — and, where one is attached, the identity.

    This is a person's identification, not the system's. Nothing automatic can
    reach this endpoint, and it does not rewrite the identity registry.
    """
    return _decision_endpoint(review_id, "confirm")


@behaviour_bp.route("/review-queue/<review_id>/deny", methods=["POST"])
def deny_review(review_id):
    """A reviewer records the flag as a false positive. A reason is required."""
    return _decision_endpoint(review_id, "deny")


@behaviour_bp.route("/review-queue/<review_id>/reopen", methods=["POST"])
def reopen_review(review_id):
    """Undo a decision within the reversal window. The original is not erased."""
    payload = request.get_json(silent=True) or {}

    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        result = reopen(
            conn, review_id,
            reviewer_id=payload.get("reviewer_id"),
            reason=payload.get("reason"),
        )
        return jsonify(result), 200
    except DecisionError as exc:
        conn.rollback()
        return jsonify({"error": str(exc), "fields": exc.fields}), exc.status
    except Exception:
        conn.rollback()
        logger.exception("review reopen failed")
        return jsonify({"error": "Internal error reopening the review."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/review-queue/<review_id>/clip", methods=["POST"])
def upload_review_clip(review_id):
    """Attach the footage around a flag, so a reviewer can see what the
    explanations describe rather than take them on trust.

    The blob NAME is stored, never a signed URL — the card signs a fresh
    short-lived one on each read.
    """
    if "file" not in request.files:
        return jsonify({"error": "No clip provided in 'file' field."}), 400

    upload = request.files["file"]
    data = upload.read()

    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT clip_url FROM behavioural_reviews WHERE review_id = %s", (review_id,)
            )
            if cursor.fetchone() is None:
                return jsonify({"error": f"No review {review_id}."}), 404

        blob_name = upload_clip(review_id, data, upload.filename or "clip.mp4")

        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE behavioural_reviews SET clip_url = %s WHERE review_id = %s",
                (blob_name, review_id),
            )
        conn.commit()
        return jsonify({"review_id": review_id, "clip": blob_name, "bytes": len(data)}), 201
    except ValueError as exc:
        conn.rollback()
        return jsonify({"error": str(exc)}), 400
    except ClipStorageUnavailable as exc:
        conn.rollback()
        # The review still stands without footage — the explanations are the record.
        logger.warning("Clip storage unavailable for %s: %s", review_id, exc)
        return jsonify({"error": "Clip storage unavailable.", "code": "storage_unavailable"}), 503
    except Exception:
        conn.rollback()
        logger.exception("clip upload failed")
        return jsonify({"error": "Internal error storing the clip."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/live-frame", methods=["POST"])
def ingest_live_frame():
    """One annotated frame from a running pipeline.

    The same image the module draws in its debug window. Held in memory only and
    overwritten by the next frame — see behaviour_live_service for why this one
    is never persisted when clips are.
    """
    camera_id = request.args.get("camera_id") or request.form.get("camera_id")

    if "file" in request.files:
        jpeg = request.files["file"].read()
    else:
        # Raw body is what a pipeline pushing every frame wants — multipart
        # framing on a per-frame POST is overhead for no benefit.
        jpeg = request.get_data()

    try:
        return jsonify(behaviour_live_service.store_frame(camera_id, jpeg)), 202
    except LiveFrameError as exc:
        return jsonify({"error": str(exc)}), 400


@behaviour_bp.route("/live/status", methods=["GET"])
def live_status():
    """Whether a camera is streaming right now.

    The card asks this before rendering a feed. Without it the browser would
    hold an <img> open against a camera that stopped hours ago and show a
    spinner where the answer is "nothing is watching this".
    """
    return jsonify(behaviour_live_service.status(request.args.get("camera_id"))), 200


@behaviour_bp.route("/live", methods=["GET"])
def live_stream():
    """The annotated feed as MJPEG.

    MJPEG rather than anything cleverer because it is an <img src> on the other
    end — no player, no codec negotiation, no websocket to keep alive. At the
    frame rates this pipeline achieves on a CPU, the bandwidth argument for a
    real codec does not arise.
    """
    camera_id = request.args.get("camera_id")
    if not camera_id:
        return jsonify({"error": "A camera_id is required."}), 400

    if behaviour_live_service.latest(camera_id) is None:
        # 503 rather than an empty stream: "no camera is streaming" is a real
        # answer, and a hanging image element cannot express it.
        return jsonify({
            "error": f"No live feed for {camera_id}.",
            "code": "not_streaming",
        }), 503

    boundary = "frame"

    def frames():
        last_seq = None
        idle_since = time.monotonic()

        while True:
            current = behaviour_live_service.latest(camera_id)
            if current is None:
                # The pipeline stopped. Ending the response is what makes the
                # browser's onerror fire, which is how the card learns to stop
                # showing a LIVE badge over a feed that is no longer live.
                return

            jpeg, seq = current
            if seq != last_seq:
                last_seq = seq
                idle_since = time.monotonic()
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            elif time.monotonic() - idle_since > behaviour_live_service.LIVE_STALE_SECONDS:
                return

            # Poll faster than the pipeline produces, so a new frame goes out as
            # soon as it exists rather than on a fixed cadence of our own.
            time.sleep(0.08)

    return Response(
        stream_with_context(frames()),
        mimetype=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


@behaviour_bp.route("/review-queue/<review_id>/history", methods=["GET"])
def review_history(review_id):
    """Every decision ever taken on this review. Append-only."""
    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        return jsonify({"review_id": review_id, "decisions": history(conn, review_id)}), 200
    except Exception:
        logger.exception("review history read failed")
        return jsonify({"error": "Internal error reading the decision history."}), 500
    finally:
        release_conn(conn)


@behaviour_bp.route("/events", methods=["GET"])
def read_events():
    """Recent events. The read side of the ingest, so step 2 is verifiable on
    its own before the review queue endpoints exist."""
    get_conn, release_conn = _db()
    try:
        conn = get_conn()
    except OperationalError:
        return jsonify({"error": "Unavailable.", "code": "db_unavailable"}), 503

    try:
        events = list_events(
            conn,
            camera_id=request.args.get("camera_id"),
            review_only=request.args.get("review_only") in ("1", "true", "yes"),
            limit=request.args.get("limit", 50),
        )
        return jsonify({"events": events, "count": len(events)}), 200
    except Exception:
        logger.exception("behaviour event read failed")
        return jsonify({"error": "Internal error reading behavioural events."}), 500
    finally:
        release_conn(conn)
