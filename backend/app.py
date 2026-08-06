import json
import os
import logging
import time
from pathlib import Path

import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

from config import Config

from routes.behaviour_routes import behaviour_bp
from routes.claim_routes import claim_bp
from routes.cpu_routes import cpu_bp
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from routes.member_score_routes import member_score_bp
from routes.route_routes import route_bp
from routes.safety_routes import safety_bp
from routes.user_routes import user_bp
from services import alerts_service
from services.claims_service import warm_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy in (
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = ROOT_DIR.parent / "frontend" / "dist"
DEMO_STATIC_DIR = ROOT_DIR / "static"

# The Azure Vision client now lives in services.plate_vision, which owns the
# free-tier call budget alongside it — two clients would mean two unmetered
# callers on one subscription.
_db_pool = None


def get_db_connection():
    global _db_pool
    if _db_pool is None:
        _db_pool = SimpleConnectionPool(
            Config.DB_POOL_MIN_CONN,
            Config.DB_POOL_MAX_CONN,
            Config.DATABASE_URL,
            connect_timeout=Config.DB_CONNECT_TIMEOUT_SECONDS,
            keepalives=1,
            keepalives_idle=Config.DB_KEEPALIVES_IDLE_SECONDS,
            keepalives_interval=Config.DB_KEEPALIVES_INTERVAL_SECONDS,
            keepalives_count=Config.DB_KEEPALIVES_COUNT,
        )
    return _db_pool.getconn()


def release_db_connection(conn):
    global _db_pool
    if conn is None:
        return
    if _db_pool is None:
        conn.close()
        return
    _db_pool.putconn(conn)


def _normalize_detection_status(raw_status):
    status = (raw_status or "").strip().lower()
    return status if status in {"suspect", "offender"} else None


def _attach_face_alert(result):
    status = _normalize_detection_status(result.get("status"))
    if not status or not result.get("is_known_user"):
        result["alert_event"] = None
        result["alerts"] = []
        return result

    person = result.get("person") or {}
    event = alerts_service.record_detection(
        match_label=status,
        entity_type="person",
        title=f"{status.title()} identified by facial recognition",
        detail=f"{person.get('full_name') or 'A person'} matched as {status}.",
        meta={
            "person_id": person.get("id"),
            "full_name": person.get("full_name"),
        },
    )
    result["alert_event"] = event
    result["alerts"] = [event] if event else []
    return result


def _attach_plate_alert(result, *, source_endpoint):
    plate = result.get("plate") or {}
    status = _normalize_detection_status(plate.get("status"))
    if not status or not result.get("match_found"):
        result["alert_event"] = None
        result["alerts"] = []
        return result

    # A tolerant match — confusable characters, one edit away, or a fragment
    # that only one registry entry could be — is shown to the operator as
    # probable but is deliberately kept out of the alert feed. Raising a
    # stolen-vehicle alert on a guessed character is worse than raising none.
    if result.get("match_confidence") == "probable":
        result["alert_event"] = None
        result["alerts"] = []
        result["alert_suppressed"] = "probable_match_requires_confirmation"
        return result

    plate_number = plate.get("plate_number") or result.get("extracted_text") or "Unknown plate"
    event = alerts_service.record_detection(
        match_label=status,
        entity_type="vehicle",
        title=f"{status.title()} plate identified",
        detail=f"Vehicle plate {plate_number} matched as {status} via {source_endpoint}.",
        meta={
            "plate_id": plate.get("id"),
            "plate_number": plate.get("plate_number"),
            "owner_name": plate.get("owner_name"),
            "source_endpoint": source_endpoint,
        },
    )
    result["alert_event"] = event
    result["alerts"] = [event] if event else []
    return result


def create_app(config_object=Config):
    app = Flask(
        __name__,
        static_folder=str(DEMO_STATIC_DIR),
        static_url_path="/static"
    )

    app.config.from_object(config_object)

    # Enable CORS for all routes (allows Vercel deployment to communicate with Railway API)
    allowed_origins = [
        "https://guardian-collect2.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ]
    # Optionally pull additional origins from environment variable if present
    custom_origin = os.environ.get("ALLOWED_ORIGIN")
    if custom_origin:
        allowed_origins.append(custom_origin)

    CORS(
        app,
        resources={r"/*": {"origins": allowed_origins}},
        supports_credentials=True,
    )

    # Register all routes
    app.register_blueprint(health_bp)
    app.register_blueprint(hotspot_bp)
    app.register_blueprint(claim_bp)
    app.register_blueprint(route_bp)
    app.register_blueprint(cpu_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(safety_bp)
    app.register_blueprint(member_score_bp)
    app.register_blueprint(behaviour_bp)

    @app.route("/demos", methods=["GET"])
    def demos_page():
        """Consolidated entry-point linking the biometric demo pages."""
        return send_from_directory(app.static_folder, "demos.html")

    @app.route("/test-scan", methods=["GET"])
    def test_scan_page():
        """Upload form for facial-recognition endpoint testing."""
        return send_from_directory(app.static_folder, "scan_test.html")

    @app.route("/test-plate", methods=["GET"])
    def test_plate_page():
        """Upload form for EasyOCR license plate endpoint testing."""
        return send_from_directory(app.static_folder, "plate_test.html")

    @app.route("/test-azure-plate", methods=["GET"])
    def test_plate_azure_page():
        """Upload form for Azure OCR license plate endpoint testing."""
        return send_from_directory(app.static_folder, "azure_plate_test.html")

    def _read_roi(field_name="roi"):
        """Parse the browser's plate-region hint, if it sent one."""
        raw = request.form.get(field_name)
        if not raw:
            return None
        try:
            roi = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return roi if isinstance(roi, dict) else None

    def _run_plate_scan(*, engine, source_endpoint, roi=None, max_passes=None):
        """Shared body for every plate endpoint.

        All three differ only in which engine they ask for and what they call
        themselves in the alert feed; the conditioning, the plate grammar and
        the registry matching are one code path.
        """
        if "file" not in request.files:
            return jsonify({"error": "No image file provided in 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()

        try:
            conn = get_db_connection()
        except OperationalError as e:
            logger.warning("plate scan DB unavailable: %s", e)
            return jsonify({"error": "Plate registry temporarily unavailable.", "code": "db_unavailable"}), 503

        try:
            from services.plate_recognition import scan_plate_image

            result = scan_plate_image(
                image_bytes,
                conn,
                roi=roi,
                engine=engine,
                max_passes=max_passes,
            )

            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]

            if result.get("throttled"):
                return jsonify(result), 200

            result = _attach_plate_alert(result, source_endpoint=source_endpoint)
            logger.info(
                "plate scan: engine=%s detected=%s text=%s match=%s confidence=%s passes=%s quota_left=%s",
                result.get("engine"),
                result.get("plate_detected"),
                result.get("extracted_text"),
                result.get("match_found"),
                result.get("match_confidence"),
                result.get("passes_used"),
                result.get("azure_calls_remaining"),
            )
            return jsonify(result), 200
        except OperationalError as e:
            logger.warning("plate scan DB operation failed: %s", e)
            return jsonify({"error": "Plate registry temporarily unavailable.", "code": "db_unavailable"}), 503
        except Exception as e:
            logger.error(f"Error processing plate scan: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500
        finally:
            release_db_connection(conn)

    @app.route("/api/v1/scan-face", methods=["POST"])
    def scan_face():
        """Facial recognition scanning endpoint (multipart under 'file')."""
        if "file" not in request.files:
            return jsonify({"error": "No image file provided in 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()
        request_started = time.perf_counter()

        # Optional per-frame context. All fields are optional, so a caller that
        # sends none of them gets exactly the previous behaviour. When a face
        # box IS supplied it is echoed back normalised, which is what lets the
        # behavioural module work out which tracked body this face belongs to
        # (see BEHAVIOUR_REVIEW_API.md §1).
        from services.frame_context import build_frame_context

        frame_context = build_frame_context(request.form)

        logger.info(
            "scan-face request received: filename=%s bytes=%s remote=%s camera=%s face_box=%s",
            file.filename,
            len(image_bytes),
            request.remote_addr,
            frame_context["camera_id"],
            "yes" if frame_context["face_box"] else "no",
        )
        try:
            conn = get_db_connection()
        except OperationalError as e:
            logger.warning("scan-face DB unavailable: %s", e)
            return jsonify({"error": "Face registry temporarily unavailable.", "code": "db_unavailable"}), 503

        try:
            from services.recognition import process_incoming_face_image

            result = process_incoming_face_image(
                image_bytes=image_bytes,
                db_conn=conn,
                model_name=Config.FACE_MODEL,
                threshold=Config.MATCH_THRESHOLD,
            )

            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]

            result = _attach_face_alert(result)
            result.update(frame_context)

            # THE JOIN. With a face box and a camera we can ask which tracked
            # body that face was inside, and attach this match to it. The
            # correlator refuses whenever it is not sure — no body tracking at
            # that moment, the face outside every box, or two people too close
            # to tell apart — because a wrong join puts one person's identity on
            # another person's behaviour, which is worse than no match at all.
            if frame_context.get("face_box_centre"):
                from services.behaviour_track_service import correlate_scan

                result["behavioural_link"] = correlate_scan(conn, result, frame_context)
            total_ms = round((time.perf_counter() - request_started) * 1000, 2)
            result["request_timing_ms"] = total_ms
            logger.info(
                "scan-face result: success=%s known=%s status=%s distance=%s total_ms=%s stage_ms=%s",
                result.get("success"),
                result.get("is_known_user"),
                result.get("status"),
                result.get("match_distance"),
                total_ms,
                result.get("timings_ms"),
            )
            return jsonify(result), 200
        except OperationalError as e:
            logger.warning("scan-face DB operation failed: %s", e)
            return jsonify({"error": "Face registry temporarily unavailable.", "code": "db_unavailable"}), 503
        except Exception as e:
            logger.error(f"Error processing facial scan: {e}")
            return jsonify({"error": "Internal processing error during scan"}), 500
        finally:
            release_db_connection(conn)

    @app.route("/api/v1/scan-plate", methods=["POST"])
    def scan_plate():
        """Still-image plate scan. Azure Vision, with the local engine behind it."""
        return _run_plate_scan(engine="auto", source_endpoint="/api/v1/scan-plate")

    @app.route("/api/v1/scan-plate-azure", methods=["POST"])
    def scan_plate_azure():
        """Still-image plate scan, pinned to Azure Vision.

        An upload has no live loop pacing it, so it gets both conditioning
        passes rather than the adaptive budget the live endpoint runs on.
        """
        return _run_plate_scan(
            engine="azure",
            source_endpoint="/api/v1/scan-plate-azure",
            max_passes=2,
        )

    @app.route("/api/v1/scan-plate-live", methods=["POST"])
    def scan_plate_live():
        """Live-camera plate scan.

        The browser sends a cropped, upscaled frame plus the region its own
        detector locked onto, so the server is refining a candidate rather than
        searching a whole doorbell frame. Pass count is left adaptive: this
        endpoint fires every few seconds and has to share one free-tier minute
        with every other scan.
        """
        return _run_plate_scan(
            engine="auto",
            source_endpoint="/api/v1/scan-plate-live",
            roi=_read_roi(),
        )

    @app.route("/api/v1/plate-scan-budget", methods=["GET"])
    def plate_scan_budget():
        """Remaining Azure Vision calls in the current minute.

        The live view reads this to pace itself and to show the operator why a
        scan was skipped, instead of silently dropping frames.
        """
        from services import plate_vision

        return jsonify({
            "azure_configured": plate_vision.azure_configured(),
            "limit_per_minute": plate_vision.quota_gate.limit,
            "remaining": plate_vision.quota_gate.remaining(),
            "retry_after_seconds": round(plate_vision.quota_gate.retry_after_seconds(), 1),
            "offline_fallback": plate_vision.easyocr_available(),
        }), 200

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def frontend(path: str):
        """Serve Vite build output at root, with SPA fallback to index.html."""
        if path.startswith("api/"):
            abort(404)

        # Serve static assets from dist when present (assets/*, icons, etc.)
        if path:
            target = FRONTEND_DIST_DIR / path
            if target.is_file():
                return send_from_directory(FRONTEND_DIST_DIR, path)

        dist_index = FRONTEND_DIST_DIR / "index.html"
        if dist_index.is_file():
            return send_from_directory(FRONTEND_DIST_DIR, "index.html")

        # Fallback so the app still runs if frontend/dist has not been built yet.
        return send_from_directory(app.static_folder, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    from services.recognition import warm_recognition_pipeline

    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm claims cache: {e}")

    try:
        warm_ms = warm_recognition_pipeline(Config.FACE_MODEL)
        logger.info("Warm recognition pipeline completed in %sms", warm_ms)
    except Exception as e:
        logger.warning("Failed to warm recognition pipeline: %s", e)

    try:
        conn = get_db_connection()
        release_db_connection(conn)
        logger.info("Database pool initialized.")
    except Exception as e:
        logger.warning("Database pool warmup failed: %s", e)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )