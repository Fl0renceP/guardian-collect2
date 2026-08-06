import os
import re
import logging
import importlib
import time
from pathlib import Path

import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import SimpleConnectionPool

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

from config import Config

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

# Reuse one Azure Vision client across requests.
_azure_client = None
_db_pool = None


def get_azure_client():
    global _azure_client
    if _azure_client is None:
        endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
        key = os.environ.get("AZURE_VISION_KEY")
        if not endpoint or not key:
            raise RuntimeError(
                "Azure Vision is not configured. Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY."
            )

        try:
            imageanalysis = importlib.import_module("azure.ai.vision.imageanalysis")
            credentials = importlib.import_module("azure.core.credentials")
        except ImportError as exc:
            raise RuntimeError(
                "Azure Vision SDK is not installed. Install requirements and retry."
            ) from exc

        _azure_client = imageanalysis.ImageAnalysisClient(
            endpoint=endpoint,
            credential=credentials.AzureKeyCredential(key),
        )
    return _azure_client


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


def _face_detection_alert(face, *, faces_in_frame=1):
    """Raise the alert feed event for one matched face, if it warrants one."""
    status = _normalize_detection_status(face.get("status"))
    if not status or not face.get("is_known_user"):
        return None

    person = face.get("person") or {}
    detail = f"{person.get('full_name') or 'A person'} matched as {status}."
    if faces_in_frame > 1:
        # An operator triaging the feed needs to know this was a group, not a
        # one-to-one scan — it changes what they're looking at on the footage.
        detail += f" One of {faces_in_frame} faces in the frame."

    return alerts_service.record_detection(
        match_label=status,
        entity_type="person",
        title=f"{status.title()} identified by facial recognition",
        detail=detail,
        meta={
            "person_id": person.get("id"),
            "full_name": person.get("full_name"),
            "faces_in_frame": faces_in_frame,
        },
    )


def _attach_face_alert(result):
    """Emit an alert per flagged identity in the frame, not just for one.

    A frame can hold several people, and each flagged one is its own sighting of
    its own person — collapsing them into a single event would drop identities
    out of the feed entirely. `alert_event` stays singular and points at the
    primary face's event so existing readers of that key still work.
    """
    faces = result.get("faces")
    if not isinstance(faces, list):
        # Pre-multi-face shape (or an error payload): treat the top level as the
        # one and only face.
        faces = [result] if result.get("is_known_user") else []

    faces_in_frame = result.get("faces_resolved") or len(faces) or 1

    events = []
    primary_event = None
    for face in faces:
        event = _face_detection_alert(face, faces_in_frame=faces_in_frame)
        if not event:
            continue
        events.append(event)
        # `faces` is ordered largest-first with is_primary marking the face the
        # top-level fields describe; fall back to the first event raised.
        if face.get("is_primary") or primary_event is None:
            primary_event = event

    result["alert_event"] = primary_event
    result["alerts"] = events
    return result


def _attach_plate_alert(result, *, source_endpoint):
    plate = result.get("plate") or {}
    status = _normalize_detection_status(plate.get("status"))
    if not status or not result.get("match_found"):
        result["alert_event"] = None
        result["alerts"] = []
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

    def extract_text_from_bytes(image_bytes: bytes) -> str:
        """Send image bytes to Azure Vision Read OCR and return extracted text."""
        try:
            models = importlib.import_module("azure.ai.vision.imageanalysis.models")
            result = get_azure_client().analyze(
                image_data=image_bytes,
                visual_features=[models.VisualFeatures.READ],
            )

            extracted_words = []
            if result.read is not None:
                for block in result.read.blocks:
                    for line in block.lines:
                        extracted_words.append(line.text)
            return " ".join(extracted_words)
        except Exception as e:
            logger.error(f"Azure OCR Error: {e}")
            return ""

    def clean_plate_text(text: str) -> str:
        """Normalise plate text so punctuation variants still match the registry."""
        return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()

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
        logger.info(
            "scan-face request received: filename=%s bytes=%s remote=%s",
            file.filename,
            len(image_bytes),
            request.remote_addr,
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
                max_faces=Config.MAX_SCAN_FACES,
            )

            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]

            result = _attach_face_alert(result)
            total_ms = round((time.perf_counter() - request_started) * 1000, 2)
            result["request_timing_ms"] = total_ms
            logger.info(
                "scan-face result: success=%s faces=%s known=%s flagged=%s "
                "primary_status=%s primary_distance=%s total_ms=%s stage_ms=%s",
                result.get("success"),
                result.get("faces_resolved"),
                result.get("faces_known"),
                result.get("faces_flagged"),
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
        """License plate recognition scanning endpoint (EasyOCR path)."""
        if "file" not in request.files:
            return jsonify({"error": "No image file provided in 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()
        conn = get_db_connection()
        try:
            from services.plate_recognition import process_incoming_plate_image

            result = process_incoming_plate_image(
                image_bytes=image_bytes,
                db_conn=conn,
            )

            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]

            result = _attach_plate_alert(result, source_endpoint="/api/v1/scan-plate")
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error processing plate scan: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500
        finally:
            release_db_connection(conn)

    @app.route("/api/v1/scan-plate-azure", methods=["POST"])
    def scan_plate_azure():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()

        raw_text = extract_text_from_bytes(image_bytes)
        cleaned_plate = clean_plate_text(raw_text)

        if not cleaned_plate:
            return jsonify({"error": "No text detected in image"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT p.id, p.plate_number, p.status, p.owner_name, f.image_url
                FROM vehicle_plates p
                LEFT JOIN vehicle_plate_images f ON p.id = f.plate_id
                WHERE regexp_replace(upper(p.plate_number), '[^A-Z0-9]', '', 'g') = %s
                ORDER BY f.created_at DESC NULLS LAST
                LIMIT 1;
                """,
                (cleaned_plate,),
            )
            match = cursor.fetchone()

            if match:
                result = {
                    "match_found": True,
                    "raw_text": raw_text,
                    "extracted_text": cleaned_plate,
                    "plate": {
                        "id": match[0],
                        "plate_number": match[1],
                        "status": match[2],
                        "owner_name": match[3],
                        "image_url": match[4],
                    },
                }
                result = _attach_plate_alert(result, source_endpoint="/api/v1/scan-plate-azure")
                return jsonify(result), 200

            return jsonify({
                "match_found": False,
                "raw_text": raw_text,
                "extracted_text": cleaned_plate,
                "message": f"Plate '{cleaned_plate}' scanned but not flagged in database.",
            }), 200
        except Exception as e:
            logger.error(f"Error processing Azure plate scan: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500
        finally:
            cursor.close()
            release_db_connection(conn)

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