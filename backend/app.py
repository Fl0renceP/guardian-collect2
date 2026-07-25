import logging
import psycopg2
from flask import Flask, request, jsonify, send_from_directory

from config import Config
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from services.claims_service import warm_cache
from services.recognition import process_incoming_face_image
from services import detection_log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Register Blueprints
app.register_blueprint(health_bp)
app.register_blueprint(hotspot_bp)


def get_db_connection():
    """Establishes and returns a database connection using application config."""
    return psycopg2.connect(Config.DATABASE_URL)


@app.route("/")
def home():
    return jsonify({"message": "Guardian Collective API is running"}), 200


@app.route("/test-scan", methods=["GET"])
def test_scan_page():
    """Serves a minimal upload form for facial-recognition endpoint testing."""
    return send_from_directory(app.static_folder, "scan_test.html")


@app.route("/api/v1/scan-face", methods=["POST"])
def scan_face():
    """
    Facial recognition scanning endpoint.
    Accepts a multipart file payload under the 'file' field.
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file provided in 'file' field"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()

    # Where the check came from. Defaults suit the file-upload demo; a real feed
    # would send its own camera id and position.
    camera_id = request.form.get("camera_id", "demo_upload").strip() or "demo_upload"

    def _coord(name):
        raw = request.form.get(name, "").strip()
        try:
            return float(raw) if raw else None
        except ValueError:
            return None

    conn = get_db_connection()
    try:
        # Process face matching against PostgreSQL + DeepFace
        result = process_incoming_face_image(
            image_bytes=image_bytes,
            db_conn=conn,
            model_name=Config.FACE_MODEL,
            threshold=Config.MATCH_THRESHOLD,
        )

        # Handle tuple responses from service (e.g. ({'error': 'No face'}, 400))
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]

        # Every check is logged, matched or not — that is the point of the
        # detections table. record() swallows its own failures, so a logging
        # problem can never stop an alert reaching the operator.
        detection_id = detection_log.record(
            conn, result,
            threshold=Config.MATCH_THRESHOLD,
            camera_id=camera_id,
            lat=_coord("location_lat"),
            lng=_coord("location_lng"),
        )
        if detection_id:
            result["detection_id"] = detection_id

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing facial scan: {e}")
        return jsonify({"error": "Internal processing error during scan"}), 500

    finally:
        conn.close()


@app.route("/api/v1/detections", methods=["GET"])
def list_detections():
    """
    Recent face checks, matched or not.

    This is the continuous dataset the identity registry deliberately isn't:
    every check is here, while only people someone chose to enrol are in persons.

    ?limit=50        how many rows
    ?alerts_only=1   only checks that met an alert condition
    """
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 500)
    except ValueError:
        return jsonify({"error": "limit must be a number"}), 400

    alerts_only = request.args.get("alerts_only", "").lower() in ("1", "true", "yes")

    conn = get_db_connection()
    try:
        return jsonify({
            "summary": detection_log.summary(conn),
            "detections": detection_log.recent(conn, limit=limit, alerts_only=alerts_only),
        }), 200
    except Exception as e:
        logger.error(f"Error reading detections: {e}")
        return jsonify({"error": "Could not read the detection log"}), 500
    finally:
        conn.close()


if __name__ == "__main__":
    # Warm up claims cache on application boot
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm cache on startup: {e}")

    # Run local dev server
    app.run(host="0.0.0.0", port=5000, debug=True)