import os
import json
import logging
import psycopg2
from queue import Empty
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context

from config import Config
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from services.claims_service import warm_cache
from services.alert_delivery_service import alert_delivery_service
from services.plate_recognition import process_incoming_plate_image  
from services.recognition import process_incoming_face_image
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

# Initialize Client
endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
key = os.environ.get("AZURE_VISION_KEY")

client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

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


def _normalize_detection_status(raw_status):
    status = (raw_status or "").strip().lower()
    return status if status in {"verified", "suspect", "offender"} else None


def _attach_face_alert_payload(result):
    status = _normalize_detection_status(result.get("status"))
    if not status or not result.get("is_known_user"):
        result["alert_event"] = None
        result["alerts"] = []
        return result

    person = result.get("person") or {}
    event = alert_delivery_service.register_detection(
        status=status,
        entity_type="person",
        entity={
            "id": person.get("id"),
            "name": person.get("full_name"),
            "status": status,
        },
        source_endpoint="/api/v1/scan-face",
        push_enabled=Config.PUSH_NOTIFICATIONS_ENABLED,
        push_dry_run=Config.PUSH_NOTIFICATIONS_DRY_RUN,
        push_min_level=Config.PUSH_MIN_LEVEL,
    )
    result["alert_event"] = event
    result["alerts"] = [event] if event else []
    return result


def _attach_plate_alert_payload(result):
    plate = result.get("plate") or {}
    status = _normalize_detection_status(plate.get("status"))
    if not status or not result.get("match_found"):
        result["alert_event"] = None
        result["alerts"] = []
        return result

    event = alert_delivery_service.register_detection(
        status=status,
        entity_type="vehicle",
        entity={
            "id": str(plate.get("id")) if plate.get("id") is not None else None,
            "plate_number": plate.get("plate_number"),
            "owner_name": plate.get("owner_name"),
            "status": status,
        },
        source_endpoint="/api/v1/scan-plate",
        push_enabled=Config.PUSH_NOTIFICATIONS_ENABLED,
        push_dry_run=Config.PUSH_NOTIFICATIONS_DRY_RUN,
        push_min_level=Config.PUSH_MIN_LEVEL,
    )
    result["alert_event"] = event
    result["alerts"] = [event] if event else []
    return result


@app.get("/api/v1/alerts")
def list_alerts():
    """List recent alerts/push events filtered by audience and channel."""
    audience = (request.args.get("audience") or "crime_prevention").strip().lower()
    channel = (request.args.get("channel") or "alerts").strip().lower()
    try:
        limit = int(request.args.get("limit", "50"))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    if audience not in {"member", "crime_prevention"}:
        return jsonify({"error": "audience must be one of: member, crime_prevention"}), 400
    if channel not in {"alerts", "push"}:
        return jsonify({"error": "channel must be one of: alerts, push"}), 400

    items = alert_delivery_service.list_events(audience=audience, channel=channel, limit=limit)
    return jsonify(
        {
            "audience": audience,
            "channel": channel,
            "count": len(items),
            "items": items,
        }
    )


@app.get("/api/v1/alerts/stream")
def stream_alerts():
    """Server-Sent Events stream for live alert/push updates."""
    audience = (request.args.get("audience") or "crime_prevention").strip().lower()
    channel = (request.args.get("channel") or "alerts").strip().lower()

    if audience not in {"member", "crime_prevention"}:
        return jsonify({"error": "audience must be one of: member, crime_prevention"}), 400
    if channel not in {"alerts", "push"}:
        return jsonify({"error": "channel must be one of: alerts, push"}), 400

    sub = alert_delivery_service.subscribe(audience=audience, channel=channel)

    @stream_with_context
    def generate():
        try:
            ready = {"audience": audience, "channel": channel, "ready": True}
            yield f"event: ready\ndata: {json.dumps(ready)}\n\n"
            while True:
                try:
                    event = alert_delivery_service.queue_get(sub, timeout_seconds=20)
                    yield f"event: notification\ndata: {json.dumps(event)}\n\n"
                except Empty:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            alert_delivery_service.unsubscribe(sub)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/")
def home():
    return jsonify({"message": "Guardian Collective API is running"}), 200


@app.route("/test-scan", methods=["GET"])
def test_scan_page():
    """Serves a minimal upload form for facial-recognition endpoint testing."""
    return send_from_directory(app.static_folder, "scan_test.html")

@app.route("/test-plate", methods=["GET"])
def test_plate_page():
    """Serves a minimal upload form for license plate recognition endpoint testing."""
    return send_from_directory(app.static_folder, "plate_test.html")

@app.route("/test-azure-plate", methods=["GET"])
def test_plate_azure_page():
    """Serves upload form for Azure license plate OCR testing."""
    return send_from_directory(app.static_folder, "azure_plate_test.html")


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

        result = _attach_face_alert_payload(result)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing facial scan: {e}")
        return jsonify({"error": "Internal processing error during scan"}), 500

    finally:
        conn.close()

@app.route("/api/v1/scan-plate", methods=["POST"])
def scan_plate():
    """License plate recognition scanning endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No image file provided in 'file' field"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()
    conn = get_db_connection()

    try:
        result = process_incoming_plate_image(
            image_bytes=image_bytes,
            db_conn=conn
        )

        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]

        result = _attach_plate_alert_payload(result)
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing plate scan: {e}")
        return jsonify({"error": "Internal processing error during plate scan"}), 500

    finally:
        conn.close()

def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Sends image bytes to Azure AI Vision Read OCR and returns extracted text."""
    try:
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.READ]  # Correct Enum
        )

        extracted_words = []
        if result.read is not None:  # Correct Property
            for block in result.read.blocks:
                for line in block.lines:
                    extracted_words.append(line.text)

        return " ".join(extracted_words)

    except Exception as e:
        logger.error(f"Azure OCR Error: {e}")
        return ""

@app.route("/api/v1/scan-plate-azure", methods=["POST"])
def scan_plate_azure():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    # Extract text using Azure Vision
    raw_text = extract_text_from_bytes(image_bytes)
    
    # Clean string (remove spaces/special characters for database matching)
    cleaned_plate = "".join([c for c in raw_text if c.isalnum()]).upper()

    return jsonify({
        "raw_text": raw_text,
        "cleaned_plate": cleaned_plate
    }), 200

if __name__ == "__main__":
    # Warm up claims cache on application boot
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm cache on startup: {e}")

    # Run local dev server
    app.run(host="0.0.0.0", port=5000, debug=True)