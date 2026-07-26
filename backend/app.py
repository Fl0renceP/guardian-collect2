import os
import re
import logging
import psycopg2
from flask import Flask, request, jsonify, send_from_directory

from config import Config
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from services.claims_service import warm_cache
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
    #Establishes and returns a database connection using application config.
    return psycopg2.connect(Config.DATABASE_URL)


@app.route("/")
def home():
    return jsonify({"message": "Guardian Collective API is running"}), 200


@app.route("/demos", methods=["GET"])
def demos_page():
    """Consolidated entry-point linking the biometric demo pages."""
    return send_from_directory(app.static_folder, "demos.html")


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

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error processing plate scan: {e}")
        return jsonify({"error": "Internal processing error during plate scan"}), 500

    finally:
        conn.close()

def extract_text_from_bytes(image_bytes: bytes) -> str:
    #Sends image bytes to Azure AI Vision Read OCR and returns extracted text.
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


def clean_plate_text(text: str) -> str:
    """Normalise plate text so punctuation variants still match the registry."""
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()

@app.route("/api/v1/scan-plate-azure", methods=["POST"])
def scan_plate_azure():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()

    # Extract text using Azure Vision
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
            return jsonify({
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
            }), 200

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