import logging
import psycopg2
from flask import Flask, request, jsonify, send_from_directory

from config import Config
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from services.claims_service import warm_cache
from services.recognition import process_incoming_face_image

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


if __name__ == "__main__":
    # Warm up claims cache on application boot
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm cache on startup: {e}")

    # Run local dev server
    app.run(host="0.0.0.0", port=5000, debug=True)