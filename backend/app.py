"""Guardian Collective backend — Flask app factory.

MERGE NOTE (resolved by hand): this file had a structural conflict between the
app-factory branch and the plate-recognition branch — one used `create_app()`,
the other declared routes against a module-level `app`. Both sides are kept:
the factory pattern stays (PROJECT_CONTEXT §9 documents it as the convention)
and every route from the plate branch now lives inside it, logic unchanged.

Optional subsystems are imported lazily and degrade to a 503 rather than
stopping the server from booting:

  * psycopg2 / DeepFace / plate recognition — a ~600MB ML install that only the
    biometric endpoints need. The hot-spot map, claims, routing, alerts, patrol
    and safety score use none of it.
  * Azure AI Vision — needs AZURE_VISION_ENDPOINT and AZURE_VISION_KEY. Built on
    first use; constructing the client at import time with an unset key raised
    and took the whole API down with it.

The rule: a missing optional dependency disables its own endpoint, never the app.
"""

import logging
import os
import re

from flask import Flask, jsonify, request, send_from_directory

from config import Config
from routes.claim_routes import claim_bp
from routes.cpu_routes import cpu_bp
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from routes.member_score_routes import member_score_bp
from routes.route_routes import route_bp
from routes.safety_routes import safety_bp
from routes.user_routes import user_bp
from services.claims_service import warm_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy in (
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Optional subsystems
# ---------------------------------------------------------------------------

class SubsystemUnavailable(RuntimeError):
    """An optional dependency isn't installed or configured."""


def get_db_connection():
    """PostgreSQL connection for the biometric registries (faces, plates)."""
    try:
        import psycopg2
    except ImportError as exc:
        raise SubsystemUnavailable(
            "psycopg2 is not installed — install requirements.txt to enable "
            "the biometric endpoints."
        ) from exc
    return psycopg2.connect(Config.DATABASE_URL)


_vision_client = None


def get_vision_client():
    """Azure AI Vision client, built on first use.

    Returns None when the SDK isn't installed or the credentials aren't set, so
    the caller can answer 503 instead of the server failing to start.
    """
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    endpoint = os.environ.get("AZURE_VISION_ENDPOINT")
    key = os.environ.get("AZURE_VISION_KEY")
    if not endpoint or not key:
        logger.warning("Azure AI Vision not configured — /scan-plate-azure disabled.")
        return None

    try:
        from azure.ai.vision.imageanalysis import ImageAnalysisClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError as exc:
        logger.warning("azure-ai-vision-imageanalysis not installed: %s", exc)
        return None

    _vision_client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    return _vision_client


def extract_text_from_bytes(image_bytes: bytes) -> str:
    """Send image bytes to Azure AI Vision Read OCR and return the extracted text."""
    client = get_vision_client()
    if client is None:
        raise SubsystemUnavailable(
            "Azure AI Vision is not configured. Set AZURE_VISION_ENDPOINT and "
            "AZURE_VISION_KEY, and install azure-ai-vision-imageanalysis."
        )

    try:
        from azure.ai.vision.imageanalysis.models import VisualFeatures

        result = client.analyze(image_data=image_bytes, visual_features=[VisualFeatures.READ])

        extracted_words = []
        if result.read is not None:
            for block in result.read.blocks:
                for line in block.lines:
                    extracted_words.append(line.text)

        return " ".join(extracted_words)

    except SubsystemUnavailable:
        raise
    except Exception as e:
        logger.error(f"Azure OCR Error: {e}")
        return ""


def clean_plate_text(text: str) -> str:
    """Normalise plate text so punctuation variants still match the registry."""
    return re.sub(r"[^A-Za-z0-9]", "", text or "").upper()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config_object=Config):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config.from_object(config_object)

    app.register_blueprint(health_bp)
    app.register_blueprint(hotspot_bp)
    app.register_blueprint(claim_bp)
    app.register_blueprint(route_bp)
    app.register_blueprint(cpu_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(safety_bp)
    app.register_blueprint(member_score_bp)

    # Pull the claims collection on boot so the first visitor doesn't wait.
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm claims cache: {e}")

    # ---------------- pages ----------------

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/safety-score.html")
    def safety_score_page():
        return send_from_directory(app.static_folder, "safety-score.html")

    @app.route("/demos", methods=["GET"])
    def demos_page():
        """Consolidated entry-point linking the biometric demo pages."""
        return send_from_directory(app.static_folder, "demos.html")

    @app.route("/test-scan", methods=["GET"])
    def test_scan_page():
        """Minimal upload form for facial-recognition endpoint testing."""
        return send_from_directory(app.static_folder, "scan_test.html")

    @app.route("/test-plate", methods=["GET"])
    def test_plate_page():
        """Minimal upload form for licence-plate recognition endpoint testing."""
        return send_from_directory(app.static_folder, "plate_test.html")

    @app.route("/test-azure-plate", methods=["GET"])
    def test_plate_azure_page():
        """Upload form for Azure licence-plate OCR testing."""
        return send_from_directory(app.static_folder, "azure_plate_test.html")

    # ---------------- biometric endpoints ----------------

    def _uploaded_file():
        """Shared multipart validation. Returns (bytes, None) or (None, response)."""
        if "file" not in request.files:
            return None, (jsonify({"error": "No image file provided in 'file' field"}), 400)
        file = request.files["file"]
        if file.filename == "":
            return None, (jsonify({"error": "No selected file"}), 400)
        return file.read(), None

    @app.route("/api/v1/scan-face", methods=["POST"])
    def scan_face():
        """Facial recognition. Accepts a multipart file payload under 'file'."""
        image_bytes, error = _uploaded_file()
        if error:
            return error

        try:
            from services.recognition import process_incoming_face_image

            conn = get_db_connection()
        except (SubsystemUnavailable, ImportError) as e:
            logger.error(f"Facial recognition unavailable: {e}")
            return jsonify({"error": "Facial recognition is not available on this server.",
                            "detail": str(e)}), 503

        try:
            result = process_incoming_face_image(
                image_bytes=image_bytes,
                db_conn=conn,
                model_name=Config.FACE_MODEL,
                threshold=Config.MATCH_THRESHOLD,
            )
            # Services may return ({'error': ...}, status) tuples.
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
        """Licence plate recognition."""
        image_bytes, error = _uploaded_file()
        if error:
            return error

        try:
            from services.plate_recognition import process_incoming_plate_image

            conn = get_db_connection()
        except (SubsystemUnavailable, ImportError) as e:
            logger.error(f"Plate recognition unavailable: {e}")
            return jsonify({"error": "Plate recognition is not available on this server.",
                            "detail": str(e)}), 503

        try:
            result = process_incoming_plate_image(image_bytes=image_bytes, db_conn=conn)
            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Error processing plate scan: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500

        finally:
            conn.close()

    @app.route("/api/v1/scan-plate-azure", methods=["POST"])
    def scan_plate_azure():
        """Licence plate OCR via Azure AI Vision, matched against the registry."""
        image_bytes, error = _uploaded_file()
        if error:
            return error

        try:
            raw_text = extract_text_from_bytes(image_bytes)
        except SubsystemUnavailable as e:
            return jsonify({"error": "Azure licence-plate OCR is not available on this server.",
                            "detail": str(e)}), 503

        cleaned_plate = clean_plate_text(raw_text)
        if not cleaned_plate:
            return jsonify({"error": "No text detected in image"}), 400

        try:
            conn = get_db_connection()
        except SubsystemUnavailable as e:
            return jsonify({"error": "Plate registry is not available on this server.",
                            "detail": str(e)}), 503

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

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
