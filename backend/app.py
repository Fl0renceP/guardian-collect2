import logging

from flask import Flask, jsonify, request, send_from_directory

from config import Config

from routes.claim_routes import claim_bp
from routes.cpu_routes import cpu_bp
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from routes.route_routes import route_bp
from routes.user_routes import user_bp
from routes.safety_routes import safety_bp
from routes.member_score_routes import member_score_bp

from services.claims_service import warm_cache

# NOTE: psycopg2 and services.recognition (-> DeepFace -> TensorFlow) are
# imported inside scan_face() rather than here on purpose. Imported at module
# level they made the whole API refuse to start without a ~600MB ML install —
# the hot-spot map, claims, routing and alerts all use none of it. The scan
# endpoint behaves identically; it just resolves its own dependencies on first
# call instead of at boot. Nothing in the recognition logic itself is changed.


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for noisy in (
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "urllib3",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)


def create_app(config_object=Config):
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static"
    )

    app.config.from_object(config_object)

    # Register all routes
    app.register_blueprint(health_bp)
    app.register_blueprint(hotspot_bp)
    app.register_blueprint(claim_bp)
    app.register_blueprint(route_bp)
    app.register_blueprint(cpu_bp)
    app.register_blueprint(user_bp)
    app.register_blueprint(safety_bp)
    app.register_blueprint(member_score_bp)

    # Warm claims cache
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm claims cache: {e}")

    @app.route("/")
    def index():
        return send_from_directory(
            app.static_folder,
            "index.html"
        )

    @app.route("/safety-score.html")
    def safety_score_page():
        return send_from_directory(
            app.static_folder,
            "safety-score.html"
        )

    @app.route("/api/v1/scan-face", methods=["POST"])
    def scan_face():
        """
        Facial recognition scanning endpoint.
        Accepts multipart file payload under 'file'.
        """

        if "file" not in request.files:
            return jsonify(
                {"error": "No image file provided in 'file' field"}
            ), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify(
                {"error": "No selected file"}
            ), 400

        # Deferred so the rest of the API can boot without the ML stack.
        try:
            import psycopg2
            from services.recognition import process_incoming_face_image
        except ImportError as e:
            logger.error(f"Facial recognition stack unavailable: {e}")
            return jsonify(
                {
                    "error": "Facial recognition is not available on this server.",
                    "detail": "Install the DeepFace/TensorFlow dependencies "
                              "from requirements.txt to enable it.",
                }
            ), 503

        image_bytes = file.read()

        conn = psycopg2.connect(Config.DATABASE_URL)

        try:
            result = process_incoming_face_image(
                image_bytes=image_bytes,
                db_conn=conn,
                model_name=Config.FACE_MODEL,
                threshold=Config.MATCH_THRESHOLD,
            )

            if isinstance(result, tuple):
                return jsonify(result[0]), result[1]

            return jsonify(result), 200

        except Exception as e:
            logger.error(
                f"Error processing facial scan: {e}"
            )
            return jsonify(
                {"error": "Internal processing error during scan"}
            ), 500

        finally:
            conn.close()

    return app


app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )