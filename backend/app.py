import os
import logging
import psycopg2
from flask import Flask, request, jsonify, send_from_directory

from config import Config
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from services.claims_service import warm_cache
from services.plate_recognition import process_incoming_plate_image  
from services.recognition import process_incoming_face_image
from services import detection_log
from services import camera_poller
from services import media_analysis
from services import plate_ocr
from services import face_tracker

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

@app.route("/test-plate", methods=["GET"])
def test_plate_page():
    """Serves a minimal upload form for license plate recognition endpoint testing."""
    return send_from_directory(app.static_folder, "plate_test.html")

@app.route("/test-azure-plate", methods=["GET"])
def test_plate_azure_page():
    """Serves upload form for Azure license plate OCR testing."""
    return send_from_directory(app.static_folder, "azure_plate_test.html")


@app.route("/live", methods=["GET"])
def live_page():
    """Continuous scanning from a camera — the no-manual-upload path.

    Browsers only grant camera access on https:// or localhost, so open this as
    http://localhost:5000/live, not by LAN IP.
    """
    return send_from_directory(app.static_folder, "live.html")


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
            probable_threshold=Config.PROBABLE_THRESHOLD,
            # Needed to associate faces across frames; stripped before responding.
            include_embeddings=True,
        )

        # Handle tuple responses from service (e.g. ({'error': 'No face'}, 400))
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]

        # Track-level decisioning. Frames from one camera arrive as separate
        # requests, so the tracker holds state per camera_id between them.
        # A per-frame answer is still returned — this adds the aggregated view
        # rather than replacing it, so a caller sending unrelated stills is
        # unaffected.
        try:
            result["track_decisions"] = face_tracker.registry.update(
                camera_id, result.get("faces") or [],
                Config.MATCH_THRESHOLD, Config.PROBABLE_THRESHOLD,
            )
        except Exception as e:
            logger.warning(f"Tracking failed (per-frame result unaffected): {e}")
            result["track_decisions"] = []

        # 512 floats per face is ~8KB of JSON no client needs.
        for face in result.get("faces") or []:
            face.pop("embedding", None)

        # Every check is logged, matched or not — that is the point of the
        # detections table. record() swallows its own failures, so a logging
        # problem can never stop an alert reaching the operator.
        detection_ids = detection_log.record(
            conn, result,
            threshold=Config.MATCH_THRESHOLD,
            camera_id=camera_id,
            lat=_coord("location_lat"),
            lng=_coord("location_lng"),
        )
        if detection_ids:
            # One id per identified face, in the same order as result["faces"].
            result["detection_ids"] = detection_ids
            result["detection_id"] = detection_ids[0]

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

@app.route("/api/v1/scan-plate-azure", methods=["POST"])
def scan_plate_azure():
    """
    Licence plate recognition via Azure AI Vision Read.

    The OCR, plate-shape filtering and registry lookup live in
    services/plate_ocr.py. An earlier version of this route flattened every line
    of text in the image into one string and stripped the punctuation, which
    turned a photo containing a dealer sticker and a plate into a single
    meaningless run of characters that could never match anything.
    """
    if "file" not in request.files:
        return jsonify({"error": "No image file provided in 'file' field"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()

    conn = get_db_connection()
    try:
        result = plate_ocr.process_plate_image(image_bytes, db_conn=conn)
        return jsonify(result), 200 if result.get("success") else 502
    except Exception as e:
        logger.error(f"Error processing Azure plate scan: {e}")
        return jsonify({"error": "Internal processing error during plate scan"}), 500
    finally:
        conn.close()

@app.route("/media", methods=["GET"])
def media_page():
    """Upload images and videos, identify everyone, compare across files."""
    return send_from_directory(app.static_folder, "media.html")


@app.route("/api/v1/media/analyse", methods=["POST"])
def media_analyse():
    """
    Analyse one or more uploaded images/videos.

    Runs on a background thread and returns a job id — a minute of video is
    minutes of scanning, which is far longer than a request should be held open.
    Poll /api/v1/media/job/<id>.
    """
    files = request.files.getlist("files") or request.files.getlist("file")
    if not files:
        return jsonify({"error": "No files provided in 'files'"}), 400

    try:
        interval = float(request.form.get("sample_interval", 1.0))
    except ValueError:
        return jsonify({"error": "sample_interval must be a number"}), 400
    interval = min(max(interval, 0.2), 10.0)

    uploads = []
    for item in files:
        if not item.filename:
            continue
        uploads.append((item.filename, item.read()))
    if not uploads:
        return jsonify({"error": "No usable files provided"}), 400

    try:
        job_id = media_analysis.submit(uploads, sample_interval=interval)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"job_id": job_id, "files": len(uploads)}), 202


@app.route("/api/v1/media/job/<job_id>", methods=["GET"])
def media_job(job_id):
    state = media_analysis.job(job_id)
    if state is None:
        return jsonify({"error": "No such job"}), 404
    return jsonify(state), 200


@app.route("/api/v1/camera/test", methods=["POST"])
def camera_test():
    """One-shot fetch, so the UI can explain why a camera will not connect."""
    url = (request.json or {}).get("url", "").strip()
    try:
        return jsonify(camera_poller.test_connection(url)), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Could not reach the camera: {e}",
            "hint": "Check the phone and this machine are on the same network. "
                    "Guest and conference wifi usually block device-to-device "
                    "traffic — a phone hotspot avoids that.",
        }), 502


@app.route("/api/v1/camera/start", methods=["POST"])
def camera_start():
    """Begin pulling frames from a network camera. Nobody uploads anything."""
    body = request.json or {}
    try:
        status = camera_poller.start(
            url=(body.get("url") or "").strip(),
            camera_id=(body.get("camera_id") or "IPCAM-01").strip(),
            interval=float(body.get("interval", 1.5)),
            motion_threshold=float(body.get("motion_threshold", 25.0)),
        )
        return jsonify(status), 200
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/v1/camera/stop", methods=["POST"])
def camera_stop():
    return jsonify(camera_poller.stop()), 200


@app.route("/api/v1/camera/status", methods=["GET"])
def camera_status():
    return jsonify(camera_poller.status()), 200


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


def warm_face_model():
    """Load the detector and embedding model before the first real request.

    Facenet512 takes 20-40s to load, and it loads lazily on first use. Without
    this, the first scan after every restart blocks for that long: pressing
    "Start ingest" produced nothing for half a minute and looked broken, when it
    was only loading. Paying the cost at boot moves the wait somewhere nobody is
    watching a live camera.

    A synthetic grey square is enough — the models load on the call, whether or
    not it finds a face.
    """
    import numpy as _np
    from services import face_geometry as _fg
    from services import recognition as _rec

    blank = _np.full((160, 160, 3), 128, dtype=_np.uint8)
    _fg.detect_faces(blank)                    # loads the YuNet detector
    _rec.DeepFace.represent(                   # loads Facenet512
        img_path=blank, model_name=Config.FACE_MODEL,
        detector_backend="skip", enforce_detection=False, align=False,
    )


if __name__ == "__main__":
    # Warm up claims cache on application boot
    try:
        warm_cache()
        logger.info("Successfully warmed claims cache.")
    except Exception as e:
        logger.warning(f"Failed to warm cache on startup: {e}")

    try:
        import time as _time
        _started = _time.perf_counter()
        warm_face_model()
        logger.info("Face model warmed in %.1fs — first scan will be fast.",
                    _time.perf_counter() - _started)
    except Exception as e:
        logger.warning(f"Could not warm the face model (first scan will be slow): {e}")

    # Run local dev server
    app.run(host="0.0.0.0", port=5000, debug=True)