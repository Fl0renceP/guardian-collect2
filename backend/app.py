import logging
import importlib
import os
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
from services.plate_recognition import process_incoming_plate_image
from services.recognition import process_incoming_face_image
from services import detection_log
from services import camera_poller
from services import media_analysis
from services import plate_ocr
from services import plate_video
from services import face_tracker

# Configure logging

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

    # A probable match still alerts — on a watchlist a likely offender is worth
    # a look — but the alert must not read like a certainty when the recogniser
    # has already said it is not one.
    if face.get("confidence") == "probable":
        detail += " Probable match, not confirmed — needs a human to check."
    if faces_in_frame > 1:
        # An operator triaging the feed needs to know this was a group, not a
        # one-to-one scan — it changes what they are looking at on the footage.
        detail += f" One of {faces_in_frame} faces in the frame."

    return alerts_service.record_detection(
        match_label=status,
        entity_type="person",
        title=f"{status.title()} identified by facial recognition",
        detail=detail,
        meta={
            "person_id": person.get("id"),
            "full_name": person.get("full_name"),
            "match_confidence": face.get("confidence"),
            "needs_review": bool(face.get("needs_review")),
            "faces_in_frame": faces_in_frame,
        },
    )


def _attach_face_alert(result):
    """Emit an alert per flagged identity in the frame, not just for one.

    The recogniser resolves every face, but the top-level fields describe only
    the primary one. Alerting off those alone meant a frame holding a verified
    resident and a flagged stranger raised an alert about the resident's match
    and none at all about the stranger — identified, then silently dropped.

    `alert_event` stays singular and points at the primary face's event, so
    existing readers of that key are unaffected.
    """
    faces = result.get("faces")
    if not isinstance(faces, list) or not faces:
        # Error payload, or a caller still on the single-face shape: treat the
        # top level as the one and only face.
        faces = [result] if result.get("is_known_user") else []

    faces_in_frame = len(faces)
    primary_index = result.get("primary_face_index")

    events = []
    primary_event = None
    for face in faces:
        event = _face_detection_alert(face, faces_in_frame=faces_in_frame)
        if not event:
            continue
        events.append(event)
        if face.get("index") == primary_index or primary_event is None:
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

    @app.route("/live", methods=["GET"])
    def live_page():
        """Continuous scanning from a camera — the no-manual-upload path.

        Browsers only grant camera access on https:// or localhost, so open this
        as http://localhost:5000/live, not by LAN IP.
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

        # Where the check came from. Defaults suit the file-upload demo; a real
        # feed would send its own camera id and position.
        camera_id = request.form.get("camera_id", "demo_upload").strip() or "demo_upload"

        def _coord(name):
            raw = request.form.get(name, "").strip()
            try:
                return float(raw) if raw else None
            except ValueError:
                return None

        # Client-side blink check (build guide §5). Tri-state on purpose: a
        # caller that says nothing is recorded as None ("not checked"), which
        # is a different fact from False ("checked, no blink seen") and must
        # not be silently collapsed into it.
        _liveness_raw = request.form.get("liveness_confirmed", "").strip().lower()
        liveness_confirmed = (
            True if _liveness_raw in ("1", "true", "yes")
            else False if _liveness_raw in ("0", "false", "no")
            else None
        )

        request_started = time.perf_counter()
        logger.info(
            "scan-face request received: filename=%s bytes=%s remote=%s",
            file.filename,
            len(image_bytes),
            request.remote_addr,
        )
        # One acquisition only. This used to take a connection here AND again
        # below, orphaning the first: never released, never closed. At
        # DB_POOL_MAX_CONN=8 the pool was dry after eight scans, and every
        # DB-backed endpoint then failed with "connection pool exhausted" —
        # including ones that had nothing to do with faces.
        try:
            conn = get_db_connection()
        except OperationalError as e:
            logger.warning("scan-face DB unavailable: %s", e)
            return jsonify({"error": "Face registry temporarily unavailable.", "code": "db_unavailable"}), 503

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
                liveness_confirmed=liveness_confirmed,
            )
            if detection_ids:
                # One id per identified face, in the same order as result["faces"].
                result["detection_ids"] = detection_ids
                result["detection_id"] = detection_ids[0]

            result = _attach_face_alert(result)
            total_ms = round((time.perf_counter() - request_started) * 1000, 2)
            result["request_timing_ms"] = total_ms
            summary = result.get("summary") or {}
            logger.info(
                "scan-face result: success=%s faces=%s known=%s alerts=%s raised=%s "
                "primary_status=%s primary_distance=%s total_ms=%s stage_ms=%s",
                result.get("success"),
                summary.get("total"),
                summary.get("known"),
                summary.get("alerts"),
                len(result.get("alerts") or []),
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
        """
        Licence plate recognition via Azure AI Vision Read.

        The OCR, plate-shape filtering and registry lookup live in
        services/plate_ocr.py. An earlier version of this route flattened every
        line of text in the image into one string and stripped the punctuation,
        which turned a photo containing a dealer sticker and a plate into a single
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
            if result.get("success"):
                result = _attach_plate_alert(
                    result, source_endpoint="/api/v1/scan-plate-azure"
                )
                return jsonify(result), 200
            return jsonify(result), 502
        except Exception as e:
            logger.error(f"Error processing Azure plate scan: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500
        finally:
            # release, not close: conn came from the pool, and closing it drains
            # one slot permanently instead of handing it back.
            release_db_connection(conn)

    @app.route("/api/v1/scan-plate-live", methods=["POST"])
    def scan_plate_live():
        """One frame of a live video stream, read locally through EasyOCR.

        Separate from /api/v1/scan-plate rather than a flag on it, because the
        two have opposite priorities. A single uploaded photo should try its
        hardest on the one image it has. A video frame is disposable — the next
        one arrives in a moment — so it locates the plate before reading, and
        withholds judgement until several frames agree.

        stream_id partitions the frame-to-frame vote, so two cameras (or two
        operators) scanning at once do not pool their readings into one verdict.
        """
        if "file" not in request.files:
            return jsonify({"error": "No image file provided in 'file' field"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        stream_id = request.form.get("stream_id") or "default"
        image_bytes = file.read()

        conn = get_db_connection()
        try:
            result = plate_video.read_frame(
                image_bytes, db_conn=conn, stream_id=stream_id
            )
            if not result.get("success"):
                return jsonify(result), 502

            # Only on the frame where the vote lands. Without this the operator
            # gets an alert per frame for as long as the car sits in view.
            if result.get("alert") and result.get("stability", {}).get("newly_confirmed"):
                result = _attach_plate_alert(
                    result, source_endpoint="/api/v1/scan-plate-live"
                )
            else:
                result["alert_event"] = None
                result["alerts"] = []
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error processing live plate frame: {e}")
            return jsonify({"error": "Internal processing error during plate scan"}), 500
        finally:
            release_db_connection(conn)

    @app.route("/api/v1/scan-plate-live/reset", methods=["POST"])
    def scan_plate_live_reset():
        """Forget a stream's vote history, so a new session starts clean."""
        stream_id = (request.get_json(silent=True) or {}).get("stream_id") or "default"
        plate_video.reset_stream(stream_id)
        return jsonify({"success": True, "stream_id": stream_id}), 200

    @app.route("/media", methods=["GET"])
    def media_page():
        """Upload images and videos, identify everyone, compare across files."""
        return send_from_directory(app.static_folder, "media.html")

    @app.route("/api/v1/media/analyse", methods=["POST"])
    def media_analyse():
        """
        Analyse one or more uploaded images/videos.

        Runs on a background thread and returns a job id — a minute of video is
        minutes of scanning, which is far longer than a request should be held
        open. Poll /api/v1/media/job/<id>.
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
        every check is here, while only people someone chose to enrol are in
        persons.

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
        import time as _time
        _started = _time.perf_counter()
        warm_face_model()
        logger.info("Face model warmed in %.1fs — first scan will be fast.",
                    _time.perf_counter() - _started)
    except Exception as e:
        logger.warning(f"Could not warm the face model (first scan will be slow): {e}")

    try:
        warm_ms = warm_recognition_pipeline(Config.FACE_MODEL)
        logger.info("Warm recognition pipeline completed in %sms", warm_ms)
    except Exception as e:
        logger.warning("Failed to warm recognition pipeline: %s", e)

    try:
        warm_ms = plate_video.warm_plate_reader()
        logger.info("EasyOCR plate reader warmed in %sms.", warm_ms)
    except Exception as e:
        logger.warning("Could not warm the plate reader (first frame will be slow): %s", e)

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
