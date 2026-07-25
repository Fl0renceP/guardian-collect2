"""Pull frames from a network camera and scan them, with no human involved.

This is the ingestion path proper. The browser page can only scan a camera the
laptop itself owns — drawing a cross-origin video into a canvas taints it and the
upload is blocked — so a remote camera has to be polled server-side. That is also
how real CCTV works: the camera exposes a URL, something pulls from it.

Works with any camera that serves a JPEG over HTTP. For the Android IP Webcam app
that is:

    http://<phone-ip>:8080/shot.jpg

A motion gate sits in front of the expensive work, same idea as the browser page:
consecutive frames are compared as 32x24 greyscale thumbnails, which costs
microseconds, and the ~1s face scan only runs when the scene actually changes. An
empty room costs nothing.
"""

import logging
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np
import psycopg2
import requests

from config import Config
from services import detection_log
from services.recognition import process_incoming_face_image

logger = logging.getLogger(__name__)

_THUMB = (32, 24)


class CameraPoller:
    """One camera, polled on a background thread."""

    def __init__(self, url, camera_id, interval=1.5, motion_threshold=25.0):
        self.url = url
        self.camera_id = camera_id
        self.interval = max(0.5, float(interval))
        self.motion_threshold = float(motion_threshold)

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.started_at = None
        self.frames = 0
        self.scans = 0
        self.alerts = 0
        self.errors = 0
        self.last_error = None
        self.last_result = None
        self.last_scan_at = None
        self._last_thumb = None

    # -- lifecycle ---------------------------------------------------------

    def start(self):
        if self._thread and self._thread.is_alive():
            return False
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def running(self):
        return bool(self._thread and self._thread.is_alive())

    # -- work --------------------------------------------------------------

    def _grab(self):
        response = requests.get(self.url, timeout=5)
        response.raise_for_status()
        return response.content

    def _motion_score(self, image_bytes):
        """Mean absolute difference against the previous frame, 0-255."""
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            return None
        thumb = cv2.resize(frame, _THUMB, interpolation=cv2.INTER_AREA).astype(np.int16)
        previous, self._last_thumb = self._last_thumb, thumb
        if previous is None:
            return 999.0  # always scan the first frame
        return float(np.mean(np.abs(thumb - previous)))

    def _scan(self, image_bytes):
        conn = psycopg2.connect(Config.DATABASE_URL)
        try:
            result = process_incoming_face_image(
                image_bytes=image_bytes,
                db_conn=conn,
                model_name=Config.FACE_MODEL,
                threshold=Config.MATCH_THRESHOLD,
                probable_threshold=Config.PROBABLE_THRESHOLD,
            )
            if isinstance(result, tuple):
                result = result[0]

            detection_log.record(
                conn, result,
                threshold=Config.MATCH_THRESHOLD,
                camera_id=self.camera_id,
            )
            return result
        finally:
            conn.close()

    def _run(self):
        logger.info("Camera %s polling %s every %.1fs", self.camera_id, self.url, self.interval)
        while not self._stop.is_set():
            cycle_started = time.time()
            try:
                image_bytes = self._grab()
                with self._lock:
                    self.frames += 1

                score = self._motion_score(image_bytes)
                if score is not None and score >= self.motion_threshold:
                    result = self._scan(image_bytes)
                    with self._lock:
                        self.scans += 1
                        self.last_scan_at = time.time()
                        if result.get("alert"):
                            self.alerts += 1
                        # Carry every face through, not just the leading one —
                        # a frame with two people is two identifications.
                        self.last_result = {
                            "success": bool(result.get("success")),
                            "alert": bool(result.get("alert")),
                            "message": result.get("message") or result.get("error"),
                            "faces_detected": result.get("faces_detected", 0),
                            "summary": result.get("summary"),
                            "faces": [
                                {
                                    "index": f.get("index"),
                                    "is_known_user": f.get("is_known_user"),
                                    "alert": f.get("alert"),
                                    "confidence": f.get("confidence"),
                                    "needs_review": f.get("needs_review"),
                                    "status": f.get("status"),
                                    "full_name": (f.get("person") or {}).get("full_name"),
                                    "match_distance": f.get("match_distance"),
                                    "bbox": f.get("bbox"),
                                }
                                for f in (result.get("faces") or [])
                            ],
                            "at": self.last_scan_at,
                        }
            except Exception as exc:
                with self._lock:
                    self.errors += 1
                    self.last_error = str(exc)[:200]
                logger.warning("Camera %s: %s", self.camera_id, exc)
                # Back off on a dead camera rather than hammering it.
                self._stop.wait(2.0)

            elapsed = time.time() - cycle_started
            self._stop.wait(max(0.0, self.interval - elapsed))

        logger.info("Camera %s stopped", self.camera_id)

    def status(self):
        with self._lock:
            return {
                "running": self.running,
                "url": self.url,
                "camera_id": self.camera_id,
                "interval": self.interval,
                "motion_threshold": self.motion_threshold,
                "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
                "frames": self.frames,
                "scans": self.scans,
                "alerts": self.alerts,
                "errors": self.errors,
                "last_error": self.last_error,
                "last_result": self.last_result,
            }


# One camera at a time keeps this honest about what it is: a demo ingestion path,
# not a fleet manager. Several cameras would want a worker per camera and a shared
# queue so they cannot starve each other of CPU.
_active = None
_active_lock = threading.Lock()


def validate_url(url):
    """Reject anything that is not a plain http(s) URL.

    The server fetches whatever it is given here, so this is the boundary where a
    typo — or something worse — turns into the backend making arbitrary outbound
    requests. Kept deliberately narrow.
    """
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Camera URL must start with http:// or https://")
    if not parsed.hostname:
        raise ValueError("Camera URL has no host")
    return url


def start(url, camera_id="IPCAM-01", interval=1.5, motion_threshold=25.0):
    global _active
    validate_url(url)
    with _active_lock:
        if _active and _active.running:
            raise RuntimeError(f"Camera '{_active.camera_id}' is already running — stop it first.")
        _active = CameraPoller(url, camera_id, interval, motion_threshold)
        _active.start()
        return _active.status()


def stop():
    global _active
    with _active_lock:
        if not _active:
            return {"running": False}
        _active.stop()
        status = _active.status()
        _active = None
        return status


def status():
    with _active_lock:
        return _active.status() if _active else {"running": False}


def test_connection(url):
    """One-shot fetch so the UI can say why a camera will not connect."""
    validate_url(url)
    started = time.perf_counter()
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    array = np.frombuffer(response.content, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("That URL returned data, but it is not a decodable image.")
    return {
        "ok": True,
        "width": frame.shape[1],
        "height": frame.shape[0],
        "bytes": len(response.content),
        "fetch_ms": round((time.perf_counter() - started) * 1000, 1),
    }
