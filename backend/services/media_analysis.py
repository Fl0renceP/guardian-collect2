"""Analyse uploaded images and videos, and compare faces across them.

Three jobs:

1. An image: identify every face in it.
2. A video: sample frames across the clip and identify every face in each.
3. Across several files: work out who appears in more than one.

Point 3 is the part the identity registry cannot do on its own. Matching a face
against the gallery answers "is this a known offender". It cannot answer "is the
stranger in this video the same stranger as in that photo", because neither
sighting is enrolled. So unknown faces are also compared directly against each
other and grouped, which is what the spec means by comparing faces across
multiple different pictures or videos.

Videos are sampled, not decoded frame by frame. At ~1.4s per scan, a 30-second
clip at 25fps would be 750 frames and 17 minutes of work for maybe five distinct
seconds of content. Sampling once a second, with a motion gate to skip static
stretches, gets the same answer in a fraction of the time.
"""

import logging
import os
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np
import psycopg2

from config import Config
from services.recognition import process_incoming_face_image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

# How close two unknown faces must be to be called the same person. Deliberately
# STRICTER than MATCH_THRESHOLD: claiming two strangers are the same individual
# is a claim with no reference photo behind it and no human having confirmed it,
# so it should need more evidence than matching against a curated gallery entry.
UNKNOWN_CLUSTER_THRESHOLD = float(os.getenv("UNKNOWN_CLUSTER_THRESHOLD", "0.12"))

# Seconds between sampled video frames.
DEFAULT_SAMPLE_INTERVAL = float(os.getenv("VIDEO_SAMPLE_INTERVAL", "1.0"))

# Skip a sampled frame if it looks like the previous one. Same trick as the live
# page: compare 32x24 greyscale thumbnails.
MOTION_THRESHOLD = float(os.getenv("VIDEO_MOTION_THRESHOLD", "6.0"))

_jobs = {}
_jobs_lock = threading.Lock()


def media_type_for(filename):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def _cosine(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _scan(image_bytes, conn):
    return process_incoming_face_image(
        image_bytes=image_bytes,
        db_conn=conn,
        model_name=Config.FACE_MODEL,
        threshold=Config.MATCH_THRESHOLD,
        include_embeddings=True,
    )


def _collect(result, source, timestamp, people, unknowns):
    """Fold one frame's faces into the running per-file totals.

    A person seen in forty frames is one person with forty sightings, not forty
    detections — the caller wants "who was in this video", not a frame dump.
    """
    for face in result.get("faces") or []:
        if face.get("is_known_user") and face.get("person"):
            person = face["person"]
            entry = people.setdefault(person["id"], {
                "person_id": person["id"],
                "full_name": person["full_name"],
                "status": person["status"],
                "alert": bool(face.get("alert")),
                "sightings": 0,
                "best_distance": None,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "image_url": person.get("image_url"),
            })
            entry["sightings"] += 1
            entry["last_seen"] = timestamp
            distance = face.get("match_distance")
            if distance is not None and (entry["best_distance"] is None
                                         or distance < entry["best_distance"]):
                entry["best_distance"] = distance
        else:
            embedding = face.get("embedding")
            if embedding:
                unknowns.append({
                    "source": source,
                    "timestamp": timestamp,
                    "embedding": np.asarray(embedding, dtype=np.float64),
                    "quality": (face.get("capture_quality") or {}).get("quality_score"),
                    "bbox": face.get("bbox"),
                })


def analyse_image(image_bytes, filename, conn):
    people, unknowns = {}, []
    result = _scan(image_bytes, conn)
    if result.get("success"):
        _collect(result, filename, 0.0, people, unknowns)

    return {
        "filename": filename,
        "media_type": "image",
        "frames_sampled": 1,
        "frames_with_faces": 1 if result.get("faces") else 0,
        "faces_detected": result.get("faces_detected", 0),
        "people": sorted(people.values(), key=lambda p: p["best_distance"] or 1),
        "unknown_faces": unknowns,
        "error": None if result.get("success") else result.get("error"),
    }


def analyse_video(path, filename, conn, sample_interval=DEFAULT_SAMPLE_INTERVAL,
                  progress=None):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        return {
            "filename": filename, "media_type": "video", "error":
            "Could not open that video. The file may be corrupt, or use a codec "
            "OpenCV was not built with.",
            "people": [], "unknown_faces": [], "frames_sampled": 0,
        }

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0.0
    step = max(1, int(round(fps * sample_interval)))

    people, unknowns, timeline = {}, [], []
    sampled = with_faces = skipped_static = 0
    last_thumb = None
    index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step:
                index += 1
                continue

            timestamp = round(index / fps, 2) if fps else float(index)
            sampled += 1

            # Motion gate — a locked-off camera on an empty room produces hundreds
            # of identical frames, and scanning each costs a second for nothing.
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA).astype(np.int16)
            if last_thumb is not None and np.mean(np.abs(thumb - last_thumb)) < MOTION_THRESHOLD:
                skipped_static += 1
                last_thumb = thumb
                index += 1
                continue
            last_thumb = thumb

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                result = _scan(buffer.tobytes(), conn)
                if result.get("success") and result.get("faces"):
                    with_faces += 1
                    _collect(result, filename, timestamp, people, unknowns)
                    timeline.append({
                        "t": timestamp,
                        "faces": [
                            {
                                "full_name": (f.get("person") or {}).get("full_name"),
                                "status": f.get("status"),
                                "alert": f.get("alert"),
                                "distance": f.get("match_distance"),
                            }
                            for f in result["faces"]
                        ],
                    })

            if progress:
                progress(sampled, timestamp, duration)
            index += 1
    finally:
        capture.release()

    return {
        "filename": filename,
        "media_type": "video",
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "sample_interval": sample_interval,
        "frames_sampled": sampled,
        "frames_scanned": sampled - skipped_static,
        "frames_skipped_static": skipped_static,
        "frames_with_faces": with_faces,
        "people": sorted(people.values(), key=lambda p: p["best_distance"] or 1),
        "unknown_faces": unknowns,
        "timeline": timeline,
        "error": None,
    }


def cluster_unknowns(all_unknowns, threshold=UNKNOWN_CLUSTER_THRESHOLD):
    """Group unrecognised faces that are probably the same person.

    Greedy single-link clustering against each group's first member. Fine for the
    tens-of-faces scale here; a real system would use proper agglomerative
    clustering with a linkage criterion, because single-link can chain (A~B, B~C,
    therefore A~C) even when A and C are not actually alike.
    """
    groups = []
    for face in all_unknowns:
        placed = False
        for group in groups:
            if _cosine(group["_centroid"], face["embedding"]) < threshold:
                group["appearances"].append({
                    "source": face["source"],
                    "timestamp": face["timestamp"],
                    "quality": face["quality"],
                })
                group["_members"].append(face["embedding"])
                # Averaging as the group grows makes it less sensitive to whichever
                # face happened to arrive first.
                group["_centroid"] = np.mean(group["_members"], axis=0)
                placed = True
                break
        if not placed:
            groups.append({
                "_centroid": face["embedding"],
                "_members": [face["embedding"]],
                "appearances": [{
                    "source": face["source"],
                    "timestamp": face["timestamp"],
                    "quality": face["quality"],
                }],
            })

    out = []
    for i, group in enumerate(groups):
        sources = sorted({a["source"] for a in group["appearances"]})
        out.append({
            "group": f"Unknown person {chr(ord('A') + i)}" if i < 26 else f"Unknown person {i + 1}",
            "appearances": group["appearances"],
            "sources": sources,
            "sighting_count": len(group["appearances"]),
            "in_multiple_media": len(sources) > 1,
        })
    # Faces appearing across several files first — that is the interesting case.
    return sorted(out, key=lambda g: (not g["in_multiple_media"], -g["sighting_count"]))


def build_cross_media(analyses):
    """Who appears in more than one of the analysed files."""
    by_person = {}
    for analysis in analyses:
        for person in analysis.get("people", []):
            entry = by_person.setdefault(person["person_id"], {
                "person_id": person["person_id"],
                "full_name": person["full_name"],
                "status": person["status"],
                "alert": person["alert"],
                "sources": {},
            })
            entry["sources"][analysis["filename"]] = {
                "sightings": person["sightings"],
                "best_distance": person["best_distance"],
                "first_seen": person["first_seen"],
                "last_seen": person["last_seen"],
            }

    known = []
    for entry in by_person.values():
        entry["source_count"] = len(entry["sources"])
        entry["in_multiple_media"] = entry["source_count"] > 1
        known.append(entry)
    known.sort(key=lambda e: (not e["in_multiple_media"], -e["source_count"]))

    unknowns = [u for a in analyses for u in a.get("unknown_faces", [])]
    return {
        "known_people": known,
        "unknown_groups": cluster_unknowns(unknowns),
        "shared_known": [e for e in known if e["in_multiple_media"]],
    }


# -- job runner ------------------------------------------------------------
# Video analysis takes far longer than a request should, so uploads run on a
# background thread and the page polls. Jobs live in memory: this is a demo
# ingestion path, and a restart losing a job result is an acceptable trade for
# not standing up a queue.

def _run_job(job_id, files, sample_interval):
    conn = psycopg2.connect(Config.DATABASE_URL)
    try:
        analyses = []
        for position, (filename, path, kind) in enumerate(files):
            with _jobs_lock:
                _jobs[job_id]["current_file"] = filename
                _jobs[job_id]["files_done"] = position

            def progress(sampled, timestamp, duration):
                with _jobs_lock:
                    _jobs[job_id]["frames_done"] = sampled
                    _jobs[job_id]["position_seconds"] = timestamp
                    _jobs[job_id]["duration_seconds"] = duration

            try:
                if kind == "video":
                    analysis = analyse_video(path, filename, conn, sample_interval, progress)
                else:
                    with open(path, "rb") as handle:
                        analysis = analyse_image(handle.read(), filename, conn)
            except Exception as exc:
                logger.exception("Analysis failed for %s", filename)
                analysis = {"filename": filename, "media_type": kind, "error": str(exc),
                            "people": [], "unknown_faces": []}
            analyses.append(analysis)

        cross = build_cross_media(analyses)

        # Embeddings are working data, not output — 8KB of floats per face would
        # dwarf the actual answer.
        for analysis in analyses:
            analysis["unknown_face_count"] = len(analysis.get("unknown_faces", []))
            analysis.pop("unknown_faces", None)

        with _jobs_lock:
            _jobs[job_id].update({
                "state": "done",
                "files_done": len(files),
                "finished_at": time.time(),
                "result": {"files": analyses, "cross_media": cross},
            })
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        with _jobs_lock:
            _jobs[job_id].update({"state": "failed", "error": str(exc)})
    finally:
        conn.close()
        for _, path, _ in files:
            try:
                os.remove(path)
            except OSError:
                pass


def submit(uploads, sample_interval=DEFAULT_SAMPLE_INTERVAL):
    """uploads: list of (filename, bytes). Returns a job id to poll."""
    files = []
    for filename, data in uploads:
        kind = media_type_for(filename)
        if kind is None:
            raise ValueError(f"'{filename}' is not a supported image or video type.")
        suffix = os.path.splitext(filename)[1]
        handle, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        files.append((filename, path, kind))

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "state": "running",
            "started_at": time.time(),
            "files_total": len(files),
            "files_done": 0,
            "current_file": files[0][0] if files else None,
            "frames_done": 0,
            "position_seconds": 0,
            "duration_seconds": 0,
            "result": None,
        }
    threading.Thread(target=_run_job, args=(job_id, files, sample_interval), daemon=True).start()
    return job_id


def job(job_id):
    with _jobs_lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None
