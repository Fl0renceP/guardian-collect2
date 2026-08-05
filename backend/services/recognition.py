import logging
import os
import tempfile
import time

# retina-face and deepface are written against the legacy tf.keras API. From
# TensorFlow 2.16 on, `tensorflow.keras` resolves to Keras 3 instead, and
# building the RetinaFace graph dies with "A KerasTensor cannot be used as
# input to a TensorFlow function". This flag points `tensorflow.keras` back at
# tf_keras (a requirements.txt dependency) and must be set before TensorFlow is
# imported — which the DeepFace import below does transitively.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import psycopg2
import numpy as np
from deepface import DeepFace

from config import Config
from services.blob_storage import BlobStorageService
from services.face_quality import assess as assess_quality

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Cosine distance threshold (pgvector `<=>`), NOT Euclidean. Facenet512 embeddings
# are unnormalised — measured norms are ~21-22 — so a raw L2 threshold has to be
# on the order of 23 to mean anything, and it drifts with embedding magnitude.
# Cosine is scale-invariant, so the number below means the same thing regardless.
#
# Measured on the seed photos: same person (re-encoded / dimmed / half-res) lands
# at 0.002-0.023, different people at 0.669-1.124. 0.30 is DeepFace's own tuned
# value for Facenet512 and sits in that gap with room on both sides.
#
# If you change the operator in the query below, change this to match.
MATCH_THRESHOLD = 0.30
_blob_service = None


def warm_recognition_pipeline(model_name="Facenet512", detector_backend=None):
    """Warm both DeepFace models so the first live scan avoids model boot latency.

    Warming the embedder alone left the DETECTOR to load on the first real scan,
    so that scan paid a model download/build that this function exists to absorb.
    A 1x1 pixel is enough to force the detector to instantiate; enforce_detection
    is off because there is obviously no face in it and we only want the load.
    """
    started = time.perf_counter()
    DeepFace.build_model(model_name)
    backends = [detector_backend or Config.FACE_DETECTOR]
    # The fallback too: it loads on the first frame that escalates to it, and
    # that frame is by definition an awkward one already paying the slow path.
    if Config.FACE_DETECTOR_FALLBACK and Config.FACE_DETECTOR_FALLBACK not in backends:
        backends.append(Config.FACE_DETECTOR_FALLBACK)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, np.zeros((1, 1, 3), dtype=np.uint8))
        for backend in backends:
            try:
                DeepFace.represent(
                    img_path=tmp_path,
                    model_name=model_name,
                    detector_backend=backend,
                    enforce_detection=False,
                )
            except Exception:
                # Warming is best-effort: a failure here costs latency on the first
                # scan, not correctness, and must never stop the app from booting.
                logger.info("Could not pre-warm detector '%s'.", backend)
    finally:
        os.remove(tmp_path)
    return round((time.perf_counter() - started) * 1000, 2)


def _face_area(face):
    """Pixel area of a detected face, used to pick the subject in a group shot."""
    area = face.get("facial_area") or {}
    return area.get("w", 0) * area.get("h", 0)


def represent_face(img_path, model_name=None, detector_backend=None, fallback_backend=None):
    """Embed the faces in an image, escalating detectors until one finds something.

    THE ONE PLACE any face is turned into a vector — the live scan, enrolment,
    the seed importer and the re-embed script all come through here. That is
    deliberate: the detector picks the crop and the crop moves the embedding, so
    a gallery built by one path and probed by another would be comparing subtly
    different things. Keeping the cascade in a single function means the two
    cannot drift apart without someone editing this line.

    Returns (objs, detector_used). Raises whatever the last detector raised if
    none of them find a face, so callers keep their existing failure handling.
    """
    model_name = model_name or Config.FACE_MODEL
    primary = detector_backend or Config.FACE_DETECTOR
    fallback = fallback_backend if fallback_backend is not None else Config.FACE_DETECTOR_FALLBACK

    chain = [primary] + ([fallback] if fallback and fallback != primary else [])
    last_error = None
    for backend in chain:
        try:
            objs = DeepFace.represent(
                img_path=img_path,
                model_name=model_name,
                detector_backend=backend,
                enforce_detection=True,  # Fails cleanly if no face is detected
            )
            if objs:
                return objs, backend
        except Exception as exc:
            last_error = exc
            if backend != chain[-1]:
                logger.info("Detector '%s' found no face; escalating.", backend)

    raise last_error or ValueError("No face detected.")


def _blob_signer():
    global _blob_service
    if _blob_service is None:
        try:
            _blob_service = BlobStorageService()
        except Exception:
            _blob_service = False
    return _blob_service or None


def _readable_face_url(url):
    signer = _blob_signer()
    if not url or signer is None:
        return url
    try:
        return signer.sign_stored_url(url)
    except Exception:
        return url


def process_incoming_face_image(image_bytes, db_conn=None, model_name="Facenet", threshold=MATCH_THRESHOLD,
                                detector_backend=None):
    """
    Core facial recognition logic:
    1. Extracts an embedding for the largest face in the incoming image.
    2. Queries pgvector for nearest match.
    3. If match found (< MATCH_THRESHOLD cosine distance): returns person details & alert status.
    4. If no match found: reports the face as unknown and stops there.

    An unrecognised face is deliberately NOT written to the identity registry.
    Auto-labelling strangers as 'verified' is a rejected design (PROJECT_CONTEXT.md
    section 5): one bad-angle photo of an offender would whitelist them for good,
    and it quietly builds a biometric record of every passer-by. Enrolment belongs
    in a separate, deliberate admin action.
    """
    timings_ms = {}
    # Must match the detector the gallery was embedded with — see Config.FACE_DETECTOR.
    detector_backend = detector_backend or Config.FACE_DETECTOR

    # 1. Save uploaded image bytes to a temporary file for DeepFace processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        embed_started = time.perf_counter()
        # 2. Extract the 512-dim vector embedding (Facenet512 by default)
        embeddings, detector_used = represent_face(
            tmp_path, model_name=model_name, detector_backend=detector_backend
        )
        timings_ms["embedding"] = round((time.perf_counter() - embed_started) * 1000, 2)
        # embeddings[0] is whichever face the detector happened to return first,
        # which in a group shot is arbitrary. Take the largest instead: the person
        # nearest the camera is the one being scanned.
        subject = max(embeddings, key=_face_area)
        query_vector = subject["embedding"]
        faces_detected = len(embeddings)
        face_confidence = subject.get("face_confidence")

        # Advisory only — a scan is NEVER refused for poor quality. You always
        # want to try to identify whoever is in front of the camera, however bad
        # the frame. The score tells the operator how much to trust the answer,
        # which is a different question from whether the image is fit to become a
        # stored reference (that gate lives in enrolment).
        quality_started = time.perf_counter()
        probe_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        capture_quality = (
            assess_quality(probe_image, subject.get("facial_area"), face_confidence)
            if probe_image is not None
            else None
        )
        timings_ms["quality"] = round((time.perf_counter() - quality_started) * 1000, 2)

    except Exception as e:
        os.remove(tmp_path)
        return {
            "success": False,
            "error": "No face detected in the image.",
            "details": str(e)
        }

    # Clean up temp file
    os.remove(tmp_path)

    # 3. Query PostgreSQL using pgvector's cosine distance operator (<=>)
    conn = db_conn or psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        query_started = time.perf_counter()
        # Convert list to string format expected by pgvector '[x1, x2, ...]'
        vector_str = str(query_vector)

        # Compare against EVERY stored photo of every person and keep the single
        # closest. A person enrolled with several photos is recognised if any one
        # of them is close enough, so one bad angle no longer defines an identity.
        #
        # The UNION also folds in persons.face_embedding, the older one-vector-per-
        # person column. That keeps anyone whose photo rows have no embedding yet
        # visible to the search instead of silently dropping out of the gallery.
        # use_for_matching excludes captures kept purely as case evidence — a
        # blurry 40px CCTV grab is real evidence and a terrible reference.
        #
        # Collapsing to one row per PERSON (not per capture) matters: the runner-up
        # has to be a different human for the margin below to mean anything.
        # Top 2 people, so the caller can see how decisive the win was.
        query = """
            WITH candidates AS (
                SELECT p.id, p.full_name, p.status,
                       (pf.embedding <=> %(vec)s::vector) AS distance,
                       pf.image_url
                FROM persons p
                JOIN person_faces pf ON pf.person_id = p.id
                WHERE pf.embedding IS NOT NULL
                  AND pf.use_for_matching

                UNION ALL

                SELECT p.id, p.full_name, p.status,
                       (p.face_embedding <=> %(vec)s::vector) AS distance,
                       NULL AS image_url
                FROM persons p
                WHERE p.face_embedding IS NOT NULL
            ),
            best_per_person AS (
                SELECT DISTINCT ON (id) id, full_name, status, distance, image_url
                FROM candidates
                ORDER BY id, distance ASC
            )
            SELECT id, full_name, status, distance, image_url
            FROM best_per_person
            ORDER BY distance ASC
            LIMIT 2;
        """
        cursor.execute(query, {"vec": vector_str})
        ranked = cursor.fetchall()
        timings_ms["query_nearest"] = round((time.perf_counter() - query_started) * 1000, 2)
        result = ranked[0] if ranked else None

        # Distance to the nearest DIFFERENT person. A tiny margin means the scan
        # sat almost equally close to two people, which is worth showing even when
        # the winner is under threshold.
        margin_to_next = None
        next_person = None
        if len(ranked) > 1:
            margin_to_next = round(float(ranked[1][3]) - float(ranked[0][3]), 4)
            next_person = {"full_name": ranked[1][1], "status": ranked[1][2],
                           "distance": round(float(ranked[1][3]), 4)}

        # 4. Check if we found a match within threshold
        if result and result[3] < threshold:
            person_id, full_name, status, distance, image_url = result

            # Determine alert condition
            is_flagged = status in ["offender", "suspect"]

            # Pull every capture of the matched person with its own distance. The
            # answer is one person, but a security decision needs the evidence
            # behind it: which sightings agreed, from which camera, and how well.
            cursor.execute(
                """
                SELECT id, image_url, source, camera_id, captured_at, incident_ref,
                       quality_score, (embedding <=> %(vec)s::vector) AS distance
                FROM person_faces
                WHERE person_id = %(pid)s
                  AND embedding IS NOT NULL
                  AND use_for_matching
                ORDER BY distance ASC;
                """,
                {"vec": vector_str, "pid": person_id},
            )
            support_started = time.perf_counter()
            captures = [
                {
                    "id": str(row[0]),
                    "image_url": _readable_face_url(row[1]),
                    "source": row[2],
                    "camera_id": row[3],
                    "captured_at": row[4].isoformat() if row[4] else None,
                    "incident_ref": row[5],
                    "quality_score": row[6],
                    "distance": round(float(row[7]), 4),
                    "agrees": float(row[7]) < threshold,
                }
                for row in cursor.fetchall()
            ]
            timings_ms["query_supporting"] = round((time.perf_counter() - support_started) * 1000, 2)
            agreeing = sum(1 for c in captures if c["agrees"])

            return {
                "success": True,
                "is_known_user": True,
                "alert": is_flagged,  # True for offender/suspect, False for verified
                "status": status,
                "person": {
                    "id": str(person_id),
                    "full_name": full_name,
                    "status": status,
                    "image_url": _readable_face_url(image_url)
                },
                "match_distance": round(distance, 4),
                "matched_against_photos": len(captures),
                "agreeing_captures": agreeing,
                "margin_to_next_person": margin_to_next,
                "next_closest_person": next_person,
                "supporting_captures": captures,
                "faces_detected": faces_detected,
                "face_confidence": face_confidence,
                "detector_used": detector_used,
                "capture_quality": capture_quality,
                "timings_ms": timings_ms,
                "message": f"ALERT: {status.upper()} DETECTED!" if is_flagged else f"Member '{full_name}' is verified."
            }

        # 5. NO MATCH -> report as unknown. Nothing is written to the registry.
        # The nearest distance is still returned: it's the only way to tell a
        # complete stranger from someone who just missed the threshold.
        return {
            "success": True,
            "is_known_user": False,
            "alert": False,
            "status": None,
            "person": None,
            "match_distance": round(result[3], 4) if result else None,
            "nearest_person": {"full_name": result[1], "status": result[2]} if result else None,
            "faces_detected": faces_detected,
            "face_confidence": face_confidence,
            "detector_used": detector_used,
            "capture_quality": capture_quality,
            "timings_ms": timings_ms,
            "registered": False,
            "message": "Unknown face — not in the registry. Not added; enrolment is a separate admin action."
        }

    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        cursor.close()
        if db_conn is None:
            conn.close()