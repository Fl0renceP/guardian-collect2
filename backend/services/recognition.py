import os
import tempfile
import time
import cv2
import psycopg2
import numpy as np
from deepface import DeepFace

from services.blob_storage import BlobStorageService
from services.face_quality import assess as assess_quality

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

# Hard cap on how many faces a single scan will resolve, largest first.
# Every extra face is two more pgvector round trips, and the faces past the
# nearest few in a live frame are usually too small to identify anyway — a shot
# with thirty faces in it is a crowd photo, not a scan. Bounding this keeps the
# worst-case latency of the live feed predictable.
MAX_FACES_PER_SCAN = 8

_blob_service = None


def warm_recognition_pipeline(model_name="Facenet512"):
    """Warm the DeepFace model so the first live scan avoids model boot latency."""
    started = time.perf_counter()
    DeepFace.build_model(model_name)
    return round((time.perf_counter() - started) * 1000, 2)


def _face_area(face):
    """Pixel area of a detected face, used to order the faces in a group shot."""
    area = face.get("facial_area") or {}
    return area.get("w", 0) * area.get("h", 0)


def _significance(face):
    """Sort key for choosing which face the top-level response describes.

    The largest face is the obvious subject of a *deliberate* scan, but on a live
    feed the face that matters is the flagged one: an offender standing three
    metres behind the person at the camera is exactly the detection this system
    exists to make. If the top-level fields always described the biggest face,
    every existing consumer of them would report "no match" while the alert sat
    unread in faces[3]. So rank flagged first, then known, then by size.

    The full per-face list is returned regardless, in size order — nothing is
    hidden by this choice, it only decides which face gets the legacy fields.
    """
    if face.get("alert"):
        tier = 2
    elif face.get("is_known_user"):
        tier = 1
    else:
        tier = 0
    return (tier, face.get("face_area") or 0)


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


# Compare against EVERY stored photo of every person and keep the single
# closest. A person enrolled with several photos is recognised if any one of
# them is close enough, so one bad angle no longer defines an identity.
#
# The UNION also folds in persons.face_embedding, the older one-vector-per-
# person column. That keeps anyone whose photo rows have no embedding yet
# visible to the search instead of silently dropping out of the gallery.
# use_for_matching excludes captures kept purely as case evidence — a blurry
# 40px CCTV grab is real evidence and a terrible reference.
#
# Collapsing to one row per PERSON (not per capture) matters: the runner-up has
# to be a different human for the margin below to mean anything. Top 2 people,
# so the caller can see how decisive the win was.
_NEAREST_PERSONS_SQL = """
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

_SUPPORTING_CAPTURES_SQL = """
    SELECT id, image_url, source, camera_id, captured_at, incident_ref,
           quality_score, (embedding <=> %(vec)s::vector) AS distance
    FROM person_faces
    WHERE person_id = %(pid)s
      AND embedding IS NOT NULL
      AND use_for_matching
    ORDER BY distance ASC;
"""


def _accumulate(timings_ms, key, started):
    """Add one stage's elapsed ms onto a running per-frame total.

    Stages are additive here rather than overwritten because a frame with four
    faces runs the query stages four times, and what an operator wants to see in
    the timing breakdown is what the whole frame cost.
    """
    elapsed = (time.perf_counter() - started) * 1000
    timings_ms[key] = round(timings_ms.get(key, 0.0) + elapsed, 2)


def _match_one_face(cursor, query_vector, threshold, timings_ms):
    """Resolve one face embedding against the identity registry.

    Returns only the fields that describe *that* face — identity, distance,
    evidence. It knows nothing about how many other faces shared the frame;
    ordering, primary selection and frame-wide totals are the caller's job.
    """
    # Convert list to string format expected by pgvector '[x1, x2, ...]'
    vector_str = str(query_vector)

    query_started = time.perf_counter()
    cursor.execute(_NEAREST_PERSONS_SQL, {"vec": vector_str})
    ranked = cursor.fetchall()
    _accumulate(timings_ms, "query_nearest", query_started)
    result = ranked[0] if ranked else None

    # Distance to the nearest DIFFERENT person. A tiny margin means the face sat
    # almost equally close to two people, which is worth showing even when the
    # winner is under threshold.
    margin_to_next = None
    next_person = None
    if len(ranked) > 1:
        margin_to_next = round(float(ranked[1][3]) - float(ranked[0][3]), 4)
        next_person = {"full_name": ranked[1][1], "status": ranked[1][2],
                       "distance": round(float(ranked[1][3]), 4)}

    # Check if we found a match within threshold
    if result and result[3] < threshold:
        person_id, full_name, status, distance, image_url = result

        # Determine alert condition
        is_flagged = status in ["offender", "suspect"]

        # Pull every capture of the matched person with its own distance. The
        # answer is one person, but a security decision needs the evidence
        # behind it: which sightings agreed, from which camera, and how well.
        support_started = time.perf_counter()
        cursor.execute(_SUPPORTING_CAPTURES_SQL, {"vec": vector_str, "pid": person_id})
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
        _accumulate(timings_ms, "query_supporting", support_started)
        agreeing = sum(1 for c in captures if c["agrees"])

        return {
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
            "message": f"ALERT: {status.upper()} DETECTED!" if is_flagged else f"Member '{full_name}' is verified.",
        }

    # NO MATCH -> report as unknown. Nothing is written to the registry.
    # The nearest distance is still returned: it's the only way to tell a
    # complete stranger from someone who just missed the threshold.
    return {
        "is_known_user": False,
        "alert": False,
        "status": None,
        "person": None,
        "match_distance": round(result[3], 4) if result else None,
        "nearest_person": {"full_name": result[1], "status": result[2]} if result else None,
        "registered": False,
        "message": "Unknown face — not in the registry. Not added; enrolment is a separate admin action.",
    }


def process_incoming_face_image(image_bytes, db_conn=None, model_name="Facenet",
                                threshold=MATCH_THRESHOLD, max_faces=MAX_FACES_PER_SCAN):
    """
    Core facial recognition logic:
    1. Extracts an embedding for EVERY face in the incoming image.
    2. Queries pgvector for the nearest match to each one, independently.
    3. Returns one entry per face in `faces`, each with its own identity,
       distance, box and quality — plus the alert status for that person.
    4. Mirrors the most significant face into the top-level fields, so callers
       written against the single-face response keep working (see _significance).

    Resolving every face rather than only the largest is what makes this usable
    on a live feed: people arrive in groups, and the person a camera most needs
    to identify is rarely the one who walked closest to it. `faces` is ordered
    largest-first and capped at `max_faces` — `faces_truncated` says how many
    were dropped, so a full frame never silently looks like an empty one.

    An unrecognised face is deliberately NOT written to the identity registry.
    Auto-labelling strangers as 'verified' is a rejected design (PROJECT_CONTEXT.md
    section 5): one bad-angle photo of an offender would whitelist them for good,
    and it quietly builds a biometric record of every passer-by. Enrolment belongs
    in a separate, deliberate admin action.
    """
    timings_ms = {}

    # 1. Save uploaded image bytes to a temporary file for DeepFace processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        embed_started = time.perf_counter()
        # 2. Extract a 512-dim vector embedding per face (Facenet512 by default).
        # DeepFace.represent already returns one entry per detected face; the old
        # code threw all but one away.
        embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=model_name,
            detector_backend="retinaface",
            enforce_detection=True  # Fails cleanly if no face is detected
        )
        _accumulate(timings_ms, "embedding", embed_started)

        # Largest first: with a cap in play, the faces worth spending queries on
        # are the ones with enough pixels to identify. This is also the order the
        # UI draws and lists them in.
        ordered = sorted(embeddings, key=_face_area, reverse=True)
        faces_detected = len(ordered)
        faces_truncated = max(0, faces_detected - max_faces)
        ordered = ordered[:max_faces]

        # Advisory only — a scan is NEVER refused for poor quality. You always
        # want to try to identify whoever is in front of the camera, however bad
        # the frame. The score tells the operator how much to trust the answer,
        # which is a different question from whether the image is fit to become a
        # stored reference (that gate lives in enrolment).
        quality_started = time.perf_counter()
        probe_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        # The client scales the frame down before upload, so every box below is
        # in the uploaded image's pixel space. Report those dimensions or the
        # caller has no way to map a box back onto its own video element.
        image_size = (
            {"width": int(probe_image.shape[1]), "height": int(probe_image.shape[0])}
            if probe_image is not None
            else None
        )
        qualities = [
            assess_quality(probe_image, face.get("facial_area"), face.get("face_confidence"))
            if probe_image is not None
            else None
            for face in ordered
        ]
        _accumulate(timings_ms, "quality", quality_started)

    except Exception as e:
        os.remove(tmp_path)
        return {
            "success": False,
            "error": "No face detected in the image.",
            "details": str(e)
        }

    # Clean up temp file
    os.remove(tmp_path)

    # 3. Query PostgreSQL using pgvector's cosine distance operator (<=>), once
    # per face, over a single shared connection and cursor.
    conn = db_conn or psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        faces = []
        for index, (face, quality) in enumerate(zip(ordered, qualities)):
            entry = {
                "index": index,
                "facial_area": face.get("facial_area"),
                "face_area": _face_area(face),
                "face_confidence": face.get("face_confidence"),
                "capture_quality": quality,
            }
            entry.update(_match_one_face(cursor, face["embedding"], threshold, timings_ms))
            faces.append(entry)

        # 4. Pick the face the top-level fields describe, and say so explicitly
        # in the payload so a multi-face-aware caller never has to guess.
        primary = max(faces, key=_significance) if faces else None
        for entry in faces:
            entry["is_primary"] = entry is primary

        flagged = [f for f in faces if f["alert"]]
        known = [f for f in faces if f["is_known_user"]]

        response = {
            "success": True,
            "faces": faces,
            "faces_detected": faces_detected,
            "faces_resolved": len(faces),
            "faces_truncated": faces_truncated,
            "faces_known": len(known),
            "faces_flagged": len(flagged),
            # Any flagged face anywhere in the frame. Distinct from the
            # top-level `alert`, which belongs to the primary face only — though
            # _significance makes the primary a flagged face whenever one exists,
            # so in practice these agree. Kept separate because a caller reading
            # "is anyone in this frame flagged?" shouldn't depend on that.
            "any_alert": bool(flagged),
            "image_size": image_size,
            "timings_ms": timings_ms,
        }

        if primary:
            # Legacy single-face shape, mirrored from the primary face. Every
            # key the pre-multi-face response had still means what it meant.
            response.update({k: v for k, v in primary.items()
                             if k not in {"index", "face_area", "is_primary"}})
            response["primary_face_index"] = primary["index"]
            if len(faces) > 1:
                others = len(faces) - 1
                response["message"] = (
                    f"{response['message']} "
                    f"({others} other face{'s' if others != 1 else ''} in frame.)"
                )

        return response

    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        cursor.close()
        if db_conn is None:
            conn.close()
