import os
import tempfile
import time
import cv2
import psycopg2
import numpy as np
from deepface import DeepFace

from services.face_quality import assess as assess_quality
from services import face_geometry

DATABASE_URL = os.getenv("DATABASE_URL")

# Face detector. This was ~96% of scan latency: retinaface, the previous choice,
# took ~24s per scan on this hardware against yunet's ~0.77s — a 31x difference
# for no accuracy the threshold can see. Benchmarked over the seed probes, every
# backend still matched the correct person; yunet's worst was 0.0425 against a
# 0.30 cut-off, while retinaface managed 0.0161.
#
#   yunet 1.25s | ssd 1.43s | opencv 1.60s | mtcnn 3.61s | centerface 4.37s
#   retinaface 23.97s
#
# opencv is fast but crops loosely — its worst probe was 0.1008, ~4x looser than
# ssd — so speed alone is not the criterion.
#
# Of yunet's ~770ms, roughly 180ms finds the face (63ms on a 640px frame) and
# ~590ms embeds it. That split is why a live feed should detect on every frame
# but embed only when a face actually appears.
#
# Keep enroll_face.py and init_db.py on the same detector: different backends
# crop differently, so a mismatch leaves every stored reference slightly out of
# step with every scan. After changing this, run reembed_references.py.
FACE_DETECTOR = os.getenv("FACE_DETECTOR", "yunet")

# Alignment is OFF, which is not the usual advice. Two measured reasons:
#
# 1. It silently loses faces. In a two-person frame, represent() returned only
#    one face with alignment on and both with it off — extract_faces behaves the
#    same way, so it is the alignment step itself, not the detector. Padding the
#    image did not help. A group shot quietly becoming a single identification is
#    exactly what the multi-face requirement forbids.
#
# 2. It makes matching WORSE here, measured on the same references and probes:
#
#        align=True    worst genuine 0.0412   impostor 0.2960    7.2x
#        align=False   worst genuine 0.0204   impostor 0.3609   17.7x
#
#    Alignment rotates the crop using the detector's eye landmarks. yunet's
#    landmarks are coarse, so that rotation injects error rather than removing
#    it. With a landmark-strong detector like retinaface the trade-off may well
#    reverse — re-measure before turning this on.
#
# Changing this changes the embeddings, so run reembed_references.py after.
FACE_ALIGN = os.getenv("FACE_ALIGN", "false").strip().lower() in ("1", "true", "yes")

# Cosine distance threshold (pgvector `<=>`), NOT Euclidean. Facenet512 embeddings
# are unnormalised — measured norms are ~21-22 — so a raw L2 threshold has to be
# on the order of 23 to mean anything, and it drifts with embedding magnitude.
# Cosine is scale-invariant, so the number below means the same thing regardless.
#
# 0.30 was DeepFace's published value for Facenet512 and it is TOO LOOSE for this
# gallery. The one out-of-gallery face available for testing lands at 0.296-0.364
# depending on detector, i.e. essentially on top of 0.30 — under yunet it matched
# an offender outright, flagging an innocent person. Measured, references and
# probes embedded with the same detector throughout:
#
#   detector     worst genuine   nearest impostor   ratio
#   yunet               0.0412             0.2960    7.2x
#   ssd                 0.0535             0.3091    5.8x
#   mtcnn               0.0499             0.3129    6.3x
#   retinaface          0.0161             0.3635   22.5x
#
# 0.15 sits ~3.6x above the worst genuine match and ~2x below that impostor.
#
# THIS WINDOW IS NARROWER THAN IT LOOKS. Every genuine figure above comes from a
# synthetic degradation of the reference image — a re-encode, a tilt, a dimming.
# A genuinely different photograph of the same person (different day, clothes,
# angle) will land much further out, plausibly 0.15-0.25, which is uncomfortably
# close to that impostor at 0.296. Two things follow:
#
#   1. Test with real second photographs before trusting any value here.
#   2. Enrolling several varied references per person is the lever that widens
#      the window — it pulls the worst genuine distance down without moving
#      impostors.
#
# If you change the operator in the query below, change this to match.
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.15"))

# Second, looser band. Real cameras do not produce reference-quality images, and
# a system that only recognises people under ideal conditions is not much use on
# CCTV. Measured on this gallery, live captures of enrolled people came back at
# 0.1552, 0.1774 and 0.2023 — recognisably the right person, but outside the
# strict cut-off, so they were reported as strangers.
#
# So a match below MATCH_THRESHOLD is reported as "confirmed", and one between
# that and PROBABLE_THRESHOLD as "probable": named, alerted on, but explicitly
# marked as needing a human to verify.
#
# 0.25 is not arbitrary. It sits above the worst genuine live capture (0.2023)
# and below the nearest known impostor (0.2960) — a real person NOT in the
# registry. Widen it and that impostor starts being named as an offender.
# Everything above the band is still reported as unknown, with the distance, so
# nothing is hidden either way.
PROBABLE_THRESHOLD = float(os.getenv("PROBABLE_THRESHOLD", "0.25"))

# Each face costs its own ~590ms embedding, so a crowd scene is priced per head.
# Faces are processed largest-first, so this cap drops the most distant and least
# identifiable people rather than an arbitrary subset.
MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES_PER_FRAME", "10"))

# Which per-identity template drives the decision: "min", "centroid" or "both"
# (both = whichever is closer). All are computed and reported regardless; this
# only chooses what the match is judged on.
TEMPLATE_STRATEGY = os.getenv("TEMPLATE_STRATEGY", "min").strip().lower()

# Proper 5-point similarity alignment before embedding, via services.face_geometry.
#
# Not the same thing as DeepFace's align=, which rotates on two eye points and
# measurably hurt this gallery. This maps five landmarks onto the ArcFace
# canonical template, normalising rotation, scale and translation together.
#
# Measured: on unrotated degradations it costs a little (worst genuine 0.0204 ->
# 0.0311, resampling noise on faces that were already upright), but on a ROTATED
# probe — the case a live camera actually produces — it gained 39% (0.0928 ->
# 0.0565). Live captures here are re-photographed prints at an angle, so the
# rotated case is the representative one. Settle it properly with
# calibrate_threshold.py against real camera data.
#
# Changing this changes the embeddings: re-run reembed_references.py after.
USE_5POINT_ALIGN = os.getenv("USE_5POINT_ALIGN", "true").strip().lower() in ("1", "true", "yes")

# Refuse to identify a face the system cannot actually read, rather than naming
# somebody from an unreadable image. See services.face_geometry.assess_probe.
ENFORCE_PROBE_QUALITY = os.getenv("ENFORCE_PROBE_QUALITY", "true").strip().lower() in ("1", "true", "yes")

# Which face leads the response when several are present. An offender in the
# background matters more than a verified member in the foreground.
_SEVERITY = {"offender": 3, "suspect": 2, "verified": 1}


def warm_recognition_pipeline(model_name="Facenet512"):
    """Warm the DeepFace model so the first live scan avoids model boot latency."""
    started = time.perf_counter()
    DeepFace.build_model(model_name)
    return round((time.perf_counter() - started) * 1000, 2)


def _face_area(face):
    """Pixel area of a detected face — proxy for how close they are to the camera."""
    area = face.get("facial_area") or {}
    return area.get("w", 0) * area.get("h", 0)


def _plain_crop(image, bbox, size=160):
    """Bounding-box crop with no alignment, for USE_5POINT_ALIGN=false."""
    x, y, w, h = bbox
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


# Collapsing to one row per PERSON (not per capture) matters: the runner-up has
# to be a different human for the margin to mean anything.
#
# Matching reads person_faces only. An earlier version also UNIONed in
# persons.face_embedding as a safety net so nobody could drop out of the gallery,
# but that arm had no use_for_matching filter — the legacy column carries no
# quality metadata to filter on — so it silently readmitted vectors the quality
# gate had rejected. A safety net that bypasses the safety check is not one.
#
# use_for_matching excludes captures kept purely as case evidence: a blurry 40px
# CCTV grab is real evidence and a terrible reference.
# Both per-identity templates are computed every time, because they answer
# different questions and each is cheap once the rows are already scanned:
#
#   min      distance to the person's CLOSEST enrolled capture. Robust to a bad
#            enrolment — one poor photo cannot hide a good one — but a single
#            unlucky reference can also drag an impostor in.
#   centroid distance to the AVERAGE of their captures. Smooths per-photo noise
#            and is the more stable template as enrolments accumulate, but one
#            bad enrolment pollutes it permanently and it drifts if a person's
#            appearance genuinely changes.
#
# TEMPLATE_STRATEGY decides which drives the decision; both are always reported,
# so the calibration script can compare them without re-running anything.
_GALLERY_QUERY = """
    WITH per_capture AS (
        SELECT p.id, p.full_name, p.status,
               min(pf.embedding <=> %(vec)s::vector) AS min_distance,
               count(*) AS capture_count
        FROM persons p
        JOIN person_faces pf ON pf.person_id = p.id
        WHERE pf.embedding IS NOT NULL
          AND pf.use_for_matching
        GROUP BY p.id, p.full_name, p.status
    ),
    centroids AS (
        SELECT p.id,
               (avg(pf.embedding)::vector <=> %(vec)s::vector) AS centroid_distance
        FROM persons p
        JOIN person_faces pf ON pf.person_id = p.id
        WHERE pf.embedding IS NOT NULL
          AND pf.use_for_matching
        GROUP BY p.id
    )
    SELECT c.id, c.full_name, c.status,
           CASE %(strategy)s
               WHEN 'centroid' THEN t.centroid_distance
               WHEN 'both'     THEN least(c.min_distance, t.centroid_distance)
               ELSE c.min_distance
           END AS distance,
           c.min_distance,
           t.centroid_distance,
           c.capture_count
    FROM per_capture c
    JOIN centroids t ON t.id = c.id
    ORDER BY distance ASC
    LIMIT 2;
"""

_SUPPORTING_QUERY = """
    SELECT id, image_url, source, camera_id, captured_at, incident_ref,
           quality_score, (embedding <=> %(vec)s::vector) AS distance
    FROM person_faces
    WHERE person_id = %(pid)s
      AND embedding IS NOT NULL
      AND use_for_matching
    ORDER BY distance ASC;
"""


def embed_image(image, model_name=None):
    """Detect, align and embed the largest face — the single shared entry point.

    Enrolment, re-embedding and scanning all go through this, because a reference
    embedded by a different route than the probe is a reference that no longer
    lines up. Every previous accuracy problem in this project traced back to two
    code paths disagreeing about detection, alignment or threshold units.

    Returns (embedding, face, quality) or (None, None, None) when no face is found.
    """
    from config import Config as _Config  # local: avoids a circular import at module load

    faces = face_geometry.detect_faces(image)
    if not faces:
        return None, None, None

    face = faces[0]
    aligned = (face_geometry.align(image, face.landmarks, 160)
               if USE_5POINT_ALIGN else _plain_crop(image, face.bbox))
    if aligned is None:
        return None, None, None

    quality = face_geometry.assess_probe(image, face, aligned)
    embedded = DeepFace.represent(
        img_path=aligned, model_name=model_name or _Config.FACE_MODEL,
        detector_backend="skip", enforce_detection=False, align=False,
    )
    return embedded[0]["embedding"], face, quality


def _match_message(full_name, status, is_flagged, confirmed):
    """Wording that does not overstate a probable match as a certainty."""
    if is_flagged:
        if confirmed:
            return f"ALERT: {status.upper()} DETECTED!"
        return f"LIKELY {status.upper()}: {full_name} — verify before acting."
    if confirmed:
        return f"Member '{full_name}' is verified."
    return f"Probably '{full_name}' (verified member) — low-confidence match."


def _match_face(cursor, vector_str, threshold, probable_threshold=None):
    """Identify a single face embedding against the gallery.

    Returns the per-face verdict. Never raises for "no match" — an unknown face
    is a normal, expected outcome, not an error.

    Two bands: under `threshold` is a confirmed identification, between that and
    `probable_threshold` is a probable one that a human should confirm.
    """
    if probable_threshold is None:
        probable_threshold = max(threshold, PROBABLE_THRESHOLD)
    cursor.execute(_GALLERY_QUERY, {"vec": vector_str, "strategy": TEMPLATE_STRATEGY})
    ranked = cursor.fetchall()
    best = ranked[0] if ranked else None
    templates = None
    if best:
        templates = {
            "strategy": TEMPLATE_STRATEGY,
            "min_distance": round(float(best[4]), 4),
            "centroid_distance": round(float(best[5]), 4),
            "enrolled_captures": int(best[6]),
        }

    # Distance to the nearest DIFFERENT person. A tiny margin means the face sat
    # almost equally close to two people — worth surfacing even when the winner
    # is under threshold.
    margin_to_next = None
    next_person = None
    if len(ranked) > 1:
        margin_to_next = round(float(ranked[1][3]) - float(ranked[0][3]), 4)
        next_person = {"full_name": ranked[1][1], "status": ranked[1][2],
                       "distance": round(float(ranked[1][3]), 4)}

    if best and best[3] < probable_threshold:
        person_id, full_name, status, distance = best[0], best[1], best[2], best[3]
        is_flagged = status in ("offender", "suspect")
        confirmed = distance < threshold
        confidence = "confirmed" if confirmed else "probable"

        # The answer is one person, but a security decision needs the evidence:
        # which stored sightings agreed, from which camera, and how strongly.
        cursor.execute(_SUPPORTING_QUERY, {"vec": vector_str, "pid": person_id})
        captures = [
            {
                "id": str(row[0]),
                "image_url": row[1],
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

        return {
            "is_known_user": True,
            "alert": is_flagged,
            # A probable match still raises the alert — on a watchlist, a likely
            # offender is worth a look — but it is labelled so nobody mistakes it
            # for a certainty.
            "confidence": confidence,
            "needs_review": not confirmed,
            "status": status,
            "person": {
                "id": str(person_id),
                "full_name": full_name,
                "status": status,
                # The capture that actually produced the winning distance, so the
                # UI shows the reference the decision rested on.
                "image_url": captures[0]["image_url"] if captures else None,
            },
            "templates": templates,
            "match_distance": round(float(distance), 4),
            "matched_against_photos": len(captures),
            "agreeing_captures": sum(1 for c in captures if c["agrees"]),
            "margin_to_next_person": margin_to_next,
            "next_closest_person": next_person,
            "supporting_captures": captures,
            "nearest_person": None,
            "message": _match_message(full_name, status, is_flagged, confirmed),
        }

    # No match. Nothing is written to the registry — auto-labelling strangers as
    # 'verified' is a rejected design (PROJECT_CONTEXT.md section 5). The nearest
    # distance is still returned: it is the only way to tell a complete stranger
    # from someone who just missed the threshold.
    return {
        "is_known_user": False,
        "alert": False,
        "confidence": "none",
        "needs_review": False,
        "status": None,
        "person": None,
        "match_distance": round(float(best[3]), 4) if best else None,
        "matched_against_photos": 0,
        "agreeing_captures": 0,
        "margin_to_next_person": margin_to_next,
        "next_closest_person": next_person,
        "supporting_captures": [],
        "nearest_person": ({"full_name": best[1], "status": best[2]} if best else None),
        "templates": templates,
        "registered": False,
        "message": ("Unknown face — not in the registry. Not added; "
                    "enrolment is a separate admin action."),
        "nearest_distance": round(float(best[3]), 4) if best else None,
    }


def _frame_message(faces):
    """One line describing the whole frame, not just one person in it."""
    if not faces:
        return "No faces detected."
    if len(faces) == 1:
        return faces[0]["message"]

    alerting = [f for f in faces if f["alert"]]
    verified = [f for f in faces if f["is_known_user"] and not f["alert"]]
    undecided = [f for f in faces if f.get("confidence") == "no_decision"]
    unknown = [f for f in faces if not f["is_known_user"]
               and f.get("confidence") != "no_decision"]

    parts = []
    if alerting:
        names = ", ".join(f"{f['person']['full_name']} ({f['status']})" for f in alerting)
        parts.append(f"ALERT: {names}")
    if verified:
        parts.append(f"{len(verified)} verified")
    if unknown:
        parts.append(f"{len(unknown)} unknown")
    if undecided:
        parts.append(f"{len(undecided)} unreadable")
    return f"{len(faces)} faces — " + "; ".join(parts)


def _pick_primary(faces):
    """The face the response leads with.

    Severity first, then closeness to the camera. An offender at the back of a
    frame outranks a verified member at the front — the opposite of what picking
    the largest face would give you.
    """
    return max(
        faces,
        key=lambda f: (
            _SEVERITY.get(f.get("status") or "", 0),
            1 if f["is_known_user"] else 0,
            f.get("_area", 0),
        ),
    )


def process_incoming_face_image(image_bytes, db_conn=None, model_name="Facenet",
                                threshold=MATCH_THRESHOLD, include_embeddings=False,
                                probable_threshold=None):
    """
    Identify EVERY face in the image, not just the most prominent one.

    1. Detect all faces and embed each one.
    2. Match each against pgvector independently.
    3. Return a per-face verdict, plus a frame-level summary.

    include_embeddings adds each face's raw 512-dim vector to its entry. Off by
    default because it is ~8KB of JSON per face that no UI needs. Media analysis
    turns it on so unknown faces can be compared against EACH OTHER across files —
    "the same stranger appears in both videos" is a question the gallery cannot
    answer, since neither sighting is in it.

    The response also mirrors the most significant face at the top level
    (`person`, `match_distance`, `alert`, ...), so callers written against the
    single-face shape keep working. `faces` is the complete answer; the top-level
    fields are a convenience view of one entry in it.

    An unrecognised face is deliberately NOT written to the identity registry.
    Auto-labelling strangers as 'verified' is a rejected design (PROJECT_CONTEXT.md
    section 5): one bad-angle photo of an offender would whitelist them for good,
    and it quietly builds a biometric record of every passer-by. Enrolment belongs
    in a separate, deliberate admin action.
    """
    probe_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if probe_image is None:
        return {"success": False, "error": "Image could not be decoded.",
                "faces_detected": 0, "faces": []}

    # Detection now comes from services.face_geometry rather than DeepFace, because
    # it returns the five landmarks that alignment and the quality gate both need.
    # DeepFace is used for embedding only.
    detected = face_geometry.detect_faces(probe_image)
    if not detected:
        return {"success": False, "error": "No face detected in the image.",
                "faces_detected": 0, "faces": []}

    timings_ms = {}

    # 1. Save uploaded image bytes to a temporary file for DeepFace processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        embed_started = time.perf_counter()
        # 2. Extract the 512-dim vector embedding (Facenet512 by default)
        embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=model_name,
            detector_backend="retinaface",  # Faster for single face detection
            enforce_detection=True  # Fails cleanly if no face is detected
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
            "faces_detected": 0,
            "faces": [],
        }

    total_detected = len(detected)          # already sorted largest first
    truncated = max(0, total_detected - MAX_FACES_PER_FRAME)
    detected = detected[:MAX_FACES_PER_FRAME]

    conn = db_conn or psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        faces = []
        for index, face in enumerate(detected):
            box = face.as_dict()
            aligned = (face_geometry.align(probe_image, face.landmarks, 160)
                       if USE_5POINT_ALIGN else _plain_crop(probe_image, face.bbox))
            quality = face_geometry.assess_probe(probe_image, face, aligned)

            base = {
                "index": index,
                "bbox": {k: box[k] for k in ("x", "y", "w", "h")},
                "face_confidence": box["detector_score"],
                "quality": quality,
                "aligned": bool(USE_5POINT_ALIGN),
                "_area": face.area,
            }

            # A probe the system cannot read gets NO DECISION, not a guess.
            # Returning the closest name from an unreadable image produces an
            # answer indistinguishable from a real identification, which is the
            # worst possible failure mode for a watchlist.
            if ENFORCE_PROBE_QUALITY and not quality["decidable"]:
                base.update({
                    "is_known_user": False,
                    "alert": False,
                    "confidence": "no_decision",
                    "needs_review": True,
                    "status": None,
                    "person": None,
                    "match_distance": None,
                    "matched_against_photos": 0,
                    "agreeing_captures": 0,
                    "margin_to_next_person": None,
                    "next_closest_person": None,
                    "supporting_captures": [],
                    "nearest_person": None,
                    "templates": None,
                    "message": "NO DECISION — " + "; ".join(quality["reasons"]),
                })
                faces.append(base)
                continue

            if aligned is None:
                base.update({
                    "is_known_user": False, "alert": False, "confidence": "no_decision",
                    "needs_review": True, "status": None, "person": None,
                    "match_distance": None, "matched_against_photos": 0,
                    "agreeing_captures": 0, "margin_to_next_person": None,
                    "next_closest_person": None, "supporting_captures": [],
                    "nearest_person": None, "templates": None,
                    "message": "NO DECISION — the face could not be cropped for embedding.",
                })
                faces.append(base)
                continue

            # detector_backend="skip": the crop IS the face, already aligned.
            embedded = DeepFace.represent(
                img_path=aligned, model_name=model_name,
                detector_backend="skip", enforce_detection=False, align=False,
            )
            vector = embedded[0]["embedding"]

            verdict = _match_face(cursor, str(vector), threshold, probable_threshold)
            verdict.update(base)
            # Kept for callers written against the old field name.
            verdict["capture_quality"] = {
                "quality_score": round(min(quality["sharpness"] / 150.0, 1.0), 3),
                "face_pixels": min(box["w"], box["h"]),
                "blur_variance": quality["sharpness"],
                "blur_directional_ratio": quality["balance"],
                "det_confidence": quality["detector_score"],
                "passes": quality["decidable"],
                "reasons": quality["reasons"],
            }
            if include_embeddings:
                verdict["embedding"] = [float(v) for v in vector]
            faces.append(verdict)

        summary = {
            "total": len(faces),
            "known": sum(1 for f in faces if f["is_known_user"]),
            "unknown": sum(1 for f in faces if not f["is_known_user"]),
            "alerts": sum(1 for f in faces if f["alert"]),
            "confirmed": sum(1 for f in faces if f.get("confidence") == "confirmed"),
            "probable": sum(1 for f in faces if f.get("confidence") == "probable"),
            "no_decision": sum(1 for f in faces if f.get("confidence") == "no_decision"),
            "needs_review": sum(1 for f in faces if f.get("needs_review")),
            "truncated": truncated,
        }

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
                    "image_url": row[1],
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
                    "image_url": image_url
                },
                "match_distance": round(distance, 4),
                "matched_against_photos": len(captures),
                "agreeing_captures": agreeing,
                "margin_to_next_person": margin_to_next,
                "next_closest_person": next_person,
                "supporting_captures": captures,
                "faces_detected": faces_detected,
                "face_confidence": face_confidence,
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
            "capture_quality": capture_quality,
            "timings_ms": timings_ms,
            "registered": False,
            "message": "Unknown face — not in the registry. Not added; enrolment is a separate admin action."
        }

        primary = _pick_primary(faces)
        for face in faces:
            face.pop("_area", None)

        response = {
            "success": True,
            "faces_detected": total_detected,
            "faces": faces,
            "summary": summary,
            "alert": any(f["alert"] for f in faces),
            "message": _frame_message(faces),
        }

        # Mirror the leading face at the top level for single-face callers.
        for key in ("is_known_user", "status", "person", "match_distance",
                    "matched_against_photos", "agreeing_captures",
                    "margin_to_next_person", "next_closest_person",
                    "supporting_captures", "nearest_person", "capture_quality",
                    "face_confidence", "confidence", "needs_review"):
            response[key] = primary.get(key)
        response["registered"] = False
        response["primary_face_index"] = primary["index"]

        return response

    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e), "faces_detected": 0, "faces": []}
    finally:
        cursor.close()
        if db_conn is None:
            conn.close()
