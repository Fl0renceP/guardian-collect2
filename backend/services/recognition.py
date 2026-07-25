import os
import tempfile
import cv2
import psycopg2
import numpy as np
from deepface import DeepFace

from services.face_quality import assess as assess_quality

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

# Each face costs its own ~590ms embedding, so a crowd scene is priced per head.
# Faces are processed largest-first, so this cap drops the most distant and least
# identifiable people rather than an arbitrary subset.
MAX_FACES_PER_FRAME = int(os.getenv("MAX_FACES_PER_FRAME", "10"))

# Which face leads the response when several are present. An offender in the
# background matters more than a verified member in the foreground.
_SEVERITY = {"offender": 3, "suspect": 2, "verified": 1}


def _face_area(face):
    """Pixel area of a detected face — proxy for how close they are to the camera."""
    area = face.get("facial_area") or {}
    return area.get("w", 0) * area.get("h", 0)


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
_GALLERY_QUERY = """
    WITH candidates AS (
        SELECT p.id, p.full_name, p.status,
               (pf.embedding <=> %(vec)s::vector) AS distance,
               pf.image_url
        FROM persons p
        JOIN person_faces pf ON pf.person_id = p.id
        WHERE pf.embedding IS NOT NULL
          AND pf.use_for_matching
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

_SUPPORTING_QUERY = """
    SELECT id, image_url, source, camera_id, captured_at, incident_ref,
           quality_score, (embedding <=> %(vec)s::vector) AS distance
    FROM person_faces
    WHERE person_id = %(pid)s
      AND embedding IS NOT NULL
      AND use_for_matching
    ORDER BY distance ASC;
"""


def _match_face(cursor, vector_str, threshold):
    """Identify a single face embedding against the gallery.

    Returns the per-face verdict. Never raises for "no match" — an unknown face
    is a normal, expected outcome, not an error.
    """
    cursor.execute(_GALLERY_QUERY, {"vec": vector_str})
    ranked = cursor.fetchall()
    best = ranked[0] if ranked else None

    # Distance to the nearest DIFFERENT person. A tiny margin means the face sat
    # almost equally close to two people — worth surfacing even when the winner
    # is under threshold.
    margin_to_next = None
    next_person = None
    if len(ranked) > 1:
        margin_to_next = round(float(ranked[1][3]) - float(ranked[0][3]), 4)
        next_person = {"full_name": ranked[1][1], "status": ranked[1][2],
                       "distance": round(float(ranked[1][3]), 4)}

    if best and best[3] < threshold:
        person_id, full_name, status, distance, image_url = best
        is_flagged = status in ("offender", "suspect")

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
            "status": status,
            "person": {
                "id": str(person_id),
                "full_name": full_name,
                "status": status,
                "image_url": image_url,
            },
            "match_distance": round(float(distance), 4),
            "matched_against_photos": len(captures),
            "agreeing_captures": sum(1 for c in captures if c["agrees"]),
            "margin_to_next_person": margin_to_next,
            "next_closest_person": next_person,
            "supporting_captures": captures,
            "nearest_person": None,
            "message": (f"ALERT: {status.upper()} DETECTED!" if is_flagged
                        else f"Member '{full_name}' is verified."),
        }

    # No match. Nothing is written to the registry — auto-labelling strangers as
    # 'verified' is a rejected design (PROJECT_CONTEXT.md section 5). The nearest
    # distance is still returned: it is the only way to tell a complete stranger
    # from someone who just missed the threshold.
    return {
        "is_known_user": False,
        "alert": False,
        "status": None,
        "person": None,
        "match_distance": round(float(best[3]), 4) if best else None,
        "matched_against_photos": 0,
        "agreeing_captures": 0,
        "margin_to_next_person": margin_to_next,
        "next_closest_person": next_person,
        "supporting_captures": [],
        "nearest_person": ({"full_name": best[1], "status": best[2]} if best else None),
        "registered": False,
        "message": ("Unknown face — not in the registry. Not added; "
                    "enrolment is a separate admin action."),
    }


def _frame_message(faces):
    """One line describing the whole frame, not just one person in it."""
    if not faces:
        return "No faces detected."
    if len(faces) == 1:
        return faces[0]["message"]

    alerting = [f for f in faces if f["alert"]]
    verified = [f for f in faces if f["is_known_user"] and not f["alert"]]
    unknown = [f for f in faces if not f["is_known_user"]]

    parts = []
    if alerting:
        names = ", ".join(f"{f['person']['full_name']} ({f['status']})" for f in alerting)
        parts.append(f"ALERT: {names}")
    if verified:
        parts.append(f"{len(verified)} verified")
    if unknown:
        parts.append(f"{len(unknown)} unknown")
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
                                threshold=MATCH_THRESHOLD, include_embeddings=False):
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
    # DeepFace wants a path, so the upload goes to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        detected = DeepFace.represent(
            img_path=tmp_path,
            model_name=model_name,
            detector_backend=FACE_DETECTOR,
            enforce_detection=True,  # Fails cleanly if no face is detected
            align=FACE_ALIGN,        # off — see the note above, it drops faces
        )
    except Exception as e:
        os.remove(tmp_path)
        return {
            "success": False,
            "error": "No face detected in the image.",
            "details": str(e),
            "faces_detected": 0,
            "faces": [],
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Largest first, so the cap below drops the most distant faces.
    detected.sort(key=_face_area, reverse=True)
    total_detected = len(detected)
    truncated = max(0, total_detected - MAX_FACES_PER_FRAME)
    detected = detected[:MAX_FACES_PER_FRAME]

    # Quality is advisory here and a scan is NEVER refused for it. You always want
    # to try to identify whoever is in front of the camera, however poor the
    # frame. Whether an image is fit to become a stored reference is a different
    # question, and that gate lives in enrolment.
    probe_image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    conn = db_conn or psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        faces = []
        for index, face in enumerate(detected):
            box = face.get("facial_area") or {}
            verdict = _match_face(cursor, str(face["embedding"]), threshold)
            verdict.update({
                "index": index,
                "bbox": {
                    "x": int(box.get("x", 0)), "y": int(box.get("y", 0)),
                    "w": int(box.get("w", 0)), "h": int(box.get("h", 0)),
                },
                "face_confidence": face.get("face_confidence"),
                "capture_quality": (
                    assess_quality(probe_image, box, face.get("face_confidence"))
                    if probe_image is not None else None
                ),
                "_area": _face_area(face),
            })
            if include_embeddings:
                verdict["embedding"] = [float(v) for v in face["embedding"]]
            faces.append(verdict)

        summary = {
            "total": len(faces),
            "known": sum(1 for f in faces if f["is_known_user"]),
            "unknown": sum(1 for f in faces if not f["is_known_user"]),
            "alerts": sum(1 for f in faces if f["alert"]),
            "truncated": truncated,
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
                    "face_confidence"):
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
