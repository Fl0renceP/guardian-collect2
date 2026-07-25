import os
import tempfile
import psycopg2
import numpy as np
from deepface import DeepFace
from services.blob_storage import BlobStorageService

DATABASE_URL = os.getenv("DATABASE_URL")
MATCH_THRESHOLD = 0.40  # L2 Euclidean distance threshold for FaceNet (Lower = stricter)

def process_incoming_face_image(image_bytes, db_conn=None, model_name="Facenet", threshold=MATCH_THRESHOLD, filename="scan_input.jpg"):
    """
    Core facial recognition logic:
    1. Extracts embedding from incoming image.
    2. Queries pgvector for nearest match.
    3. If match found (< 0.40 distance): returns person details & alert status.
    4. If no match found: saves image to Azure Blob, registers new person as 'verified'.
    """
    # 1. Save uploaded image bytes to a temporary file for DeepFace processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        # 2. Extract 128-dim vector embedding using FaceNet
        embeddings = DeepFace.represent(
            img_path=tmp_path,
            model_name=model_name,
            enforce_detection=True  # Fails cleanly if no face is detected
        )
        query_vector = embeddings[0]["embedding"]

    except Exception as e:
        os.remove(tmp_path)
        return {
            "success": False,
            "error": "No face detected in the image.",
            "details": str(e)
        }

    # Clean up temp file
    os.remove(tmp_path)

    # 3. Query PostgreSQL using pgvector L2 distance operator (<->)
    conn = db_conn or psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        # Convert list to string format expected by pgvector '[x1, x2, ...]'
        vector_str = str(query_vector)

        # Query top 1 closest face vector in DB
        query = """
            SELECT 
                p.id, 
                p.full_name, 
                p.status, 
                (p.face_embedding <-> %s::vector) AS distance,
                f.image_url
            FROM persons p
            LEFT JOIN person_faces f ON p.id = f.person_id
            ORDER BY distance ASC
            LIMIT 1;
        """
        cursor.execute(query, (vector_str,))
        result = cursor.fetchone()

        # 4. Check if we found a match within threshold
        if result and result[3] < threshold:
            person_id, full_name, status, distance, image_url = result
            
            # Determine alert condition
            is_flagged = status in ["offender", "suspect"]

            return {
                "success": True,
                "is_known_user": True,
                "alert": is_flagged,  # True for offender/suspect, False for verified
                "status": status,
                "person": {
                    "id": person_id,
                    "full_name": full_name,
                    "status": status,
                    "image_url": image_url
                },
                "match_distance": round(distance, 4),
                "message": f"ALERT: {status.upper()} DETECTED!" if is_flagged else f"Member '{full_name}' is verified."
            }

        # 5. NO MATCH FOUND -> Auto-register new person as 'verified'
        blob_service = BlobStorageService()
        blob_url = blob_service.upload_image(image_bytes, filename=f"auto_registered/{filename}")

        # Insert new person record with default status 'verified'
        insert_person_sql = """
            INSERT INTO persons (full_name, status, face_embedding)
            VALUES (%s, %s, %s::vector)
            RETURNING id, full_name, status;
        """
        cursor.execute(insert_person_sql, ("New Community Member", "verified", vector_str))
        new_person = cursor.fetchone()
        conn.commit()

        # Link face image URL
        insert_face_sql = """
            INSERT INTO person_faces (person_id, image_url)
            VALUES (%s, %s);
        """
        cursor.execute(insert_face_sql, (new_person[0], blob_url))
        conn.commit()

        return {
            "success": True,
            "is_known_user": False,
            "alert": False,
            "status": "verified",
            "person": {
                "id": new_person[0],
                "full_name": new_person[1],
                "status": "verified",
                "image_url": blob_url
            },
            "message": "Unrecognized face. Automatically registered new member with 'verified' status."
        }

    except Exception as e:
        conn.rollback()
        return {"success": False, "error": str(e)}
    finally:
        cursor.close()
        if db_conn is None:
            conn.close()