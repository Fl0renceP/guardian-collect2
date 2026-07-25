import psycopg2
from deepface import DeepFace
from services.blob_storage import BlobStorageService

# Initialize Blob Storage Helper
blob_service = BlobStorageService()

def process_incoming_face_image(image_bytes, db_conn, model_name="Facenet", threshold=0.40):
    """
    Core Logic:
    1. Extract 128-dimension vector embedding using DeepFace.
    2. Query PostgreSQL (pgvector) to find closest matching embedding.
    3. If distance < threshold:
       - 'offender' or 'suspect' -> Alert!
       - 'verified' -> Member verified.
    4. If no match -> Upload photo to Azure Blob + auto-enroll in Postgres as 'verified'.
    """
    
    # 1. Generate face embedding array using DeepFace
    try:
        embedding_objs = DeepFace.represent(
            img_path=image_bytes, 
            model_name=model_name, 
            enforce_detection=True
        )
        scanned_vector = embedding_objs[0]["embedding"]
    except Exception as e:
        print(f"Face detection failed: {e}")
        return {"error": "No face detected in image"}, 400

    # Convert list to string format required by pgvector query
    vector_str = str(scanned_vector)

    # 2. Query PostgreSQL for vector similarity (Cosine Distance: <=>)
    with db_conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, full_name, status, (face_embedding <=> %s::vector) AS distance
            FROM persons
            WHERE face_embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT 1;
            """,
            (vector_str,)
        )
        match = cursor.fetchone()

    # 3. Check if closest match falls within our threshold
    if match and match[3] < threshold:
        person_id, name, status, distance = match
        confidence = round((1 - distance) * 100, 2)
        
        if status in ['offender', 'suspect']:
            send_alert(person_id, name, status, confidence)
            return {
                "flagged": True, 
                "status": status, 
                "name": name, 
                "confidence": confidence
            }
        else:
            print(f"Member {name} is verified.")
            return {
                "flagged": False, 
                "status": "verified", 
                "name": name, 
                "confidence": confidence
            }

    # 4. No match found -> Auto-enroll new subject as 'verified'
    print("Unknown face detected. Auto-enrolling as verified member...")
    new_person = auto_enroll_new_subject(image_bytes, scanned_vector, db_conn)
    return {
        "flagged": False, 
        "status": "verified", 
        "name": new_person["name"], 
        "enrolled": True
    }


def auto_enroll_new_subject(image_bytes, face_vector, db_conn):
    """Saves raw photo to Azure Blob Storage and registers new user in Postgres."""
    # Upload image to Azure Blob Storage
    image_url = blob_service.upload_image(image_bytes)
    
    with db_conn.cursor() as cursor:
        # Insert into persons table
        cursor.execute(
            """
            INSERT INTO persons (full_name, status, face_embedding)
            VALUES (%s, %s, %s::vector)
            RETURNING id, full_name;
            """,
            ("New Member (Auto-Enrolled)", "verified", str(face_vector))
        )
        person_id, name = cursor.fetchone()
        
        # Save face image link
        cursor.execute(
            "INSERT INTO person_faces (person_id, image_url) VALUES (%s, %s);",
            (person_id, image_url)
        )
        db_conn.commit()
        
    return {"id": person_id, "name": name}


def send_alert(person_id, name, status, confidence):
    print(f"🚨 SECURITY ALERT: Detected {status.upper()} '{name}' (ID: {person_id}) with {confidence}% confidence!")