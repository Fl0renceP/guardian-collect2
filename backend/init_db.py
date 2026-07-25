import os
import psycopg2
from deepface import DeepFace
from dotenv import load_dotenv
from services.blob_storage import BlobStorageService

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CREATE_SCHEMA_SQL = """
DROP TABLE IF EXISTS person_faces CASCADE;
DROP TABLE IF EXISTS persons CASCADE;
DROP TYPE IF EXISTS person_status CASCADE;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE person_status AS ENUM ('offender', 'suspect', 'verified');

CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(255) NOT NULL,
    status person_status NOT NULL DEFAULT 'verified',
    face_embedding vector(512),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- One row per captured image. A person accumulates several over time (CCTV
-- gives many sightings of the same individual), and each carries its own
-- embedding because matching compares against every stored capture.
--
-- Keep this in step with migrate_multi_face.py and migrate_capture_metadata.py:
-- those add these same columns to an existing database, this creates them fresh.
CREATE TABLE person_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    embedding vector(512),
    source TEXT,

    -- FALSE means "evidence only": kept against the person, never used to
    -- identify anyone. A blurry 40px CCTV grab is real evidence and a harmful
    -- matching reference.
    use_for_matching BOOLEAN NOT NULL DEFAULT TRUE,
    quality_score REAL,
    face_pixels INTEGER,
    det_confidence REAL,
    blur_variance REAL,
    blur_directional_ratio REAL,

    camera_id TEXT,
    captured_at TIMESTAMP WITH TIME ZONE,
    incident_ref TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_person_faces_person_id ON person_faces(person_id);
CREATE INDEX idx_person_faces_matchable
    ON person_faces(person_id) WHERE use_for_matching AND embedding IS NOT NULL;

CREATE INDEX idx_persons_status ON persons(status);
"""

# Seed definition linking to blob filenames
SEED_RECORDS = [
    {
        "full_name": "Tinashe Madanire",
        "status": "offender",
        "blob_name": "seed_offender.jpeg",
        "local_fallback_path": "seed_photos/offender.jpeg"
    },
    {
        "full_name": "Victoria Armstrong",
        "status": "suspect",
        "blob_name": "seed_suspect.jpeg",
        "local_fallback_path": "seed_photos/suspect.jpeg"
    },
    {
        "full_name": "Tadiwa Banda",
        "status": "verified",
        "blob_name": "seed_verified.jpeg",
        "local_fallback_path": "seed_photos/verified.jpeg"
    }
]

def setup_and_seed():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL missing in .env")

    # 1. Initialize Azure Blob Storage container and upload photos
    blob_service = BlobStorageService()
    
    print("--- Step 1: Schema Creation & Azure Blob Sync ---")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        cursor.execute(CREATE_SCHEMA_SQL)
        
        print("--- Step 2: Processing Embeddings & Seeding DB ---")
        for record in SEED_RECORDS:
            local_path = os.path.join(os.path.dirname(__file__), record["local_fallback_path"])
            
            # Upload image to blob container
            image_url = blob_service.upload_image(local_path, filename=record["blob_name"])
            print(f"Uploaded {record['full_name']} image to {image_url}")

            # Generate the 512-dim Facenet512 embedding from the local image file.
            #
            # enforce_detection stays True to match services/recognition.py. With
            # False, a seed photo whose face isn't found is silently embedded as
            # the WHOLE image, and that garbage vector becomes the permanent
            # reference for this person — a failure nothing downstream can detect.
            # Better to fail loudly here, while someone is watching.
            print(f"Extracting facial embedding for {record['full_name']}...")
            objs = DeepFace.represent(img_path=local_path, model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)

            # Largest face, not objs[0], which is arbitrary if the photo has more
            # than one person in it.
            def _area(face):
                box = face.get("facial_area") or {}
                return box.get("w", 0) * box.get("h", 0)

            subject = max(objs, key=_area)
            vector_embedding = str(subject["embedding"])

            # Insert into persons table
            cursor.execute(
                """
                INSERT INTO persons (full_name, status, face_embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING id;
                """,
                (record["full_name"], record["status"], vector_embedding)
            )
            person_id = cursor.fetchone()[0]

            # Link the face image, carrying its own embedding. Matching reads
            # person_faces, not persons.face_embedding — a person can hold many
            # reference photos and each needs its own vector. Seeding without one
            # would leave a freshly initialised database with nothing to match.
            cursor.execute(
                """
                INSERT INTO person_faces
                    (person_id, image_url, embedding, source, use_for_matching)
                VALUES (%s, %s, %s::vector, 'seed_data', TRUE);
                """,
                (person_id, image_url, vector_embedding)
            )

        conn.commit()
        print("✅ Azure Storage Container 'face-db2' ready & Database fully seeded with real face vectors!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during setup: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_and_seed()