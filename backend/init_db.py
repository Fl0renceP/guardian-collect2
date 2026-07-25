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
    face_embedding vector(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE person_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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

            # Generate 128-dim FaceNet embedding via DeepFace using local image file
            print(f"Extracting facial embedding for {record['full_name']}...")
            objs = DeepFace.represent(img_path=local_path, model_name="Facenet", enforce_detection=False)
            vector_embedding = str(objs[0]["embedding"])

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

            # Insert metadata record
            cursor.execute(
                """
                INSERT INTO person_faces (person_id, image_url)
                VALUES (%s, %s);
                """,
                (person_id, image_url)
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