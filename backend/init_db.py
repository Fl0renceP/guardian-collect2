import os
import psycopg2
from deepface import DeepFace
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

CREATE_SCHEMA_SQL = """
-- Drop tables if resetting for clean setup
DROP TABLE IF EXISTS person_faces CASCADE;
DROP TABLE IF EXISTS persons CASCADE;
DROP TYPE IF EXISTS person_status CASCADE;

-- 1. Enable pgvector extension for local face embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create custom ENUM for member status
CREATE TYPE person_status AS ENUM ('offender', 'suspect', 'verified');

-- 3. Create primary persons table with 128-dim vector embedding
CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name VARCHAR(255) NOT NULL,
    status person_status NOT NULL DEFAULT 'verified',
    face_embedding vector(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Create person_faces metadata table
CREATE TABLE person_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    person_id UUID NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_persons_status ON persons(status);
"""

# Seed records (Name, Status, Placeholder URL)
SEED_DATA = [
    {
        "full_name": "John Doe (Test Offender)",
        "status": "offender",
        "image_url": "https://guardiansa5001.blob.core.windows.net/face-images/seed_offender.jpg"
    },
    {
        "full_name": "Jane Smith (Test Suspect)",
        "status": "suspect",
        "image_url": "https://guardiansa5001.blob.core.windows.net/face-images/seed_suspect.jpg"
    },
    {
        "full_name": "Alex Johnson (Verified Member)",
        "status": "verified",
        "image_url": "https://guardiansa5001.blob.core.windows.net/face-images/seed_verified.jpg"
    }
]

def generate_dummy_embedding():
    """Generates a dummy 128-float vector if real seed image is not yet loaded."""
    import random
    return [random.uniform(-1, 1) for _ in range(128)]

def initialize_database():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing.")
        
    print("Connecting to Azure Cosmos DB for PostgreSQL ('citus' database)...")
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        print("Creating tables, ENUM types, and enabling pgvector...")
        cursor.execute(CREATE_SCHEMA_SQL)
        
        print("Seeding initial records...")
        for item in SEED_DATA:
            # Generate vector embedding (Replace with DeepFace.represent() when actual image files exist)
            embedding_vector = str(generate_dummy_embedding())

            # Insert into persons table
            cursor.execute(
                """
                INSERT INTO persons (full_name, status, face_embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING id;
                """,
                (item["full_name"], item["status"], embedding_vector)
            )
            person_id = cursor.fetchone()[0]

            # Insert associated face record
            cursor.execute(
                """
                INSERT INTO person_faces (person_id, image_url)
                VALUES (%s, %s);
                """,
                (person_id, item["image_url"])
            )

        conn.commit()
        print("✅ Cosmos DB for PostgreSQL successfully initialized and seeded!")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error during database initialization: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    initialize_database()