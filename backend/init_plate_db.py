import os
import psycopg2
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "plate-images"

CREATE_PLATE_SCHEMA_SQL = """
DROP TABLE IF EXISTS vehicle_plate_images CASCADE;
DROP TABLE IF EXISTS vehicle_plates CASCADE;

CREATE TABLE vehicle_plates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_number VARCHAR(32) UNIQUE NOT NULL,
    status person_status NOT NULL DEFAULT 'verified',
    owner_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE vehicle_plate_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plate_id UUID NOT NULL REFERENCES vehicle_plates(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_vehicle_plates_number ON vehicle_plates(plate_number);
"""

# Seed data for demo
SEED_PLATES = [
    {
        "plate_number": "CA 123-456",
        "status": "offender",
        "owner_name": "Stolen Vehicle - Alert",
        "blob_name": "seed_offender_plate.jpeg"
    },
    {
        "plate_number": "CY 987-654",
        "status": "suspect",
        "owner_name": "Under Investigation",
        "blob_name": "seed_suspect_plate.jpeg"
    },
    {
        "plate_number": "CL 456-789",
        "status": "verified",
        "owner_name": "Registered Resident",
        "blob_name": "seed_verified_plate.jpeg"
    }
]

def init_plate_storage_and_db():
    # 1. Create 'plate-images' blob container in Azure
    if AZURE_STORAGE_CONNECTION_STRING:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        try:
            blob_service_client.create_container(CONTAINER_NAME, public_access="container")
            print(f"Created Azure container: '{CONTAINER_NAME}'")
        except Exception as e:
            print(f"Container '{CONTAINER_NAME}' check/notice: {e}")

    # 2. Setup PostgreSQL schema
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    try:
        cursor.execute(CREATE_PLATE_SCHEMA_SQL)
        
        # 3. Seed records
        for plate in SEED_PLATES:
            # Normalize plate number (uppercase, stripped spaces)
            clean_plate = plate["plate_number"].replace("-", "").replace(" ", "").upper()
            
            cursor.execute("""
                INSERT INTO vehicle_plates (plate_number, status, owner_name)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (clean_plate, plate["status"], plate["owner_name"]))
            
            plate_id = cursor.fetchone()[0]
            print(f"Seeded plate: {clean_plate} ({plate['status']})")

        conn.commit()
        print("✅ Vehicle plates schema initialized and seeded!")
    except Exception as e:
        conn.rollback()
        print(f"Error seeding plates: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    init_plate_storage_and_db()