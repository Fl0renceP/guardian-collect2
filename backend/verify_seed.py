import os
import psycopg2
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "face-db2"  # Your container name

def verify_azure_blob():
    print("\n--- 1. Checking Azure Blob Storage Container ---")
    if not AZURE_STORAGE_CONNECTION_STRING:
        print("❌ AZURE_STORAGE_CONNECTION_STRING missing.")
        return

    blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(CONTAINER_NAME)
    
    blobs = list(container_client.list_blobs())
    print(f"📁 Found {len(blobs)} image(s) in container '{CONTAINER_NAME}':")
    for blob in blobs:
        print(f"  • Name: {blob.name} | Size: {blob.size / 1024:.1f} KB")

def verify_postgresql():
    print("\n--- 2. Checking PostgreSQL Database Records ---")
    if not DATABASE_URL:
        print("❌ DATABASE_URL missing.")
        return

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        # Query joined person metadata, status, vector length, and face image URL
        query = """
            SELECT 
                p.id, 
                p.full_name, 
                p.status, 
                vector_dims(p.face_embedding) AS embedding_dim,
                f.image_url,
                p.created_at
            FROM persons p
            JOIN person_faces f ON p.id = f.person_id;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        print(f"🗄️ Found {len(rows)} record(s) in PostgreSQL:")
        for r in rows:
            person_id, name, status, vector_dim, url, created_at = r
            print(f"  • Name: {name}")
            print(f"    - ID: {person_id}")
            print(f"    - Status: {status}")
            print(f"    - Vector Dimensions: {vector_dim}-dim")
            print(f"    - Blob URL: {url}")
            print(f"    - Created: {created_at}\n")

    except Exception as e:
        print(f"❌ Error querying database: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    verify_azure_blob()
    verify_postgresql()