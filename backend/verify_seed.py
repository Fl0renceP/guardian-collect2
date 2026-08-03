import os
import psycopg2
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from urllib.parse import urlparse

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

    print("\n  Metadata sample (first 5 blobs):")
    for blob in blobs[:5]:
        props = container_client.get_blob_client(blob.name).get_blob_properties()
        meta = props.metadata or {}
        print(f"  • {blob.name}")
        if not meta:
            print("      (no metadata)")
            continue
        for key in sorted(meta.keys()):
            print(f"      {key}: {meta[key]}")


def _column_exists(cursor, table_name, column_name):
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s;
        """,
        (table_name, column_name),
    )
    return cursor.fetchone() is not None


def _blob_name_from_url(url):
    parsed = urlparse(url)
    path = parsed.path.lstrip("/")
    if not path:
        return None
    parts = path.split("/", 1)
    if len(parts) != 2:
        return None
    return parts[1]

def verify_postgresql():
    print("\n--- 2. Checking PostgreSQL Database Records ---")
    if not DATABASE_URL:
        print("❌ DATABASE_URL missing.")
        return

    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    try:
        has_embedding = _column_exists(cursor, "person_faces", "embedding")
        has_use_for_matching = _column_exists(cursor, "person_faces", "use_for_matching")
        has_source = _column_exists(cursor, "person_faces", "source")

        summary_query = """
            SELECT
                p.id,
                p.full_name,
                p.status,
                count(f.id) AS captures,
                count(*) FILTER (WHERE f.embedding IS NOT NULL) AS captures_with_embedding,
                count(*) FILTER (
                    WHERE (%(has_use_for_matching)s IS FALSE) OR (f.use_for_matching AND f.embedding IS NOT NULL)
                ) AS usable_references,
                count(*) FILTER (
                    WHERE (%(has_use_for_matching)s IS TRUE) AND (NOT f.use_for_matching)
                ) AS evidence_only,
                min(f.created_at) AS first_capture,
                max(f.created_at) AS last_capture
            FROM persons p
            LEFT JOIN person_faces f ON p.id = f.person_id
            GROUP BY p.id, p.full_name, p.status
            ORDER BY p.full_name;
        """
        if not has_embedding:
            summary_query = summary_query.replace(
                "count(*) FILTER (WHERE f.embedding IS NOT NULL) AS captures_with_embedding,",
                "0 AS captures_with_embedding,",
            )
            summary_query = summary_query.replace(
                "WHERE (%(has_use_for_matching)s IS FALSE) OR (f.use_for_matching AND f.embedding IS NOT NULL)",
                "TRUE",
            )

        cursor.execute(summary_query, {"has_use_for_matching": has_use_for_matching})
        rows = cursor.fetchall()

        print(f"🗄️ Found {len(rows)} person(s) in PostgreSQL:")
        for r in rows:
            person_id, name, status, captures, with_embedding, refs, evidence_only, first_cap, last_cap = r
            print(f"  • Name: {name}")
            print(f"    - ID: {person_id}")
            print(f"    - Status: {status}")
            print(f"    - Captures: {captures}")
            print(f"    - With embedding: {with_embedding}")
            print(f"    - Usable references: {refs}")
            print(f"    - Evidence-only: {evidence_only}")
            print(f"    - First capture: {first_cap}")
            print(f"    - Last capture: {last_cap}\n")

        sample_query = """
            SELECT
                p.full_name,
                p.status,
                f.image_url,
                COALESCE(f.source, 'n/a') AS source,
                f.use_for_matching,
                f.quality_score,
                f.created_at
            FROM persons p
            JOIN person_faces f ON p.id = f.person_id
            ORDER BY f.created_at DESC
            LIMIT 8;
        """
        if not has_source:
            sample_query = sample_query.replace("COALESCE(f.source, 'n/a') AS source,", "'n/a' AS source,")
        if not has_use_for_matching:
            sample_query = sample_query.replace("f.use_for_matching,", "TRUE AS use_for_matching,")

        cursor.execute(sample_query)
        sample_rows = cursor.fetchall()

        print("Recent capture rows:")
        for full_name, status, image_url, source, use_for_matching, quality_score, created_at in sample_rows:
            print(f"  • {full_name} ({status})")
            print(f"    - Source: {source}")
            print(f"    - use_for_matching: {use_for_matching}")
            print(f"    - quality_score: {quality_score}")
            print(f"    - created_at: {created_at}")
            print(f"    - image_url: {image_url}")
            print(f"    - blob_name: {_blob_name_from_url(image_url)}\n")

    except Exception as e:
        print(f"❌ Error querying database: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    verify_azure_blob()
    verify_postgresql()