"""Add pgvector ANN indexes for face matching queries.

Usage:
    python backend/scripts/add_face_vector_indexes.py
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def main():
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set.")

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # HNSW is preferred for cosine nearest-neighbor retrieval when supported.
        try:
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_person_faces_embedding_hnsw
                ON person_faces
                USING hnsw (embedding vector_cosine_ops)
                WHERE embedding IS NOT NULL AND use_for_matching;
                """
            )
            print("Created/verified HNSW index: idx_person_faces_embedding_hnsw")
        except Exception as hnsw_error:
            print(f"HNSW index creation failed: {hnsw_error}")
            print("Falling back to IVFFlat index.")
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_person_faces_embedding_ivfflat
                ON person_faces
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
                WHERE embedding IS NOT NULL AND use_for_matching;
                """
            )
            print("Created/verified IVFFlat index: idx_person_faces_embedding_ivfflat")

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_persons_face_embedding_hnsw
            ON persons
            USING hnsw (face_embedding vector_cosine_ops)
            WHERE face_embedding IS NOT NULL;
            """
        )
        print("Created/verified HNSW index: idx_persons_face_embedding_hnsw")

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
