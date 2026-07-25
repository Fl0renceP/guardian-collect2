"""Add capture provenance and quality columns to person_faces.

Additive and re-runnable. Nothing is dropped.

Two jobs:

1. Provenance — a face lifted from CCTV needs to carry where and when it came
   from. Without camera_id / captured_at / incident_ref you cannot answer "why did
   you match this person", which is the question that matters in a security
   product.

2. Quality — use_for_matching separates two different roles a stored image can
   play. A blurry 40px grab is perfectly good evidence for a case file and
   completely unusable as a matching reference. Previously every stored image was
   forced to be both.

Existing rows are treated as trusted references (use_for_matching = TRUE), since
the seed photos were curated by hand.

    python migrate_capture_metadata.py
"""

import psycopg2

from config import Config

MIGRATE_SQL = """
ALTER TABLE person_faces
    ADD COLUMN IF NOT EXISTS use_for_matching BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS quality_score REAL,
    ADD COLUMN IF NOT EXISTS face_pixels INTEGER,
    ADD COLUMN IF NOT EXISTS det_confidence REAL,
    ADD COLUMN IF NOT EXISTS blur_variance REAL,
    ADD COLUMN IF NOT EXISTS blur_directional_ratio REAL,
    ADD COLUMN IF NOT EXISTS camera_id TEXT,
    ADD COLUMN IF NOT EXISTS captured_at TIMESTAMP WITH TIME ZONE,
    ADD COLUMN IF NOT EXISTS incident_ref TEXT;

-- The matching query filters on these two columns on every scan.
CREATE INDEX IF NOT EXISTS idx_person_faces_person_id ON person_faces(person_id);
CREATE INDEX IF NOT EXISTS idx_person_faces_matchable
    ON person_faces(person_id) WHERE use_for_matching AND embedding IS NOT NULL;
"""


def migrate():
    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute(MIGRATE_SQL)
        conn.commit()
        print("person_faces capture/quality columns ready")

        cur.execute(
            """
            SELECT p.full_name, p.status,
                   count(pf.id) AS captures,
                   count(*) FILTER (WHERE pf.use_for_matching AND pf.embedding IS NOT NULL)
                       AS usable_references
            FROM persons p
            LEFT JOIN person_faces pf ON pf.person_id = p.id
            GROUP BY p.id, p.full_name, p.status
            ORDER BY p.full_name;
            """
        )
        print("\ngallery:")
        for name, status, captures, usable in cur.fetchall():
            print(f"  {name:20s} ({status:8s}) captures={captures} usable_as_reference={usable}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
