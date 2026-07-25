"""Give person_faces its own embedding, so one person can have several photos.

Additive and re-runnable — nothing is dropped. persons.face_embedding stays where
it is and is still matched against, so this cannot break the existing endpoint.

Why: matching against a single photo per person means one bad angle is the whole
identity. With several photos, a scan is compared against every photo of every
person and the closest one wins, so a person recognised from any one of their
photos is recognised.

The `source` column is from DATA_SCHEMA.md section 1 — it records where a photo
came from ('seed_data', 'enrolled', ...).

    python migrate_multi_face.py
"""

import psycopg2

from config import Config

MIGRATE_SQL = """
ALTER TABLE person_faces
    ADD COLUMN IF NOT EXISTS embedding vector(512),
    ADD COLUMN IF NOT EXISTS source TEXT;
"""

# Backfill only where a person has exactly one stored photo. In that case the
# embedding on persons was generated from that very file by init_db.py, so it can
# be copied across rather than recomputed. Anyone with several photos is left
# alone — we can't tell which of them produced the single stored vector.
BACKFILL_SQL = """
UPDATE person_faces pf
SET embedding = p.face_embedding,
    source = COALESCE(pf.source, 'seed_data')
FROM persons p
WHERE pf.person_id = p.id
  AND pf.embedding IS NULL
  AND p.face_embedding IS NOT NULL
  AND (SELECT count(*) FROM person_faces x WHERE x.person_id = p.id) = 1;
"""


def migrate():
    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute(MIGRATE_SQL)
        conn.commit()
        print("person_faces.embedding and person_faces.source ready")

        cur.execute(BACKFILL_SQL)
        conn.commit()
        print(f"backfilled {cur.rowcount} photo embeddings from persons.face_embedding")

        cur.execute(
            """
            SELECT p.full_name, p.status,
                   count(pf.id) AS photos,
                   count(pf.embedding) AS photos_with_embedding
            FROM persons p
            LEFT JOIN person_faces pf ON pf.person_id = p.id
            GROUP BY p.id, p.full_name, p.status
            ORDER BY p.full_name;
            """
        )
        print("\ngallery:")
        for name, status, photos, with_embedding in cur.fetchall():
            print(f"  {name:20s} ({status:8s}) photos={photos} usable={with_embedding}")

        cur.execute("SELECT count(*) FROM person_faces WHERE embedding IS NULL")
        missing = cur.fetchone()[0]
        if missing:
            print(
                f"\nNOTE: {missing} photo row(s) still have no embedding. They are "
                "ignored when matching. Re-add them with enroll_face.py."
            )
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
