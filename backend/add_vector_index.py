"""Add an HNSW index to the face embedding columns — WHEN THE GALLERY IS BIG ENOUGH.

Do not run this yet.

With a handful of references, a sequential scan over every embedding is already
sub-millisecond and the planner will ignore an index entirely. Building one now
costs memory and adds write overhead on every enrolment, in exchange for nothing.
Measured on the current gallery, the vector query is ~50ms of a ~25 SECOND scan;
face detection is essentially all of it. Indexing here optimises 0.2% of the work.

Run it when the number of matchable references passes roughly a thousand, or when
the query time reported below stops being noise next to detection time.

One thing to know before you do: HNSW is APPROXIMATE. It trades a small chance of
missing the true nearest neighbour for speed. For a watchlist that means a real
offender can occasionally fail to surface. Tune with ef_search, and verify recall
against a sequential scan before trusting it in production.

The operator class must match the distance operator in the query. recognition.py
uses `<=>` (cosine), so the index below is vector_cosine_ops. If you ever switch
back to `<->`, this index silently stops being used.

    python add_vector_index.py --check    # report size and current query time
    python add_vector_index.py --create   # actually build it
"""

import argparse
import time

import psycopg2

from config import Config

RECOMMENDED_MINIMUM_ROWS = 1000

CREATE_SQL = """
CREATE INDEX IF NOT EXISTS idx_person_faces_embedding_hnsw
    ON person_faces USING hnsw (embedding vector_cosine_ops);
"""


def gallery_size(cur):
    cur.execute(
        "SELECT count(*) FROM person_faces WHERE embedding IS NOT NULL AND use_for_matching"
    )
    return cur.fetchone()[0]


def check(cur):
    rows = gallery_size(cur)
    print(f"matchable references: {rows}")

    cur.execute("SELECT embedding FROM person_faces WHERE embedding IS NOT NULL LIMIT 1")
    sample = cur.fetchone()
    if not sample:
        print("no embeddings stored yet — nothing to index")
        return rows

    started = time.perf_counter()
    cur.execute(
        """
        SELECT id FROM person_faces
        WHERE embedding IS NOT NULL AND use_for_matching
        ORDER BY embedding <=> %s::vector
        LIMIT 1;
        """,
        (sample[0],),
    )
    cur.fetchone()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f"nearest-neighbour query: {elapsed_ms:.1f} ms (sequential scan)")

    cur.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE tablename = 'person_faces' AND indexdef ILIKE '%%hnsw%%';
        """
    )
    existing = [r[0] for r in cur.fetchall()]
    print(f"hnsw indexes present: {existing or 'none'}")

    if rows < RECOMMENDED_MINIMUM_ROWS:
        print(
            f"\nVERDICT: do not index yet. {rows} rows is far below the ~{RECOMMENDED_MINIMUM_ROWS} "
            "where HNSW starts earning its keep, and at this size Postgres will not use it anyway."
        )
    else:
        print(f"\nVERDICT: {rows} rows — worth indexing. Re-run with --create.")
    return rows


def create(cur, conn, force):
    rows = gallery_size(cur)
    if rows < RECOMMENDED_MINIMUM_ROWS and not force:
        raise SystemExit(
            f"Only {rows} matchable references. An HNSW index would not be used and would "
            f"slow enrolment down. Pass --force if you really want it anyway."
        )
    print("building HNSW index (this locks writes on person_faces while it runs)...")
    started = time.perf_counter()
    cur.execute(CREATE_SQL)
    conn.commit()
    print(f"done in {time.perf_counter() - started:.1f}s")
    print("Now verify recall against a sequential scan before relying on it — HNSW is approximate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report size and query time (default)")
    parser.add_argument("--create", action="store_true", help="build the index")
    parser.add_argument("--force", action="store_true", help="build it even on a small gallery")
    args = parser.parse_args()

    connection = psycopg2.connect(Config.DATABASE_URL)
    try:
        with connection.cursor() as cursor:
            if args.create:
                create(cursor, connection, args.force)
            else:
                check(cursor)
    finally:
        connection.close()
