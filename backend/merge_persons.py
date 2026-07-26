"""Merge two registry records that turn out to be the same human.

You will get duplicates. Someone is entered twice under a slightly different name,
or captures are filed against a new record before anyone realises who it is. Left
alone, the duplicates split that person's sightings across two identities and both
match worse than one combined record would.

This moves every capture from --source onto --target, then deletes the now-empty
source record. Dry run by default; pass --confirm to actually write.

    python merge_persons.py --source "New Community Member" --target "Tinashe Madanire"
    python merge_persons.py --source "..." --target "..." --confirm
"""

import argparse

import psycopg2

from config import Config


def _resolve(cur, name_or_id):
    """Accept either an exact full_name or a UUID."""
    cur.execute(
        "SELECT id, full_name, status FROM persons WHERE full_name = %s OR id::text = %s",
        (name_or_id, name_or_id),
    )
    rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No person matching '{name_or_id}'.")
    if len(rows) > 1:
        raise SystemExit(f"'{name_or_id}' matches {len(rows)} records — use the UUID instead.")
    return rows[0]


def merge(source_key, target_key, confirm):
    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    try:
        source_id, source_name, source_status = _resolve(cur, source_key)
        target_id, target_name, target_status = _resolve(cur, target_key)

        if source_id == target_id:
            raise SystemExit("Source and target are the same record.")

        cur.execute("SELECT count(*) FROM person_faces WHERE person_id = %s", (source_id,))
        moving = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM person_faces WHERE person_id = %s", (target_id,))
        existing = cur.fetchone()[0]

        print(f"source: {source_name} ({source_status})  {moving} capture(s)")
        print(f"target: {target_name} ({target_status})  {existing} capture(s)")
        print(f"result: {target_name} would hold {moving + existing} capture(s)")

        if source_status != target_status:
            print(f"\nNOTE: statuses differ ('{source_status}' vs '{target_status}'). "
                  f"The merged record keeps the target's status: '{target_status}'. "
                  "Check that is the one you want — this decides whether they trigger alerts.")

        # persons.face_embedding is the legacy single-vector column and is not
        # carried across: deleting the source row drops it. That only loses
        # something if the source has captures whose own embedding is missing.
        cur.execute(
            """
            SELECT count(*) FROM person_faces
            WHERE person_id = %s AND embedding IS NULL
            """,
            (source_id,),
        )
        orphaned = cur.fetchone()[0]
        if orphaned:
            print(f"\nWARNING: {orphaned} of the source's captures have no embedding of their "
                  "own. They will move across but stay unusable for matching until re-enrolled.")

        if not confirm:
            print("\nDry run — nothing written. Re-run with --confirm to apply.")
            return

        cur.execute(
            "UPDATE person_faces SET person_id = %s WHERE person_id = %s",
            (target_id, source_id),
        )
        moved = cur.rowcount
        cur.execute("DELETE FROM persons WHERE id = %s", (source_id,))
        conn.commit()

        cur.execute(
            """
            SELECT count(*), count(*) FILTER (WHERE use_for_matching AND embedding IS NOT NULL)
            FROM person_faces WHERE person_id = %s
            """,
            (target_id,),
        )
        total, refs = cur.fetchone()
        print(f"\nMoved {moved} capture(s) onto {target_name}; deleted record '{source_name}'.")
        print(f"{target_name} now has {total} capture(s), {refs} usable as references.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="record to merge FROM (will be deleted)")
    parser.add_argument("--target", required=True, help="record to merge INTO (kept)")
    parser.add_argument("--confirm", action="store_true", help="actually apply the merge")
    args = parser.parse_args()
    merge(args.source, args.target, args.confirm)
