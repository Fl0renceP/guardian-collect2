"""POPIA retention purge for biometric data.

Biometric data — facial images and facial templates — is *special personal
information* under POPIA, in the same class as health and criminal-behaviour
data. Holding it indefinitely is the default failure mode of a system like this
one, so expiry has to be a scheduled job rather than a good intention.

Two kinds of data expire differently, and conflating them is the mistake this
script exists to avoid:

1. CAPTURES (person_faces rows sourced from a camera). These are the special
   personal information. Past the window they are deleted outright, embedding
   and image both.

2. THE DETECTION LOG (detections rows). This is the audit trail, and §8 of the
   build guide wants it kept — it is the evidence that the system was used
   properly. So the row survives, but it is DE-IDENTIFIED: the matched person
   id, the matched name and the precise coordinates are cleared. What remains
   is "a check happened at this time, at this camera, with this outcome",
   which answers the audit question without naming anybody.

Enrolled reference photos are NOT purged. They are the registry itself,
gathered through a deliberate enrolment step, and deleting them would empty the
gallery rather than minimise retained data. They are identified by their
provenance label — see PROTECTED_SOURCES.

Usage:
    python backend/scripts/purge_biometrics.py                 # dry run, changes nothing
    python backend/scripts/purge_biometrics.py --apply         # perform the purge
    python backend/scripts/purge_biometrics.py --apply --delete-blobs
    python backend/scripts/purge_biometrics.py --capture-days 30 --detection-days 14

Scheduling (Windows Task Scheduler, daily):
    schtasks /create /tn "GuardianPurge" /tr "<venv>\\python.exe <repo>\\backend\\scripts\\purge_biometrics.py --apply --delete-blobs" /sc daily /st 03:00
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402

# Provenance labels that mark a row as an enrolled reference rather than a
# transient capture. Rows carrying one of these are never purged.
PROTECTED_SOURCES = tuple(
    s.strip()
    for s in os.getenv("BIOMETRIC_PROTECTED_SOURCES", "seed_data,enroll,enrolment,enrollment").split(",")
    if s.strip()
)

DEFAULT_CAPTURE_RETENTION_DAYS = int(os.getenv("BIOMETRIC_RETENTION_DAYS", "90"))
DEFAULT_DETECTION_RETENTION_DAYS = int(os.getenv("DETECTION_IDENTITY_RETENTION_DAYS", "30"))


SELECT_EXPIRED_CAPTURES = """
    SELECT id, image_url, source, created_at
    FROM person_faces
    WHERE COALESCE(source, '') <> ALL(%(protected)s)
      AND COALESCE(captured_at, created_at) < now() - (%(days)s || ' days')::interval
    ORDER BY COALESCE(captured_at, created_at);
"""

DELETE_CAPTURES = """
    DELETE FROM person_faces
    WHERE id = ANY(%(ids)s);
"""

COUNT_IDENTIFIED_DETECTIONS = """
    SELECT count(*)
    FROM detections
    WHERE detected_at < now() - (%(days)s || ' days')::interval
      AND (matched_person_id IS NOT NULL
           OR matched_name IS NOT NULL
           OR location_lat IS NOT NULL
           OR location_lng IS NOT NULL);
"""

DEIDENTIFY_DETECTIONS = """
    UPDATE detections
       SET matched_person_id = NULL,
           matched_name      = NULL,
           location_lat      = NULL,
           location_lng      = NULL
     WHERE detected_at < now() - (%(days)s || ' days')::interval
       AND (matched_person_id IS NOT NULL
            OR matched_name IS NOT NULL
            OR location_lat IS NOT NULL
            OR location_lng IS NOT NULL);
"""


def purge(capture_days, detection_days, apply=False, delete_blobs=False):
    if not Config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    conn = psycopg2.connect(Config.DATABASE_URL)
    blob = None
    if delete_blobs:
        # Imported lazily: a purge that only touches the database should not
        # require storage credentials to be present.
        from services.blob_storage import BlobStorageService

        blob = BlobStorageService()

    removed_blobs = 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                SELECT_EXPIRED_CAPTURES,
                {"protected": list(PROTECTED_SOURCES), "days": capture_days},
            )
            expired = cur.fetchall()

            cur.execute(COUNT_IDENTIFIED_DETECTIONS, {"days": detection_days})
            (identified_detections,) = cur.fetchone()

            print(f"Capture retention   : {capture_days} days")
            print(f"Detection retention : {detection_days} days (de-identify, row kept)")
            print(f"Protected sources   : {', '.join(PROTECTED_SOURCES) or '(none)'}")
            print()
            print(f"Expired captures to delete    : {len(expired)}")
            print(f"Detections to de-identify     : {identified_detections}")

            if expired:
                print()
                print("  Oldest few:")
                for row in expired[:5]:
                    print(f"    {row[3]}  source={row[2] or '(none)'}  {row[1][:70]}")

            if not apply:
                print()
                print("Dry run — nothing was changed. Re-run with --apply to perform the purge.")
                return

            if expired:
                if blob is not None:
                    for _, image_url, _, _ in expired:
                        if blob.delete_stored_url(image_url):
                            removed_blobs += 1

                cur.execute(DELETE_CAPTURES, {"ids": [r[0] for r in expired]})

            cur.execute(DEIDENTIFY_DETECTIONS, {"days": detection_days})
            deidentified = cur.rowcount

        conn.commit()
        print()
        print(f"Deleted captures     : {len(expired)}")
        print(f"Deleted blobs        : {removed_blobs}" if delete_blobs else "Deleted blobs        : skipped (--delete-blobs not set)")
        print(f"De-identified rows   : {deidentified}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Perform the purge. Without this the script only reports.")
    parser.add_argument("--delete-blobs", action="store_true", help="Also delete the stored image for each expired capture.")
    parser.add_argument("--capture-days", type=int, default=DEFAULT_CAPTURE_RETENTION_DAYS)
    parser.add_argument("--detection-days", type=int, default=DEFAULT_DETECTION_RETENTION_DAYS)
    args = parser.parse_args()

    purge(
        capture_days=args.capture_days,
        detection_days=args.detection_days,
        apply=args.apply,
        delete_blobs=args.delete_blobs,
    )


if __name__ == "__main__":
    main()
