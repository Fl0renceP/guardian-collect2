"""Recompute every stored face embedding with the current Config.FACE_DETECTOR.

Run this after changing FACE_DETECTOR. The detector decides the crop and the
crop decides the embedding, so a gallery embedded with one detector and probed
with another is comparing slightly different things. Measured on the seed set
the mismatch still matched (yunet probe vs retinaface gallery landed at
0.018-0.049, well under the 0.30 threshold), but there is no reason to run with
a known inconsistency, and the margin is not guaranteed to hold on a larger
gallery.

Images come from Azure Blob via the URL already stored on each row, so this does
not depend on backend/seed_photos still being present.

    python backend/scripts/reembed_faces.py --dry-run
    python backend/scripts/reembed_faces.py
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg2  # noqa: E402
import requests  # noqa: E402
from deepface import DeepFace  # noqa: E402

from config import Config  # noqa: E402
from services.blob_storage import BlobStorageService  # noqa: E402
from services.recognition import represent_face  # noqa: E402


def _largest_face(objs):
    def area(face):
        box = face.get("facial_area") or {}
        return int(box.get("w", 0)) * int(box.get("h", 0))

    return max(objs, key=area)


def _embed_url(url, signer):
    """Download one stored image and return its embedding under the new detector."""
    readable = url
    if signer is not None:
        try:
            readable = signer.sign_stored_url(url)
        except Exception:
            readable = url

    response = requests.get(readable, timeout=30)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    try:
        # The same cascade the scan endpoint uses. It matters here: the off-angle
        # references are exactly the ones the primary detector cannot see, so
        # embedding them through the cascade is what keeps them comparable to the
        # probe that will also have escalated to reach them.
        objs, detector = represent_face(tmp_path, model_name=Config.FACE_MODEL)
    finally:
        os.remove(tmp_path)

    if not objs:
        raise ValueError("No face detected.")
    return _largest_face(objs)["embedding"], detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Re-embed and report, but roll back instead of committing.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N face rows.")
    args = parser.parse_args()

    if not Config.DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set.")

    print(f"detector: {Config.FACE_DETECTOR}   model: {Config.FACE_MODEL}")
    if args.dry_run:
        print("DRY RUN — no changes will be committed.\n")

    try:
        signer = BlobStorageService()
    except Exception as exc:
        print(f"Blob service unavailable ({exc}); using stored URLs as-is.")
        signer = None

    DeepFace.build_model(Config.FACE_MODEL)

    conn = psycopg2.connect(Config.DATABASE_URL)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT pf.id, pf.image_url, p.full_name
        FROM person_faces pf JOIN persons p ON p.id = pf.person_id
        WHERE pf.image_url IS NOT NULL
        ORDER BY p.full_name, pf.id;
        """
    )
    rows = cursor.fetchall()
    if args.limit:
        rows = rows[: args.limit]

    updated = failed = 0
    started = time.perf_counter()

    escalated = 0
    for face_id, image_url, full_name in rows:
        try:
            embedding, detector = _embed_url(image_url, signer)
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {full_name:<24} {str(exc)[:70]}")
            continue

        cursor.execute(
            "UPDATE person_faces SET embedding = %(vec)s::vector WHERE id = %(id)s;",
            {"vec": str(embedding), "id": face_id},
        )
        updated += 1
        if detector != Config.FACE_DETECTOR:
            escalated += 1
        print(f"  ok    {full_name:<24} {str(face_id)[:8]}  [{detector}]")

    # persons.face_embedding is the older one-vector-per-person column. The
    # matching query UNIONs it in, so leaving it on the old detector would put a
    # stale-crop vector back into the same search this script exists to align.
    cursor.execute(
        """
        SELECT p.id, p.full_name, MIN(pf.image_url)
        FROM persons p JOIN person_faces pf ON pf.person_id = p.id
        WHERE p.face_embedding IS NOT NULL AND pf.image_url IS NOT NULL
        GROUP BY p.id, p.full_name;
        """
    )
    legacy = cursor.fetchall()
    for person_id, full_name, image_url in legacy:
        try:
            embedding, _ = _embed_url(image_url, signer)
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {full_name:<24} (persons.face_embedding) {str(exc)[:50]}")
            continue
        cursor.execute(
            "UPDATE persons SET face_embedding = %(vec)s::vector WHERE id = %(id)s;",
            {"vec": str(embedding), "id": person_id},
        )
        updated += 1
        print(f"  ok    {full_name:<24} (persons.face_embedding)")

    elapsed = time.perf_counter() - started
    if escalated:
        print(f"\n{escalated} image(s) needed the '{Config.FACE_DETECTOR_FALLBACK}' fallback — "
              f"'{Config.FACE_DETECTOR}' found no face in them.")
    if args.dry_run:
        conn.rollback()
        print(f"\nDRY RUN complete — rolled back. {updated} would update, {failed} failed, {elapsed:.1f}s")
    else:
        conn.commit()
        print(f"\nCommitted. {updated} embeddings updated, {failed} failed, {elapsed:.1f}s")

    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()
