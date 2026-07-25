"""Recompute stored reference embeddings with the current detector.

Run this after changing FACE_DETECTOR.

Detectors crop faces differently — a slightly different box, a slightly different
alignment — so an embedding made with one and compared against a scan made with
another carries a small systematic offset. Measured after switching retinaface to
yunet, the same probes moved from 0.0074-0.0161 to 0.0337-0.0425 against a 0.30
threshold: still a comfortable match, but the drift is real and it eats margin
you would rather keep for genuine variation in the person's appearance.

Each capture is re-fetched from blob storage and re-embedded, so the reference
ends up in exactly the space live scans produce.

Only rows with use_for_matching are touched. Evidence-only captures are left
alone — they never take part in matching, so their vectors do not matter.

    python reembed_references.py            # dry run, reports what would change
    python reembed_references.py --confirm  # write
"""

import argparse

import psycopg2

from config import Config
from services import recognition
from services.blob_storage import BlobStorageService


def blob_name_from_url(url, container_name):
    marker = f"/{container_name}/"
    return url.split(marker, 1)[1] if marker in url else None


def largest(objs):
    def area(face):
        box = face.get("facial_area") or {}
        return box.get("w", 0) * box.get("h", 0)
    return max(objs, key=area)


def reembed(confirm):
    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    blob = BlobStorageService()
    container = blob.container_client.container_name

    print(f"detector: yunet (5-point)  align: {recognition.USE_5POINT_ALIGN}  "
          f"model: {Config.FACE_MODEL}")
    print(f"container: {container}\n")

    try:
        cur.execute(
            """
            SELECT pf.id, p.full_name, pf.image_url, pf.embedding IS NOT NULL
            FROM person_faces pf
            JOIN persons p ON p.id = pf.person_id
            WHERE pf.use_for_matching
            ORDER BY p.full_name, pf.created_at;
            """
        )
        rows = cur.fetchall()
        if not rows:
            print("no matchable references found")
            return

        changed = failed = 0
        for face_id, full_name, image_url, had_embedding in rows:
            blob_name = blob_name_from_url(image_url, container)
            if not blob_name:
                print(f"  SKIP  {full_name}: {image_url} is not in container '{container}'")
                continue

            try:
                image_bytes = blob.container_client.download_blob(blob_name).readall()
            except Exception as exc:
                print(f"  FAIL  {full_name}: could not download {blob_name}: {exc}")
                failed += 1
                continue

            # Exactly the same call a live scan makes — detect, 5-point align,
            # embed. A reference produced by any other route stops lining up with
            # the probes it is compared against.
            import cv2
            import numpy as np

            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"  FAIL  {full_name}: {blob_name} is not a decodable image")
                failed += 1
                continue

            embedding, _, _ = recognition.embed_image(image)
            if embedding is None:
                print(f"  FAIL  {full_name}: no face found in {blob_name}")
                failed += 1
                continue

            new_vector = str(embedding)

            # How far the reference moved. A large shift means the two detectors
            # disagreed considerably about where this face is.
            drift = None
            if had_embedding:
                cur.execute(
                    "SELECT embedding <=> %s::vector FROM person_faces WHERE id = %s",
                    (new_vector, face_id),
                )
                drift = float(cur.fetchone()[0])

            label = f"{full_name} / {blob_name.rsplit('/', 1)[-1]}"
            drift_text = f"moved {drift:.4f}" if drift is not None else "was empty"
            print(f"  {'OK  ' if confirm else 'DRY '}  {label:52s} {drift_text}")

            if confirm:
                cur.execute(
                    "UPDATE person_faces SET embedding = %s::vector WHERE id = %s",
                    (new_vector, face_id),
                )
                conn.commit()
            changed += 1

        print(f"\n{'updated' if confirm else 'would update'} {changed} reference(s), {failed} failed")
        if not confirm:
            print("Dry run — nothing written. Re-run with --confirm to apply.")
        else:
            # persons.face_embedding is the legacy column and is no longer used
            # for matching, so it is deliberately left as-is.
            print("Note: persons.face_embedding is legacy and unused by matching; left untouched.")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--confirm", action="store_true", help="actually write the new embeddings")
    args = parser.parse_args()
    reembed(args.confirm)
