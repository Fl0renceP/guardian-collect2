"""Incrementally import seed face photos into PostgreSQL + Azure Blob Storage.

Expected filename format inside backend/seed_photos:
    Full Name - angle label - status.ext
Example:
    Tadiwa Banda - front 1 - verified.jpeg

Behavior:
- Existing person rows keep their current status (DB status is canonical).
- New person rows are created from the filename's status.
- Each image is uploaded to Blob with metadata labels and inserted into person_faces.
- Quality gate is enforced for matching references; failures are stored as evidence only.
- Re-runs are idempotent via deterministic blob names + duplicate checks.

Usage:
    python backend/scripts/import_seed_faces.py
    python backend/scripts/import_seed_faces.py --dry-run
    python backend/scripts/import_seed_faces.py --limit 10
"""

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import psycopg2
from deepface import DeepFace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config  # noqa: E402
from services.blob_storage import BlobStorageService, CONTAINER_NAME  # noqa: E402
from services.face_quality import assess as assess_quality  # noqa: E402

VALID_STATUSES = {"offender", "suspect", "verified"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@dataclass
class ParsedSeedFile:
    full_name: str
    angle_label: str
    status: str
    path: Path


def _normalize_spaces(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", _normalize_spaces(value).lower()).strip("_")


def _parse_seed_filename(path):
    stem = _normalize_spaces(path.stem)
    match = re.match(r"^(?P<name>.+?)\s*-\s*(?P<angle>.+?)\s*-\s*(?P<status>offender|suspect|verified)$", stem, re.IGNORECASE)
    if not match:
        return None
    return ParsedSeedFile(
        full_name=_normalize_spaces(match.group("name")),
        angle_label=_normalize_spaces(match.group("angle")),
        status=match.group("status").lower(),
        path=path,
    )


def _iter_seed_files(seed_dir):
    for path in sorted(seed_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _read_image_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _embedding_from_file(path):
    objs = DeepFace.represent(
        img_path=str(path),
        model_name=Config.FACE_MODEL,
        detector_backend="retinaface",
        enforce_detection=True,
    )
    if not objs:
        raise ValueError("No face detected by DeepFace.")

    def area(face):
        box = face.get("facial_area") or {}
        return int(box.get("w", 0)) * int(box.get("h", 0))

    subject = max(objs, key=area)
    return subject


def _build_blob_name(parsed, image_bytes):
    digest = hashlib.sha1(image_bytes).hexdigest()[:16]
    person_slug = _slug(parsed.full_name)
    angle_slug = _slug(parsed.angle_label)
    ext = parsed.path.suffix.lower() or ".jpg"
    return f"faces/seed/{person_slug}/{parsed.status}/{digest}_{angle_slug}{ext}"


def _schema_columns(cur, table_name):
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s;
        """,
        (table_name,),
    )
    return {row[0] for row in cur.fetchall()}


def _ensure_schema(cur):
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
    if not cur.fetchone():
        raise RuntimeError("pgvector extension is not installed. Run CREATE EXTENSION vector.")

    person_cols = _schema_columns(cur, "persons")
    face_cols = _schema_columns(cur, "person_faces")

    required_persons = {"id", "full_name", "status"}
    required_faces = {
        "person_id",
        "image_url",
        "embedding",
        "source",
        "use_for_matching",
        "quality_score",
        "face_pixels",
        "det_confidence",
        "blur_variance",
        "blur_directional_ratio",
    }

    missing_persons = sorted(required_persons - person_cols)
    missing_faces = sorted(required_faces - face_cols)

    if missing_persons or missing_faces:
        parts = []
        if missing_persons:
            parts.append(f"persons missing: {', '.join(missing_persons)}")
        if missing_faces:
            parts.append(f"person_faces missing: {', '.join(missing_faces)}")
        raise RuntimeError(
            "Schema is not ready for multi-image import (" + "; ".join(parts) + "). "
            "Run migrate_multi_face.py and migrate_capture_metadata.py first."
        )


def _resolve_person(cur, full_name, parsed_status):
    cur.execute("SELECT id, status FROM persons WHERE full_name = %s", (full_name,))
    rows = cur.fetchall()
    if len(rows) > 1:
        raise RuntimeError(f"Name '{full_name}' is not unique in persons; merge duplicates first.")

    if rows:
        person_id, existing_status = rows[0]
        return person_id, existing_status, False

    cur.execute(
        """
        INSERT INTO persons (full_name, status)
        VALUES (%s, %s)
        RETURNING id, status;
        """,
        (full_name, parsed_status),
    )
    person_id, status = cur.fetchone()
    return person_id, status, True


def _already_imported(cur, person_id, source, blob_name):
    cur.execute(
        """
        SELECT 1
        FROM person_faces
        WHERE person_id = %s
          AND source = %s
          AND image_url LIKE %s
        LIMIT 1;
        """,
        (person_id, source, f"%/{blob_name}"),
    )
    return cur.fetchone() is not None


def run_import(seed_dir, source, dry_run=False, limit=None, force_references=False):
    if not Config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    blob = None
    if not dry_run:
        blob = BlobStorageService()

    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()

    stats = {
        "files_seen": 0,
        "parsed": 0,
        "invalid_name": 0,
        "new_people": 0,
        "imported": 0,
        "skipped_duplicate": 0,
        "mismatched_status": 0,
        "quality_reference": 0,
        "quality_evidence_only": 0,
        "failed": 0,
    }

    rejects = []

    try:
        _ensure_schema(cur)

        for path in _iter_seed_files(seed_dir):
            if limit is not None and stats["files_seen"] >= limit:
                break

            stats["files_seen"] += 1
            parsed = _parse_seed_filename(path)
            if not parsed:
                stats["invalid_name"] += 1
                rejects.append((path.name, "invalid filename format"))
                continue
            if parsed.status not in VALID_STATUSES:
                stats["invalid_name"] += 1
                rejects.append((path.name, f"invalid status '{parsed.status}'"))
                continue

            stats["parsed"] += 1

            try:
                image_bytes = _read_image_bytes(path)
                blob_name = _build_blob_name(parsed, image_bytes)

                person_id, db_status, created = _resolve_person(cur, parsed.full_name, parsed.status)
                if created:
                    stats["new_people"] += 1

                if db_status != parsed.status:
                    stats["mismatched_status"] += 1

                if _already_imported(cur, person_id, source, blob_name):
                    stats["skipped_duplicate"] += 1
                    continue

                subject = _embedding_from_file(path)
                image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("OpenCV could not decode image bytes.")
                quality = assess_quality(image, subject.get("facial_area"), subject.get("face_confidence"))

                use_for_matching = bool(quality["passes"])
                if force_references:
                    use_for_matching = True

                metadata = {
                    "full_name": parsed.full_name,
                    "person_status": db_status,
                    "source": source,
                    "angle_label": parsed.angle_label,
                    "original_filename": path.name,
                }

                if dry_run:
                    blob_url = f"https://<storage-account>.blob.core.windows.net/{CONTAINER_NAME}/{blob_name}"
                else:
                    if blob is None:
                        raise RuntimeError("Blob client not initialized.")
                    blob_url = blob.upload_image(image_bytes, filename=blob_name, metadata=metadata)

                cur.execute(
                    """
                    INSERT INTO person_faces
                        (person_id, image_url, embedding, source, use_for_matching,
                         quality_score, face_pixels, det_confidence, blur_variance,
                         blur_directional_ratio)
                    VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        person_id,
                        blob_url,
                        str(subject["embedding"]),
                        source,
                        use_for_matching,
                        quality["quality_score"],
                        quality["face_pixels"],
                        quality["det_confidence"],
                        quality["blur_variance"],
                        quality["blur_directional_ratio"],
                    ),
                )
                cur.fetchone()

                # Keep legacy one-vector-per-person column populated for older paths.
                cur.execute(
                    """
                    UPDATE persons
                    SET face_embedding = COALESCE(face_embedding, %s::vector),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s;
                    """,
                    (str(subject["embedding"]), person_id),
                )

                stats["imported"] += 1
                if use_for_matching:
                    stats["quality_reference"] += 1
                else:
                    stats["quality_evidence_only"] += 1

            except Exception as exc:
                stats["failed"] += 1
                rejects.append((path.name, str(exc)))

        if dry_run:
            conn.rollback()
        else:
            conn.commit()

    finally:
        cur.close()
        conn.close()

    print("\n=== Seed Face Import Summary ===")
    print(f"seed_dir: {seed_dir}")
    print(f"dry_run: {dry_run}")
    print(f"files_seen: {stats['files_seen']}")
    print(f"parsed: {stats['parsed']}")
    print(f"invalid_name: {stats['invalid_name']}")
    print(f"new_people: {stats['new_people']}")
    print(f"imported: {stats['imported']}")
    print(f"skipped_duplicate: {stats['skipped_duplicate']}")
    print(f"mismatched_status(db_wins): {stats['mismatched_status']}")
    print(f"quality_reference: {stats['quality_reference']}")
    print(f"quality_evidence_only: {stats['quality_evidence_only']}")
    print(f"failed: {stats['failed']}")

    if rejects:
        print("\nRejected/failed files:")
        for name, reason in rejects:
            print(f"  - {name}: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dir",
        default=str(Path(__file__).resolve().parent.parent / "seed_photos"),
        help="Directory containing seed face images.",
    )
    parser.add_argument(
        "--source",
        default="seed_data",
        help="Provenance label stored in person_faces.source.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse + evaluate only; rollback all DB writes.")
    parser.add_argument("--limit", type=int, help="Process at most N files.")
    parser.add_argument(
        "--force-references",
        action="store_true",
        help="Store every image as use_for_matching=True even if quality fails.",
    )
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    if not seed_dir.exists() or not seed_dir.is_dir():
        raise SystemExit(f"Seed directory not found: {seed_dir}")

    run_import(
        seed_dir=seed_dir,
        source=args.source,
        dry_run=args.dry_run,
        limit=args.limit,
        force_references=args.force_references,
    )


if __name__ == "__main__":
    main()
