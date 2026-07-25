"""Attach a captured face to a person in the identity registry.

This is the deliberate admin action that replaced auto-registration: adding
someone to the registry is a decision a human makes, never a side-effect of a
camera scan.

Built for the CCTV case — one person, many sightings over time. Each capture is
stored with its own embedding plus where it came from, and the scan endpoint
matches against all of them, so every confirmed sighting makes the next
identification easier.

Captures are quality-gated. One that fails is still stored as case evidence but
marked use_for_matching = FALSE, because a 40px motion-blurred grab is genuine
evidence and a genuinely harmful matching reference.

    python enroll_face.py --person "Victoria Armstrong" --image frame.jpg \
        --camera-id CAM-07 --captured-at 2026-07-20T21:14:00 --incident-ref INC-0142
    python enroll_face.py --person "Tinashe Madanire" --image blurry.jpg --force
    python enroll_face.py --list
    python enroll_face.py --show "Victoria Armstrong"
"""

import argparse
import os
import uuid

import cv2
import numpy as np
import psycopg2
from deepface import DeepFace

from config import Config
from services.blob_storage import BlobStorageService
from services.face_quality import assess


def list_gallery(cur):
    cur.execute(
        """
        SELECT p.full_name, p.status,
               count(pf.id) AS captures,
               count(*) FILTER (WHERE pf.use_for_matching AND pf.embedding IS NOT NULL) AS refs
        FROM persons p
        LEFT JOIN person_faces pf ON pf.person_id = p.id
        GROUP BY p.id, p.full_name, p.status
        ORDER BY p.full_name;
        """
    )
    print(f"{'name':24s} {'status':10s} {'captures':>9s} {'references':>11s}")
    for name, status, captures, refs in cur.fetchall():
        print(f"{name:24s} {status:10s} {captures:>9d} {refs:>11d}")


def show_person(cur, person_name):
    cur.execute("SELECT id, full_name, status FROM persons WHERE full_name = %s", (person_name,))
    row = cur.fetchone()
    if not row:
        print(f"No person named '{person_name}'.")
        return
    person_id, full_name, status = row
    print(f"{full_name} ({status})\n")
    cur.execute(
        """
        SELECT source, camera_id, captured_at, incident_ref, quality_score,
               face_pixels, use_for_matching, image_url
        FROM person_faces WHERE person_id = %s ORDER BY created_at;
        """,
        (person_id,),
    )
    for src, cam, at, ref, q, px, usable, url in cur.fetchall():
        flag = "reference" if usable else "EVIDENCE ONLY"
        when = at.strftime("%Y-%m-%d %H:%M") if at else "-"
        print(f"  [{flag:13s}] source={src or '-'} camera={cam or '-'} at={when} "
              f"incident={ref or '-'} quality={q if q is not None else '-'} px={px or '-'}")
        print(f"                  {url}")


def enroll(person_name, image_path, source, camera_id, captured_at, incident_ref, force):
    if not os.path.isfile(image_path):
        raise SystemExit(f"No such file: {image_path}")

    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, full_name, status FROM persons WHERE full_name = %s", (person_name,))
        rows = cur.fetchall()
        if not rows:
            print(f"No person named '{person_name}'. Known people:\n")
            list_gallery(cur)
            raise SystemExit(1)
        if len(rows) > 1:
            raise SystemExit(f"'{person_name}' matches {len(rows)} people — names are not unique.")
        person_id, full_name, status = rows[0]

        image = cv2.imread(image_path)
        if image is None:
            raise SystemExit(f"Could not read {image_path} as an image.")

        # Same model and detector the scan endpoint uses, so the stored vector is
        # directly comparable to what a live scan produces.
        objs = DeepFace.represent(
            img_path=image_path,
            model_name=Config.FACE_MODEL,
            detector_backend="retinaface",
            enforce_detection=True,
        )
        if not objs:
            raise SystemExit("No face detected — nothing enrolled.")

        def area(face):
            box = face.get("facial_area") or {}
            return box.get("w", 0) * box.get("h", 0)

        subject = max(objs, key=area)
        if len(objs) > 1:
            print(f"WARNING: {len(objs)} faces found; using the largest. Crop the frame if that's wrong.")

        quality = assess(image, subject.get("facial_area"), subject.get("face_confidence"))

        print(f"quality: score={quality['quality_score']} face={quality['face_pixels']}px "
              f"sharpness={quality['blur_variance']} balance={quality['blur_directional_ratio']} "
              f"confidence={quality['det_confidence']}")

        use_for_matching = quality["passes"]
        if not quality["passes"]:
            print("\nThis capture FAILED the quality gate:")
            for reason in quality["reasons"]:
                print(f"  - {reason}")
            if force:
                use_for_matching = True
                print("\n--force given: enrolling it as a matching reference anyway.")
                print("Be aware a poor reference can pull unrelated people into false matches.")
            else:
                print("\nStoring it as evidence only (use_for_matching = FALSE).")
                print("It will be kept against this person but never used to identify anyone.")
                print("Re-run with --force to override.")

        with open(image_path, "rb") as fh:
            image_bytes = fh.read()
        ext = os.path.splitext(image_path)[1] or ".jpg"
        blob_name = f"captures/{full_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}{ext}"
        blob_url = BlobStorageService().upload_image(image_bytes, filename=blob_name)

        cur.execute(
            """
            INSERT INTO person_faces
                (person_id, image_url, embedding, source, use_for_matching,
                 quality_score, face_pixels, det_confidence, blur_variance,
                 blur_directional_ratio, camera_id, captured_at, incident_ref)
            VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                person_id, blob_url, str(subject["embedding"]), source, use_for_matching,
                quality["quality_score"], quality["face_pixels"], quality["det_confidence"],
                quality["blur_variance"], quality["blur_directional_ratio"],
                camera_id, captured_at, incident_ref,
            ),
        )
        face_id = cur.fetchone()[0]
        conn.commit()

        cur.execute(
            """
            SELECT count(*) FILTER (WHERE use_for_matching AND embedding IS NOT NULL),
                   count(*)
            FROM person_faces WHERE person_id = %s
            """,
            (person_id,),
        )
        refs, total = cur.fetchone()
        role = "matching reference" if use_for_matching else "evidence only"
        print(f"\nStored capture {face_id} for {full_name} ({status}) as {role}")
        print(f"  {blob_url}")
        print(f"  {full_name} now has {total} capture(s), {refs} usable as references")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person", help="exact full_name of an existing person")
    parser.add_argument("--image", help="path to the captured frame")
    parser.add_argument("--source", default="camera_capture",
                        help="provenance tag, e.g. seed_data / camera_capture / SAPS_reference")
    parser.add_argument("--camera-id", help="which camera or stream this came from")
    parser.add_argument("--captured-at", help="ISO timestamp of the capture, e.g. 2026-07-20T21:14:00")
    parser.add_argument("--incident-ref", help="linked claim/incident reference, e.g. INC-0142")
    parser.add_argument("--force", action="store_true",
                        help="enrol as a matching reference even if it fails the quality gate")
    parser.add_argument("--list", action="store_true", help="list people with capture counts")
    parser.add_argument("--show", help="show every stored capture for one person")
    args = parser.parse_args()

    if args.list or args.show:
        c = psycopg2.connect(Config.DATABASE_URL)
        try:
            with c.cursor() as cursor:
                if args.show:
                    show_person(cursor, args.show)
                else:
                    list_gallery(cursor)
        finally:
            c.close()
    elif args.person and args.image:
        enroll(args.person, args.image, args.source, args.camera_id,
               args.captured_at, args.incident_ref, args.force)
    else:
        parser.error("need --person and --image (or --list / --show)")
