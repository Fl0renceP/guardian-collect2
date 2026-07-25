"""Measure the real error rates of this system on YOUR camera, and pick a threshold.

Every threshold in this project so far has been argued from a handful of
synthetic probes — degraded copies of the seed photos. That is enough to catch a
gross mistake and nowhere near enough to state an error rate. This script
replaces the argument with a measurement.

Give it a folder per person, containing several photos of that person taken on
the camera you will deploy:

    backend/seed_photos/
        keziah/       phone_01.jpg  phone_02.jpg  reprint_01.jpg ...
        thabo/        phone_01.jpg  phone_02.jpg ...
        florence/     ...

Loose files at the top level are ignored, so the existing seed images do not
interfere. Every pair of images WITHIN a folder is a genuine pair; every pair
ACROSS folders is an impostor pair. No pair list to maintain.

What it reports:

    FMR   false match rate — impostor pairs wrongly accepted. The security error:
          an innocent person identified as someone on the watchlist.
    FNMR  false non-match rate — genuine pairs wrongly rejected. The usability
          error: the right person not recognised.

The two trade off directly, which is why a single "accuracy" number is
meaningless here and a target FMR has to be chosen deliberately.

    python calibrate_threshold.py
    python calibrate_threshold.py --root path/to/folders --target-fmr 0.001
    python calibrate_threshold.py --csv results.csv
"""

import argparse
import itertools
import os
import sys

import cv2
import numpy as np

from config import Config
from services import face_geometry, recognition

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_photos")


def load_identities(root, enforce_quality):
    """Embed every image, grouped by the folder it sits in."""
    identities, skipped = {}, []
    if not os.path.isdir(root):
        raise SystemExit(f"No such folder: {root}")

    folders = sorted(
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name)) and not name.startswith(".")
    )
    if not folders:
        raise SystemExit(
            f"No per-person subfolders found in {root}.\n"
            "Expected e.g. seed_photos/keziah/*.jpg — several photos per person, "
            "taken on the deployment camera."
        )

    for person in folders:
        folder = os.path.join(root, person)
        vectors = []
        for filename in sorted(os.listdir(folder)):
            if os.path.splitext(filename)[1].lower() not in IMAGE_EXTENSIONS:
                continue
            path = os.path.join(folder, filename)
            image = cv2.imread(path)
            if image is None:
                skipped.append((person, filename, "unreadable file"))
                continue

            faces = face_geometry.detect_faces(image)
            if not faces:
                skipped.append((person, filename, "no face detected"))
                continue

            aligned = face_geometry.align(image, faces[0].landmarks, 160)
            quality = face_geometry.assess_probe(image, faces[0], aligned)
            if enforce_quality and not quality["decidable"]:
                skipped.append((person, filename, "; ".join(quality["reasons"])))
                continue

            embedding, _, _ = recognition.embed_image(image)
            if embedding is None:
                skipped.append((person, filename, "embedding failed"))
                continue
            vectors.append((filename, np.asarray(embedding, dtype=np.float64)))

        if vectors:
            identities[person] = vectors
        print(f"  {person:20s} {len(vectors):3d} usable image(s)")

    return identities, skipped


def cosine(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def build_pairs(identities):
    genuine, impostor = [], []
    for person, vectors in identities.items():
        for (fa, va), (fb, vb) in itertools.combinations(vectors, 2):
            genuine.append((cosine(va, vb), person, fa, fb))
    for (pa, va_list), (pb, vb_list) in itertools.combinations(identities.items(), 2):
        for fa, va in va_list:
            for fb, vb in vb_list:
                impostor.append((cosine(va, vb), f"{pa} vs {pb}", fa, fb))
    return genuine, impostor


def rates(genuine, impostor, threshold):
    """FMR and FNMR at one threshold. Distances BELOW threshold are accepted."""
    fmr = float(np.mean([d < threshold for d, *_ in impostor])) if impostor else 0.0
    fnmr = float(np.mean([d >= threshold for d, *_ in genuine])) if genuine else 0.0
    return fmr, fnmr


def summarise(genuine, impostor, target_fmr, csv_path=None):
    g = np.array([d for d, *_ in genuine])
    i = np.array([d for d, *_ in impostor])

    print(f"\ngenuine pairs : {len(g):5d}   min {g.min():.4f}  median {np.median(g):.4f}  max {g.max():.4f}")
    print(f"impostor pairs: {len(i):5d}   min {i.min():.4f}  median {np.median(i):.4f}  max {i.max():.4f}")

    overlap = (g.max() >= i.min())
    print(f"\ndistributions overlap: {'YES' if overlap else 'no'}"
          + ("  — no threshold can separate them perfectly" if overlap
             else "  — cleanly separable on this data"))

    # Candidate thresholds: every distance actually observed, so the sweep lands
    # exactly on the points where a decision flips.
    candidates = np.unique(np.concatenate([g, i]))
    rows = [(float(t), *rates(genuine, impostor, t)) for t in candidates]

    print(f"\n{'threshold':>10s} {'FMR':>9s} {'FNMR':>9s}   (FMR = innocent people matched)")
    step = max(1, len(rows) // 18)
    for t, fmr, fnmr in rows[::step]:
        print(f"{t:10.4f} {fmr:9.4f} {fnmr:9.4f}")

    # Equal error rate — where the two curves cross. A neutral reference point,
    # not a recommendation: it weighs a false alarm the same as a miss, and on a
    # watchlist those are rarely equally costly.
    eer_t, eer = min(((t, max(fmr, fnmr)) for t, fmr, fnmr in rows),
                     key=lambda r: abs(r[1]))
    eer_row = min(rows, key=lambda r: abs(r[1] - r[2]))
    print(f"\nEER ~ {max(eer_row[1], eer_row[2]):.4f} at threshold {eer_row[0]:.4f}")

    # ROC AUC via the rank statistic — equivalent to the Mann-Whitney U, and
    # exact without needing to integrate a sampled curve.
    if len(g) and len(i):
        combined = np.concatenate([g, i])
        order = combined.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(combined) + 1)
        # Genuine should rank LOWER (smaller distance), so invert for AUC.
        rank_sum = ranks[:len(g)].sum()
        auc = 1.0 - (rank_sum - len(g) * (len(g) + 1) / 2) / (len(g) * len(i))
        print(f"ROC AUC = {auc:.4f}")

    # The operating point: strictest useful threshold whose FMR stays within target.
    acceptable = [r for r in rows if r[1] <= target_fmr]
    print(f"\n--- operating point at target FMR <= {target_fmr} ---")
    if not acceptable:
        print("  No threshold achieves that false match rate on this data.")
        print(f"  The closest impostor pair is {i.min():.4f}; any threshold at or above")
        print("  that misidentifies someone. Enrol more varied references, improve")
        print("  capture quality, or accept a higher FMR.")
        return
    best = max(acceptable, key=lambda r: r[0])   # loosest threshold still within target
    print(f"  confirmed threshold : {best[0]:.4f}")
    print(f"  expected FMR        : {best[1]:.4f}   ({best[1] * 100:.2f}% of impostor pairs)")
    print(f"  expected FNMR       : {best[2]:.4f}   ({best[2] * 100:.2f}% of genuine pairs missed)")
    print(f"\n  MATCH_THRESHOLD={best[0]:.3f}")

    # A second, looser band for "probable" matches, at ten times the FMR budget.
    loose_target = min(target_fmr * 10, 1.0)
    loose = [r for r in rows if r[1] <= loose_target]
    if loose:
        loose_best = max(loose, key=lambda r: r[0])
        if loose_best[0] > best[0]:
            print(f"  PROBABLE_THRESHOLD={loose_best[0]:.3f}   "
                  f"(FMR {loose_best[1]:.4f}, FNMR {loose_best[2]:.4f})")

    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as fh:
            fh.write("threshold,fmr,fnmr\n")
            for t, fmr, fnmr in rows:
                fh.write(f"{t:.6f},{fmr:.6f},{fnmr:.6f}\n")
        print(f"\n  full sweep written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=DEFAULT_ROOT,
                        help=f"folder containing one subfolder per person (default: {DEFAULT_ROOT})")
    parser.add_argument("--target-fmr", type=float, default=0.01,
                        help="acceptable false match rate, e.g. 0.01 = 1%% of impostor pairs")
    parser.add_argument("--csv", help="write the full threshold sweep here")
    parser.add_argument("--no-quality-gate", action="store_true",
                        help="include images that fail the probe quality gate")
    parser.add_argument("--worst", type=int, default=5,
                        help="how many worst-case pairs to list")
    args = parser.parse_args()

    print(f"root: {args.root}")
    print(f"align={recognition.USE_5POINT_ALIGN}  model={Config.FACE_MODEL}  "
          f"quality gate={'off' if args.no_quality_gate else 'on'}\n")

    identities, skipped = load_identities(args.root, not args.no_quality_gate)

    if skipped:
        print(f"\nskipped {len(skipped)} image(s):")
        for person, filename, reason in skipped[:12]:
            print(f"  {person}/{filename}: {reason}")

    usable = {p: v for p, v in identities.items() if len(v) >= 1}
    with_pairs = {p: v for p, v in usable.items() if len(v) >= 2}
    if len(usable) < 2:
        raise SystemExit("\nNeed at least two people to form impostor pairs.")
    if not with_pairs:
        raise SystemExit("\nNeed at least one person with two or more photos to form genuine pairs.")

    genuine, impostor = build_pairs(usable)
    if not genuine:
        raise SystemExit("\nNo genuine pairs — every person has only one photo.")

    summarise(genuine, impostor, args.target_fmr, args.csv)

    if args.worst:
        print(f"\nworst genuine pairs (hardest to match):")
        for d, person, fa, fb in sorted(genuine, reverse=True)[:args.worst]:
            print(f"  {d:.4f}  {person}: {fa} vs {fb}")
        print(f"\nclosest impostor pairs (most dangerous):")
        for d, pair, fa, fb in sorted(impostor)[:args.worst]:
            print(f"  {d:.4f}  {pair}: {fa} vs {fb}")


if __name__ == "__main__":
    sys.exit(main())
