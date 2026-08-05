"""Demo script — run the pipeline on a sample clip with the debug window open.

    python run_sample.py                    # first clip found in ../behaviour analysis/
    python run_sample.py --list             # show the clips it can see
    python run_sample.py --clip 2 --face-confidence 0.8 --face-label verified

This is the one to run in front of judges. It opens an OpenCV window showing
bounding boxes, pose skeletons and zone overlays, prints every detected event
with its plain-English explanation, and finishes with a summary.

A note on speed: YOLO runs on the CPU here at roughly one frame per second on a
laptop without a GPU, so the default `frame_stride` in config.yaml skips frames
to keep the window moving. Timestamps come from the video, not the wall clock,
so skipping frames costs temporal resolution but never changes what a threshold
means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import main as cli  # noqa: E402

# Where the team's sample clips live. Not moved into this module on purpose —
# they are shared demo material, not part of the code.
SAMPLE_DIRS = (
    MODULE_DIR.parent / "behaviour analysis",
    MODULE_DIR / "sample_videos",
)
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def find_clips() -> list[Path]:
    clips: list[Path] = []
    for directory in SAMPLE_DIRS:
        if directory.is_dir():
            clips.extend(
                sorted(p for p in directory.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
            )
    return clips


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the behavioural analysis demo on a sample clip.",
    )
    parser.add_argument("--clip", type=int, default=1,
                        help="Which sample clip to use, 1-based (default 1).")
    parser.add_argument("--list", action="store_true", help="List available clips and exit.")
    parser.add_argument("--source", default=None,
                        help="Use this file or camera index instead of a sample clip.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip the debug window (headless run).")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--save-json", default="sample_events.json")
    parser.add_argument("--face-confidence", type=float, default=None)
    parser.add_argument("--face-label", default=None,
                        choices=["offender", "suspect", "verified"])
    args = parser.parse_args(argv)

    clips = find_clips()
    if args.list or (not clips and not args.source):
        if not clips:
            print("No sample clips found. Looked in:")
            for directory in SAMPLE_DIRS:
                print(f"  - {directory}")
            print("\nPass one directly:  python run_sample.py --source path/to/clip.mp4")
            return 0 if args.list else 1
        print("Sample clips:")
        for i, clip in enumerate(clips, start=1):
            size_mb = clip.stat().st_size / (1024 * 1024)
            print(f"  {i}. {clip.name}  ({size_mb:.1f} MB)")
        return 0

    if args.source:
        source = args.source
    else:
        index = max(1, min(args.clip, len(clips)))
        source = str(clips[index - 1])
        print(f"Using sample clip {index}: {clips[index - 1].name}")

    argv_out = ["--source", source, "--save-json", args.save_json]
    if not args.no_show:
        argv_out.append("--show")
    if args.max_seconds is not None:
        argv_out += ["--max-seconds", str(args.max_seconds)]
    if args.stride is not None:
        argv_out += ["--stride", str(args.stride)]
    if args.face_confidence is not None:
        argv_out += ["--face-confidence", str(args.face_confidence)]
    if args.face_label is not None:
        argv_out += ["--face-label", args.face_label]

    return cli.main(argv_out)


if __name__ == "__main__":
    raise SystemExit(main())
