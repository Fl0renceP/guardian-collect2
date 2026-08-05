"""Tests for the rolling clip buffer.

The property that matters: a clip must contain the seconds BEFORE the flag.
A behavioural flag fires at the end of the behaviour — by the time `loitering`
triggers, the dwell that caused it is already past — so a buffer that starts
recording at the trigger captures the aftermath and misses the event.

    python tests/test_clip_buffer.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from clip_buffer import ClipBuffer  # noqa: E402

FPS = 3.0   # the pipeline samples a few frames a second


def frame(index):
    return np.full((240, 320, 3), (index * 3) % 255, dtype=np.uint8)


def feed(buf, seconds, start=0.0, request_at=None):
    """Push `seconds` of frames; optionally request a clip partway through."""
    count = int(seconds * FPS)
    for i in range(count):
        timestamp = start + i / FPS
        buf.add(timestamp, frame(i))
        if request_at is not None and abs(timestamp - request_at) < (0.5 / FPS):
            buf.request("clip-1", timestamp)
            request_at = None
    return start + (count - 1) / FPS


def test_a_clip_contains_footage_from_before_the_trigger():
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=6.0, post_seconds=3.0, output_dir=out, fps=6.0)
        last = feed(buf, 20.0, request_at=12.0)
        written = buf.tick(last)

        assert written, "the clip was never written"
        path = written[0][1]
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        # 6s before + 3s after at 3fps = ~27 frames, give or take a boundary one.
        assert 24 <= frames <= 30, f"expected ~27 frames of footage, got {frames}"
        assert path.stat().st_size > 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_a_clip_is_not_written_until_the_post_roll_has_passed():
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=5.0, post_seconds=4.0, output_dir=out, fps=6.0)
        feed(buf, 10.0, request_at=8.0)

        # One second after the trigger: still collecting.
        assert buf.tick(9.0) == [], "written before the post-roll completed"
        assert buf.pending_count == 1

        # Four seconds after: ready.
        feed(buf, 4.0, start=10.0)
        assert buf.tick(12.5), "not written once the post-roll had passed"
        assert buf.pending_count == 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_nothing_is_written_without_a_request():
    """Footage is only saved for flags that reached a human. Everything else
    stays in memory and is never written anywhere."""
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=5.0, post_seconds=2.0, output_dir=out, fps=6.0)
        last = feed(buf, 30.0)
        assert buf.tick(last) == []
        assert not list(Path(out).glob("*.mp4"))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_buffer_stays_within_its_byte_cap():
    """A camera running for hours must not slowly consume the machine."""
    out = tempfile.mkdtemp()
    try:
        cap_bytes = 256 * 1024
        buf = ClipBuffer(pre_seconds=600.0, post_seconds=5.0, output_dir=out,
                         max_bytes=cap_bytes)
        feed(buf, 120.0)
        assert buf.stats()["megabytes"] * 1024 * 1024 <= cap_bytes * 1.05
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_old_frames_fall_out_of_the_window():
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=5.0, post_seconds=2.0, output_dir=out)
        feed(buf, 60.0)
        # Only the pre+post window (plus a small margin) is retained.
        assert buf.stats()["seconds_buffered"] <= 5.0 + 2.0 + 3.0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_flush_writes_a_clip_whose_post_roll_never_arrived():
    """A flag in the final seconds of a run still gets whatever footage exists."""
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=5.0, post_seconds=10.0, output_dir=out, fps=6.0)
        feed(buf, 12.0, request_at=11.0)
        assert buf.tick(11.5) == [], "post-roll had not passed"
        written = buf.flush()
        assert written, "flush did not write the outstanding clip"
        assert buf.pending_count == 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_a_request_with_no_footage_is_skipped_quietly():
    out = tempfile.mkdtemp()
    try:
        buf = ClipBuffer(pre_seconds=5.0, post_seconds=1.0, output_dir=out)
        buf.request("empty", 0.0)
        assert buf.tick(10.0) == []
        assert not list(Path(out).glob("*.mp4"))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
