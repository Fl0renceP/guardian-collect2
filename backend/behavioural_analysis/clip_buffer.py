"""A rolling clip buffer, so a flag can be seen as well as read.

THE PROBLEM THIS SOLVES: a behavioural flag fires at the END of the behaviour.
By the time `loitering` triggers, the 45 seconds that caused it are already
past. Recording from the trigger onward captures the aftermath and misses the
event entirely — so frames are held in a ring buffer continuously, and a clip is
cut from BEFORE the trigger once the trailing seconds have also been seen.

Frames are held JPEG-encoded, not raw. A raw 848x480 frame is 1.2MB, so thirty
seconds at 3fps is over 100MB of RAM; the same frames as JPEG are about 4MB.
The buffer is also byte-capped, because a long-running camera must not slowly
consume the machine it is watching from.

RETENTION: the buffer is memory only and holds a rolling window of seconds. It
is not a recording of the day. Clips are written only for flags that reached a
human, and the backend deletes those on its own schedule.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PendingClip:
    """A clip that has been requested but is still collecting its trailing seconds."""

    clip_id: str
    trigger_time: float
    ready_at: float          # stream time when the post-roll is complete
    label: str = ""


@dataclass
class ClipBuffer:
    """Ring buffer of recent frames, and the clips cut from it."""

    pre_seconds: float = 12.0
    post_seconds: float = 5.0
    jpeg_quality: int = 70
    max_bytes: int = 96 * 1024 * 1024
    output_dir: Path = field(default_factory=lambda: Path("clips"))
    fps: float = 6.0          # playback rate for written clips

    _frames: Deque[Tuple[float, bytes]] = field(default_factory=deque, init=False)
    _bytes: int = field(default=0, init=False)
    _pending: List[PendingClip] = field(default_factory=list, init=False)
    _size: Optional[Tuple[int, int]] = field(default=None, init=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    # -- capture -----------------------------------------------------------
    def add(self, timestamp: float, image: np.ndarray) -> None:
        """Hold one frame. Cheap enough to call on every processed frame."""
        ok, encoded = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        )
        if not ok:
            return

        payload = encoded.tobytes()
        self._frames.append((timestamp, payload))
        self._bytes += len(payload)
        if self._size is None:
            height, width = image.shape[:2]
            self._size = (width, height)

        # Drop anything older than the pre-roll needs, then enforce the byte cap.
        horizon = timestamp - (self.pre_seconds + self.post_seconds + 2.0)
        while self._frames and self._frames[0][0] < horizon:
            self._bytes -= len(self._frames.popleft()[1])
        while self._frames and self._bytes > self.max_bytes:
            self._bytes -= len(self._frames.popleft()[1])

    # -- requesting --------------------------------------------------------
    def request(self, clip_id: str, trigger_time: float, label: str = "") -> None:
        """Ask for a clip around `trigger_time`. Written once the post-roll passes."""
        if any(p.clip_id == clip_id for p in self._pending):
            return
        self._pending.append(
            PendingClip(
                clip_id=clip_id,
                trigger_time=trigger_time,
                ready_at=trigger_time + self.post_seconds,
                label=label,
            )
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # -- writing -----------------------------------------------------------
    def tick(self, now: float) -> List[Tuple[str, Path]]:
        """Write any clips whose trailing seconds have now been captured."""
        if not self._pending:
            return []

        written: List[Tuple[str, Path]] = []
        still_waiting: List[PendingClip] = []

        for pending in self._pending:
            if now < pending.ready_at:
                still_waiting.append(pending)
                continue
            path = self._write(pending)
            if path is not None:
                written.append((pending.clip_id, path))

        self._pending = still_waiting
        return written

    def flush(self) -> List[Tuple[str, Path]]:
        """Write every outstanding clip, however short. Used at end of run so a
        flag in the final seconds still gets whatever footage exists."""
        written = []
        for pending in self._pending:
            path = self._write(pending)
            if path is not None:
                written.append((pending.clip_id, path))
        self._pending = []
        return written

    def _write(self, pending: PendingClip) -> Optional[Path]:
        start = pending.trigger_time - self.pre_seconds
        end = pending.trigger_time + self.post_seconds
        selected = [(t, data) for t, data in self._frames if start <= t <= end]

        if len(selected) < 2:
            logger.info("No footage buffered for clip %s; skipping.", pending.clip_id)
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{pending.clip_id}.mp4"

        first = cv2.imdecode(np.frombuffer(selected[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None
        height, width = first.shape[:2]

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, self.fps), (width, height)
        )
        if not writer.isOpened():
            logger.warning("Could not open a video writer for %s", path)
            return None

        try:
            for _, data in selected:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()

        span = selected[-1][0] - selected[0][0]
        logger.info(
            "Wrote clip %s — %d frames covering %.1fs (%.1fs before the trigger)",
            pending.clip_id, len(selected), span, pending.trigger_time - selected[0][0],
        )
        return path

    def stats(self) -> Dict[str, float]:
        span = (self._frames[-1][0] - self._frames[0][0]) if len(self._frames) > 1 else 0.0
        return {
            "frames": len(self._frames),
            "seconds_buffered": round(span, 1),
            "megabytes": round(self._bytes / (1024 * 1024), 1),
            "pending_clips": len(self._pending),
        }
