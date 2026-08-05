"""Stage 1 — frame ingestion.

Reads a video file or a live camera via OpenCV and yields `Frame` objects.

The one decision worth explaining here is the timestamp. For a video file the
timestamp comes from the frame index and the file's FPS, not from the wall
clock, because every heuristic downstream is time-based (dwell seconds, speed
in body-heights per second, oscillation frequency in Hz). If timestamps came
from the wall clock, a laptop that decodes slowly would turn a person walking
past into a person loitering. File time keeps the analysis identical whether the
machine runs at 4 fps or 40.

For a live camera the wall clock IS the truth, so that path uses it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameSourceError(RuntimeError):
    """Raised when a video file or camera cannot be opened or read."""


@dataclass
class Frame:
    """One frame of video, with the time it represents."""

    index: int          # frame number in the source
    timestamp: float    # seconds since the start of the stream
    image: np.ndarray   # BGR, as OpenCV delivers it
    width: int
    height: int
    # True for a live camera. Heuristics do not care, but the audit trail does:
    # a live run cannot be replayed, a file run can.
    is_live: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height


@dataclass
class SourceInfo:
    """What we know about the stream before reading it."""

    source: str
    is_live: bool
    fps: float
    frame_count: int      # 0 for a live camera
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 and self.frame_count else 0.0

    def describe(self) -> str:
        if self.is_live:
            return f"live camera {self.source} — {self.width}x{self.height} @ {self.fps:.1f}fps"
        return (
            f"{Path(self.source).name} — {self.width}x{self.height} @ {self.fps:.1f}fps, "
            f"{self.frame_count} frames ({self.duration_seconds:.1f}s)"
        )


# A file whose FPS metadata is missing or absurd would silently corrupt every
# duration in the system, so we fall back to a stated assumption instead.
FALLBACK_FPS = 25.0
MAX_PLAUSIBLE_FPS = 240.0


def open_source(source: Union[str, int, Path]) -> tuple[cv2.VideoCapture, SourceInfo]:
    """Open a video file path or a camera index, and report what it is."""
    is_live = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    handle: Union[str, int]

    if is_live:
        handle = int(source)
    else:
        path = Path(source)
        if not path.is_file():
            raise FrameSourceError(f"Video file not found: {path}")
        handle = str(path)

    capture = cv2.VideoCapture(handle)
    if not capture.isOpened():
        raise FrameSourceError(
            f"OpenCV could not open {source!r}. For a file, check the codec; "
            f"for a camera, check the index and that nothing else holds the device."
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (0 < fps <= MAX_PLAUSIBLE_FPS):
        logger.warning(
            "Source %r reports fps=%s, which is not usable. Assuming %.1f fps — "
            "all durations and speeds will be scaled by whatever the real rate is.",
            source, fps, FALLBACK_FPS,
        )
        fps = FALLBACK_FPS

    info = SourceInfo(
        source=str(source),
        is_live=is_live,
        fps=fps,
        frame_count=0 if is_live else int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    return capture, info


def iter_frames(
    source: Union[str, int, Path],
    *,
    stride: int = 1,
    start_seconds: float = 0.0,
    max_seconds: Optional[float] = None,
    capture: Optional[cv2.VideoCapture] = None,
    info: Optional[SourceInfo] = None,
) -> Iterator[Frame]:
    """Yield frames from `source`.

    `stride` processes every Nth frame — the escape hatch when the demo laptop
    cannot keep up. Timestamps stay true regardless of stride, so raising it
    costs temporal resolution but never shifts a threshold's meaning.
    """
    owns_capture = capture is None
    if owns_capture:
        capture, info = open_source(source)
    assert capture is not None and info is not None

    stride = max(1, int(stride))
    started_wall = time.monotonic()

    try:
        if start_seconds > 0 and not info.is_live:
            capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

        index = int(start_seconds * info.fps) if not info.is_live else 0
        consecutive_failures = 0

        while True:
            ok, image = capture.read()
            if not ok:
                # A live camera can drop a frame without the stream being over;
                # a file that stops reading has ended.
                if info.is_live and consecutive_failures < 5:
                    consecutive_failures += 1
                    continue
                break
            consecutive_failures = 0

            if (index - int(start_seconds * info.fps)) % stride == 0:
                timestamp = (
                    time.monotonic() - started_wall if info.is_live else index / info.fps
                )

                if max_seconds is not None and timestamp - start_seconds > max_seconds:
                    break

                height, width = image.shape[:2]
                yield Frame(
                    index=index,
                    timestamp=timestamp,
                    image=image,
                    width=width,
                    height=height,
                    is_live=info.is_live,
                )

            index += 1
    finally:
        if owns_capture:
            capture.release()
