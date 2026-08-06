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
import sys
import threading
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

# Windows' default capture backend (Media Foundation) has a failure mode worth
# naming: it OPENS a camera happily — `isOpened()` is True, the resolution and
# fps read back correctly — and then returns nothing from every `read()`. The
# live-camera retry guard below sees five failures in a row and concludes the
# stream ended, so the run reports "processed 0 frames" a tenth of a second in
# with no error anywhere. DirectShow reads the same device fine.
#
# So a live camera is opened by trying backends in order and keeping the first
# that actually hands over a frame. File paths keep the default backend, which
# decodes them correctly and where this failure mode does not arise.
CAMERA_BACKENDS = (
    ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_ANY, "default"))
    if sys.platform == "win32"
    else ((cv2.CAP_ANY, "default"),)
)


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a live camera on the first backend that actually yields a frame."""
    attempts: list[str] = []

    for backend, name in CAMERA_BACKENDS:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            attempts.append(f"{name} would not open the device")
            continue

        # `isOpened()` is exactly the check the MSMF failure mode passes, so it
        # proves nothing on its own. Only a frame in hand does. The probe costs
        # one frame of a live stream, which is not a meaningful loss.
        ok, _ = capture.read()
        if ok:
            logger.info("Camera %s opened via %s.", index, name)
            return capture

        capture.release()
        attempts.append(f"{name} opened the device but returned no frames")

    raise FrameSourceError(
        f"OpenCV opened camera {index} but could not read from it. Tried: "
        + "; ".join(attempts)
        + ". Check the camera index, that no other application holds the device "
        "(video calls are the usual culprit), and that camera access is permitted "
        "for desktop apps in the system privacy settings."
    )


class LiveCamera:
    """A live camera drained by its own thread, keeping only the newest frame.

    THE PROBLEM THIS SOLVES IS LATENCY, NOT THROUGHPUT. A webcam produces about
    30 frames a second. Detection on a CPU manages roughly one. Reading the
    stream in order therefore hands the analysis a frame that is already old,
    and it gets older every second the run continues — after a minute you are
    detecting a person who left the drive long ago, and the debug view shows a
    scene that has not existed for some time. Dropping frames is the correct
    behaviour for a live source: the newest frame is the only one that
    describes the present, and the ones behind it have been overtaken.

    A file source must never do this — every frame of a recording matters, and
    the timestamps are what make dwell times mean anything. Hence this wraps
    live captures only.

    It duck-types the `read`/`release`/`get` surface of cv2.VideoCapture so the
    pipeline does not need to know which kind of source it has.
    """

    def __init__(self, capture: cv2.VideoCapture):
        self._capture = capture
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._last_delivered = 0
        self._failures = 0
        self._stopped = threading.Event()
        self._new_frame = threading.Condition(self._lock)
        self._thread = threading.Thread(
            target=self._pump, name="camera-pump", daemon=True
        )

    def start(self) -> "LiveCamera":
        self._thread.start()
        return self

    def _pump(self) -> None:
        """Read as fast as the device allows, keeping only the last frame."""
        while not self._stopped.is_set():
            ok, image = self._capture.read()
            if not ok:
                # A live camera can drop a frame without the stream being over.
                self._failures += 1
                if self._failures > 150:
                    logger.warning("Camera stopped delivering frames.")
                    break
                time.sleep(0.01)
                continue

            self._failures = 0
            with self._new_frame:
                self._frame = image
                self._seq += 1
                self._new_frame.notify_all()

    def peek(self) -> Optional[np.ndarray]:
        """The newest frame, without waiting and without consuming it.

        This is what the live relay uses: it wants whatever is most current,
        as often as it asks, regardless of what the analysis is doing.
        """
        with self._lock:
            return self._frame

    def read(self, timeout: float = 5.0) -> tuple[bool, Optional[np.ndarray]]:
        """Wait for a frame the caller has not already seen."""
        deadline = time.monotonic() + timeout
        with self._new_frame:
            while self._seq == self._last_delivered:
                if self._stopped.is_set() or not self._thread.is_alive():
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._new_frame.wait(remaining)

            self._last_delivered = self._seq
            return True, self._frame

    def get(self, prop: int) -> float:
        return self._capture.get(prop)

    def release(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=2.0)
        self._capture.release()


# The relay in main.py needs the newest camera frame without going through the
# analysis loop, and the capture is owned several layers down. A module-level
# handle is blunt, but the alternative is threading a camera reference through
# open_source, iter_frames, Pipeline.run and on_frame purely so one optional
# feature can reach it. Only ever one live camera per process.
_active_camera: Optional[LiveCamera] = None


def active_camera() -> Optional[LiveCamera]:
    """The live camera this process is reading, if it is reading one."""
    return _active_camera


def open_source(source: Union[str, int, Path]) -> tuple[cv2.VideoCapture, SourceInfo]:
    """Open a video file path or a camera index, and report what it is."""
    is_live = isinstance(source, int) or (isinstance(source, str) and source.isdigit())

    if is_live:
        capture = open_camera(int(source))
    else:
        path = Path(source)
        if not path.is_file():
            raise FrameSourceError(f"Video file not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FrameSourceError(
                f"OpenCV could not open {source!r}. Check the codec — the file "
                f"exists but no available backend can decode it."
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

    if is_live:
        # Ask the driver for a shallow buffer as well. Not every backend honours
        # it, which is exactly why the draining thread exists rather than this
        # being the whole fix.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        global _active_camera
        _active_camera = LiveCamera(capture).start()
        return _active_camera, info

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
