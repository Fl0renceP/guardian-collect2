"""Stage 3 — skeletal keypoint extraction (MediaPipe Pose).

Runs MediaPipe Pose on each person crop from stage 2 and returns 33 body
landmarks converted into whole-frame pixel coordinates, plus the derived
geometry the heuristics actually reason about (torso height, shoulder width,
knee angle, head yaw, how visible the face is).

Two implementation notes that matter:

1. **MediaPipe 1.0 removed the legacy `mp.solutions.pose` API.** Only the Tasks
   API remains, which needs a `.task` model bundle downloaded once. This module
   fetches it on first use and caches it next to the code.

2. **IMAGE running mode, not VIDEO.** VIDEO mode carries temporal state and
   demands monotonically increasing timestamps from a single subject. We feed it
   a different person's crop on every call, so that state would be smoothing one
   person's pose onto another's. Tracking continuity already comes from
   ByteTrack upstream; asking MediaPipe for it too would corrupt both.

Nothing here identifies anyone. Landmarks are body geometry — where a wrist is
relative to a shoulder — and are held in memory for the length of a track's
history, then dropped. They are not written to the audit log by default (see
`audit.store_raw_keypoints` in config.yaml).
"""

from __future__ import annotations

import logging
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILENAME = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# BlazePose's 33 landmarks. Named so the heuristics read like English.
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

FACE_LANDMARKS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)

# Drawn by debug_overlay; kept here because it belongs with the landmark indices.
SKELETON_EDGES = (
    (LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
)


@dataclass
class Landmark:
    """One body point, in whole-frame pixel coordinates."""

    x: float
    y: float           # image convention: y grows DOWNWARD, so "higher up" = smaller y
    z: float
    visibility: float

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class PoseKeypoints:
    """A person's pose in one frame, plus the geometry derived from it."""

    landmarks: List[Landmark]
    visibility_floor: float
    bbox_height: float                 # the person's own scale reference, pixels
    raw_normalised: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # -- access ------------------------------------------------------------
    def point(self, index: int) -> Optional[Landmark]:
        """A landmark, or None if MediaPipe could not actually see it.

        Returning None rather than a guessed coordinate is the point. MediaPipe
        happily extrapolates occluded joints, and a heuristic that trusts those
        guesses will invent a climbing posture out of a blurry frame.
        """
        if index >= len(self.landmarks):
            return None
        landmark = self.landmarks[index]
        return landmark if landmark.visibility >= self.visibility_floor else None

    def visible(self, *indices: int) -> bool:
        return all(self.point(i) is not None for i in indices)

    def midpoint(self, a: int, b: int) -> Optional[Tuple[float, float]]:
        pa, pb = self.point(a), self.point(b)
        if pa is None or pb is None:
            return None
        return ((pa.x + pb.x) / 2.0, (pa.y + pb.y) / 2.0)

    # -- derived geometry --------------------------------------------------
    @property
    def shoulder_width(self) -> Optional[float]:
        left, right = self.point(LEFT_SHOULDER), self.point(RIGHT_SHOULDER)
        if left is None or right is None:
            return None
        return abs(left.x - right.x) or None

    @property
    def torso_height(self) -> Optional[float]:
        """Vertical shoulder-to-hip distance in pixels.

        The crouch measure. Compared only against this same person's own
        standing baseline, never against a population average.
        """
        shoulders = self.midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
        hips = self.midpoint(LEFT_HIP, RIGHT_HIP)
        if shoulders is None or hips is None:
            return None
        return abs(hips[1] - shoulders[1])

    @property
    def face_visibility(self) -> float:
        """0..1 — how well the camera can see this person's face.

        Low means the face is turned away or covered. That is a statement about
        the camera's view, NOT about the person: hoods, hats, masks, sun, dust
        and religious dress all produce exactly this reading. It is used only to
        mark "facial recognition cannot help here", never as evidence of intent.
        """
        scores = [self.landmarks[i].visibility for i in FACE_LANDMARKS if i < len(self.landmarks)]
        return float(np.mean(scores)) if scores else 0.0

    @property
    def head_yaw_degrees(self) -> Optional[float]:
        """Rough head rotation away from the camera, in degrees.

        Approximated from how far the nose sits off the midpoint between the
        ears, scaled by ear separation. It is a coarse estimate — there is no
        3D head model here — so it is only ever used against a wide threshold
        ("clearly turned away"), never for a fine measurement.
        """
        left_ear, right_ear, nose = (
            self.point(LEFT_EAR), self.point(RIGHT_EAR), self.point(NOSE)
        )
        if nose is None:
            return None

        if left_ear is not None and right_ear is not None:
            separation = abs(left_ear.x - right_ear.x)
            if separation < 1e-6:
                return 90.0
            ear_mid_x = (left_ear.x + right_ear.x) / 2.0
            offset = (nose.x - ear_mid_x) / (separation / 2.0)
            return math.degrees(math.asin(max(-1.0, min(1.0, offset))))

        # Only one ear visible at all — the head is side-on to the camera.
        if left_ear is not None or right_ear is not None:
            return 75.0
        return None

    def limb_extension(self, shoulder: int, wrist: int) -> Optional[float]:
        """How far a wrist reaches from its shoulder, in pixels."""
        s, w = self.point(shoulder), self.point(wrist)
        if s is None or w is None:
            return None
        return math.hypot(w.x - s.x, w.y - s.y)

    def knee_angle(self, hip: int, knee: int, ankle: int) -> Optional[float]:
        """Interior angle at the knee in degrees. 180 = straight leg."""
        a, b, c = self.point(hip), self.point(knee), self.point(ankle)
        if a is None or b is None or c is None:
            return None

        v1 = (a.x - b.x, a.y - b.y)
        v2 = (c.x - b.x, c.y - b.y)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return None

        cosine = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    @property
    def min_knee_angle(self) -> Optional[float]:
        angles = [
            a for a in (
                self.knee_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
                self.knee_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
            ) if a is not None
        ]
        return min(angles) if angles else None

    def hands_above_shoulders(self) -> Tuple[bool, bool]:
        """(left_hand_high, right_hand_high). Remember y grows downward."""
        out = []
        for wrist, shoulder in ((LEFT_WRIST, LEFT_SHOULDER), (RIGHT_WRIST, RIGHT_SHOULDER)):
            w, s = self.point(wrist), self.point(shoulder)
            out.append(bool(w is not None and s is not None and w.y < s.y))
        return (out[0], out[1])

    def to_audit_dict(self, include_raw: bool = False) -> Dict[str, object]:
        """What the audit trail records about a pose.

        Derived measurements only by default — those are the numbers a decision
        was actually made from. The full 33-point skeleton adds no audit value
        and is body-geometry data, so it stays out unless deliberately enabled.
        """
        summary: Dict[str, object] = {
            "face_visibility": round(self.face_visibility, 3),
            "head_yaw_degrees": (
                round(self.head_yaw_degrees, 1) if self.head_yaw_degrees is not None else None
            ),
            "torso_height_px": round(self.torso_height, 1) if self.torso_height else None,
            "min_knee_angle": (
                round(self.min_knee_angle, 1) if self.min_knee_angle is not None else None
            ),
            "hands_above_shoulders": list(self.hands_above_shoulders()),
        }
        if include_raw:
            summary["raw_landmarks_normalised"] = [
                [round(v, 4) for v in point] for point in self.raw_normalised
            ]
        return summary


def ensure_model(destination: Optional[Path] = None) -> Path:
    """Download the MediaPipe pose bundle once, then reuse it."""
    target = Path(destination) if destination else MODEL_DIR / MODEL_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe pose model to %s (one time, ~6MB)", target)
    partial = target.with_suffix(".partial")
    try:
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(target)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the MediaPipe pose model from {MODEL_URL}. "
            f"With no network, download it manually and place it at {target}."
        ) from exc
    return target


class PoseExtractor:
    """MediaPipe Pose over person crops."""

    def __init__(
        self,
        *,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
        visibility_floor: float = 0.35,
        model_path: Optional[Path] = None,
        # Crops smaller than this are not worth running a pose model over. The
        # same reasoning as MIN_FACE_PIXELS in backend/config.py: upscaling a
        # 30px figure to the model's input size returns 33 confident-looking
        # coordinates that describe nothing.
        #
        # Height and width are gated SEPARATELY and the height is what matters.
        # A person's box is naturally tall and narrow — roughly 2.5:1 — so a
        # single "smallest dimension" gate silently discards almost every
        # genuine person at distance while claiming they were too small.
        min_crop_height: int = 64,
        min_crop_width: int = 24,
        # Person boxes clip at the wrists and ankles; a small margin gives
        # MediaPipe the context it needs to place the joints it is being asked about.
        crop_margin: float = 0.08,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.visibility_floor = visibility_floor
        self.model_path = model_path
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width
        self.crop_margin = crop_margin
        self._landmarker = None
        self._mp = None

    def _load(self):
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_file = ensure_model(self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_file)),
                running_mode=vision.RunningMode.IMAGE,   # see module docstring
                num_poses=1,                             # one crop, one person
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self._mp = mp
        return self._landmarker

    def warm_up(self) -> None:
        self._load()

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseExtractor":
        self._load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def extract(
        self,
        image: np.ndarray,
        bbox: Sequence[float],
    ) -> Optional[PoseKeypoints]:
        """Pose for the person inside `bbox`, in whole-frame pixel coordinates."""
        frame_h, frame_w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        margin_x = (x2 - x1) * self.crop_margin
        margin_y = (y2 - y1) * self.crop_margin
        cx1 = int(max(0, x1 - margin_x))
        cy1 = int(max(0, y1 - margin_y))
        cx2 = int(min(frame_w, x2 + margin_x))
        cy2 = int(min(frame_h, y2 + margin_y))

        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            return None
        if crop_h < self.min_crop_height or crop_w < self.min_crop_width:
            return None

        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        landmarker = self._load()
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        )

        try:
            result = landmarker.detect(mp_image)
        except Exception:
            logger.exception("MediaPipe pose failed on a crop; treating as no pose.")
            return None

        if not result.pose_landmarks:
            return None

        # MediaPipe returns coordinates normalised to the CROP. Everything
        # downstream works in whole-frame pixels, so convert once, here.
        raw = result.pose_landmarks[0]
        landmarks: List[Landmark] = []
        raw_normalised: List[Tuple[float, float, float, float]] = []
        for point in raw:
            visibility = float(getattr(point, "visibility", 0.0) or 0.0)
            landmarks.append(
                Landmark(
                    x=cx1 + point.x * crop_w,
                    y=cy1 + point.y * crop_h,
                    z=float(getattr(point, "z", 0.0) or 0.0),
                    visibility=visibility,
                )
            )
            raw_normalised.append((point.x, point.y, float(getattr(point, "z", 0.0) or 0.0), visibility))

        return PoseKeypoints(
            landmarks=landmarks,
            visibility_floor=self.visibility_floor,
            bbox_height=float(y2 - y1),
            raw_normalised=raw_normalised,
        )
