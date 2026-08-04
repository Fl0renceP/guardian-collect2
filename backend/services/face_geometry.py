"""Detection, 5-point alignment and probe quality — the geometry layer.

DeepFace is used for embedding only from here on. Its detector wrapper exposes
just two eye points and its align= applies a 2-point rotation, which measurably
hurt accuracy on this gallery (separation 7.2x aligned vs 17.7x unaligned) and
silently dropped faces in group shots. Proper face alignment is not a rotation;
it is a similarity transform that maps five landmarks onto a canonical template,
normalising scale, rotation AND translation together.

OpenCV's YuNet returns those five points directly — both eyes, nose tip, both
mouth corners — plus a detection score, which also gives the quality gate the
inter-ocular distance and head pose it needs.

Landmark order from YuNet is (right eye, left eye, nose, right mouth, left
mouth), where "right" means the subject's right, i.e. the LEFT side of the image.
That matches the ArcFace canonical template's ordering, so the two line up
without reordering.
"""

import logging
import math
import os
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace's canonical 5-point template, defined for a 112x112 crop. Scaled below
# to whatever crop size the embedding model wants.
_ARCFACE_TEMPLATE_112 = np.array([
    [38.2946, 51.6963],   # subject's right eye  (image left)
    [73.5318, 51.5014],   # subject's left eye   (image right)
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # right mouth corner
    [70.7299, 92.2041],   # left mouth corner
], dtype=np.float32)

# Rough 3D face model for pose estimation, in arbitrary units, y increasing
# downward to match image coordinates. Generic rather than person-specific, so
# the angles are approximate — good enough to reject a profile view, not good
# enough to quote to a degree.
_MODEL_3D = np.array([
    [-33.0, -34.0, -30.0],
    [33.0, -34.0, -30.0],
    [0.0, 0.0, 0.0],
    [-28.0, 40.0, -30.0],
    [28.0, 40.0, -30.0],
], dtype=np.float64)

_YUNET_MODEL = os.path.expanduser(
    os.getenv("YUNET_MODEL_PATH",
              os.path.join("~", ".deepface", "weights", "face_detection_yunet_2023mar.onnx"))
)

_detector = None
_detector_lock = threading.Lock()

DETECT_SCORE_THRESHOLD = float(os.getenv("YUNET_SCORE_THRESHOLD", "0.6"))
DETECT_NMS_THRESHOLD = float(os.getenv("YUNET_NMS_THRESHOLD", "0.3"))


class Face:
    """One detected face: where it is, its landmarks, and how sure we are."""

    __slots__ = ("bbox", "landmarks", "score")

    def __init__(self, bbox, landmarks, score):
        self.bbox = bbox              # (x, y, w, h) ints
        self.landmarks = landmarks    # 5x2 float array
        self.score = float(score)

    @property
    def area(self):
        return self.bbox[2] * self.bbox[3]

    def as_dict(self):
        x, y, w, h = self.bbox
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "detector_score": round(self.score, 4)}


def _get_detector():
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                if not os.path.exists(_YUNET_MODEL):
                    raise RuntimeError(
                        f"YuNet model not found at {_YUNET_MODEL}. It ships with DeepFace — "
                        "run any scan once to download it, or set YUNET_MODEL_PATH."
                    )
                _detector = cv2.FaceDetectorYN.create(
                    _YUNET_MODEL, "", (320, 320),
                    DETECT_SCORE_THRESHOLD, DETECT_NMS_THRESHOLD, 5000,
                )
                logger.info("YuNet detector ready (%s)", _YUNET_MODEL)
    return _detector


def detect_faces(image):
    """Every face in a BGR image, largest first.

    setInputSize mutates detector state, so the whole detect call is serialised —
    the Flask dev server is threaded and the live camera poller runs concurrently.
    """
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    detector = _get_detector()
    with _detector_lock:
        detector.setInputSize((width, height))
        _, raw = detector.detect(image)
    if raw is None:
        return []

    faces = []
    for row in raw:
        x, y, w, h = (int(round(v)) for v in row[:4])
        # YuNet can return boxes that overhang the frame edge.
        x, y = max(0, x), max(0, y)
        w, h = min(w, width - x), min(h, height - y)
        if w <= 0 or h <= 0:
            continue
        faces.append(Face((x, y, w, h), row[4:14].reshape(5, 2).astype(np.float32), row[14]))

    faces.sort(key=lambda f: f.area, reverse=True)
    return faces


def align(image, landmarks, size=160):
    """Warp a face onto the canonical template using its five landmarks.

    A similarity transform (rotation + uniform scale + translation, no shear)
    is the right family here: it removes the nuisance variation of where the head
    sits and how it is tilted, without distorting the face itself the way a full
    affine or perspective fit would when the landmarks are noisy.
    """
    template = _ARCFACE_TEMPLATE_112 * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), template, method=cv2.LMEDS,
    )
    if matrix is None:
        # Degenerate landmarks. Fall back to a plain crop rather than failing:
        # an unaligned face still embeds, just less consistently.
        x, y, w, h = _bbox_from_landmarks(landmarks, image.shape)
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderValue=0.0)


def equalize(face_crop, clip_limit=2.0, tile_grid=8):
    """Even out lighting across an aligned face crop.

    CLAHE on the L channel of LAB, not cv2.equalizeHist on a greyscale copy.
    Two reasons, and both matter here:

    1. COLOUR IS KEPT. Facenet512 is trained on RGB. Converting to greyscale
       before embedding throws away a channel the model expects and degrades
       matching — the greyscale-first pipelines in the classical-CV literature
       are feeding LDA and eigenface methods, not a deep embedding network.

    2. LOCAL, NOT GLOBAL. Doorbell footage is typically a face lit from one
       side against a bright sky. Global equalisation spends its dynamic range
       on the sky and leaves the face as flat as it found it; CLAHE works per
       tile, so the shadowed cheek is lifted without blowing out the
       background. The clip limit is what stops it amplifying sensor noise in
       the flat regions.

    Applied for EMBEDDING ONLY. Quality assessment must see the real exposure —
    equalising first would make an unreadably dark capture look well-lit and
    walk it straight past the gate that exists to refuse it.
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    if face_crop.ndim == 2:
        # Already single-channel; equalise it directly.
        return cv2.createCLAHE(clipLimit=clip_limit,
                               tileGridSize=(tile_grid, tile_grid)).apply(face_crop)

    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _bbox_from_landmarks(landmarks, shape):
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    span = max(float(np.ptp(xs)), float(np.ptp(ys))) * 2.0
    half = max(span / 2.0, 10.0)
    x = int(max(0, cx - half))
    y = int(max(0, cy - half))
    w = int(min(shape[1] - x, half * 2))
    h = int(min(shape[0] - y, half * 2))
    return x, y, w, h


def interocular_pixels(landmarks):
    """Distance between the eye centres — the standard biometric scale measure.

    Better than box size: a bounding box grows with hair and chin, while the
    inter-ocular distance tracks the part of the face the encoder actually uses.
    """
    return float(np.linalg.norm(landmarks[0] - landmarks[1]))


def estimate_pose(landmarks, image_shape):
    """Approximate yaw, pitch and roll in degrees.

    solvePnP against a generic 3D face, with focal length assumed equal to the
    image width — the usual stand-in when the camera is uncalibrated. Good enough
    to reject a profile shot; not good enough to quote to the degree.
    """
    height, width = image_shape[:2]
    focal = float(width)
    camera = np.array([[focal, 0, width / 2.0],
                       [0, focal, height / 2.0],
                       [0, 0, 1]], dtype=np.float64)
    # SQPNP, not ITERATIVE: with no initial guess the iterative solver falls back
    # to DLT, which needs six correspondences, and a 5-point landmark set is five.
    # SQPNP is well conditioned from three points upward.
    try:
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_3D, landmarks.astype(np.float64), camera, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        try:
            ok, rvec, _ = cv2.solvePnP(
                _MODEL_3D, landmarks.astype(np.float64), camera, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        yaw = math.degrees(math.atan2(-rotation[2, 0], sy))
        roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rotation[1, 2], rotation[1, 1]))
        yaw = math.degrees(math.atan2(-rotation[2, 0], sy))
        roll = 0.0

    # Pitch comes back near +/-180 for a face looking straight at the camera,
    # because the model's nose axis points away from the viewer. Fold it so that
    # "level" reads as roughly zero.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180

    return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}


# --- probe quality gate -----------------------------------------------------
# Thresholds are set from measurements on this test set rather than round
# numbers, and every one of them is env-overridable:
#
#   case                 iocular   sharp   balance   verdict
#   seed references      137-148   388+    0.43-0.88  pass
#   degraded but usable      139   106+    0.79-0.85  pass
#   rotated probe            175     905      0.494   pass
#   60px CCTV face          29.7    80.5      0.857   pass  <- matches correctly today
#   motion-blurred          148.7   155.4     0.313   NO_DECISION
#   stranger (good photo)    185    1534      0.771   pass -> decided as unknown
#
# Two deliberate choices:
#
# Inter-ocular floor is 24px, well below the 30-60px that biometric guidance
# normally recommends. The 60px CCTV face measures 29.7px and identifies
# correctly, so a stricter gate would be discarding a working decision — and the
# brief here is that poor images should still be attempted. Raise it once the
# calibration script has real numbers from the deployment camera.
#
# Sharpness alone cannot catch motion blur: the blurred face scores 155 while a
# perfectly usable dim photo scores 106. The gradient-balance ratio separates
# them cleanly (0.313 vs 0.426+), which is what the balance floor is for.
MIN_INTEROCULAR_PX = float(os.getenv("MIN_INTEROCULAR_PX", "24"))
MIN_PROBE_SHARPNESS = float(os.getenv("MIN_PROBE_SHARPNESS", "50"))
MIN_PROBE_BALANCE = float(os.getenv("MIN_PROBE_BALANCE", "0.35"))
MAX_PROBE_YAW = float(os.getenv("MAX_PROBE_YAW", "35"))
# Pitch is measured against a generic 3D face, and reads roughly 15 degrees low
# on known-frontal faces here (-9.8, -14.2, -21.6). The limit is therefore
# deliberately loose and yaw carries most of the pose gating.
MAX_PROBE_PITCH = float(os.getenv("MAX_PROBE_PITCH", "35"))
MIN_DETECTOR_SCORE = float(os.getenv("MIN_PROBE_DETECTOR_SCORE", "0.70"))


def assess_probe(image, face, aligned=None):
    """Decide whether this face is good enough to be identified at all.

    Returns a dict with the measurements, a `decidable` flag and, when it is
    false, the reasons. A probe that fails should produce NO_DECISION rather than
    a match: naming someone from an image the system cannot actually read is
    worse than admitting it cannot tell, because the answer looks identical to a
    real identification.

    This is a different question from the reference-quality gate in enrolment.
    That one asks "is this image fit to represent a person for ever". This asks
    "can we read this one frame well enough to answer at all".
    """
    if aligned is None:
        aligned = align(image, face.landmarks, 160)

    iod = interocular_pixels(face.landmarks)
    pose = estimate_pose(face.landmarks, image.shape) or {}
    sharp = sharpness(aligned) if aligned is not None else 0.0
    balance = directional_sharpness_ratio(aligned) if aligned is not None else 0.0

    reasons = []
    if iod < MIN_INTEROCULAR_PX:
        reasons.append(
            f"eyes are only {iod:.0f}px apart, below the {MIN_INTEROCULAR_PX:.0f}px "
            "minimum — too few pixels across the face to identify anyone"
        )
    if sharp < MIN_PROBE_SHARPNESS:
        reasons.append(f"image is out of focus (sharpness {sharp:.0f} < {MIN_PROBE_SHARPNESS:.0f})")
    if balance < MIN_PROBE_BALANCE:
        reasons.append(
            f"edge detail is lopsided ({balance:.2f} < {MIN_PROBE_BALANCE:.2f}) — "
            "motion blur from a moving subject"
        )
    if abs(pose.get("yaw", 0.0)) > MAX_PROBE_YAW:
        reasons.append(f"head is turned too far ({pose.get('yaw'):.0f} deg yaw, limit {MAX_PROBE_YAW:.0f})")
    if abs(pose.get("pitch", 0.0)) > MAX_PROBE_PITCH:
        reasons.append(f"head is tilted too far ({pose.get('pitch'):.0f} deg pitch, limit {MAX_PROBE_PITCH:.0f})")
    if face.score < MIN_DETECTOR_SCORE:
        reasons.append(f"detector is unsure this is a face ({face.score:.2f} < {MIN_DETECTOR_SCORE:.2f})")

    return {
        "interocular_px": round(iod, 1),
        "sharpness": round(sharp, 1),
        "balance": round(balance, 3),
        "detector_score": round(face.score, 4),
        "pose": pose,
        "decidable": not reasons,
        "reasons": reasons,
    }


def sharpness(aligned_face):
    """Laplacian variance on the aligned crop.

    Measured after alignment, so every face is the same size and the number means
    the same thing regardless of how far away the subject was.
    """
    grey = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def directional_sharpness_ratio(aligned_face):
    """Gradient balance between axes — exposes motion blur.

    Motion blur is directional: a subject walking past smears vertical edges and
    leaves horizontal ones intact, so total edge energy stays high while the two
    axes fall badly out of balance.
    """
    grey = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
    gx = float(cv2.Sobel(grey, cv2.CV_64F, 1, 0, ksize=3).var())
    gy = float(cv2.Sobel(grey, cv2.CV_64F, 0, 1, ksize=3).var())
    strongest = max(gx, gy)
    return (min(gx, gy) / strongest) if strongest else 0.0
