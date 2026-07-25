"""Decide whether a detected face is good enough to become a reference photo.

This exists because of CCTV. Footage from a public-space camera routinely gives
faces 20-60 pixels across, motion-blurred, at an angle. DeepFace will happily
embed one: it upscales the crop to the model's input size and returns 512 numbers
like normal. Nothing errors. But those numbers encode upscaling artefacts rather
than a face, so the vector lands roughly equidistant from every identity in the
gallery — which makes it a magnet that drags unrelated people into false matches.

One bad enrolment can therefore poison an identity permanently, and nothing in the
response would tell you it happened.

So enrolment is gated. A capture that fails is still stored — it is evidence for
the case file — but marked use_for_matching = FALSE so it never acts as a
reference. Scanning is deliberately NOT gated: you always want to try to identify
whoever is in front of the camera, however poor the image.
"""

import cv2
import numpy as np

from config import Config

# The crop is resized to this before measuring sharpness, so the blur number is
# comparable between a 4K still and a low-res camera grab.
_BLUR_NORMALISE_TO = 160


def _laplacian_variance(gray):
    """Classic focus measure: sharp edges produce a wide spread of second derivatives."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(image, facial_area, det_confidence=None):
    """Score one detected face.

    image: BGR numpy array the face was detected in.
    facial_area: DeepFace's {'x','y','w','h'} box.
    det_confidence: detector confidence, if the backend reported one.

    Returns a dict with the raw measurements, an overall 0-1 score, a pass/fail
    flag, and human-readable reasons for any failure.
    """
    box = facial_area or {}
    x, y = int(box.get("x", 0)), int(box.get("y", 0))
    w, h = int(box.get("w", 0)), int(box.get("h", 0))

    # Short side, not area: a wide-but-short box is still a low-detail face.
    face_pixels = min(w, h) if w and h else 0

    blur = 0.0
    brightness = 0.0
    directional_ratio = 1.0
    if w > 0 and h > 0:
        ih, iw = image.shape[:2]
        crop = image[max(0, y):min(ih, y + h), max(0, x):min(iw, x + w)]
        if crop.size:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (_BLUR_NORMALISE_TO, _BLUR_NORMALISE_TO),
                              interpolation=cv2.INTER_AREA)
            blur = _laplacian_variance(gray)
            brightness = float(np.mean(gray))

            # Motion blur is directional, so it hides from the isotropic measure
            # above: a subject walking past smears vertical edges and leaves
            # horizontal ones intact, keeping total edge energy high. Comparing
            # the two gradient axes exposes it — a clean face is roughly balanced,
            # a smeared one is not.
            grad_x = float(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3).var())
            grad_y = float(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3).var())
            strongest = max(grad_x, grad_y)
            directional_ratio = min(grad_x, grad_y) / strongest if strongest else 0.0

    confidence = float(det_confidence) if det_confidence is not None else 1.0

    reasons = []
    if face_pixels < Config.MIN_FACE_PIXELS:
        reasons.append(
            f"face is {face_pixels}px across, below the {Config.MIN_FACE_PIXELS}px minimum "
            "— too little detail to be a reliable reference"
        )
    if blur < Config.MIN_BLUR_VARIANCE:
        reasons.append(
            f"sharpness {blur:.1f} is below {Config.MIN_BLUR_VARIANCE} — out of focus"
        )
    if directional_ratio < Config.MIN_BLUR_DIRECTIONAL_RATIO:
        reasons.append(
            f"edge detail is lopsided ({directional_ratio:.2f} vs "
            f"{Config.MIN_BLUR_DIRECTIONAL_RATIO} minimum) — looks like motion blur "
            "from a moving subject"
        )
    if confidence < Config.MIN_DET_CONFIDENCE:
        reasons.append(
            f"detector confidence {confidence:.2f} is below {Config.MIN_DET_CONFIDENCE}"
        )
    # Not a hard gate on its own, but worth surfacing: a crushed or blown-out crop
    # gives the encoder very little to work with.
    if brightness < 30 or brightness > 225:
        reasons.append(f"exposure looks extreme (mean brightness {brightness:.0f})")

    # Weighted so size and sharpness dominate — those are what actually destroy an
    # embedding. Kept deliberately simple: you should be able to read the score and
    # know why it is what it is.
    size_score = min(face_pixels / 160.0, 1.0)
    sharp_score = min(blur / 150.0, 1.0)
    balance_score = min(directional_ratio / 0.5, 1.0)
    score = round(
        0.35 * size_score + 0.35 * sharp_score + 0.15 * confidence + 0.15 * balance_score, 3
    )

    return {
        "face_pixels": face_pixels,
        "blur_variance": round(blur, 1),
        "blur_directional_ratio": round(directional_ratio, 3),
        "brightness": round(brightness, 1),
        "det_confidence": round(confidence, 4),
        "quality_score": score,
        "passes": not reasons,
        "reasons": reasons,
    }
