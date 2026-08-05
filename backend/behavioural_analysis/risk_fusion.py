"""Stage 6 — scoring, and fusing behaviour with facial recognition.

Two scores are produced here, and the difference between them is the point of
the whole module:

  behavioural_risk_score  — how unusual the MOVEMENT was, on its own
  composite_risk_score    — that, combined with whatever the face module said

Why fuse at all. Facial recognition fails in two directions, and each failure
has a different victim:

  * FALSE POSITIVE — it matches an innocent person to an offender record. The
    cost lands on someone who has done nothing. Behaviour is the brake: a face
    match paired with completely ordinary movement gets DAMPED, not escalated.

  * FALSE NEGATIVE — it misses a real offender because the face is covered, the
    light is poor, or the person is not in the database at all. Behaviour is the
    catch: strong behavioural evidence with NO face match still reaches a human.

Neither score is a verdict. The only decision this module makes is whether a
person should look, and that decision is `requires_human_review`.

===========================================================================
THE FORMULA, IN ONE PLACE
===========================================================================
  1. Behaviour       B = 1 - Π(1 - wᵢ·cᵢ)          (noisy-OR over triggers)
  2. Hot-spot lift   B' = B · (1 + zone_weight · zone_risk)     capped at 1
  3. Composite       C = 0.50·B' + 0.35·F + 0.15·(B'·F)
  4a. Face matched, behaviour ordinary   → C ×= 0.60
  4b. Face is a KNOWN RESIDENT           → C ×= 0.35
  4c. Strong behaviour, no face match    → review anyway
  5. Review          requires_human_review = C ≥ 0.50  (or 4c)

Every number above lives in config.yaml. Every step appends a plain-English
sentence to the reasoning trail that ships with the event.
===========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from heuristics import HeuristicResult
from settings import Settings

# Face labels as produced by services/recognition.py in the existing backend.
LABEL_OFFENDER = "offender"
LABEL_SUSPECT = "suspect"
LABEL_VERIFIED = "verified"


@dataclass
class FaceSignal:
    """What the SEPARATE facial recognition module concluded.

    Deliberately thin. This module receives a confidence and a coarse label and
    nothing else — no name, no person id, no embedding, no image reference.
    Identity stays in the face module (PROJECT_CONTEXT §9, POPIA).
    """

    confidence: Optional[float] = None
    label: Optional[str] = None

    @property
    def has_match(self) -> bool:
        return self.confidence is not None

    @property
    def is_verified_resident(self) -> bool:
        return (self.label or "").lower() == LABEL_VERIFIED


@dataclass
class FusionResult:
    behavioural_risk_score: float
    composite_risk_score: float
    requires_human_review: bool
    face_match_confidence: Optional[float]
    zone_risk: float
    reasoning: List[str] = field(default_factory=list)
    contributions: List[Dict[str, object]] = field(default_factory=list)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def behavioural_risk_score(
    triggered: Sequence[HeuristicResult],
    settings: Settings,
) -> tuple[float, List[Dict[str, object]], List[str]]:
    """Combine triggered heuristics into one 0..1 behavioural score.

    NOISY-OR: B = 1 - Π(1 - wᵢ·cᵢ)

    Chosen over a sum or a max because it behaves the way evidence does. Two
    independent observations accumulate (0.5 and 0.5 give 0.75, not 0.5), the
    result can never exceed 1 so no normalisation constant has to be invented,
    and no single heuristic can reach 1.0 alone — its weight caps it. That last
    property is deliberate: one heuristic should never be enough on its own.
    """
    if not triggered:
        return 0.0, [], ["No behavioural heuristics triggered."]

    product = 1.0
    contributions: List[Dict[str, object]] = []
    trail: List[str] = []

    for result in triggered:
        weight = settings.heuristic_weight(result.name)
        contribution = weight * float(result.confidence)
        product *= 1.0 - contribution
        contributions.append({
            "type": result.name,
            "weight": round(weight, 3),
            "confidence": round(float(result.confidence), 3),
            "contribution": round(contribution, 3),
        })

    score = _clamp(1.0 - product)
    names = ", ".join(f"{c['type']} ({c['confidence']:.2f} × weight {c['weight']:.2f})"
                      for c in contributions)
    trail.append(
        f"Behavioural score {score:.2f} from {len(contributions)} triggered "
        f"heuristic(s): {names}. Combined with noisy-OR, so independent signals "
        f"accumulate without any one of them reaching 1.0 alone."
    )
    return score, contributions, trail


def apply_zone_context(
    behaviour: float,
    zone_risk: float,
    settings: Settings,
) -> tuple[float, List[str]]:
    """Let claims hot-spot history amplify existing behavioural evidence.

    MULTIPLICATIVE, not additive, and that is the safeguard: at B = 0 the lift
    is 0 no matter how bad the area's claims history is. Living in, walking
    through, or working in a high-claims suburb can never by itself raise
    anyone's score. Context amplifies evidence; it does not create it.

    Same reasoning as ROUTE_MIN_RISK_REDUCTION in backend/config.py — in the
    South African context, treating people as riskier for where they are is a
    direct route to redlining.
    """
    if behaviour <= 0 or zone_risk <= 0:
        return behaviour, []

    weight = float(settings.fusion.zone_weight)
    lifted = _clamp(behaviour * (1.0 + weight * zone_risk))
    if lifted <= behaviour:
        return behaviour, []

    return lifted, [
        f"Behavioural score lifted {behaviour:.2f} → {lifted:.2f}: this camera's zone "
        f"carries a claims-derived risk of {zone_risk:.2f} (weight {weight}). This can "
        f"only amplify behaviour already observed — it adds nothing on its own."
    ]


def fuse_with_face(
    behaviour_score: float,
    face: Optional[FaceSignal],
    settings: Settings,
    *,
    zone_risk: float = 0.0,
    contributions: Optional[List[Dict[str, object]]] = None,
    trail: Optional[List[str]] = None,
) -> FusionResult:
    """Fuse the behavioural score with an external facial-match confidence.

    `face` is None when the face module had no match or was not run — which is
    exactly the case this module exists to cover.
    """
    cfg = settings.fusion
    reasoning: List[str] = list(trail or [])

    behaviour_effective, zone_trail = apply_zone_context(behaviour_score, zone_risk, settings)
    reasoning.extend(zone_trail)

    face_confidence = face.confidence if face and face.has_match else None
    face_component = float(face_confidence) if face_confidence is not None else 0.0

    behaviour_weight = float(cfg.behaviour_weight)
    face_weight = float(cfg.face_weight)
    agreement_weight = float(cfg.agreement_weight)

    composite = (
        behaviour_weight * behaviour_effective
        + face_weight * face_component
        + agreement_weight * (behaviour_effective * face_component)
    )
    composite = _clamp(composite)

    if face_confidence is None:
        reasoning.append(
            f"Composite {composite:.2f} = {behaviour_weight}×behaviour({behaviour_effective:.2f}) "
            f"with no facial-match confidence available to contribute."
        )
    else:
        reasoning.append(
            f"Composite {composite:.2f} = {behaviour_weight}×behaviour({behaviour_effective:.2f}) "
            f"+ {face_weight}×face({face_component:.2f}) "
            f"+ {agreement_weight}×agreement({behaviour_effective * face_component:.2f}). "
            f"The agreement term rewards two independent signals pointing the same way."
        )

    # --- Rule (a): the false-positive brake on facial recognition -----------
    normal_ceiling = float(cfg.normal_behaviour_ceiling)
    trust_threshold = float(cfg.face_trust_threshold)
    if face_confidence is not None and face_confidence >= trust_threshold and behaviour_effective <= normal_ceiling:
        damping = float(cfg.benign_behaviour_damping)
        before = composite
        composite = _clamp(composite * damping)
        reasoning.append(
            f"DOWNWEIGHTED {before:.2f} → {composite:.2f}: the face module reported a "
            f"confident match ({face_confidence:.2f}) but this person's movement was entirely "
            f"ordinary (behaviour {behaviour_effective:.2f} ≤ {normal_ceiling}). A face match "
            f"alone is not a reason to escalate someone who is behaving normally."
        )

    # --- Rule (b): a known resident is not an event -------------------------
    if face is not None and face.is_verified_resident:
        damping = float(cfg.verified_damping)
        before = composite
        composite = _clamp(composite * damping)
        reasoning.append(
            f"DOWNWEIGHTED {before:.2f} → {composite:.2f}: the face module matched this "
            f"person as a known/verified resident. A resident crouching beside their own "
            f"car is not an event."
        )

    # --- Rule (c): the false-negative catch ---------------------------------
    review_threshold = float(cfg.review_threshold)
    behaviour_only_threshold = float(cfg.behaviour_only_review_threshold)
    requires_review = composite >= review_threshold

    if face_confidence is None and behaviour_effective >= behaviour_only_threshold:
        if not requires_review:
            reasoning.append(
                f"FLAGGED FOR REVIEW despite a composite of {composite:.2f}: behavioural "
                f"evidence alone reached {behaviour_effective:.2f} (threshold "
                f"{behaviour_only_threshold}) with no facial match available. This is the "
                f"case facial recognition cannot cover — a covered face, poor light, or a "
                f"person not in any database."
            )
        requires_review = True

    reasoning.append(
        f"{'REQUIRES HUMAN REVIEW' if requires_review else 'No review required'} "
        f"(composite {composite:.2f} vs threshold {review_threshold}). "
        f"This module never acts on its own: a flagged event goes to a review queue "
        f"for a person to assess, and nothing else happens automatically."
    )

    return FusionResult(
        behavioural_risk_score=round(behaviour_score, 3),
        composite_risk_score=round(composite, 3),
        requires_human_review=requires_review,
        face_match_confidence=(
            round(float(face_confidence), 3) if face_confidence is not None else None
        ),
        zone_risk=round(zone_risk, 3),
        reasoning=reasoning,
        contributions=contributions or [],
    )


def score_event(
    triggered: Sequence[HeuristicResult],
    settings: Settings,
    *,
    face: Optional[FaceSignal] = None,
    zone_risk: float = 0.0,
) -> FusionResult:
    """The whole pipeline: triggers in, scored and explained event out."""
    behaviour, contributions, trail = behavioural_risk_score(triggered, settings)
    return fuse_with_face(
        behaviour,
        face,
        settings,
        zone_risk=zone_risk,
        contributions=contributions,
        trail=trail,
    )


# ---------------------------------------------------------------------------
# Adapter for the existing facial recognition module
# ---------------------------------------------------------------------------
# services/recognition.py returns a COSINE DISTANCE, not a confidence: smaller
# means more similar, and Config.MATCH_THRESHOLD (0.30) is the largest distance
# still considered a match. Feeding that number in raw would invert the entire
# fusion — a perfect match (distance ~0) would read as zero confidence.
def face_signal_from_recognition(
    result: Optional[Dict[str, object]],
    *,
    match_threshold: float = 0.30,
) -> Optional[FaceSignal]:
    """Convert a `process_incoming_face_image` result into a FaceSignal.

    Confidence = 1 - (distance / threshold), clamped to 0..1. So a distance of 0
    is confidence 1.0, and a distance at exactly the threshold is 0.0.

    Only the confidence and the coarse label cross this boundary. The person's
    name and id in the source dict are deliberately left behind.
    """
    if not result or not result.get("is_known_user"):
        return None

    distance = result.get("match_distance")
    if distance is None:
        return None

    confidence = max(0.0, min(1.0, 1.0 - (float(distance) / max(match_threshold, 1e-6))))
    label = result.get("status")
    return FaceSignal(confidence=confidence, label=str(label).lower() if label else None)
