"""Decide who someone is from several frames, not from one.

A per-frame decision throws away the strongest evidence a video feed offers:
that the same person is visible repeatedly. One unlucky frame — a blink, a
turn, a passing shadow — can produce a wrong name or a missed alert, and the
system has no way to notice.

Tracking associates detections across frames into a track, accumulates the
per-frame distances, and decides on the aggregate. That changes the failure
characteristics in two useful ways:

  - A single bad frame cannot raise an alert on its own.
  - A person who is genuinely present accumulates evidence, so a run of
    marginal frames can still add up to a confident identification.

Association is IoU on the bounding box plus a sanity check on the embedding.
IoU alone confuses people who cross paths; the embedding check stops a track
hopping from one person to another when their boxes overlap.

Aggregation uses the MEDIAN distance rather than the mean or the minimum. The
minimum is what a per-frame system already effectively does and is the most
optimistic reading available — one flattering frame decides everything. The
mean is dragged around by outliers. The median asks "what does this track
usually look like", which is the question that matters.
"""

import logging
import os
import threading
import time
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# Box overlap needed to call two detections the same person between frames.
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
# A track with no matching detection for this long is closed.
#
# This has to be generous relative to the SCAN interval, not the frame rate. A
# scan costs ~1.4s and the live page waits at least 1.2s between them, so
# consecutive observations of the same person arrive 2.5-4s apart. An earlier
# 3.0s value expired every track between frames, so each frame started a fresh
# track and no track ever reached the evidence threshold — tracking silently did
# nothing. Keep this well above the real inter-scan gap.
TRACK_TTL_SECONDS = float(os.getenv("TRACK_TTL_SECONDS", "12.0"))
# Frames of evidence before a track is willing to commit to an identity.
MIN_FRAMES_FOR_DECISION = int(os.getenv("TRACK_MIN_FRAMES", "3"))
# An identity must be the closest candidate in at least this share of the
# track's frames before it is accepted — stops one identity being declared on
# the strength of a single frame in a long, otherwise-unknown track.
MIN_VOTE_SHARE = float(os.getenv("TRACK_MIN_VOTE_SHARE", "0.5"))
# Embeddings this far apart are not the same person, whatever the boxes say.
SAME_PERSON_MAX_DISTANCE = float(os.getenv("TRACK_SAME_PERSON_MAX", "0.45"))


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    overlap = (x2 - x1) * (y2 - y1)
    return overlap / float(aw * ah + bw * bh - overlap)


def _cosine(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Track:
    """One person, followed across frames."""

    _next_id = 1
    _id_lock = threading.Lock()

    def __init__(self, bbox, embedding, now):
        with Track._id_lock:
            self.id = Track._next_id
            Track._next_id += 1
        self.bbox = bbox
        self.embedding = np.asarray(embedding, dtype=np.float64) if embedding is not None else None
        self.created_at = now
        self.last_seen = now
        self.frames = 0
        # identity id -> list of distances observed for it on this track
        self.votes = defaultdict(list)
        self.identity_meta = {}
        self.no_decision_frames = 0
        self.unknown_frames = 0
        self.announced = None       # last decision emitted, to avoid repeating it

    def update(self, bbox, embedding, face_result, now):
        self.bbox = bbox
        self.last_seen = now
        self.frames += 1
        if embedding is not None:
            vector = np.asarray(embedding, dtype=np.float64)
            # Running average keeps the track's identity anchored without letting
            # the most recent frame redefine it.
            self.embedding = vector if self.embedding is None else (self.embedding * 0.7 + vector * 0.3)

        confidence = face_result.get("confidence")
        if confidence == "no_decision":
            self.no_decision_frames += 1
            return
        person = face_result.get("person") or {}
        distance = face_result.get("match_distance")
        if person.get("id") and distance is not None:
            self.votes[person["id"]].append(float(distance))
            self.identity_meta[person["id"]] = {
                "full_name": person.get("full_name"),
                "status": person.get("status"),
                "image_url": person.get("image_url"),
            }
        else:
            self.unknown_frames += 1

    def decide(self, threshold, probable_threshold):
        """Aggregate verdict for this track, or None while evidence is thin."""
        decided_frames = self.frames - self.no_decision_frames
        if decided_frames < MIN_FRAMES_FOR_DECISION:
            return None

        if not self.votes:
            return {
                "track_id": self.id, "frames": self.frames,
                "decided_frames": decided_frames, "confidence": "none",
                "is_known_user": False, "alert": False, "person": None,
                "distance": None, "vote_share": 0.0,
                "message": f"Unknown person — {decided_frames} frames, no match.",
            }

        # Winner by number of frames it led, ties broken by median distance.
        identity = max(self.votes.items(),
                       key=lambda kv: (len(kv[1]), -float(np.median(kv[1]))))[0]
        distances = self.votes[identity]
        share = len(distances) / float(decided_frames)
        median = float(np.median(distances))
        meta = self.identity_meta.get(identity, {})

        if share < MIN_VOTE_SHARE or median >= probable_threshold:
            return {
                "track_id": self.id, "frames": self.frames,
                "decided_frames": decided_frames, "confidence": "none",
                "is_known_user": False, "alert": False, "person": None,
                "distance": round(median, 4), "vote_share": round(share, 2),
                "message": (f"Unknown person — best candidate {meta.get('full_name')} "
                            f"led only {len(distances)}/{decided_frames} frames."),
            }

        confirmed = median < threshold
        flagged = meta.get("status") in ("offender", "suspect")
        return {
            "track_id": self.id,
            "frames": self.frames,
            "decided_frames": decided_frames,
            "confidence": "confirmed" if confirmed else "probable",
            "needs_review": not confirmed,
            "is_known_user": True,
            "alert": flagged,
            "status": meta.get("status"),
            "person": {"id": identity, **meta},
            "distance": round(median, 4),
            "best_distance": round(min(distances), 4),
            "worst_distance": round(max(distances), 4),
            "vote_share": round(share, 2),
            "supporting_frames": len(distances),
            "message": _track_message(meta, confirmed, flagged, len(distances), decided_frames),
        }


def _track_message(meta, confirmed, flagged, supporting, total):
    name = meta.get("full_name")
    status = (meta.get("status") or "").upper()
    evidence = f"{supporting} of {total} frames"
    if flagged:
        if confirmed:
            return f"ALERT: {status} — {name}, confirmed over {evidence}."
        return f"LIKELY {status}: {name} over {evidence} — verify before acting."
    if confirmed:
        return f"{name} recognised over {evidence}."
    return f"Probably {name} over {evidence} — low confidence."


class TrackerRegistry:
    """One tracker per camera. Frames from a camera arrive independently, so
    state has to persist between requests for tracking to mean anything."""

    def __init__(self):
        self._cameras = defaultdict(list)
        self._lock = threading.Lock()

    def update(self, camera_id, faces, threshold, probable_threshold, now=None):
        """Feed one frame's faces in; get back any track-level decisions.

        `faces` are entries from recognition's `faces` list, which must include
        their embeddings — call the recogniser with include_embeddings=True.
        """
        now = now if now is not None else time.time()
        decisions = []

        with self._lock:
            tracks = [t for t in self._cameras[camera_id] if now - t.last_seen <= TRACK_TTL_SECONDS]

            unmatched = list(tracks)
            for face in faces:
                bbox = face.get("bbox") or {}
                box = (bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0))
                embedding = face.get("embedding")

                best_track, best_iou = None, IOU_THRESHOLD
                for track in unmatched:
                    overlap = iou(box, track.bbox)
                    if overlap < best_iou:
                        continue
                    # Boxes can overlap when two people cross. The embedding is
                    # the tiebreaker that stops a track jumping between them.
                    if (embedding is not None and track.embedding is not None
                            and _cosine(track.embedding, np.asarray(embedding, dtype=np.float64))
                            > SAME_PERSON_MAX_DISTANCE):
                        continue
                    best_track, best_iou = track, overlap

                if best_track is None:
                    best_track = Track(box, embedding, now)
                    tracks.append(best_track)
                else:
                    unmatched.remove(best_track)

                best_track.update(box, embedding, face, now)

                verdict = best_track.decide(threshold, probable_threshold)
                if verdict:
                    # Only surface a track when its verdict changes, so a person
                    # standing still does not re-announce every frame.
                    signature = (verdict["confidence"], (verdict.get("person") or {}).get("id"))
                    if best_track.announced != signature:
                        best_track.announced = signature
                        decisions.append(verdict)

            self._cameras[camera_id] = tracks

        return decisions

    def active(self, camera_id, now=None):
        now = now if now is not None else time.time()
        with self._lock:
            return [t for t in self._cameras.get(camera_id, [])
                    if now - t.last_seen <= TRACK_TTL_SECONDS]

    def reset(self, camera_id=None):
        with self._lock:
            if camera_id is None:
                self._cameras.clear()
            else:
                self._cameras.pop(camera_id, None)


registry = TrackerRegistry()
