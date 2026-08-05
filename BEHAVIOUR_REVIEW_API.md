# Behavioural Review API — contract

The agreed JSON shapes between three parts that get built separately:

| Part | Owner | What it produces |
|---|---|---|
| **Facial recognition** | `services/recognition.py`, `LiveScanDemo.jsx` | A match: confidence + label (offender / suspect / verified) |
| **Behavioural analysis** | `backend/behavioural_analysis/` | A movement score + plain-English explanations |
| **Review card** | `frontend/src/views/BehaviourReview.jsx` | A human decision: confirm or deny |

This document is what each side codes against so they can be built in parallel.

**Built so far:** everything except clip buffering (step 5 of the build order) —
the behavioural module, §1 the face-to-body join, §2 the face box round-trip,
§3 event ingest, §4–5 the queue and card, §6 confirm/deny/reopen, §7 storage.

---

## 0. The core idea

Facial recognition alone produces both false positives (an innocent person
matched to an offender record) and false negatives (a real offender missed
because their face is covered or absent from the database). Behaviour is an
independent second signal on the same person:

- face match **+ ordinary movement** → score pushed **down**
- **no face match +** unusual movement → **still reaches a human**

The two are fused into one `composite_risk_score`, and the only thing that
score does is decide whether a person sees the card. The human decides
everything else.

---

## 1. The join — which face belongs to which body ✅ BUILT

> Implemented in `backend/services/behaviour_track_service.py`, called from
> `/api/v1/scan-face`. Body positions arrive via `POST /api/v1/behaviour/tracks`
> (published by the behavioural module under `--push`). Tests:
> `backend/tests/test_behaviour_join.py`.
>
> **The join needs a third input the rest of this document missed:** where the
> BODIES are. A face scan knows where the face was; only the behavioural
> pipeline knows where the bodies were. Neither can do the join alone, so the
> pipeline now publishes normalised body boxes to `behavioural_tracks`, which
> the correlator reads. Those rows expire after 30 minutes — a continuous
> record of where every body stood is far more intrusive than the sparse events
> it exists to support.
>
> **Track ids are per-run.** Restart the module and `person-1` is a different
> human being. Links are therefore only applied to events within
> `LINK_VALIDITY_MINUTES` (30) of the scan that produced them; without that
> bound a later run's `person-1` would silently inherit an earlier person's
> identity. The proper fix is a per-run session id on both events and links.

This is the one genuinely new piece of engineering, and everything else depends
on it.

The face module returns a match for a **face crop**. The behavioural module
tracks **whole-body boxes** with anonymous track IDs. To fuse them you must know
they describe the same person. With one person in frame it is trivial. With two
(as in the sample garden clip) a face match with no coordinates cannot be
attached to either body without guessing.

**Rule:** a face match belongs to the person track whose bounding box contains
the centre of the face box, in the same frame.

```
face_centre = (face_box.x + face_box.w/2, face_box.y + face_box.h/2)
track        = the person track whose bbox contains face_centre
            (if several contain it, the smallest box wins — that is the nearer person)
            (if none contain it, the match is UNATTACHED and must not be fused)
```

An unattached match is not an error and must not be silently attached to the
nearest track. It means "we matched a face but cannot say whose body it is",
and the card should say exactly that.

**The correlator refuses in four cases**, each reported with a reason:

| Reason | Meaning |
|---|---|
| `no_body_tracking_at_that_moment` | Nothing was tracking bodies on that camera within 1s |
| `face_outside_every_body_box` | The face sat in no body — a reflection, a photo, or an unconfirmed track |
| `ambiguous_several_bodies_overlap` | Two bodies of similar size both contain the face. **Refused, not guessed** |
| `correlation_error` | Something failed; the face match still stands |

The ambiguity refusal is the important one. Where two people overlap at similar
distance there is no honest way to choose, and a wrong join produces a
confident, specific, wrong accusation — worse than either signal failing alone.
"No facial match" is a perfectly good answer; a wrong name is not.

An automatic link is labelled `source: "automatic"` on the card, with its
evidence (time delta, how many bodies were in frame), so a reviewer weighs it
rather than reads it as fact. A human's decision always outranks it.

**Good news:** `LiveScanDemo.jsx` already computes this box. Line 293:

```js
faceBoxRef.current = { x: b.originX, y: b.originY, w: b.width, h: b.height }
```

It is currently used only to draw the reticle and is discarded. Sending it with
the scan is the whole change on the frontend side.

### Coordinate space

`face_box` is in **source video pixels** (`video.videoWidth` × `videoHeight`),
not displayed CSS pixels — `LiveScanDemo` already stores the unscaled values.
Send `frame_width` and `frame_height` alongside so the backend can normalise.

---

## 2. `POST /api/v1/scan-face` — extended (backwards compatible) ✅ BUILT

> Implemented in `backend/services/frame_context.py` (parsing/normalising),
> `backend/app.py` (echo), and `frontend/src/components/LiveScanDemo.jsx`
> (sending). Tests: `backend/tests/test_frame_context.py`.

The existing endpoint, with optional new fields. **Every new field is optional**;
omitting them gives exactly today's behaviour, so nothing that calls it now breaks.

**Request** (`multipart/form-data`)

| Field | Type | Req | Notes |
|---|---|---|---|
| `file` | file | yes | The frame. Unchanged. |
| `face_box` | JSON string | no | `{"x":123,"y":80,"w":64,"h":64}` in source video pixels |
| `frame_width` / `frame_height` | int | no | Source video dimensions |
| `camera_id` | string | no | Which camera. Defaults to `"demo_upload"` |
| `captured_at` | iso8601 | no | When the frame was grabbed, for correlating with behaviour |

**Response** — today's body, plus:

```jsonc
{
  "success": true,
  "is_known_user": true,
  "status": "suspect",              // offender | suspect | verified
  "match_distance": 0.1834,         // cosine DISTANCE — smaller is a better match
  "person": { "id": 12, "full_name": "…", "image_url": "…" },

  // NEW — only present when face_box was supplied
  "face_box": { "x": 123, "y": 80, "w": 64, "h": 64 },
  "frame_size": { "w": 720, "h": 540 },
  "captured_at": "2026-08-05T14:22:07Z",
  "camera_id": "gate_cam_01"
}
```

> **`match_distance` is a distance, not a confidence.** Smaller means a better
> match, and `Config.MATCH_THRESHOLD` (0.30) is the largest distance still
> counted as a match. Converting is `confidence = 1 - (distance / threshold)`,
> clamped to 0–1. `risk_fusion.face_signal_from_recognition()` already does this.
> Passing the raw distance into the fusion would invert it — a perfect match
> would read as zero confidence.

---

## 3. `POST /api/v1/behaviour/events` — behaviour module → backend ✅ BUILT

> Implemented in `backend/routes/behaviour_routes.py`,
> `backend/services/behaviour_events_service.py`, tables created by
> `backend/init_behaviour_db.py`. `push_to_flask_api()` posts here when `--push`
> is passed and `output.flask_api_url` is set (wired in `config.demo.yaml`,
> deliberately left `null` in `config.yaml`).

What `behavioural_analysis/api_output.py::push_to_flask_api()` posts.

**Request body** — the event exactly as the module already emits it:

```jsonc
{
  "event_id": "person-4@2026-08-05T14:22:07+00:00",
  "track_id": "person-4",                    // anonymous, per-run. Not a person id.
  "timestamp": "2026-08-05T14:22:07+00:00",
  "location_zone_id": "gate_cam_01",
  "behavioural_risk_score": 0.71,
  "triggered_heuristics": [
    {
      "type": "crouched_near_vehicle",
      "confidence": 0.82,
      "explanation": "Track held a crouched posture — 62% of their own standing height, threshold 72% — beside a car for 9.4s, rather than passing at walking height."
    }
  ],
  "face_match_confidence": null,             // null when no face match was available
  "composite_risk_score": 0.58,
  "requires_human_review": true,
  "reasoning": ["…step-by-step how the score was reached…"]
}
```

**Response** `201` when stored, `200` when it was already held (so a retrying
pusher can tell the difference):

```jsonc
{
  "event_id": "…",
  "review_id": "rev-000123",   // null when the event did not need a human
  "queued_for_review": true,
  "duplicate": false
}
```

`queued_for_review` mirrors `requires_human_review`. Events below the threshold
are still stored (they are the denominator for measuring false-positive rate)
but do not create a card.

**Rules — enforced, not assumed**
- `explanation` is **required** on every triggered heuristic. An event missing
  one is rejected `400` — a score with no sentence a human can read is not
  reviewable, only obeyable.
- Identity fields (`full_name`, `person_id`, `embedding`, `image_url`, …) are
  **rejected** `400`, not quietly stripped. Silently dropping them would hide a
  caller that is trying to send identity, and that caller needs to know.
- Ingest is **idempotent on `event_id`** — a retry after a network failure
  returns the original `review_id` rather than opening a second review.

`GET /api/v1/behaviour/events?review_only=1&limit=50` reads them back, so the
ingest is verifiable before the queue endpoints in §4 exist.

---

## 4. `GET /api/v1/behaviour/review-queue` — the list

**Query:** `status` = `pending` | `confirmed` | `denied` (default `pending`),
`camera_id`, `limit`.

```jsonc
{
  "reviews": [
    {
      "review_id": "rev-000123",
      "status": "pending",
      "opened_at": "2026-08-05T14:22:09Z",
      "camera_id": "gate_cam_01",
      "composite_risk_score": 0.58,
      "face": { "label": "suspect", "confidence": 0.61, "attached": true },
      "headline": "Crouched at a vehicle for 9.4s",   // the top heuristic, for the list row
      "still_url": "…/sas-signed.jpg"
    }
  ],
  "counts": { "pending": 3, "confirmed": 11, "denied": 6 }
}
```

---

## 5. `GET /api/v1/behaviour/review-queue/{review_id}` — one card

Everything the card renders. This is the join, assembled.

```jsonc
{
  "review_id": "rev-000123",
  "status": "pending",
  "opened_at": "2026-08-05T14:22:09Z",
  "camera_id": "gate_cam_01",
  "suburb": "MIDDELBURG",

  // TOP OF CARD — who the face module thinks this is
  "identity": {
    "attached": true,                  // false = face matched but not tied to this body
    "label": "suspect",                // offender | suspect | verified | null
    "confidence": 0.61,                // 0..1, already converted from cosine distance
    "match_distance": 0.1834,
    "person_id": 12,
    "full_name": "Victoria Armstrong",
    "reference_image_url": "…",        // SAS-signed, short-lived
    "still_url": "…",                  // the frame that triggered the scan
    "first_seen_label": "First match on this camera"
  },

  // MIDDLE — the behaviour, and why it was flagged
  "behaviour": {
    "track_id": "person-4",
    "behavioural_risk_score": 0.71,
    "composite_risk_score": 0.58,
    "clip_url": "…/clip.mp4",          // buffered ~30s around the trigger, SAS-signed
    "live_stream_url": "…",            // null when the person has left frame
    "triggered_heuristics": [
      { "type": "crouched_near_vehicle", "confidence": 0.82, "explanation": "…" }
    ],
    "reasoning": ["…"]
  },

  // BOTTOM — what a decision would mean
  "decision": {
    "options": ["confirm", "deny"],
    "confirm_effect": "Records that this person is a suspect, and alerts Crime Prevention Units.",
    "deny_effect": "Records this as a false flag. No alert is sent."
  }
}
```

### Live *and* buffered

`live_stream_url` is useful — an operator watching a person right now is the
real use case for armed response. But by the time someone opens the card the
person may be gone, so `clip_url` (a rolling buffer around the trigger) is what
makes the card work at all. Build the clip; treat live as a bonus that is
`null` whenever the track is no longer in frame.

---

## 6. `POST .../confirm`, `/deny`, `/reopen`, `GET .../history` ✅ BUILT

> Implemented in `backend/services/behaviour_decision_service.py`. Decisions are
> append-only in `behavioural_review_decisions`; `behavioural_reviews` holds
> only the current state. Tests: `backend/tests/test_behaviour_decisions.py`.
>
> **Confirming does NOT write to the identity registry.** `identity_written` is
> always `false`. The judgement is recorded against the review, and
> `persons.status` is untouched — a movement heuristic must not be a one-click
> path into the curated registry that facial recognition matches against. That
> is the auto-labelling pattern PROJECT_CONTEXT §5 already rejected once, and
> arriving at it via a behaviour score does not make it a different pattern.
> Promotion stays a separate, deliberate act.

Mirrors `/api/claims/{id}/approve` and `/deny`.

**Confirm**

```jsonc
{ "reviewer_id": "emp-004", "label": "suspect", "note": "Seen trying door handles." }
```

**Deny**

```jsonc
{ "reviewer_id": "emp-004", "reason": "Resident fetching something from their own car." }
```

`reason` is **required** on deny — same rule as declining a claim. It is also
the most valuable data the whole system produces: a denied flag is a measured
false positive, and the only honest way to show the fusion reduces them.

**Response** `200`

```jsonc
{
  "review_id": "rev-000123",
  "status": "confirmed",
  "decided_by": "emp-004",
  "decided_at": "2026-08-05T14:24:31Z",
  "reversible_until": "2026-08-06T14:24:31Z",
  "identity_written": true,          // whether the face registry was updated
  "alerts_sent": ["cpu"]
}
```

### What confirming actually does

Confirming is **a human making an identification**, not the system making one.
That is legitimate — it is the entire point of the review step — but it has to
be handled as a decision by a named person:

1. **Log who, when, and why.** `decided_by`, `decided_at`, and the note go into
   the audit trail beside the behavioural inputs. The decision must be
   reconstructable later without the video.
2. **Make it reversible.** Provide an undo window. A label applied in error
   should not be permanent.
3. **Never let the system apply the label on its own.** No threshold, however
   high, may write `offender` into the face registry without a person clicking.
   That is the auto-labelling pattern rejected in PROJECT_CONTEXT §5, and
   arriving at it through a behaviour score instead of a face score does not
   make it a different pattern.
4. **Alerts still route through `alerts_service.audience_for`.** Confirming a
   `suspect` notifies Crime Prevention Units only. Members see `offender` alerts
   and nothing else — that rule is what keeps members from false-alarm fatigue,
   and a behavioural flag must not become a way around it. A confirmation with
   **no identity attached alerts nobody** — there is no one to alert about — but
   the review still stands as a confirmed observation.

### Deciding twice, and undoing

`POST .../reopen` returns a decided review to `pending` within
`REVERSIBLE_HOURS` (24). Deciding an already-decided review is `409` until it is
reopened. The reopen appends a row saying so; it never removes the decision it
undoes. `GET .../history` returns the whole trail — a reviewer who confirmed,
reopened, then denied is three facts, and the middle one is exactly what an
audit needs to see.

> **There is no authentication yet.** `reviewer_id` is supplied by
> `session.jsx` and trusted by the API (PROJECT_CONTEXT §9). Before this feature
> is real, the reviewer identity on an identification decision has to be
> genuinely authenticated — it is the audit trail's only signature.

---

## 7. Storage ✅ BUILT

Two Postgres tables, created by `python init_behaviour_db.py`. Neither holds
biometrics. That script does **not** drop anything, unlike `init_db.py` and
`init_plate_db.py` — those reseed fixed demo data, whereas these tables hold
event history and human decisions that a setup script must never be able to erase.

> **Two corrections against the live database.** Identity lives in `persons`,
> not the `faces` table named in DATA_SCHEMA.md, so the identity FK is
> `matched_person_id UUID REFERENCES persons(id)`. And **PostGIS is not
> installed** — the only extension present is `vector` — so coordinates are
> plain `DOUBLE PRECISION` columns matching the `detections` table, rather than
> a geometry column nothing could index.

**`behavioural_events`** — every event, flagged or not.

| Column | Notes |
|---|---|
| `event_id` | PK |
| `track_id` | Anonymous, per-run |
| `camera_id` / `location_zone_id` | For the hot-spot map join |
| `occurred_at` | |
| `behavioural_risk_score`, `composite_risk_score`, `face_match_confidence` | |
| `triggered_heuristics` | JSON, explanations included |
| `reasoning` | JSON, the scoring trail |
| `requires_human_review` | |

**`behavioural_reviews`** — the human decisions.

| Column | Notes |
|---|---|
| `review_id` | PK, `rev-000123` from a sequence |
| `event_id` | FK, unique — one review per event |
| `matched_person_id` | FK to `persons`, nullable — **the only link to identity in this schema, and it lives here, not on the event** |
| `matched_label` | `person_status` enum: offender / suspect / verified |
| `status` | `pending` \| `confirmed` \| `denied` |
| `decided_by` / `decided_at` | The audit signature |
| `decision_note` / `denial_reason` | |
| `clip_url` | Private blob, served via short-lived SAS like claim media |

The behavioural module itself keeps writing its own local SQLite/JSONL audit
regardless. That is deliberate: the module must be auditable standalone, not
only when a backend happens to be reachable.

**Media follows the claim-media rules** (`services/storage_service.py`): private
container, short-lived SAS URLs per request, never public, never persisted onto
the record. Clips of people at their homes are at least as sensitive as claim
photos. Give them a retention limit — a clip whose review is closed does not
need to exist indefinitely.

---

## 8. Build order

1. ~~**Send `face_box`** from `LiveScanDemo.jsx` and echo it back.~~ **DONE.**
   The scan now returns `face_box`, `face_box_normalised`, `face_box_centre`,
   `frame_size`, `camera_id` and `captured_at`. The live scan panel displays the
   centre so the round-trip is visible.
2. ~~**`POST /api/v1/behaviour/events`** + the `behavioural_events` table.~~
   **DONE.** Both tables exist, ingest validates and is idempotent, and
   `push_to_flask_api` posts to it under `--push`. Verified end to end: the
   module analysed a sample clip, POSTed, and the event is in Postgres with its
   explanation intact.
3. **The queue endpoints** and the card (mock exists: `BehaviourReview.jsx`).
4. ~~**Confirm/deny** + the audit fields.~~ **DONE.** Confirm, deny, reopen and
   history, with an append-only decision trail. Denying requires a reason;
   deciding requires a reviewer id; confirming never touches the face registry.
5. **Clip buffering.** The heaviest piece; the card works from the explanations
   until it exists.

**Still outstanding, and it blocks going live:** there is no authentication, so
`reviewer_id` is whatever the client sends. On an identification decision that
field is the audit trail's only signature. Everything above is built and tested,
but the signature is currently unforgeable in the wrong direction — anyone can
sign as anyone.
