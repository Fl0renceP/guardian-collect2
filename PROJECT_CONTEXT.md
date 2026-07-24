# Guardian Collective — Project Context

**Team:** Ctrl+Alt+Elite
**Event:** Discovery GradHack 2026 — Theme: "AI for Safer Communities"
**Purpose of this file:** Shared context for every team member and for AI coding assistants (GitHub Copilot, Claude Code) working in this repo. Read this before generating or reviewing any code. Keep it up to date as decisions change.

---

## 1. Problem statement

Discovery Insure wants to move from reactively paying out home invasion and vehicle theft claims to proactively preventing them, using:
- Doorbell/street-facing camera footage from members' homes
- Discovery Insure's historical claims data
- Partnerships with security companies / Crime Prevention Units (armed response, SAPS)

Two things need to work together:
1. **Reactive/investigative loop:** detect faces and license plates in camera footage, compare them against a database of known offenders/suspects, and alert the right people immediately.
2. **Proactive/preventive loop:** mine claims data for hot-spots (location + time patterns), and use that to optimize where security patrols go and when.

## 2. Stakeholders and what they can do

| Stakeholder | Facing | Key actions |
|---|---|---|
| **Discovery Insure Members** | Consumer | Receive alerts when an **offender** is identified near them (not suspects — avoids false-alarm fatigue); view crime hot-spots near them; submit claims |
| **Discovery Employees** | Internal | Approve/deny member-submitted claims (approved claims flow into the claims dataset); view analytics/read-only dashboard |
| **Crime Prevention Units** (armed response / SAPS) | Internal/B2B | Receive alerts for **both** offenders and suspects (they need the fuller picture to respond); get optimized patrol/response routes; are assumed to be the source of new suspect/offender reference images in a real deployment (we simulate this with seed data for the hackathon) |

**Design rule:** members only ever see "offender" alerts (higher confidence, avoids scaring people over an unverified suspect). Crime Prevention Units see both, since they're the ones who act on the ground.

## 3. Functional requirements

1. **Facial recognition** — detect a face in an image/video frame, generate a comparable representation (embedding), and match it against a known face database.
2. **Image-to-text (LPR)** — extract license plate characters from an image.
3. **Cross-multimedia comparison** — determine whether a face or plate appearing in one image/video matches one seen elsewhere.
4. **Alerts** — push notification to members (offender only) and Crime Prevention Units (offender or suspect).
5. **Claims ingestion** — members submit claims → queryable, structured storage → Discovery employees verify/approve → approved claims join the working dataset used for hot-spot analysis.
6. **Route optimization** — suggest efficient patrol/response routes between hot-spots or toward an active alert location.
7. **Crime hot-spot visualization** — map view, filterable by location and by crime category (Theft, Hijack, Armed Robbery, Burglary, Attempted Theft, Remote Jamming).

## 4. Current build priority

**Facial recognition is the priority demo.** Get face-in → match/no-match → alert-or-not working end to end before anything else. See `DEV_ROADMAP.md` for the phase plan and `DATA_SCHEMA.md` for the exact tables involved.

## 5. Facial recognition logic (current design)

### Original team logic (as first proposed)
```
existing face_db → camera/image input → check_faces()
  → True (offender/suspect match) → send_alert()
  → False (verified) → print("Member is verified")
  → if face not in db → add to db as "verified" by default
```

### Revised logic used in this repo — and why it changed
Auto-labelling every unrecognized face as "verified" creates a security hole: an offender only has to be photographed once under bad conditions (poor angle, low light) to get permanently whitelisted, after which they'd never trigger an alert again. It also means we'd be silently building a biometric database of every stranger caught on a camera, which is a real responsible-AI/privacy problem worth avoiding, both ethically and for how judges are likely to view it.

**Revised flow:**
```
1. Seed face_db with known reference embeddings, each labelled:
   offender | suspect | verified (i.e. known/whitelisted, e.g. a household member)

2. Camera/image input arrives (file upload for the demo, simulating a live feed)

3. detect_faces(image) → 0..n face bounding boxes

4. for each detected face:
     embedding = generate_embedding(face)
     match, score = find_best_match(embedding, face_db)   # cosine similarity, threshold-based

     if match found and match.label in [offender, suspect] and score >= THRESHOLD:
         log_detection(match, score, camera_id, timestamp, location)
         send_alert(match.label, camera_id, location)      # members only see "offender"

     elif match found and match.label == verified and score >= THRESHOLD:
         log_detection(match, score, camera_id, timestamp, location)
         # no alert — known/whitelisted person

     else:
         log_detection(None, score, camera_id, timestamp, location)
         # unknown face: NOT auto-added to the identity registry.
         # Optionally flagged for manual review by a Discovery employee
         # (status = "pending_review") rather than silently trusted.
```

Every check is logged in a separate `detections` table (see `DATA_SCHEMA.md`), independent of the identity registry (`faces` table). This keeps the "who is this person" registry small and deliberate, while still building the continuous dataset Discovery wants for future use.

### Model choice for the hackathon
Use the `face_recognition` Python library (dlib-based) for face detection + 128-d embeddings + `face_recognition.face_distance` for similarity. This avoids depending on Azure's Face API identification/verification endpoints, which are Microsoft **Limited Access** features requiring a registration/approval process that will not clear in hackathon timelines. Plain face *detection* is fine to layer in from Azure AI Vision later if wanted for the demo polish, but *matching* logic should be self-contained so we're not blocked on external approval.

## 6. Tech stack

| Layer | Choice |
|---|---|
| Backend | Python, Flask (app factory pattern), SQLAlchemy |
| Face recognition | `face_recognition` (dlib) for hackathon speed; embeddings stored as vectors |
| Database | PostgreSQL (Azure Database for PostgreSQL), PostGIS for geospatial, pgvector for embedding similarity search |
| Image storage | Azure Blob Storage (store image files; keep DB rows lean) |
| Claims OCR / plates | Azure AI Vision Read/OCR API |
| Maps / hotspots / routing | Azure Maps (geocoding, route optimization, map rendering) |
| Alerts | Azure Functions (trigger logic) + Firebase Cloud Messaging (push delivery) |
| AI-generated briefings | Azure AI Foundry (agent/prompt flow) |
| Frontend | React 18 + Vite |
| Hosting | Azure App Service (backend), Azure Static Web Apps or App Service (frontend) |
| CI/CD | GitHub Actions → Azure (OIDC auth) |

## 7. Repo structure

```
backend/
  app.py            # app factory, create_app()
  wsgi.py           # entrypoint for gunicorn / Azure
  config.py         # env-based config
  extensions.py     # db = SQLAlchemy(), shared extension instances
  models/
    face.py         # Face model (identity registry)
    detection.py    # Detection model (event log)
  routes/
    face_routes.py  # /api/faces, /api/detect endpoints
    health_routes.py
  services/
    face_service.py # detection + embedding + matching + alert logic
  seed_data.py       # populate 3 test faces (offender, suspect, verified)
  requirements.txt
  .env.example
frontend/
  src/               # React app (owner: TBD per TEAM_ROLES.md)
docs/
  PROJECT_CONTEXT.md   # this file
  DEV_ROADMAP.md
  TEAM_ROLES.md
  DATA_SCHEMA.md
.github/workflows/
  azure-deploy.yml
```

## 8. Data schemas

See `docs/DATA_SCHEMA.md` for full column-level detail on:
- `faces` (identity registry: offenders/suspects/verified people)
- `detections` (log of every camera check, matched or not)
- `claims` (the Discovery Insure claims dataset: `Incident, PERIL, SUBURB, ITEM_TYPE, VEHICLE_MAKE, VEHICLE_MODEL, VEHICLE_YEAR, INCIDENT_DATE_TIME, CLAIM_AMOUNT, ITEM_CATEGORY, ITEM_PERIL_DESCR`)

## 9. Conventions for contributors (and Copilot/Claude Code)

- Backend: Flask app factory pattern, blueprints per feature area, no business logic in route handlers — put it in `services/`.
- Never commit real API keys or connection strings — use `.env` (gitignored), reference `.env.example` for required variables.
- Don't reintroduce the "auto-verify unknown faces" pattern described in Section 5 — it's a known rejected design.
- Claims data fields should be treated as read from the CSV format described above; don't assume extra columns exist (e.g. there's no lat/long — suburb names need geocoding via Azure Maps before any spatial work).
- Keep commit messages descriptive; open a PR against `main` rather than pushing directly once more than one person is working in the repo.

## 10. Open questions (raised by the team, still unresolved)

- Preferred output format for the hot-spot map: internal ops tool, member-facing app, or both? (Current working assumption: both, with different feature depth — see prior team discussion.)
- How does this connect to Discovery Insure's existing claims workflows, if at all — assumed simulated/standalone for the hackathon.
- Is there an existing security-company/SAPS data partnership to assume, or is that integration being designed from scratch? Assumed from scratch, seeded with test data for the demo.
- Permission to use Discovery's branding/badges — assume no until confirmed; use generic "Guardian Collective" branding for the demo.
