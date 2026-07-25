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
| Claims database | **Azure Cosmos DB** (`guardian-db` / `insurance-data`) — source of truth for claims and the hot-spot map |
| Face/vector database | PostgreSQL (Azure Database for PostgreSQL), PostGIS for geospatial, pgvector for embedding similarity search |
| Image storage | Azure Blob Storage (store image files; keep DB rows lean) |
| Claims OCR / plates | Azure AI Vision Read/OCR API |
| Maps / hotspots | **Leaflet + leaflet.heat**, OpenStreetMap (CARTO) tiles for rendering; **Nominatim** for one-time suburb geocoding — no API key, see note below |
| Routing | **Valhalla** (OpenStreetMap, public FOSSGIS instance — no key) with `exclude_polygons` for risk-avoiding routes; **H3** for the travel-risk surface |
| Alerts | Azure Functions (trigger logic) + Firebase Cloud Messaging (push delivery) |
| AI-generated briefings | Azure AI Foundry (agent/prompt flow) |
| Frontend | React 18 + Vite |
| Hosting | Azure App Service (backend), Azure Static Web Apps or App Service (frontend) |
| CI/CD | GitHub Actions → Azure (OIDC auth) |

**Why the map isn't Azure Maps:** the hot-spot map was built against Leaflet + OpenStreetMap because no Azure Maps subscription key exists in the project, and the map is a first-screen feature that couldn't wait on provisioning. Nothing about the choice is load-bearing — the frontend reads `GET /api/hotspots`, which returns plain `{suburb, lat, lng, count, ...}` records. Swapping in the Azure Maps control later means rewriting the render layer in `backend/static/index.html` only; the API contract and the geocode cache stay as they are.

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
    face_routes.py     # /api/faces, /api/detect endpoints
    hotspot_routes.py  # /api/hotspots, /api/filters
    health_routes.py
  services/
    face_service.py    # detection + embedding + matching + alert logic
    claims_service.py  # claims load, hot-spot aggregation, submission + review
    geocode_service.py # read access to the suburb geocode cache
    storage_service.py # claim media -> private Blob container, SAS read URLs
    users_service.py   # Cosmos `users` container: members/employees/units + opt-in location
    members_service.py # compatibility shim over users_service — prefer users_service
    risk_service.py    # H3 travel-risk surface (where x when)
    routing_service.py # Valhalla routing + risk-avoiding alternative
    alerts_service.py  # alert feed + offender/suspect audience routing
    patrol_service.py  # CPU patrol loops (coverage/allocation, not shortest path)
  scripts/
    geocode_suburbs.py # one-time (resumable) Nominatim geocoder
  static/
    index.html      # standalone hot-spot heatmap served by Flask
  data/
    suburb_geocache.json  # suburb -> lat/lng, committed so nobody re-runs geocoding
  seed_data.py       # populate 3 test faces (offender, suspect, verified)
  requirements.txt
  .env.example
frontend/            # React 18 + Vite (owner: Tadiwa; claims flows: Keziah)
  vite.config.js     # proxies /api -> Flask :5000, so no CORS setup
  src/
    main.jsx         # routes
    api.js           # fetch wrapper + shared formatters
    session.jsx      # current role/identity — STANDS IN FOR AUTH, see §9
    theme.css        # design tokens (shared palette with the backend map)
    components/
      Layout.jsx     # app bar, nav, role switcher, theme toggle
      StatusPill.jsx
    views/
      HotspotMap.jsx   # heatmap (Leaflet driven imperatively via refs)
      SafeRoute.jsx    # member: fastest vs risk-avoiding route
      SubmitClaim.jsx  # member: report an incident
      MyClaims.jsx     # member: status + decline reasons
      ReviewQueue.jsx  # employee: approve / decline
      AlertsFeed.jsx   # CPU: alerts in the unit's operating area
      PatrolPlan.jsx   # CPU: per-vehicle patrol loops
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
- Claims are read from **Cosmos DB**, not the CSV — `services/claims_service.py` is the only place that touches either. The CSV remains as an offline fallback; don't add a second reader for it.
- Claims documents still carry no lat/long, so suburb names are resolved through the committed geocode cache (`backend/data/suburb_geocache.json`); don't add per-request geocoding.
- **A claim with a `status` field only counts toward hot-spots once approved.** Absence of `status` means "historical, already part of the dataset". If you add claim submission, write `status: "pending"` and flip it on approval — that's what keeps unverified member submissions off the map.
- After any claim write, call `claims_service.apply_to_snapshot(doc)` **and** `invalidate_cache()` — the first makes the very next read correct, the second lets a background refresh reconcile. Invalidating alone is not enough: stale-while-revalidate would keep serving the pre-write snapshot.
- **A member's home location is optional and opt-in.** Always read it through `users_service.member_home()` — that function is the single place `share_location` is enforced. Never read `home_lat` off a profile directly, and never treat stored coordinates as permission to use them. Turning sharing off must delete the coordinates, not just hide them.
- **There is no authentication.** `services/users_service.py` and `frontend/src/session.jsx` supply a demo identity that the API trusts. The `auth` block on each user document is a placeholder and is stripped before any response. Don't build anything that assumes `member_id`/`employee_id` is verified — swapping in a real auth provider is a known outstanding task (see `DEV_ROADMAP.md` Phase 5).
- Claim media lives in a **private** Blob container and is served through short-lived per-request SAS URLs (`services/storage_service.py`). Never make the container public or persist a signed URL on the claim document.
- Door-camera consent is opt-in, per-incident, and timestamped. Don't default it to true, don't infer it, and don't reuse one incident's consent for another.
- **Never present the travel-risk surface as street-level.** Claims are located to suburb centroids, so risk is binned to ~5 km² H3 cells. "This area has more evening hijack claims" is supported; "avoid this road" is not.
- The risk surface separates *where* (smoothed spatial density) from *when* (a single pooled hour/day profile) because claims are far too sparse per suburb to estimate both together. Don't refactor it into per-cell-per-hour counts — that's noise, not signal.
- Risk scores normalise against a **fixed** reference peak, not the current moment's peak. Normalising per query cancels the time multiplier out algebraically and makes every hour look identical.
- **Don't loosen `ROUTE_MIN_RISK_REDUCTION` to make demos look better.** Suggesting detours on noise is how this feature turns into an app that tells people to avoid particular neighbourhoods — a real redlining risk in the South African context.
- **All alerts go through `alerts_service.audience_for`.** Members see `offender` only; Crime Prevention Units see `offender` and `suspect`. Don't bypass it when adding a new alert source, and don't widen the member set — that rule is the whole reason members don't get false-alarm fatigue.
- **Don't stub the unwired alert feeds with fake data.** `_detection_alerts` and `_predicted_alerts` return empty on purpose until Phase 1 and Azure Functions exist. An alerts panel that invents offender sightings is worse than an empty one, and the UI already labels which feeds are live.
- CPU patrol planning is a **coverage/allocation** problem, not shortest-path. Don't refactor it to reuse `routing_service.compare_routes` — they answer different questions.
- Keep commit messages descriptive; open a PR against `main` rather than pushing directly once more than one person is working in the repo.

## 10. Open questions (raised by the team, still unresolved)

- Preferred output format for the hot-spot map: internal ops tool, member-facing app, or both? (Current working assumption: both, with different feature depth — see prior team discussion.)
- How does this connect to Discovery Insure's existing claims workflows, if at all — assumed simulated/standalone for the hackathon.
- Is there an existing security-company/SAPS data partnership to assume, or is that integration being designed from scratch? Assumed from scratch, seeded with test data for the demo.
- Permission to use Discovery's branding/badges — assume no until confirmed; use generic "Guardian Collective" branding for the demo.
