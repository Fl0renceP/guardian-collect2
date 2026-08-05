# Dev Roadmap

Prioritization is deliberate: the GradHack panel advised getting facial recognition working first, so Phase 1 is scoped to be demoable on its own before anything else is built. Later phases can happen in parallel once Phase 1 is stable.

## Phase 0 — Setup (Day 1, all team)
- [ ] Repo created on GitHub, structure matches `PROJECT_CONTEXT.md` Section 7
- [ ] Azure resources provisioned: App Service (backend), Azure Database for PostgreSQL (with pgvector extension enabled), Blob Storage container for face images
- [ ] GitHub Actions → Azure deployment pipeline working (OIDC auth) — even deploying a "hello world" Flask app end to end first, before adding real logic
- [ ] `.env.example` filled in and shared (real `.env` never committed)
- [ ] Everyone can run `backend/app.py` locally against a local or dev Postgres instance

## Phase 1 — Facial recognition demo (priority — Tinashe)
Goal: file upload of an image → detect face → compare against 3 seeded faces (offender, suspect, verified) → correct match/no-match result → alert fires (or doesn't) as expected. This should work as a standalone demo before it's wired into the rest of the app.

- [ ] `face_recognition` library installed and working locally (dlib build can be finicky on some machines — confirm early, don't leave it to the night before)
- [ ] `faces` table created (schema in `DATA_SCHEMA.md`), pgvector extension enabled
- [ ] `seed_data.py` populates 3 test faces with real embeddings from 3 sample images
- [ ] `services/face_service.py`: `detect_faces()`, `generate_embedding()`, `find_best_match()` implemented
- [ ] Similarity threshold picked and tested against the 3 seed faces plus at least 2-3 "unknown" test images (confirm false-match rate is reasonable before demo day)
- [ ] `detections` table logs every check (matched or not)
- [ ] `send_alert()` stub — for Phase 1 this can just log/print "ALERT: offender detected" — real push delivery comes in Phase 4
- [ ] `POST /api/detect` endpoint: accepts an image, returns `{match_label, score, alert_sent}`
- [ ] Manual test: run all 3 seed images back through `/api/detect` and confirm each correctly matches itself
- [ ] Demo script/checklist written so this can be shown standalone if nothing else is ready in time

**Exit criteria for Phase 1:** you can upload an image of the seeded "offender," get a match with a score above threshold, and see an alert fire in the logs — repeatably, not just once by luck.

## Phase 2 — Claims data pipeline (can start in parallel once Phase 0 is done)
- [x] Claims ingested into **Azure Cosmos DB** (`guardian-db` / `insurance-data`, 15,712 docs) and read live by `services/claims_service.py` — the hot-spot map picks up new claims without a redeploy. CSV retained as an offline fallback only.
- [x] Distinct `SUBURB` values geocoded and cached — `scripts/geocode_suburbs.py` → `backend/data/suburb_geocache.json` (Nominatim, not Azure Maps; see PROJECT_CONTEXT §6)
- [x] Member claim submission endpoint — `POST /api/claims` (multipart), writes a Cosmos doc with `status: "pending"`; photo/video to Blob Storage
- [x] Discovery employee approve/deny endpoints — `POST /api/claims/<id>/approve` and `/deny`; approval joins the dataset and the hot-spot map, denial stores a reason shown to the member
- [x] Member and employee React views for both flows (see Phase 5)

## Phase 3 — Hot-spot analytics + map
- [x] `GET /api/hotspots` endpoint, filterable by crime category, item type and date range (+ `GET /api/filters` so the UI never hardcodes peril names)
- [x] Hot-spot heatmap rendering on the frontend — Leaflet + leaflet.heat at `GET /`, the app's landing screen
- [ ] Clustering pass grouped by peril category and time window (PostGIS `ST_ClusterDBSCAN` or scikit-learn DBSCAN) — current aggregation is per-suburb, which is enough for the demo but doesn't find cross-suburb clusters
- [ ] Filter by location (radius around a member's address) — currently filterable by category/type/date only

## Phase 4 — Alerts + route optimization
- [x] **Push notification delivery** — Web Push (VAPID), replacing the earlier Azure Functions / Firebase Cloud Messaging plan. No extra cloud vendor: the backend already runs as a long-lived Flask/Gunicorn process on Railway, so it signs and sends pushes itself.

  ```mermaid
  flowchart LR
      A[DeepFace / EasyOCR match] --> B["app.py: _attach_face_alert / _attach_plate_alert"]
      B --> C[alerts_service.record_detection]
      C --> D[push_service.notify_detection]
      D --> V{{"VAPID: sign request with\nour private key + claims"}}
      V --> E["pywebpush -> browser's push service\n(Chrome/Mozilla/Apple)"]
      E --> P{{"Push service verifies signature\nagainst our public key\n(from original subscription)"}}
      P --> F[Service worker on member/CPU device]
      F --> G[OS notification]
  ```

  - [x] `pywebpush` dependency + `VAPID_PUBLIC_KEY`/`VAPID_PRIVATE_KEY` config (`backend/config.py`), generated once via `backend/scripts/generate_vapid_keys.py`
  - [x] Subscriptions stored per user (`users_service.add_push_subscription` / `list_push_subscriptions`), `POST /api/push/subscribe` + `/unsubscribe` (`routes/push_routes.py`)
  - [x] `push_service.notify_detection` wired into both `_attach_face_alert` and `_attach_plate_alert` in `app.py`, reusing the alert's existing `push_audience` (member/cpu — same rule as `audience_for`)
  - [x] Frontend service worker (`frontend/public/sw.js`) + subscribe flow (`frontend/src/push.js`), with an "Enable alerts" button in the app bar for member/cpu roles
  - [ ] iOS Safari note: push only reaches an installed (Home Screen) PWA, not a regular tab — a platform limit, not something this design works around
- [x] Alert routing logic: members only get `offender` alerts; Crime Prevention Units get `offender` and `suspect` alerts — implemented in `alerts_service.audience_for`; every alert source passes through it, so wiring `/api/detect` in needs no changes to the rule
- [x] **Member route optimisation** — `GET /api/risk` + `POST /api/routes/compare`, and the "Plan a route" view: fastest vs risk-avoiding route by travel mode and departure time, via Valhalla `exclude_polygons` (not Azure Maps — no key; see PROJECT_CONTEXT §6)
- [x] **Crime Prevention Unit patrol routing** — `POST /api/patrol/plan` + the "Patrol planning" view: risk cells in the unit's area, split across vehicles by k-means, ordered into loops (nearest-neighbour + 2-opt), real road paths from Valhalla. Headline metric is risk covered per km driven
- [ ] Upgrade patrol planning from the heuristic to a real VRP solve (VROOM or OR-Tools) once shift lengths, time windows or vehicle capabilities matter
- [x] **Crime Prevention Unit alerts** — `GET /api/alerts` + the "Alerts" view, with the offender/suspect audience split enforced in `alerts_service.audience_for` so it's already correct when detections arrive
- [ ] **Live motion alerts** — `watchPosition` in the browser, evaluate the current H3 cell client-side (h3-js) so no location trail leaves the device, alert on entering an elevated cell
- [ ] Wire live/predicted alerts from Azure Functions into the risk surface alongside historical claims

## Phase 2b — User directory
- [x] Cosmos `users` container (partition key `/role`) holding all three stakeholder types; seeded with 10 members, 10 Discovery employees, 5 Crime Prevention companies via `scripts/seed_users.py`
- [x] Optional, opt-in member home location — scopes the alerts feed and seeds the route planner's origin; withdrawing consent deletes the coordinates
- [ ] **Authentication** — the `auth` block on each user document is a placeholder; nothing verifies identity yet. Biggest outstanding gap
- [ ] Containers still to add for a full audit trail: `claim_events` (`/incident_id`), `alerts` (`/recipient_id`), `patrol_runs` (`/unit_id`)

## Phase 5 — Integration, polish, demo prep
- [x] React 18 + Vite app in `frontend/` — hot-spot map, member claim submission, member claims status, employee review queue. Vite proxies `/api` to Flask (no CORS setup).
- [ ] Frontend dashboard pulling from the remaining endpoints (alerts feed, patrol routes)
- [ ] **Replace the demo identity switcher with real auth** — `members_service.py` and `session.jsx` currently trust a UI-selected identity; this is the biggest known gap
- [ ] AI-generated weekly briefing via Azure AI Foundry (nice-to-have if time allows)
- [ ] End-to-end run-through of the full demo flow, timed
- [ ] Fallback plan: if any Phase 2-4 component isn't ready, Phase 1's standalone facial recognition demo is still a complete, presentable story on its own

## Team dependency notes
- Phase 1 (facial recognition) has no hard dependency on Phases 2-4 — this is intentional, so it can be demoed even if other pieces slip.
- Phase 3 (hot-spots) depends on Phase 2's geocoding step being done first.
- Phase 4's alert routing depends on Phase 1's `match_label` output already being correct.
- See `TEAM_ROLES.md` for who owns each phase.
