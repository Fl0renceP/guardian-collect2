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
- [ ] Claims CSV cleaned and ingested into `claims` table (normalize `CLAIM_AMOUNT` decimal format, handle `NULL` vehicle fields)
- [ ] Distinct `SUBURB` values geocoded via Azure Maps, cached into `claims_geocoded`
- [ ] Member claim submission endpoint (basic form → pending row)
- [ ] Discovery employee approve/deny endpoint (approved claims join the working dataset)

## Phase 3 — Hot-spot analytics + map
- [ ] Clustering pass over `claims_geocoded` (PostGIS `ST_ClusterDBSCAN` or Python/scikit-learn DBSCAN) grouped by peril category and time window
- [ ] `GET /api/hotspots` endpoint, filterable by location and crime category
- [ ] Azure Maps rendering hot-spots on the frontend

## Phase 4 — Alerts + route optimization
- [ ] Replace the Phase 1 `send_alert()` stub with real delivery: Azure Function trigger → Firebase Cloud Messaging
- [ ] Alert routing logic: members only get `offender` alerts; Crime Prevention Units get `offender` and `suspect` alerts
- [ ] Azure Maps Route API wired up for patrol/response route suggestions between active hot-spots or toward an alert location

## Phase 5 — Integration, polish, demo prep
- [ ] Frontend dashboard pulling from all endpoints (hotspot map, alerts feed, patrol routes, claims status)
- [ ] AI-generated weekly briefing via Azure AI Foundry (nice-to-have if time allows)
- [ ] End-to-end run-through of the full demo flow, timed
- [ ] Fallback plan: if any Phase 2-4 component isn't ready, Phase 1's standalone facial recognition demo is still a complete, presentable story on its own

## Team dependency notes
- Phase 1 (facial recognition) has no hard dependency on Phases 2-4 — this is intentional, so it can be demoed even if other pieces slip.
- Phase 3 (hot-spots) depends on Phase 2's geocoding step being done first.
- Phase 4's alert routing depends on Phase 1's `match_label` output already being correct.
- See `TEAM_ROLES.md` for who owns each phase.
