# Team Roles — Ctrl+Alt+Elite

Discovery GradHack 2026, "Guardian Collective." Saved in-repo so GitHub Copilot / Claude Code have context on who owns what — reference this before generating code for a given area so suggestions match the right owner's part of the stack.

## Tinashe — Facial Recognition & Core Backend/Infra (Priority Owner)
- **Owns:** Phase 1 in full — face detection, embedding generation, matching logic, seed data, the `faces`/`detections` schema and tables
- Also owns the underlying Azure infrastructure: App Service, PostgreSQL (with pgvector/PostGIS), Blob Storage, and the GitHub Actions → Azure deployment pipeline
- **Deliverable:** a working, repeatable `/api/detect` demo (Phase 1 exit criteria in `DEV_ROADMAP.md`)
- **Depends on:** nothing — this is the first thing built
- **Feeds into:** Florence's alert routing (Phase 4) and the frontend's detection/alerts feed (Tadiwa)

## Victoria Armstrong — Claims Data Pipeline & Hot-spot Analytics
- **Owns:** Phase 2 (claims ingestion, cleaning, geocoding) and Phase 3 (clustering, hot-spot endpoint)
- Cleans and loads the claims CSV into the `claims` table, normalizes `CLAIM_AMOUNT`, geocodes distinct suburbs via Azure Maps into `claims_geocoded`
- Builds the DBSCAN/PostGIS clustering pass and the `GET /api/hotspots` endpoint (filterable by location + crime category)
- **Depends on:** Phase 0 infra being up
- **Feeds into:** Tadiwa's hot-spot map rendering, Florence's route optimization

## Tadiwa Banda — Frontend & Dashboard
- **Owns:** `frontend/` — React 18 + Vite build of the member and internal dashboards
- Builds out the "Guardian Collective" dashboard: alerts feed, hot-spot map (Azure Maps component), patrol route display, claims status view
- Coordinates with all other owners since the frontend consumes every backend endpoint
- **Depends on:** early API contracts from Tinashe (detection results shape) and Victoria (hot-spot data shape) — agree on response JSON shapes early, even before the endpoints are fully built, so frontend work isn't blocked

## Florence Phiri — Alerts & Route Optimization
- **Owns:** Phase 4 — replacing the alert stub with real Azure Functions + Firebase Cloud Messaging delivery, and Azure Maps Route API integration for patrol/response routes
- Implements the alert routing rule: members only see `offender` alerts, Crime Prevention Units see `offender` and `suspect` alerts
- **Depends on:** Tinashe's `match_label` output (Phase 1) and Victoria's hot-spot locations (Phase 3)

## Keziah Solomons — Claims Workflow & Integration/Testing
- **Owns:** the member-facing claim submission flow and the Discovery-employee approve/deny flow (part of Phase 2), plus end-to-end integration testing across all phases going into Phase 5
- Builds the "queryable form + employee verification" piece from the functional requirements
- Owns demo-day run-through: timing the full flow, identifying weak points, keeping the Phase 1 standalone fallback demo ready in case later phases slip
- **Depends on:** Victoria's claims schema being finalized early

## Cross-cutting agreements
- Agree on API response shapes (JSON field names/types) before building both sides of an integration — don't let frontend and backend drift apart on naming.
- Anyone touching the `faces` or `detections` schema should check with Tinashe first, since Phase 1 depends on it staying stable.
- Daily sync recommended given the tight timeline — even 10 minutes to flag blockers before they cost a full phase.
