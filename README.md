# Guardian Collective

Discovery GradHack 2026 — Team Ctrl+Alt+Elite. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
for the problem statement, design decisions and tech stack, and
[DEV_ROADMAP.md](DEV_ROADMAP.md) for phase status.

## Running it

Two processes. Backend first:

```bash
cd backend
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

Then the React app:

```bash
cd frontend
npm install
npm run dev              # http://localhost:5173
```

**Open <http://localhost:5173>** — Vite proxies `/api` to Flask, so there's no
CORS setup and no API base URL to configure. The Flask server also still serves
a standalone copy of the map at <http://localhost:5000> if you want the backend
demo on its own.

## The three views

Switch role with the "Viewing as" control in the top right (this stands in for
login — see *Known gaps*).

| View | Who | What |
|---|---|---|
| **Hot-spots** (`/`) | Both | Crime heatmap, filterable by category, item type and date |
| **Report an incident** (`/report`) | Member | Submit a claim: location, crime type, description, times, photo/video, door-camera permission |
| **My claims** (`/my-claims`) | Member | Status of their reports, and the **reason** when one is declined |
| **Review queue** (`/review`) | Employee | Review submissions with evidence, then approve or decline with a reason |

Approving a claim writes `status: "approved"`, which puts it into the working
dataset and onto the hot-spot map immediately. Declining requires a reason,
which is what the member is shown.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Hot-spot heatmap (standalone backend copy) |
| `GET /api/hotspots` | Per-suburb claim hot-spots. Optional filters: `peril`, `item_type`, `date_from`, `date_to` |
| `GET /api/filters` | Peril / item-type values and the dataset's date bounds |
| `GET /api/suburbs?q=` | Geocoded suburb names, for the claim form's location field |
| `GET /api/members` | Demo member/employee directory (**not** authentication) |
| `POST /api/claims` | Member submits a claim (multipart: fields + `media` files) |
| `GET /api/claims` | List submissions; filter by `status` and/or `member_id` |
| `GET /api/claims/counts` | Queue totals by status |
| `GET /api/claims/<id>` | One claim, with short-lived signed media URLs |
| `POST /api/claims/<id>/approve` | Approve — joins the dataset and the map |
| `POST /api/claims/<id>/deny` | Decline with a `denial_reason` shown to the member |
| `POST /api/claims/refresh` | Force an immediate re-read of the claims collection |
| `GET /api/health` | Liveness + live claims source + geocode coverage + media storage |

## Claim media

Photos and video go to Azure Blob Storage in a **private** container
(`claim-media`, one folder per incident). Nothing is publicly readable: the
review UI receives per-request SAS URLs that expire after
`CLAIM_MEDIA_SAS_MINUTES` (default 30), so a leaked claim document doesn't hand
out durable access to someone's incident footage.

Door-camera permission is opt-in, never pre-ticked, scoped to the single
incident, and stored with the timestamp the member gave it
(`camera_consent` / `camera_consent_at`). The review screen shows assessors
"do not pull footage for this claim" when it wasn't given.

## Known gaps

- **There is no authentication.** The role and identity are picked in the UI and
  trusted by the API (`backend/services/members_service.py`,
  `frontend/src/session.jsx`). Every endpoint taking a `member_id` or
  `employee_id` would need a real authenticated principal before this is
  anything but a demo.
- **Declines are not pushed to members.** The reason is stored and shown in "My
  claims"; real push delivery (Firebase Cloud Messaging) is Phase 4.

Filters are AND-ed; `peril` and `item_type` accept comma-separated or repeated values:

```
/api/hotspots?peril=Hijack,Armed%20Robbery&item_type=Vehicle&date_from=2025-01-01
```

## Where claims data comes from

The map reads **Azure Cosmos DB** (`guardian-db` / `insurance-data`), not the CSV,
so a claim added or approved through the app appears on the map without a
redeploy. Set `COSMOS_URI` and `COSMOS_KEY` in `.env` (see `backend/.env.example`).

Two behaviours worth knowing:

- **Claims are snapshotted in memory for `CLAIMS_CACHE_TTL_SECONDS`** (default 60).
  The filter UI re-queries on every chip click, so serving those from a snapshot
  keeps the map responsive and the RU spend flat. An expired snapshot is served
  immediately and refreshed on a background thread, so nobody waits on the
  multi-second full pull. After writing a claim, call
  `claims_service.invalidate_cache()` (or `POST /api/claims/refresh`) so it shows
  up right away.
- **A claim with a `status` field only counts once approved.** Historical records
  have no `status` at all, and absence means "already part of the dataset" — so a
  member's pending submission can't move the hot-spot map before a Discovery
  employee verifies it. Recognised approved values: `approved`, `verified`,
  `accepted`.

`CLAIMS_SOURCE` controls the source: `cosmos` (fail loudly), `csv` (offline), or
`auto` (default — prefer Cosmos, fall back to the CSV if the account is
unreachable, so a dead network on demo day degrades the map to stale-but-working
rather than broken). `GET /api/health` reports which source is actually live.

## The suburb geocode cache

`Gradhack_Insure_Data_CLEANED.csv` has suburb names but **no coordinates**, so the
2,929 distinct suburbs are geocoded once and the result committed to
`backend/data/suburb_geocache.json`. You should not need to run this — it's only
for rebuilding or extending the cache:

```bash
python backend/scripts/geocode_suburbs.py                # full run, resumable
python backend/scripts/geocode_suburbs.py --retry-misses # re-try failures only
```

`--retry-misses` broadens the query for names OpenStreetMap doesn't carry as
distinct places — private estates, agricultural holdings, cadastral portions and
directional variants (`ALBERTON NORTH` → `ALBERTON`). Those entries are stored
with `"approximate": true` and the query that matched, the API returns
`approximate` per hot-spot plus an `approximate_claims` total, and the map labels
them "Approximate location" rather than passing an estimate off as an exact pin.
The error is the offset between a sub-area and its parent suburb's centre —
immaterial at the zoom this map is read at, but don't build routing on it.

It is deliberately slow (~1 request/second, ~55 min for a full run) because
[Nominatim's usage policy](https://operations.osmfoundation.org/policies/nominatim/)
caps it there and requires a descriptive `User-Agent`. Don't parallelise it —
that gets the whole team's IP blocked. The run is safe to interrupt: it flushes
every 25 lookups and processes suburbs busiest-first.
