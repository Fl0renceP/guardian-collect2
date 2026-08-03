# Guardian Collective

Discovery GradHack 2026 — Team Ctrl+Alt+Elite. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
for the problem statement, design decisions and tech stack, and
[DEV_ROADMAP.md](DEV_ROADMAP.md) for phase status.

## Running it

Two processes, two terminals. **Backend first** — Vite proxies `/api` to it, so
the frontend shows connection errors without it.

### One-time setup

```bash
# Backend: create a virtual environment and install into it
py -3.12 -m venv backend/.venv   # Windows (recommended for TensorFlow compatibility)
# python3.12 -m venv backend/.venv   # macOS/Linux
backend\.venv\Scripts\python -m pip install -r backend/requirements.txt   # Windows
# source backend/.venv/bin/activate && pip install -r backend/requirements.txt   # macOS/Linux

# Frontend
npm --prefix frontend install
```

This pulls TensorFlow (via DeepFace, for Phase 1 facial recognition), so the
first install downloads several hundred MB and takes a while. It only happens
once.

If you use [uv](https://docs.astral.sh/uv/) instead of a system Python:

```bash
uv venv backend/.venv --python 3.12
uv pip install --python backend/.venv -r backend/requirements.txt
```

### Every time

```bash
# Terminal 1 — backend on :5000
backend\.venv\Scripts\python backend/app.py

# Terminal 2 — frontend on :5173
npm --prefix frontend run dev
```

**Open <http://localhost:5173>.** Vite proxies `/api` to Flask, so there's no
CORS setup and no API base URL to configure.

Flask also serves a standalone copy of the hot-spot map at
<http://localhost:5000> if you want to demo the backend on its own.

### What needs what

| Feature | Needs |
|---|---|
| Hot-spots, claims, routing, alerts, patrol, safety score | Cosmos + Blob credentials in `.env` — that's all |
| `POST /api/v1/scan-face` | A local PostgreSQL with pgvector at `DATABASE_URL`, plus the DeepFace/TensorFlow install |

## Multi-angle face gallery import

Use this when you have additional seed images per person (different angles / lighting).
It incrementally adds photos to Azure Blob + `person_faces` and keeps existing DB
person status as the source of truth.

### Filename format

Put images in `backend/seed_photos` using:

```text
Full Name - angle label - status.ext
```

Example:

```text
Tadiwa Banda - front 1 - verified.jpeg
```

Allowed status values are `offender`, `suspect`, `verified`.

### Import commands

```bash
# Parse + quality-check without writing
python backend/scripts/import_seed_faces.py --dry-run

# Perform upload + DB insert
python backend/scripts/import_seed_faces.py

# Optional: process only first N files
python backend/scripts/import_seed_faces.py --limit 20
```

What the importer does:

- Creates a person only if the name does not already exist.
- Preserves existing `persons.status` when a person already exists.
- Uploads each file to Blob with metadata labels:
  `full_name`, `person_status`, `source`, `angle_label`, `original_filename`.
- Computes an embedding per image and stores one `person_faces` row per image.
- Applies the quality gate:
  low-quality images are still saved, but as `use_for_matching = FALSE`
  (evidence only).
- Is rerun-safe (duplicate imports are skipped).

### Verify after import

```bash
python backend/verify_seed.py
```

The report now shows per-person capture counts, matching-reference counts,
evidence-only counts, recent capture rows, and a sample of blob metadata.

Note that `backend/app.py` imports `psycopg2` and `services.recognition` (which
imports DeepFace) **at module level**, so the whole API — including the map —
won't start unless the ML stack is installed, even though nothing else uses it.
Moving that import inside the `/scan-face` handler would let the rest of the app
run without it.

## The three views

Switch role with the "Viewing as" control in the top right (this stands in for
login — see *Known gaps*).

| View | Who | What |
|---|---|---|
| **Hot-spots** (`/`) | Both | Crime heatmap, filterable by category, item type and date |
| **Plan a route** (`/route`) | Member | Fastest route vs one avoiding elevated-risk areas, by travel mode and departure time |
| **Report an incident** (`/report`) | Member | Submit a claim: location, crime type, description, times, photo/video, door-camera permission |
| **My claims** (`/my-claims`) | Member | Status of their reports, and the **reason** when one is declined |
| **Review queue** (`/review`) | Employee | Review submissions with evidence, then approve or decline with a reason |
| **Alerts** (`/alerts`) | Crime Prevention Unit | Incidents and member reports in the unit's operating area, by severity |
| **Patrol planning** (`/patrol`) | Crime Prevention Unit | One patrol loop per vehicle across the highest-risk areas for a shift |

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
| `GET /api/users` | User directory; filter with `?role=member\|employee\|cpu` |
| `GET /api/users/<id>` | One user (the `auth` block is never returned) |
| `PATCH /api/users/<id>/location` | Set or clear a member's optional home location |
| `GET /api/units` | Crime Prevention Unit directory (**not** authentication) |
| `GET /api/alerts` | Alerts for an `audience` (`member`/`cpu`), optionally scoped to a `unit_id` |
| `POST /api/patrol/plan` | Patrol loops for a unit's vehicles at a given `hour`/`weekday` |
| `GET /api/risk` | Travel-risk cells for an `hour` / `weekday` |
| `GET /api/risk/profile` | The pooled hour and day multipliers behind the surface |
| `POST /api/routes/compare` | Fastest vs risk-avoiding route for `origin`/`destination`/`mode` |
| `GET /api/health` | Liveness + live claims source + geocode coverage + media storage |

## Travel risk and safer routes

Routing uses **Valhalla** on the public FOSSGIS instance — no API key, and it
supports `exclude_polygons`, which is what turns "route around these areas" into
one parameter instead of a custom graph build. Set `VALHALLA_URL` to a
self-hosted Docker instance (with a Geofabrik `south-africa-latest.osm.pbf`
extract) if rate limits or demo-day connectivity are a concern.

### How the risk surface is built

The design is forced by what the claims data can carry, and it's worth
understanding before anyone tunes it:

- **Suburb centroids, not streets.** Claims carry a suburb name and nothing
  finer, so risk is binned to H3 cells at resolution 7 (~5 km²). Anything finer
  would invent precision that isn't in the data. Never present this as
  street-level advice.
- **Where and when are estimated separately.** Claims are desperately sparse per
  suburb (median 2; 41% of suburbs have exactly one), so a per-cell-per-hour
  estimate would be pure noise. Spatial density is smoothed across neighbouring
  cells, then scaled by a **single pooled** hour-of-day and day-of-week profile.
  That's ~30 temporal parameters fitted on thousands of claims rather than tens
  of thousands fitted on one or two each.
- **Perils are weighted for travel, not counted equally.** Hijack, attempted
  hijack and armed robbery carry full weight; vehicle theft counts at 0.3
  because it overwhelmingly happens to parked cars; home contents claims score
  zero. Without this the 6,666 vehicle-theft claims drown out the 941 violent
  ones and the surface peaks at lunchtime instead of at 20:00.
- **Scores are normalised against a fixed reference** (worst cell, worst hour,
  worst day) rather than against the current moment. Normalising per-query would
  scale every cell equally and cancel the time multiplier out entirely, leaving
  05:00 Tuesday indistinguishable from 20:00 Saturday.

The resulting hour profile peaks at **20:00 (×1.63)** and bottoms out at
**01:00 (×0.39)**, which matches the raw hijack/armed-robbery distribution.

### How a route is chosen

1. Get the fastest route.
2. Find the high-risk cells **that route actually passes through** — not the
   worst cells nationally, which would hand Valhalla polygons in Cape Town for a
   Johannesburg trip.
3. Re-route excluding them. If that severs the only corridor, or costs more than
   `ROUTE_MAX_DETOUR_RATIO` (default 1.35× the time), step down to fewer
   exclusions and try again — avoiding the worst two areas for eight minutes is
   a real option; avoiding six for an extra hour is not.
4. Only recommend the alternative if it cuts exposure by at least
   `ROUTE_MIN_RISK_REDUCTION` (default 20%). Below that, the fastest route is
   recommended and the alternative is shown for comparison only.

That last guard is deliberate and should not be loosened casually. Detours have
a real cost, and a tool that proposes one on statistical noise stops being a
safety feature and becomes an app that tells people to avoid certain
neighbourhoods.

### What this is not

The surface shows **where travel-related claims have concentrated**, not where
crime will happen. It is **not adjusted for exposure** — more claims in a suburb
partly means more Discovery members drive through it, so busy areas look worse
per trip than they may actually be. Getting policy counts per suburb would fix
this and is the single biggest accuracy improvement available.

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

## Users

All three stakeholder types live in the Cosmos `users` container, discriminated
by `role` (`member` / `employee` / `cpu`) and **partitioned by `/role`** — the
dominant read is "everyone of this role", which becomes a single-partition
query. (At serious write throughput three logical partitions would hot-spot;
you'd repartition on `/id` and add an email→id lookup. The container is tiny
here, so the query pattern wins.)

Shared identity fields (name, email, phone, status) sit at the top level;
role-specific fields live in `member_profile` / `employee_profile` /
`unit_profile`. Every document carries an empty `auth` block so authentication
has somewhere to land without a migration — **it is never returned by the API**.

Seed or reset the demo directory (10 members, 10 employees, 5 units):

```bash
python backend/scripts/seed_users.py           # idempotent upsert
python backend/scripts/seed_users.py --wipe    # also remove users not in the seed
```

`services/members_service.py` is now a thin compatibility shim re-exporting
`users_service` — prefer `users_service` in new code.

### Optional member home location

A member can record a home location, and it is **entirely optional** — the app
works without it, they just see national alerts instead of nearby ones. Four of
the ten seeded members deliberately have no location, so that path stays tested.

Three rules the code enforces:

- **`share_location` is the switch, not the presence of coordinates.** Consumers
  must call `users_service.member_home()`, which is the single place the opt-in
  is checked. Storing a point is not permission to use it.
- **Turning sharing off deletes the coordinates and address**, rather than
  hiding them. Withdrawing consent removes the data.
- **No movement history is ever stored** — only the single point the member
  places on the map.

When it's set, it scopes the alerts feed to the member's radius and becomes the
default origin in route planning.

## Crime Prevention Units

The third stakeholder. Two screens, both scoped to the unit's operating area
(a radius around its base, set in `members_service.py`).

### Alerts

**Audience routing is the rule that matters** (PROJECT_CONTEXT §2): members only
ever see `offender` matches; units see `offender` **and** `suspect`. That's
implemented in `alerts_service.audience_for` and every alert passes through it —
so the rule is already correct the day Phase 1 starts emitting detections.

Four feeds, and the UI shows which are actually live:

| Feed | Status |
|---|---|
| Recent incidents (claims dataset) | **Live** |
| Member reports awaiting review | **Live** |
| Face / plate matches | Not wired — needs Phase 1 `/api/detect` |
| Predicted risk | Not wired — needs Azure Functions |

The unwired feeds return **nothing** rather than placeholder data. An alerts
panel that invents offender sightings is worse than an empty one, and an
operator seeing no detections should know it's because the detector isn't built.
To plug one in, return alert dicts from its `_*_alerts` function in
`alerts_service.py` — the shape is documented on `_alert`.

The default window is **90 days**, not 30. Serious incidents run at roughly 19 a
month nationally, so a 30-day window around a single city is reliably empty.

### Patrol planning

This is deliberately **not** the member routing problem. A member asks "safest
way from A to B" — shortest path with a penalty. A unit asks "given N vehicles,
where should we be to cover the most risk per kilometre?" — a coverage and
allocation problem. Solving it as a shortest path would answer the wrong
question.

1. Take the highest-risk cells in the unit's area for the shift's hour.
2. Split them across vehicles geographically (k-means on cell centres), so two
   vehicles don't shadow each other.
3. Order each vehicle's stops into a loop from base and back
   (nearest-neighbour + 2-opt), and get the real road path from Valhalla.

The headline metric is **risk covered per kilometre** — the cost-efficiency
number a controller actually manages against. It falls as vehicles are added
(0.198 at one vehicle, 0.123 at six for Sandton), which is the diminishing
return you'd expect and makes the fleet-size trade-off visible.

**It's a heuristic, not an optimal solve.** Nearest-neighbour with 2-opt is
close enough on ten stops and needs no extra dependency. A real VRP solver
(VROOM or OR-Tools, both open source) is the upgrade once shift lengths, time
windows or vehicle capabilities matter.

## Known gaps

- **There is no authentication.** The role and identity are picked in the UI and
  trusted by the API (`backend/services/members_service.py`,
  `frontend/src/session.jsx`). Every endpoint taking a `member_id`,
  `employee_id` or `unit_id` would need a real authenticated principal before
  this is anything but a demo. This matters most for the CPU views — the
  offender/suspect audience split is only meaningful if the audience is actually
  verified.
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
