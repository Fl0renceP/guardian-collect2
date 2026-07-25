# Guardian Collective

Discovery GradHack 2026 — Team Ctrl+Alt+Elite. See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)
for the problem statement, design decisions and tech stack, and
[DEV_ROADMAP.md](DEV_ROADMAP.md) for phase status.

## Running the backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Then open <http://localhost:5000> — the crime hot-spot map is the landing screen.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /` | Hot-spot heatmap (the app's landing screen) |
| `GET /api/hotspots` | Per-suburb claim hot-spots. Optional filters: `peril`, `item_type`, `date_from`, `date_to` |
| `GET /api/filters` | Peril / item-type values and the dataset's date bounds |
| `POST /api/claims/refresh` | Force an immediate re-read of the claims collection |
| `GET /api/health` | Liveness + live claims source + geocode coverage |

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
