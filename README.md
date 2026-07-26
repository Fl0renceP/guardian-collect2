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
| `GET /api/v1/alerts` | Recent alert feed filtered by audience/channel |
| `GET /api/v1/alerts/stream` | Live alert feed via Server-Sent Events (SSE) |

Filters are AND-ed; `peril` and `item_type` accept comma-separated or repeated values:

```
/api/hotspots?peril=Hijack,Armed%20Robbery&item_type=Vehicle&date_from=2025-01-01
```

## Manual Scan Alerts And Demo Push Policy

Face (`POST /api/v1/scan-face`) and plate (`POST /api/v1/scan-plate`) scans now
emit a normalized `alert_event` plus `alerts[]` in the response when a known
status is detected (`verified`, `suspect`, `offender`).

Delivery policy is audience-based:

- `member`
  - alerts: `offender` only
  - push: `offender` only
- `crime_prevention`
  - alerts: `suspect` and `offender`
  - push: disabled

This is implemented without creating external accounts/resources. Push is demo
mode by default and can be toggled with environment flags.

### Alert feed endpoints

- `GET /api/v1/alerts?audience=member|crime_prevention&channel=alerts|push&limit=50`
- `GET /api/v1/alerts/stream?audience=member|crime_prevention&channel=alerts|push`

### Push env flags

- `PUSH_NOTIFICATIONS_ENABLED` (default `false`)
- `PUSH_NOTIFICATIONS_DRY_RUN` (default `true`)
- `PUSH_MIN_LEVEL` (`low` | `medium` | `high`, default `medium`)

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

It is deliberately slow (~1 request/second, ~55 min for a full run) because
[Nominatim's usage policy](https://operations.osmfoundation.org/policies/nominatim/)
caps it there and requires a descriptive `User-Agent`. Don't parallelise it —
that gets the whole team's IP blocked. The run is safe to interrupt: it flushes
every 25 lookups and processes suburbs busiest-first.
