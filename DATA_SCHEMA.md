# Data Schema

This is the working schema for the three core datasets in Guardian Collective. The team's original face-db proposal (`img_id, image, label`) is kept conceptually but split into an **identity registry** and a **detection log**, so the "who is this person" table stays small and deliberate while every camera check still gets recorded (see `PROJECT_CONTEXT.md` Section 5 for why).

## 1. `faces` — identity registry

The known offenders/suspects/verified people we match against. Small, curated table.

| Column | Type | Notes |
|---|---|---|
| `face_id` | UUID / serial PK | |
| `label` | enum: `offender`, `suspect`, `verified`, `pending_review` | `pending_review` is new — used instead of auto-defaulting unknown faces to `verified` |
| `embedding` | vector(128) | Face embedding (pgvector column), used for cosine similarity search |
| `image_url` | text | Reference into Azure Blob Storage — don't store raw image bytes in Postgres |
| `source` | text | e.g. `"seed_data"`, `"SAPS_reference"`, `"camera_capture"` |
| `linked_incident_id` | FK → claims.incident, nullable | Optional link if this face was captured in connection with a specific claim |
| `created_at` | timestamp | |
| `updated_at` | timestamp | |
| `notes` | text, nullable | Free text, e.g. reviewer comments for `pending_review` entries |

## 2. `detections` — event log

Every single face check against a camera frame, whether matched or not. This is the "continuous dataset" Discovery's spec wants to keep building.

| Column | Type | Notes |
|---|---|---|
| `detection_id` | UUID / serial PK | |
| `camera_id` | text | Which camera/stream (or `"demo_upload"` for hackathon file-upload simulation) |
| `matched_face_id` | FK → faces.face_id, nullable | Null if no match found |
| `match_label` | enum: `offender`, `suspect`, `verified`, `no_match` | Denormalized copy of the matched face's label at detection time, for fast querying |
| `match_score` | float | Similarity score used for the match decision |
| `alert_sent` | boolean | Whether this detection triggered a push alert |
| `location_lat` / `location_lng` | float, nullable | Camera's location if known |
| `detected_at` | timestamp | |

## 3. `claims` — Discovery Insure claims data

As provided, unmodified column set:

| Column | Notes |
|---|---|
| `Incident` | Primary key / claim reference (e.g. `INC-001`) |
| `PERIL` | High-level peril category (e.g. `Theft`) |
| `SUBURB` | Free text suburb name — **no coordinates provided**, needs geocoding (Azure Maps) before any spatial/hot-spot work |
| `ITEM_TYPE` | e.g. `Contents`, `Vehicle` |
| `VEHICLE_MAKE` / `VEHICLE_MODEL` / `VEHICLE_YEAR` | Null for non-vehicle claims |
| `INCIDENT_DATE_TIME` | Date (and possibly time) of the incident |
| `CLAIM_AMOUNT` | Numeric — watch for comma-as-decimal-separator in the raw file (e.g. `391572,28`), normalize on ingest |
| `ITEM_CATEGORY` | e.g. `Home contents - Theft`, `Motor Vehicle - Theft` |
| `ITEM_PERIL_DESCR` | Descriptive peril text |

**Derived table (recommended, not in the original CSV):** `claims_geocoded` — `Incident` FK, `latitude`, `longitude`, populated by a one-time geocoding pass over distinct `SUBURB` values (cache the suburb → coordinate lookup, don't re-geocode per row).

## 4. Notes on data quality from the sample rows

- `CLAIM_AMOUNT` uses a comma as the decimal separator in the sample data (`391572,28`) — convert to a proper float/decimal on ingest, don't trust locale-based parsing blindly.
- Vehicle fields are `NULL` for contents claims — expected, not a data error.
- `SUBURB` values will need normalization (casing, whitespace) before using as a geocoding lookup key, since inconsistent formatting will produce duplicate geocode entries for the same place.
