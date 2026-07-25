import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root (this file lives in backend/)
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Load explicitly rather than relying on cwd — the app runs from the repo root
# locally and from backend/ under gunicorn. backend/.env wins where both set a key.
load_dotenv(REPO_ROOT / ".env")
load_dotenv(BASE_DIR / ".env", override=True)


def _clean(value):
    """Env values in this project are sometimes quoted and padded — strip both."""
    return value.strip().strip('"').strip("'") if isinstance(value, str) else value


class Config:
    # 1. PostgreSQL Database with pgvector enabled
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/guardian_db")

    # 2. Azure Blob Storage
    AZURE_STORAGE_CONNECTION_STRING = _clean(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
    BLOB_CONNECTION_STRING = AZURE_STORAGE_CONNECTION_STRING  # legacy alias
    BLOB_CONTAINER_NAME = "face-images"

    # Claim evidence (photos / video). Private container — the review UI reads
    # it through short-lived SAS URLs, never public links.
    CLAIM_MEDIA_CONTAINER = _clean(os.getenv("CLAIM_MEDIA_CONTAINER", "claim-media"))
    CLAIM_MEDIA_SAS_MINUTES = int(os.getenv("CLAIM_MEDIA_SAS_MINUTES", "30"))
    # Per-file upload ceiling; Flask enforces the request total separately.
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "64")) * 1024 * 1024

    # 3. DeepFace Model Configuration
    # Options: "Facenet" (128-dim), "Facenet512" (512-dim), "ArcFace" (512-dim), "VGG-Face"
    FACE_MODEL = "Facenet512"
    # Cosine distance match threshold for Facenet (lower = stricter match)
    MATCH_THRESHOLD = 0.60

    # 4. Claims / hot-spot data
    # Azure Cosmos DB is the source of truth: it's writable, so claims submitted
    # and approved through the app show up on the hot-spot map without a redeploy.
    COSMOS_URI = _clean(os.getenv("COSMOS_URI"))
    COSMOS_KEY = _clean(os.getenv("COSMOS_KEY"))
    COSMOS_DATABASE = _clean(os.getenv("COSMOS_DATABASE", "guardian-db"))
    COSMOS_CONTAINER = _clean(os.getenv("COSMOS_CONTAINER", "insurance-data"))

    # "cosmos" | "csv" | "auto". "auto" prefers Cosmos and falls back to the CSV
    # if the account is unreachable — the demo still runs on a dead network.
    CLAIMS_SOURCE = _clean(os.getenv("CLAIMS_SOURCE", "auto")).lower()

    # How long an in-memory snapshot of the claims collection is reused before
    # being re-fetched. Filters re-query on every chip click, so serving those
    # from memory keeps the map responsive and the RU spend flat; the trade-off
    # is that an externally-added claim takes up to this long to appear.
    # Writes made through this app invalidate the cache immediately.
    CLAIMS_CACHE_TTL_SECONDS = float(os.getenv("CLAIMS_CACHE_TTL_SECONDS", "60"))

    # Fallback CSV. Semicolon-delimited with comma decimal separators (SA/Excel locale).
    CLAIMS_CSV_PATH = Path(
        os.getenv("CLAIMS_CSV_PATH", REPO_ROOT / "Gradhack_Insure_Data_CLEANED.csv")
    )
    CLAIMS_CSV_DELIMITER = ";"

    # 5. Geocoding (OpenStreetMap / Nominatim)
    # Suburb -> lat/lng cache built once by scripts/geocode_suburbs.py.
    GEOCACHE_PATH = Path(os.getenv("GEOCACHE_PATH", BASE_DIR / "data" / "suburb_geocache.json"))
    NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
    # Nominatim's usage policy REQUIRES a descriptive User-Agent with contact details
    # and a hard limit of 1 request/second. Do not remove either.
    NOMINATIM_USER_AGENT = os.getenv(
        "NOMINATIM_USER_AGENT",
        "GuardianCollective-GradHack2026/1.0 (+https://github.com/ctrl-alt-elite)",
    )
    NOMINATIM_DELAY_SECONDS = float(os.getenv("NOMINATIM_DELAY_SECONDS", "1.1"))

    # Rough bounding box for South Africa, used to reject nonsense geocode hits
    # (a suburb name that also exists in another country).
    SA_BOUNDS = {"min_lat": -35.5, "max_lat": -21.5, "min_lng": 15.5, "max_lng": 33.5}

    # 6. Travel-risk surface
    # Resolution 7 is ~5 km across. Claims are located to suburb centroids, so
    # anything finer would invent precision the data doesn't have.
    RISK_H3_RESOLUTION = int(os.getenv("RISK_H3_RESOLUTION", "7"))
    # Rings of neighbouring cells a claim's weight bleeds into (halving each
    # ring), turning a centroid into a neighbourhood-sized footprint.
    RISK_SMOOTHING_RINGS = int(os.getenv("RISK_SMOOTHING_RINGS", "2"))

    # 7. Routing (Valhalla / OpenStreetMap)
    # The public FOSSGIS instance needs no key. Self-host via Docker with a
    # Geofabrik South Africa extract if rate limits or demo connectivity bite.
    VALHALLA_URL = _clean(os.getenv("VALHALLA_URL", "https://valhalla1.openstreetmap.de/route"))
    VALHALLA_TIMEOUT_SECONDS = float(os.getenv("VALHALLA_TIMEOUT_SECONDS", "30"))

    # A cell must score at least this to be worth routing around.
    ROUTE_AVOID_THRESHOLD = float(os.getenv("ROUTE_AVOID_THRESHOLD", "0.55"))
    # Hard cap on excluded areas — hand Valhalla too many and it either detours
    # absurdly or fails to find any route at all.
    ROUTE_MAX_AVOID_POLYGONS = int(os.getenv("ROUTE_MAX_AVOID_POLYGONS", "12"))
    # Past this much extra travel time, the safer route is shown but not advised.
    ROUTE_MAX_DETOUR_RATIO = float(os.getenv("ROUTE_MAX_DETOUR_RATIO", "1.35"))
    # And below this relative risk improvement it isn't worth suggesting at all —
    # detours have a real cost and shouldn't be proposed on noise.
    ROUTE_MIN_RISK_REDUCTION = float(os.getenv("ROUTE_MIN_RISK_REDUCTION", "0.20"))
    # Score at or above which a sampled point counts as "high risk".
    ROUTE_HIGH_RISK_THRESHOLD = float(os.getenv("ROUTE_HIGH_RISK_THRESHOLD", "0.55"))
