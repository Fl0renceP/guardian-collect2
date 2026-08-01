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
    # Cosine distance match threshold for Facenet (lower = stricter match).
    # This is the value the endpoint actually uses — app.py passes it into
    # recognition.process_incoming_face_image, overriding that module's default.
    # It must stay in the same units as the operator in that module's query (<=>).
    # 0.60 here was being compared against raw Euclidean distances of ~20-30, so
    # nothing but a byte-identical image ever matched.
    MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.30"))

    # 3b. Face capture quality gate.
    # CCTV frames are the problem case: a 30px face upscaled to the model's 160px
    # input still yields 512 numbers, they just carry no identity. That vector sits
    # roughly equidistant from everyone, so once enrolled it acts as a magnet for
    # false matches. Captures below these bars are still stored as case evidence,
    # but flagged use_for_matching = FALSE so they never become a reference.
    #
    # Short side of the detected face box, in pixels. Facenet512 consumes 160x160,
    # so below ~80 we are upscaling 2x or more and inventing detail.
    MIN_FACE_PIXELS = int(os.getenv("MIN_FACE_PIXELS", "80"))
    # Detector's own confidence in the box.
    MIN_DET_CONFIDENCE = float(os.getenv("MIN_DET_CONFIDENCE", "0.90"))
    # Variance of the Laplacian on the face crop, normalised to 160x160 so the
    # number is comparable across resolutions. Sharp faces run well above 100,
    # motion-blurred CCTV grabs fall below 40.
    MIN_BLUR_VARIANCE = float(os.getenv("MIN_BLUR_VARIANCE", "40.0"))
    # Motion blur defeats the measure above, because it is directional: a subject
    # walking past smears vertical edges while leaving horizontal ones sharp, so
    # total edge energy stays high. Measured on a 25px horizontal smear, the
    # x/y gradient ratio collapsed to 0.14 while every clean face sat at 0.39-0.87
    # — and that blurred face still scored a higher Laplacian variance (132) than
    # a perfectly usable dim photo (91), so no isotropic threshold separates them.
    MIN_BLUR_DIRECTIONAL_RATIO = float(os.getenv("MIN_BLUR_DIRECTIONAL_RATIO", "0.25"))

    # 4. Claims / hot-spot data
    # Azure Cosmos DB is the source of truth: it's writable, so claims submitted
    # and approved through the app show up on the hot-spot map without a redeploy.
    COSMOS_URI = _clean(os.getenv("COSMOS_URI"))
    COSMOS_KEY = _clean(os.getenv("COSMOS_KEY"))
    COSMOS_DATABASE = _clean(os.getenv("COSMOS_DATABASE", "guardian-db"))
    COSMOS_CONTAINER = _clean(os.getenv("COSMOS_CONTAINER", "insurance-data"))
    # User directory: members, Discovery employees and Crime Prevention Units,
    # partitioned by /role. Created on first use by users_service.
    COSMOS_USERS_CONTAINER = _clean(os.getenv("COSMOS_USERS_CONTAINER", "users"))
    USERS_CACHE_TTL_SECONDS = float(os.getenv("USERS_CACHE_TTL_SECONDS", "30"))

    

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

    # 8. Patrol routing — the mirror image of the member's avoidance.
    # A member routes *around* risk cells; a patrol is sent *through* them, so
    # the loop between two stops is nudged onto streets that need presence.
    # How far out of its way a vehicle may swing to sweep one extra area. Both
    # a proportional and an absolute allowance, whichever is larger, because
    # neither works alone: stops inside a cluster sit 1-3 km apart while the
    # risk cells are ~5 km across, so a purely proportional corridor rejects
    # every candidate on a short leg — and a purely absolute one lets a 60 km
    # leg wander for nothing.
    PATROL_VIA_CORRIDOR_RATIO = float(os.getenv("PATROL_VIA_CORRIDOR_RATIO", "1.35"))
    PATROL_VIA_MAX_DETOUR_KM = float(os.getenv("PATROL_VIA_MAX_DETOUR_KM", "4.0"))
    # And it must be worth the detour in the first place.
    PATROL_VIA_MIN_SCORE = float(os.getenv("PATROL_VIA_MIN_SCORE", "0.35"))
    # Cap per vehicle: every via point is another location in the Valhalla
    # request and another chance for the loop to stop looking like a patrol.
    PATROL_MAX_VIA_POINTS = int(os.getenv("PATROL_MAX_VIA_POINTS", "8"))
    # A shift is finite. Past this much longer than the plain fastest loop, the
    # detours get trimmed back rather than handed to a controller as a plan.
    PATROL_MAX_DETOUR_RATIO = float(os.getenv("PATROL_MAX_DETOUR_RATIO", "1.5"))
    # Concurrent Valhalla calls. Each vehicle now needs two routes (the plain
    # fastest loop and the risk-seeking one), so a six-vehicle plan is twelve
    # calls — sequential, that is a minute of staring at a spinner.
    PATROL_ROUTE_WORKERS = int(os.getenv("PATROL_ROUTE_WORKERS", "4"))
