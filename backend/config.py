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

    # 2. Azure Blob Storage (Still used to save image files)
    BLOB_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    BLOB_CONTAINER_NAME = "face-images"

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
