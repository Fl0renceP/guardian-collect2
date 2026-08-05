"""Storage for behavioural review clips.

A clip is the few seconds of footage around a flag, so a reviewer can see what
the explanations are describing rather than take them on trust.

IT IS THE MOST SENSITIVE THING THIS FEATURE HANDLES. Claim media is evidence a
member chose to submit about their own incident. A behavioural clip is video of
whoever happened to walk past a camera — frequently someone who has done nothing
and will never know they were recorded. So it follows the claim-media rules and
then some:

  * private container, never public
  * short-lived read-only SAS generated per request
  * **the blob NAME is stored, never a signed URL** — a persisted signed URL is
    a public link with extra steps, and it outlives the page that rendered it
  * a hard retention limit, enforced here rather than left to a policy nobody
    runs (`BEHAVIOUR_CLIP_RETENTION_DAYS`)

If blob storage is unreachable the clip simply is not attached. The review still
works — the explanations are the record, and the module keeps its own local copy.
"""

import logging
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

from config import Config

logger = logging.getLogger(__name__)

_service = None
_lock = threading.Lock()
_container_ready = False

ALLOWED_EXTENSIONS = {".mp4", ".webm", ".m4v"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
MAX_CLIP_BYTES = 64 * 1024 * 1024


class ClipStorageUnavailable(RuntimeError):
    """Blob storage isn't configured or can't be reached."""


def is_configured():
    return bool(Config.AZURE_STORAGE_CONNECTION_STRING)


def _get_service():
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                if not is_configured():
                    raise ClipStorageUnavailable(
                        "AZURE_STORAGE_CONNECTION_STRING is not set."
                    )
                try:
                    from azure.storage.blob import BlobServiceClient

                    _service = BlobServiceClient.from_connection_string(
                        Config.AZURE_STORAGE_CONNECTION_STRING
                    )
                except Exception as exc:
                    raise ClipStorageUnavailable(str(exc)) from exc
    return _service


def _ensure_container():
    """Create the container if needed. Private — no public access argument is
    passed, and none should ever be added."""
    global _container_ready
    if _container_ready:
        return
    service = _get_service()
    try:
        service.create_container(Config.BEHAVIOUR_CLIP_CONTAINER)
        logger.info("Created private blob container %s", Config.BEHAVIOUR_CLIP_CONTAINER)
    except Exception as exc:
        if "ContainerAlreadyExists" not in str(exc):
            logger.debug("Container check for clips: %s", exc)
    _container_ready = True


def upload_clip(review_id, data, filename="clip.mp4"):
    """Store one clip. Returns the blob NAME, which is what gets persisted."""
    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ".mp4"
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Clip must be one of {', '.join(sorted(ALLOWED_EXTENSIONS))}.")
    if not data:
        raise ValueError("Clip is empty.")
    if len(data) > MAX_CLIP_BYTES:
        raise ValueError(f"Clip exceeds {MAX_CLIP_BYTES // (1024 * 1024)}MB.")

    _ensure_container()
    safe_review = _SAFE_NAME.sub("-", str(review_id))[:40]
    blob_name = f"{safe_review}/{uuid.uuid4().hex}{extension}"

    try:
        from azure.storage.blob import ContentSettings

        blob = _get_service().get_blob_client(Config.BEHAVIOUR_CLIP_CONTAINER, blob_name)
        blob.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type="video/mp4"),
        )
    except ClipStorageUnavailable:
        raise
    except Exception as exc:
        raise ClipStorageUnavailable(f"Clip upload failed: {exc}") from exc

    logger.info("Stored clip %s (%d bytes) for %s", blob_name, len(data), review_id)
    return blob_name


def read_url(blob_name, minutes=None):
    """A short-lived read-only SAS URL for one clip.

    Generated per request and never stored. If this returned something that
    could be saved on the review, the container's privacy would be decorative.
    """
    if not blob_name:
        return None

    minutes = minutes or Config.BEHAVIOUR_CLIP_SAS_MINUTES
    service = _get_service()

    try:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        token = generate_blob_sas(
            account_name=service.account_name,
            container_name=Config.BEHAVIOUR_CLIP_CONTAINER,
            blob_name=blob_name,
            account_key=service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
        )
    except Exception as exc:
        logger.warning("Could not sign clip %s: %s", blob_name, exc)
        return None

    blob = service.get_blob_client(Config.BEHAVIOUR_CLIP_CONTAINER, blob_name)
    return f"{blob.url}?{token}"


def safe_read_url(blob_name, minutes=None):
    """read_url that never raises — a missing clip must not break a review card."""
    if not blob_name:
        return None
    try:
        return read_url(blob_name, minutes)
    except Exception:
        logger.warning("Clip URL unavailable for %s", blob_name, exc_info=True)
        return None


def purge_expired(older_than_days=None):
    """Delete clips past the retention limit.

    Enforced in code rather than left to a lifecycle policy somebody has to
    remember to configure. Returns how many were removed.
    """
    days = older_than_days or Config.BEHAVIOUR_CLIP_RETENTION_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    try:
        _ensure_container()
        container = _get_service().get_container_client(Config.BEHAVIOUR_CLIP_CONTAINER)
        removed = 0
        for blob in container.list_blobs():
            if blob.last_modified and blob.last_modified < cutoff:
                container.delete_blob(blob.name)
                removed += 1
        if removed:
            logger.info("Purged %d clip(s) older than %d days", removed, days)
        return removed
    except Exception:
        logger.warning("Clip purge failed", exc_info=True)
        return 0


def storage_status():
    return {
        "configured": is_configured(),
        "container": Config.BEHAVIOUR_CLIP_CONTAINER,
        "sas_minutes": Config.BEHAVIOUR_CLIP_SAS_MINUTES,
        "retention_days": Config.BEHAVIOUR_CLIP_RETENTION_DAYS,
    }
