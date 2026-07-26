"""Claim media (photos / video) upload to Azure Blob Storage.

The container is kept **private**. Claim media is evidence attached to a real
person's incident report, so blobs are never publicly readable — the review UI
gets short-lived read-only SAS URLs generated per request instead.
"""

import logging
import mimetypes
import re
import threading
import uuid
from datetime import datetime, timedelta, timezone

from config import Config

logger = logging.getLogger(__name__)

_service = None
_lock = threading.Lock()
_container_ready = False

# Kept deliberately tight — this accepts uploads from an unauthenticated demo form.
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif",
    ".mp4", ".mov", ".webm", ".m4v",
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class StorageUnavailable(RuntimeError):
    """Blob storage isn't configured or can't be reached."""


def is_configured():
    return bool(Config.AZURE_STORAGE_CONNECTION_STRING)


def _get_service():
    global _service
    if _service is not None:
        return _service

    if not is_configured():
        raise StorageUnavailable("AZURE_STORAGE_CONNECTION_STRING is not set")

    with _lock:
        if _service is not None:
            return _service
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise StorageUnavailable(f"azure-storage-blob is not installed: {exc}") from exc
        try:
            _service = BlobServiceClient.from_connection_string(
                Config.AZURE_STORAGE_CONNECTION_STRING
            )
        except Exception as exc:
            raise StorageUnavailable(f"could not connect to Blob storage: {exc}") from exc
        return _service


def _ensure_container():
    """Create the media container once per process if it isn't there yet."""
    global _container_ready
    if _container_ready:
        return
    service = _get_service()
    try:
        service.create_container(Config.CLAIM_MEDIA_CONTAINER)
        logger.info("Created blob container %s", Config.CLAIM_MEDIA_CONTAINER)
    except Exception:
        # Almost always ContainerAlreadyExists; a real permissions problem will
        # resurface on the upload call with a clearer message.
        pass
    _container_ready = True


def _safe_name(filename):
    """Reduce a user-supplied filename to something safe to use as a blob suffix."""
    name = _SAFE_NAME.sub("_", (filename or "upload").strip())[-60:]
    return name.lstrip(".") or "upload"


def extension_of(filename):
    dot = (filename or "").rfind(".")
    return (filename[dot:].lower() if dot != -1 else "")


def is_allowed(filename):
    return extension_of(filename) in ALLOWED_EXTENSIONS


def upload_claim_media(incident_id, file_storage):
    """Upload one Werkzeug FileStorage under a claim. Returns a media descriptor dict.

    The blob name is namespaced by incident so a claim's evidence stays together
    and one member's upload can never overwrite another's.
    """
    if not is_allowed(file_storage.filename):
        raise ValueError(
            f"Unsupported file type '{extension_of(file_storage.filename)}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    _ensure_container()
    service = _get_service()

    blob_name = f"{incident_id}/{uuid.uuid4().hex[:8]}-{_safe_name(file_storage.filename)}"
    content_type = (
        file_storage.mimetype
        or mimetypes.guess_type(file_storage.filename or "")[0]
        or "application/octet-stream"
    )

    try:
        from azure.storage.blob import ContentSettings

        blob = service.get_blob_client(Config.CLAIM_MEDIA_CONTAINER, blob_name)
        blob.upload_blob(
            file_storage.stream,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    except Exception as exc:
        raise StorageUnavailable(f"upload failed: {exc}") from exc

    return {
        "blob": blob_name,
        "filename": file_storage.filename,
        "content_type": content_type,
        "kind": "video" if content_type.startswith("video/") else "image",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def read_url(blob_name, minutes=None):
    """A short-lived read-only SAS URL for one blob.

    Generated per request rather than stored on the claim, so a leaked claim
    document doesn't hand out durable access to someone's incident footage.
    """
    service = _get_service()
    minutes = minutes or Config.CLAIM_MEDIA_SAS_MINUTES

    try:
        from azure.storage.blob import BlobSasPermissions, generate_blob_sas

        token = generate_blob_sas(
            account_name=service.account_name,
            container_name=Config.CLAIM_MEDIA_CONTAINER,
            blob_name=blob_name,
            account_key=service.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
        )
    except Exception as exc:
        raise StorageUnavailable(f"could not sign media URL: {exc}") from exc

    blob = service.get_blob_client(Config.CLAIM_MEDIA_CONTAINER, blob_name)
    return f"{blob.url}?{token}"


def with_read_urls(media):
    """Attach a fresh SAS `url` to each media descriptor on a claim."""
    signed = []
    for item in media or []:
        entry = dict(item)
        try:
            entry["url"] = read_url(item["blob"])
        except StorageUnavailable as exc:
            logger.warning("Could not sign %s: %s", item.get("blob"), exc)
            entry["url"] = None
        signed.append(entry)
    return signed


def storage_status():
    return {
        "configured": is_configured(),
        "container": Config.CLAIM_MEDIA_CONTAINER,
    }
