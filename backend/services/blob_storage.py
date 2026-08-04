import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from azure.storage.blob import BlobServiceClient, ContentSettings, BlobSasPermissions, generate_blob_sas
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "face-db2"
DEFAULT_READ_SAS_MINUTES = int(os.getenv("FACE_IMAGE_SAS_MINUTES", "30"))


def _content_type_for(filename):
    ext = os.path.splitext((filename or "").lower())[1]
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")


def _sanitize_metadata(metadata):
    """Azure metadata keys must be ASCII and identifier-like; values must be strings."""
    if not metadata:
        return None

    cleaned = {}
    for key, value in metadata.items():
        if value is None:
            continue
        safe_key = re.sub(r"[^a-z0-9_]", "_", str(key).strip().lower())
        if not safe_key:
            continue
        safe_value = str(value).strip()
        if not safe_value:
            continue
        cleaned[safe_key] = safe_value
    return cleaned or None


def _split_blob_url(url):
    """(container, blob_name) for a blob URL, or (None, None)."""
    parsed = urlparse(url or "")
    path = parsed.path.lstrip("/")
    if not path:
        return None, None
    parts = path.split("/", 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def _blob_name_from_url(url):
    """Blob name, but only for the face container.

    The container check is a safety guard, not a formality: delete_stored_url
    relies on it so a row pointing somewhere unexpected cannot make the purge
    delete an arbitrary blob.
    """
    container_name, blob_name = _split_blob_url(url)
    if container_name != CONTAINER_NAME:
        return None
    return blob_name


def sign_url(url, minutes=None):
    """Short-lived read link for any blob in this storage account.

    Module-level and container-agnostic because plate images live in a
    different container from faces, and both are private now — anything handed
    to a client has to be signed or it simply will not load.

    Falls back to the original URL when it cannot be signed, so a
    misconfiguration degrades to a broken image rather than a failed scan.
    """
    if not url or not AZURE_STORAGE_CONNECTION_STRING:
        return url
    container_name, blob_name = _split_blob_url(url)
    if not container_name or not blob_name:
        return url
    try:
        client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        token = generate_blob_sas(
            account_name=client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes or DEFAULT_READ_SAS_MINUTES),
        )
        return f"{client.get_blob_client(container_name, blob_name).url}?{token}"
    except Exception:
        return url

class BlobStorageService:
    def __init__(self):
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self._get_or_create_container()

    def _get_or_create_container(self):
        """Ensure the face container exists and is PRIVATE.

        These blobs are faces — special personal information under POPIA. With
        public access the URL is the only thing standing between a stored face
        and the open internet, and blob URLs leak: they land in logs, in
        database dumps, in screenshots, in a shared browser tab. Anyone holding
        one could read the image forever, with no login and no audit trail.

        Reads go through short-lived SAS links instead (see read_url), so
        access is granted per request and expires.

        The policy is re-asserted on every startup rather than only at creation
        because this container was previously created public: a fresh
        deployment would be private, but the existing one stays exposed until
        something actively closes it.
        """
        container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            container_client.create_container()  # private: no public_access
            print(f"Container '{CONTAINER_NAME}' created (private).")
            return container_client

        try:
            props = container_client.get_container_properties()
            current = props.get("public_access")
            if current:
                container_client.set_container_access_policy(signed_identifiers={}, public_access=None)
                print(
                    f"SECURITY: container '{CONTAINER_NAME}' was public "
                    f"('{current}') — access has been revoked."
                )
        except Exception as exc:
            # Loud, because failing to close this is a data-exposure problem,
            # not a cosmetic one.
            print(f"WARNING: could not verify/revoke public access on '{CONTAINER_NAME}': {exc}")

        return container_client

        return container_client

    def upload_image(self, file_path_or_bytes, filename=None, metadata=None):
        """
        Uploads a local image file or byte stream to Azure Blob Storage.
        Returns the public URL of the uploaded image.
        """
        if filename is None:
            filename = f"face_{uuid.uuid4().hex}.jpg"

        blob_client = self.container_client.get_blob_client(filename)

        # Content settings ensure browser opens image rather than downloading
        content_settings = ContentSettings(content_type=_content_type_for(filename))
        cleaned_metadata = _sanitize_metadata(metadata)

        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=content_settings,
                    metadata=cleaned_metadata,
                )
        else:
            blob_client.upload_blob(
                file_path_or_bytes,
                overwrite=True,
                content_settings=content_settings,
                metadata=cleaned_metadata,
            )

        return blob_client.url

    def read_url(self, blob_name, minutes=None):
        """Return a short-lived read-only SAS URL for one face blob."""
        minutes = minutes or DEFAULT_READ_SAS_MINUTES

        token = generate_blob_sas(
            account_name=self.blob_service_client.account_name,
            container_name=CONTAINER_NAME,
            blob_name=blob_name,
            account_key=self.blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(minutes=minutes),
        )
        blob = self.container_client.get_blob_client(blob_name)
        return f"{blob.url}?{token}"

    def sign_stored_url(self, url, minutes=None):
        """Convert a stored blob URL into a short-lived readable SAS URL."""
        blob_name = _blob_name_from_url(url)
        if not blob_name:
            return url
        return self.read_url(blob_name, minutes=minutes)

    def delete_stored_url(self, url):
        """Delete the blob a stored URL points at. Returns True if it went.

        Used by the retention purge: dropping the person_faces row without
        deleting the image would leave the face itself in storage, which is the
        thing the retention window exists to remove.

        A URL outside this container is ignored rather than followed — the
        purge must not be able to delete arbitrary blobs because a bad row
        pointed somewhere unexpected.
        """
        blob_name = _blob_name_from_url(url)
        if not blob_name:
            return False
        try:
            self.container_client.get_blob_client(blob_name).delete_blob()
            return True
        except Exception:
            # Already gone, or never existed. Either way the desired end state
            # holds, so the caller should not treat it as a failure.
            return False