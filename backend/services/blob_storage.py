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


def _blob_name_from_url(url):
    parsed = urlparse(url or "")
    path = parsed.path.lstrip("/")
    if not path:
        return None
    parts = path.split("/", 1)
    if len(parts) != 2:
        return None
    container_name, blob_name = parts
    if container_name != CONTAINER_NAME:
        return None
    return blob_name

class BlobStorageService:
    def __init__(self):
        if not AZURE_STORAGE_CONNECTION_STRING:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING is missing in .env")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )
        self.container_client = self._get_or_create_container()

    def _get_or_create_container(self):
        """Ensures the 'face-db2' container exists with public blob access."""
        container_client = self.blob_service_client.get_container_client(CONTAINER_NAME)
        if not container_client.exists():
            # Create container with public read access for individual blobs
            container_client.create_container(public_access="blob")
            print(f"✅ Container '{CONTAINER_NAME}' created successfully.")

        # Keep access policy explicit for already-existing containers as well.
        try:
            container_client.set_container_access_policy(public_access="blob")
        except Exception as exc:
            print(f"⚠️ Unable to enforce public blob access policy on '{CONTAINER_NAME}': {exc}")

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