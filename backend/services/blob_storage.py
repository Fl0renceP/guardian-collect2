import os
import re
import uuid
from azure.storage.blob import BlobServiceClient, ContentSettings
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "face-db2"


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
            container_client.create_container()
            print(f"✅ Container '{CONTAINER_NAME}' created successfully.")
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