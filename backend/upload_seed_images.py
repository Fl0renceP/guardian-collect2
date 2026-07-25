import os
from services.blob_storage import BlobStorageService

def upload_test_images():
    blob_service = BlobStorageService()
    
    # Path to local seed folder
    seed_folder = os.path.join(os.path.dirname(__file__), "seed_photos")
    
    test_files = {
        "seed_offender.jpg": os.path.join(seed_folder, "offender.jpeg"),
        "seed_suspect.jpg": os.path.join(seed_folder, "suspect.jpeg"),
        "seed_verified.jpg": os.path.join(seed_folder, "verified.jpeg")
    }

    uploaded_urls = {}

    print("Uploading test images to Azure Blob Storage ('face-db2' container)...")
    for blob_filename, local_path in test_files.items():
        if not os.path.exists(local_path):
            print(f"⚠️ Missing local image at '{local_path}'. Skipping upload.")
            continue

        url = blob_service.upload_image(local_path, filename=blob_filename)
        uploaded_urls[blob_filename] = url
        print(f"✅ Uploaded {blob_filename} -> {url}")

    return uploaded_urls

if __name__ == "__main__":
    upload_test_images()