import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 1. PostgreSQL Database with pgvector enabled
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/guardian_db")
    
    # 2. Azure Blob Storage (Still used to save image files)
    BLOB_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    BLOB_CONTAINER_NAME = "face-images"
    
    # 3. DeepFace Model Configuration
    # Options: "Facenet" (128-dim), "Facenet512" (512-dim), "ArcFace" (512-dim), "VGG-Face"
    FACE_MODEL = "Facenet"
    # Cosine distance match threshold for Facenet (lower = stricter match)
    MATCH_THRESHOLD = 0.40