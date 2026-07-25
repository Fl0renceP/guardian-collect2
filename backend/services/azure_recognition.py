import logging
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel
from config import Config

logger = logging.getLogger(__name__)

def get_azure_face_client():
    """Instantiates the Azure Face API client using app config."""
    if not Config.AZURE_FACE_KEY or not Config.AZURE_FACE_ENDPOINT:
        raise ValueError("Missing Azure Face API Key or Endpoint in config.")
    
    return FaceClient(
        endpoint=Config.AZURE_FACE_ENDPOINT,
        credential=AzureKeyCredential(Config.AZURE_FACE_KEY)
    )

def process_azure_face_scan(image_bytes, db_conn):
    """
    Detects face in incoming image bytes using Azure Face API,
    compares against seeded records in PostgreSQL, and returns match status.
    """
    client = get_azure_face_client()

    # 1. Detect face in incoming uploaded image
    detected_faces = client.detect(
        image_bytes,
        detection_model=FaceDetectionModel.DETECTION03,
        recognition_model=FaceRecognitionModel.RECOGNITION04,
        return_face_id=True
    )

    if not detected_faces:
        return {"error": "No face detected in the provided image by Azure Face API"}, 400

    scanned_face_id = detected_faces[0].face_id

    # 2. Fetch seeded person records & photo URLs from DB
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT p.id, p.full_name, p.status, f.image_url
        FROM persons p
        JOIN person_faces f ON p.id = f.person_id;
    """)
    candidates = cursor.fetchall()
    cursor.close()

    best_match = None
    highest_confidence = 0.0

    # 3. Compare scanned face with each candidate image using Azure Verify API
    for person_id, full_name, status, candidate_image_url in candidates:
        try:
            # Detect candidate face from stored Azure Blob URL
            candidate_faces = client.detect_from_url(
                url=candidate_image_url,
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=True
            )

            if not candidate_faces:
                continue

            candidate_face_id = candidate_faces[0].face_id

            # Execute 1:1 face verification
            verify_result = client.verify_from_single_instantly(
                face_id1=scanned_face_id,
                face_id2=candidate_face_id
            )

            # Check match result
            if verify_result.is_identical and verify_result.confidence > highest_confidence:
                highest_confidence = verify_result.confidence
                best_match = {
                    "id": person_id,
                    "full_name": full_name,
                    "status": status,
                    "confidence": round(verify_result.confidence, 4),
                    "image_url": candidate_image_url
                }
        except Exception as err:
            logger.warning(f"Error matching against candidate {full_name}: {err}")
            continue

    if best_match and highest_confidence >= 0.50:  # Recommended Azure threshold
        return {
            "match_found": True,
            "provider": "Azure AI Face API",
            "person": best_match
        }

    return {
        "match_found": False,
        "provider": "Azure AI Face API",
        "message": "No matching record found."
    }