import io
import re
import logging
import easyocr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (loads model once)
reader = easyocr.Reader(['en'], gpu=False)

def clean_plate_text(text: str) -> str:
    """Removes spaces, hyphens, and non-alphanumeric characters."""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()

def process_incoming_plate_image(image_bytes: bytes, db_conn):
    """
    Extracts text from license plate image using OCR and queries the database.
    """
    try:
        # Convert bytes to PIL Image -> NumPy array for EasyOCR
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)

        # Run OCR detection
        results = reader.readtext(img_np)
        
        if not results:
            return {"error": "No text or license plate detected in image"}, 400

        # Collect detected strings
        detected_strings = [res[1] for res in results]
        cleaned_candidates = [clean_plate_text(text) for text in detected_strings if len(clean_plate_text(text)) >= 3]

        if not cleaned_candidates:
            return {"error": "Could not extract valid plate text"}, 400

        cursor = db_conn.cursor()
        
        # Search DB for matched plate numbers
        for candidate in cleaned_candidates:
            cursor.execute("""
                SELECT p.id, p.plate_number, p.status, p.owner_name, f.image_url
                FROM vehicle_plates p
                LEFT JOIN vehicle_plate_images f ON p.id = f.plate_id
                WHERE p.plate_number = %s;
            """, (candidate,))
            
            match = cursor.fetchone()
            if match:
                cursor.close()
                return {
                    "match_found": True,
                    "extracted_text": candidate,
                    "plate": {
                        "id": match[0],
                        "plate_number": match[1],
                        "status": match[2],
                        "owner_name": match[3],
                        "image_url": match[4]
                    }
                }

        cursor.close()
        
        # Return unmatched detection
        return {
            "match_found": False,
            "extracted_text": cleaned_candidates[0],
            "message": f"Plate '{cleaned_candidates[0]}' scanned but not flagged in database."
        }

    except Exception as e:
        logger.error(f"Error processing plate OCR: {e}")
        return {"error": f"OCR processing failed: {str(e)}"}, 500