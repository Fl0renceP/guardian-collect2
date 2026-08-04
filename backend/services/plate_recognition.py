import io
import re
import logging
import numpy as np
from PIL import Image

from services import plate_ocr

logger = logging.getLogger(__name__)

# EasyOCR is imported lazily. Importing it at module scope took the whole
# application down when the package was absent: app.py imports this module at
# startup, so a missing optional OCR dependency stopped the face pipeline, the
# live camera and the media analysis from running at all.
#
# Constructing the Reader also downloads ~100MB of models on first use, which is
# not something an import statement should do.
_reader = None


def get_reader():
    """Load the EasyOCR model on first use. Raises if the package is absent."""
    global _reader
    if _reader is None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Either `pip install easyocr` or use the "
                "Azure Vision OCR path (services/plate_ocr.py), which needs no local model."
            ) from exc
        logger.info("Loading EasyOCR model (first use downloads weights)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader

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
        results = get_reader().readtext(img_np)
        
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
                WHERE regexp_replace(upper(p.plate_number), '[^A-Z0-9]', '', 'g') = %s
                ORDER BY f.created_at DESC NULLS LAST
                LIMIT 1;
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
                        # Signed: the plate container is private.
                        "image_url": plate_ocr.sign_image_url(match[4]),
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