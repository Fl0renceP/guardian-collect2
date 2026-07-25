import psycopg2
from flask import Flask, request, jsonify
from config import Config
from services.recognition import process_incoming_face_image

app = Flask(__name__)

# Initialize DB connection (using config)
def get_db_connection():
    return psycopg2.connect(Config.DATABASE_URL)

@app.route("/")
def home():
    return {"message": "Guardian Collective API is running"}

@app.route("/api/v1/scan-face", methods=["POST"])
def scan_face():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['file']
    image_bytes = file.read()
    
    conn = get_db_connection()
    try:
        # Process scan with DeepFace + Postgres
        result = process_incoming_face_image(
            image_bytes=image_bytes, 
            db_conn=conn, 
            model_name=Config.FACE_MODEL, 
            threshold=Config.MATCH_THRESHOLD
        )
        
        # Handle tuple error responses (e.g. 400 when no face detected)
        if isinstance(result, tuple):
            return jsonify(result[0]), result[1]
            
        return jsonify(result), 200
        
    finally:
        conn.close()

if __name__ == "__main__":
    app.run(debug=True)