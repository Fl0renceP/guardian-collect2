FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Updated package names for modern Debian images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libpq-dev \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "from deepface import DeepFace; DeepFace.build_model('Facenet512')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "python -m gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120"]