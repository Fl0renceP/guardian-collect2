# Use a lightweight official Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout for Railway logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and PostgreSQL (psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \
    && rm -rf /var/lib/apt-get/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port (Railway will override $PORT dynamically)
EXPOSE 8080

# Start Gunicorn with timeout for DeepFace model loading
CMD ["sh", "-c", "python -m gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --timeout 120"]