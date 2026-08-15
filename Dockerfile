FROM python:3.11-slim

# Install system dependencies needed for audio processing and build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy backend and frontend code
COPY backend /app/backend
COPY frontend /app/frontend

WORKDIR /app/backend

# Default port for Hugging Face Spaces is 7860, Render uses PORT env var
ENV PORT=7860
EXPOSE 7860

# Command to run FastAPI server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
