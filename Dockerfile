# ============================================================================
# SPCH2TXT Server - Dockerfile (Optimized)
# ============================================================================
# Multi-stage Docker build for the Flask audio processing server
# Includes Whisper transcription and PyAnnote diarization
# Optimized for minimal image size

FROM python:3.13-slim AS builder

WORKDIR /build

# Install build dependencies only
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    make \
    portaudio19-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Python dependencies with minimal cache
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --compile \
    'flask>=3.0.0,<4.0.0' \
    'flask-cors>=5.0.0,<6.0.0' \
    'werkzeug>=3.0.0,<4.0.0' \
    'openai-whisper>=20250625,<20250626' \
    'pyannote-audio==3.1.1' \
    'torch==2.8.*' \
    'torchaudio==2.8.*' \
    'openai>=2.6.0,<3.0.0' \
    'python-dotenv>=1.1.1,<2.0.0' \
    'ffmpeg-python>=0.2.0,<0.3.0' \
    'requests>=2.32.0,<3.0.0' \
    'watchdog>=4.0.0,<5.0.0' \
    'huggingface-hub>=0.16.0,<1.0.0' \
    'transformers>=4.30.0,<5.0.0' \
    'numpy>=1.24.0,<2.0.0'

# ============================================================================
# Runtime stage - minimal dependencies
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY src/ ./src/
COPY .env.example ./.env.example
COPY README.md ./

# Create output directories
RUN mkdir -p /app/src/saved_audio /app/src/saved_transcripts /app/src/saved_summary

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_APP=src.server.app

EXPOSE 5001

# Run the Flask server
CMD ["python", "-m", "src.server.app"]
