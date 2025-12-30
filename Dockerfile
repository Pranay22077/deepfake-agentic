# Interceptor Backend - Production Docker Image (Lightweight)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy backend requirements
COPY backend-files/requirements.txt .

# Install Python dependencies (CPU-only PyTorch to save space)
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend-files/ .

# Create necessary directories
RUN mkdir -p models temp uploads

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV HOST=0.0.0.0
ENV HF_REPO=Pran-ay-22077/interceptor-models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["python", "app.py"]
