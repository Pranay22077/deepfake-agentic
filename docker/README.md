# E-Raksha Docker Deployment Guide

## Quick Start

### Option 1: Docker Compose (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd eraksha

# Build and start
docker-compose up --build

# Access the application
# Frontend: http://localhost:3001
# Backend API: http://localhost:8000
```

### Option 2: Docker Build & Run
```bash
# Build the image
docker build -t eraksha:latest .

# Run the container
docker run -p 8000:8000 -p 3001:3001 eraksha:latest

# Access the application
# Frontend: http://localhost:3001
# Backend API: http://localhost:8000
```

## Prerequisites

- Docker installed on your system
- At least 4GB RAM available
- Internet connection for downloading dependencies

## Configuration

### Environment Variables
- `MODEL_PATH`: Path to the model file (default: `/app/fixed_deepfake_model.pt`)
- `PORT`: Backend port (default: `8000`)
- `HOST`: Backend host (default: `0.0.0.0`)

### Volumes
- `./uploads:/app/uploads` - Uploaded videos
- `./logs:/app/logs` - Application logs
- `./models:/app/models` - Model files

## Health Check

The container includes a health check that verifies the backend API is running:
```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:8000/health
```

## Troubleshooting

### Container won't start
1. Check if ports 8000 and 3001 are available
2. Ensure model file exists in the image
3. Check Docker logs: `docker logs eraksha-app`

### Model not found error
1. Ensure `fixed_deepfake_model.pt` is in the project root
2. Rebuild the image: `docker-compose build --no-cache`

### API not responding
1. Wait 60 seconds for full startup
2. Check health endpoint: `curl http://localhost:8000/health`
3. Check backend logs in container

## Usage

1. Open http://localhost:3001 in your browser
2. Upload a video file (MP4, AVI, MOV)
3. Wait for analysis to complete
4. View results with confidence scores and heatmaps

## Updates

To update the application:
```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up --build
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## Docker Hub Deployment

To deploy to Docker Hub:
```bash
# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 -t yourusername/eraksha:latest --push .

# Users can then run:
docker run -p 8000:8000 -p 3001:3001 yourusername/eraksha:latest
```