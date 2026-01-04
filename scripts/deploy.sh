#!/bin/bash
# E-Raksha Linux/macOS Deployment Script

echo "E-Raksha Deployment Script"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

echo "Docker found"

# Stop existing container
echo "Stopping existing container..."
docker stop eraksha-site 2>/dev/null || true
docker rm eraksha-site 2>/dev/null || true

# Build new image
echo "Building E-Raksha image..."
docker build -t eraksha-site -f docker/Dockerfile .

if [ $? -ne 0 ]; then
    echo "Image build failed"
    exit 1
fi

echo "Image built successfully"

# Run container
echo "Starting E-Raksha container..."
docker run -d -p 8080:80 --name eraksha-site eraksha-site

if [ $? -ne 0 ]; then
    echo "Container failed to start"
    exit 1
fi

echo "Container started successfully"
echo "Website available at: http://localhost:8080"
echo "Model downloads available at: http://localhost:8080/models/"

# Wait and test
echo "Waiting for service to be ready..."
sleep 5

echo "Deployment complete!"