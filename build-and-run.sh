#!/bin/bash
# E-Raksha Quick Build and Run Script

echo "🚀 E-Raksha Quick Deployment"
echo "============================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if model file exists
if [ ! -f "fixed_deepfake_model.pt" ]; then
    echo "❌ Model file not found: fixed_deepfake_model.pt"
    echo "   Please ensure the model file is in the project root directory."
    exit 1
fi

echo "✅ Docker is installed"
echo "✅ Docker Compose is available"

echo ""
echo "🔧 Running setup (downloading models if needed)..."
python setup.py

if [ $? -ne 0 ]; then
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi

# Check if ports are available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use. Please stop the service using it or change the port."
    echo "   To find what's using port 8000: lsof -i :8000"
fi

if lsof -Pi :3001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 3001 is already in use. Please stop the service using it or change the port."
    echo "   To find what's using port 3001: lsof -i :3001"
fi

echo ""
echo "🔧 Building and starting E-Raksha..."
echo "This may take 2-3 minutes on first run..."
echo ""

# Build and start the application
docker-compose up --build

echo ""
echo "🛑 E-Raksha has been stopped."
echo "To restart: docker-compose up"
echo "To rebuild: docker-compose up --build"