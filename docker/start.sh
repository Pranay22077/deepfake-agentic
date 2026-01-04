#!/bin/bash
# E-Raksha Docker Startup Script

echo "[RUN] Starting E-Raksha Deepfake Detection System"
echo "=============================================="

# Check if model file exists, if not try to download
if [ ! -f "/app/fixed_deepfake_model.pt" ]; then
    echo "[ERROR] Model file not found: /app/fixed_deepfake_model.pt"
    echo "[LOOP] Attempting to download model..."
    
    cd /app
    python download_model.py
    
    if [ ! -f "/app/fixed_deepfake_model.pt" ]; then
        echo "[ERROR] Model download failed. Please check:"
        echo "   1. Internet connection"
        echo "   2. Model file availability"
        echo "   3. Manual download from GitHub releases"
        exit 1
    fi
fi

echo "[OK] Model file found: $(ls -lh /app/fixed_deepfake_model.pt)"

# Start backend server in background
echo "[SETUP] Starting Backend API Server (Port 8000)..."
cd /app/backend
python app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 10

# Check if backend is running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] Backend API is running"
else
    echo "[ERROR] Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend server
echo "[WEB] Starting Frontend Server (Port 3001)..."
cd /app/frontend
python serve-enhanced.py 3001 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 5

echo ""
echo "[DONE] E-Raksha is now running!"
echo "================================"
echo "[APP] Frontend: http://localhost:3001"
echo "[SETUP] Backend API: http://localhost:8000"
echo "[STATS] API Docs: http://localhost:8000/docs"
echo "[HEALTH]  Health Check: http://localhost:8000/health"
echo ""
echo "[INFO] Usage:"
echo "1. Open http://localhost:3001 in your browser"
echo "2. Upload a video file (MP4, AVI, MOV)"
echo "3. Get deepfake detection results"
echo ""
echo "🛑 To stop: Press Ctrl+C"
echo "================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down E-Raksha..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "[OK] Shutdown complete"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGTERM SIGINT

# Keep script running
wait