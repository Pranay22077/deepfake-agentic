#!/usr/bin/env python3
"""
Startup script for E-Raksha Agentic Backend
Downloads models and starts the FastAPI server
"""

import os
import sys

# Add paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, backend_dir)

# Download models first
from model_downloader import ensure_models_downloaded
print("[STARTUP] Checking/downloading models...")
ensure_models_downloaded()

# Import and run the app
import uvicorn
from app_agentic_corrected import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"[STARTUP] Starting E-Raksha Agentic API on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
