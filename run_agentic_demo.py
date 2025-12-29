#!/usr/bin/env python3
"""
E-Raksha Agentic Demo Launcher
Starts both backend API and frontend server
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading
import os

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting Backend API Server...")
    try:
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Run the backend
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "backend.app_agentic_corrected:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("🛑 Backend server stopped")
    except Exception as e:
        print(f"❌ Backend error: {e}")

def run_frontend():
    """Run the frontend server"""
    print("🌐 Starting Frontend Server...")
    try:
        # Wait for backend to start
        time.sleep(3)
        
        # Change to project directory
        os.chdir(Path(__file__).parent)
        
        # Run the frontend
        subprocess.run([sys.executable, "frontend/serve_agentic.py"], check=True)
    except KeyboardInterrupt:
        print("🛑 Frontend server stopped")
    except Exception as e:
        print(f"❌ Frontend error: {e}")

def main():
    """Main function to start both servers"""
    print("🎬 E-RAKSHA AGENTIC SYSTEM - BIAS CORRECTED")
    print("=" * 60)
    print("🔧 Starting bias-corrected agentic deepfake detection system")
    print("📊 Features: Multi-model routing, bias correction, balanced predictions")
    print("=" * 60)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        print("⏳ Waiting for backend to initialize...")
        time.sleep(5)
        
        # Start frontend (this will block)
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()