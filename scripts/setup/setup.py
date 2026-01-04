#!/usr/bin/env python3
"""
E-Raksha Setup Script
Handles initial setup including model download
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("[ERROR] Python 3.8+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"[OK] Python {sys.version.split()[0]} detected")
    return True

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[OK] {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("[WARNING] Docker not found - manual Python setup required")
    return False

def install_requirements():
    """Install Python requirements"""
    print("[INSTALL] Installing Python requirements...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        print("[OK] Requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install requirements: {e}")
        return False

def download_models():
    """Download model files"""
    print("[DOWNLOAD] Downloading model files...")
    
    try:
        # Get the directory where setup.py is located
        setup_dir = os.path.dirname(os.path.abspath(__file__))
        download_script = os.path.join(setup_dir, 'download_model.py')
        subprocess.run([sys.executable, download_script], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Model download failed: {e}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            print("[CREATE] Creating .env file from template...")
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                content = src.read()
                # Set some defaults
                content = content.replace('your_supabase_project_url_here', '')
                content = content.replace('your_supabase_anon_key_here', '')
                dst.write(content)
            print("[OK] Created .env file (you can edit it later for database features)")
        else:
            print("[WARNING] No .env.example found")
    else:
        print("[OK] .env file already exists")

def main():
    """Main setup function"""
    print("[SETUP] E-Raksha Setup")
    print("=" * 30)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not os.path.exists('backend-files') or not os.path.exists('src'):
        print("[ERROR] Please run this script from the E-Raksha project root directory")
        sys.exit(1)
    
    # Check Docker availability
    has_docker = check_docker()
    
    if has_docker:
        print("\n[DOCKER] Docker Setup (Recommended)")
        print("=" * 30)
        
        # Download models
        if not download_models():
            print("[ERROR] Setup failed - could not download models")
            sys.exit(1)
        
        # Create env file
        create_env_file()
        
        print("\n[DONE] Setup Complete!")
        print("=" * 30)
        print("[RUN] To start E-Raksha:")
        print("   docker-compose up --build")
        print("\n[WEB] Then open: http://localhost:3001")
        
    else:
        print("\n[PYTHON] Python Setup")
        print("=" * 30)
        
        # Install requirements
        if not install_requirements():
            print("[ERROR] Setup failed - could not install requirements")
            sys.exit(1)
        
        # Download models
        if not download_models():
            print("[ERROR] Setup failed - could not download models")
            sys.exit(1)
        
        # Create env file
        create_env_file()
        
        print("\n[DONE] Setup Complete!")
        print("=" * 30)
        print("[RUN] To start E-Raksha:")
        print("   # Terminal 1 (Backend):")
        print("   cd backend-files && python app.py")
        print("   # Terminal 2 (Frontend):")
        print("   npm run dev")
        print("\n[WEB] Then open: http://localhost:5173")

if __name__ == "__main__":
    main()
