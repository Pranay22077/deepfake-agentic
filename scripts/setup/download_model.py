#!/usr/bin/env python3
"""
E-Raksha Model Downloader
Downloads the trained model file if not present
"""

import os
import requests
import sys
from pathlib import Path
from tqdm import tqdm

# Model download URLs (you'll need to host these somewhere)
MODEL_URLS = {
    'fixed_deepfake_model.pt': {
        'url': 'https://github.com/Pranay22077/deepfake-agentic/releases/download/v1.0/fixed_deepfake_model.pt',
        'size': 136246834,  # bytes
        'description': 'Main ResNet18 deepfake detection model'
    },
    'baseline_student.pkl': {
        'url': 'https://github.com/Pranay22077/deepfake-agentic/releases/download/v1.0/baseline_student.pkl',
        'size': 44974317,  # bytes
        'description': 'Legacy pickle format model (backup)'
    }
}

def download_file(url: str, filename: str, expected_size: int = None):
    """Download a file with progress bar"""
    print(f"[DOWNLOAD] Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            print(f"[WARNING] Expected {expected_size} bytes, got {total_size} bytes")
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"[OK] Downloaded {filename} ({total_size / 1024 / 1024:.1f} MB)")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")
        return False

def check_model_files():
    """Check if model files exist and download if missing"""
    print("[CHECK] Checking for model files...")
    
    missing_files = []
    for filename, info in MODEL_URLS.items():
        if not os.path.exists(filename):
            missing_files.append(filename)
            print(f"[MISSING] Missing: {filename}")
        else:
            file_size = os.path.getsize(filename)
            print(f"[OK] Found: {filename} ({file_size / 1024 / 1024:.1f} MB)")
    
    if not missing_files:
        print("[OK] All model files are present!")
        return True
    
    print(f"\n[DOWNLOAD] Need to download {len(missing_files)} model file(s)")
    
    # Download missing files
    success_count = 0
    for filename in missing_files:
        info = MODEL_URLS[filename]
        print(f"\n[INFO] {info['description']}")
        
        if download_file(info['url'], filename, info['size']):
            success_count += 1
        else:
            print(f"[ERROR] Failed to download {filename}")
    
    if success_count == len(missing_files):
        print(f"\n[OK] Successfully downloaded all {success_count} model files!")
        return True
    else:
        print(f"\n[WARNING] Downloaded {success_count}/{len(missing_files)} files")
        return False

def main():
    """Main function"""
    print("[INIT] E-Raksha Model Downloader")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('backend') or not os.path.exists('frontend'):
        print("[ERROR] Please run this script from the E-Raksha project root directory")
        sys.exit(1)
    
    # Check and download models
    if check_model_files():
        print("\n[OK] Ready to run E-Raksha!")
        print("   Run: docker-compose up --build")
        print("   Or:  python backend/app.py")
    else:
        print("\n[ERROR] Model download failed. Please check your internet connection.")
        print("   You can also manually download the model files from:")
        for filename, info in MODEL_URLS.items():
            print(f"   - {info['url']}")
        sys.exit(1)

if __name__ == "__main__":
    main()