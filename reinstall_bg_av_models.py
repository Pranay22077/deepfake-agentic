"""
E-Raksha BG/AV Model Reinstaller

Utility for reinstalling corrupted or outdated BG and AV specialist models.
Handles clean deletion and fresh download from Hugging Face repository.

Author: E-Raksha Team
"""

import os
import shutil
from pathlib import Path
import requests
from tqdm import tqdm

# Hugging Face repository configuration
HF_REPO = "Pran-ay-22077/interceptor-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Target models for reinstallation
MODELS_TO_REINSTALL = {
    "baseline_student.pt": "BG-Model N (Background)",
    "av_model_student.pt": "AV-Model N (Audio-Visual)"
}

def delete_old_models():
    """
    Delete old or corrupted BG and AV model files.
    
    Returns:
        bool: True if deletion successful
    """
    print("🗑️  CLEANING CORRUPTED MODELS")
    print("="*60)
    
    models_dir = Path("models")
    
    for filename in MODELS_TO_REINSTALL.keys():
        # Check in models/ directory
        model_path = models_dir / filename
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"🗑️  Deleting: {filename} ({size_mb:.1f} MB)")
            model_path.unlink()
            print(f"   ✅ Deleted from models/")
        
        # Check in root directory
        root_path = Path(filename)
        if root_path.exists():
            size_mb = root_path.stat().st_size / (1024 * 1024)
            print(f"🗑️  Deleting: {filename} ({size_mb:.1f} MB)")
            root_path.unlink()
            print(f"   ✅ Deleted from root")
    
    print("\n✅ Old models deleted!")

def download_model(filename, description):
    """Download a single model from Hugging Face"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / filename
    url = f"{HF_BASE_URL}/{filename}"
    
    print(f"\n⬇️  Downloading {description}...")
    print(f"   URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        print(f"   Size: {total_size / (1024*1024):.1f} MB")
        
        with open(model_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify download
        downloaded_size = model_path.stat().st_size
        if downloaded_size == total_size:
            print(f"   ✅ Downloaded successfully: {downloaded_size / (1024*1024):.1f} MB")
            return True
        else:
            print(f"   ⚠️ Size mismatch: Expected {total_size}, got {downloaded_size}")
            return False
            
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False

def verify_models():
    """Verify the downloaded models"""
    
    print("\n🔍 VERIFYING DOWNLOADED MODELS")
    print("="*60)
    
    models_dir = Path("models")
    
    for filename, description in MODELS_TO_REINSTALL.items():
        model_path = models_dir / filename
        
        if not model_path.exists():
            print(f"❌ {description}: NOT FOUND")
            continue
        
        size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Check if size is reasonable (should be > 100 MB for these models)
        if size_mb < 100:
            print(f"⚠️ {description}: {size_mb:.1f} MB (TOO SMALL - likely corrupted)")
        else:
            print(f"✅ {description}: {size_mb:.1f} MB (looks good)")
            
            # Try to load with torch to verify
            try:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        num_params = len(checkpoint['model_state_dict'])
                        print(f"   📊 Parameters: {num_params}")
                        print(f"   📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
                    else:
                        num_params = len(checkpoint)
                        print(f"   📊 Parameters: {num_params}")
                else:
                    print(f"   ⚠️ Unexpected checkpoint format")
                
                print(f"   ✅ Model loads correctly!")
                
            except Exception as e:
                print(f"   ❌ Failed to load model: {e}")

def main():
    """Main reinstallation process"""
    
    print("🔄 REINSTALL BG AND AV MODELS")
    print("="*60)
    print("This will:")
    print("  1. Delete old/corrupted BG and AV models")
    print("  2. Download fresh copies from Hugging Face")
    print("  3. Verify the downloads")
    print("="*60)
    
    input("\nPress Enter to continue...")
    
    # Step 1: Delete old models
    delete_old_models()
    
    # Step 2: Download new models
    print("\n⬇️  DOWNLOADING FRESH MODELS")
    print("="*60)
    
    success_count = 0
    for filename, description in MODELS_TO_REINSTALL.items():
        if download_model(filename, description):
            success_count += 1
    
    # Step 3: Verify
    verify_models()
    
    # Summary
    print("\n" + "="*60)
    print("📊 REINSTALLATION SUMMARY")
    print("="*60)
    print(f"✅ Successfully downloaded: {success_count}/{len(MODELS_TO_REINSTALL)} models")
    
    if success_count == len(MODELS_TO_REINSTALL):
        print("\n🎉 All models reinstalled successfully!")
        print("\nNext steps:")
        print("  1. Run: python test_full_agent_system.py")
        print("  2. Verify models load correctly")
        print("  3. Test predictions")
    else:
        print("\n⚠️ Some models failed to download")
        print("   Please check your internet connection and try again")

if __name__ == "__main__":
    main()
