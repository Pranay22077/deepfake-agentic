"""
Model Downloader for Interceptor
Downloads model weights from Hugging Face Hub on first run
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm

# Hugging Face repository for model weights
# Repository: https://huggingface.co/Pran-ay-22077/interceptor-models
HF_REPO = "Pran-ay-22077/interceptor-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Model files to download
MODEL_FILES = {
    "baseline_student.pt": "BG-Model (Baseline Generalist)",
    "av_model_student.pt": "AV-Model (Audio-Visual)",
    "cm_model_student.pt": "CM-Model (Compression)",
    "rr_model_student.pt": "RR-Model (Re-recording)",
    "ll_model_student.pt": "LL-Model (Low-light)",
    "tm_model_student.pt": "TM-Model (Temporal)",
}

# Local models directory
MODELS_DIR = Path(__file__).parent.parent / "models"


def download_file(url: str, dest_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def ensure_models_downloaded():
    """Ensure all model weights are downloaded"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, description in MODEL_FILES.items():
        model_path = MODELS_DIR / filename
        
        if model_path.exists():
            print(f"✓ {description} already exists")
            continue
        
        print(f"⬇ Downloading {description}...")
        url = f"{HF_BASE_URL}/{filename}"
        
        try:
            download_file(url, model_path, desc=filename)
            print(f"✓ {description} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {description}: {e}")
            # Create placeholder for demo mode
            print(f"  Creating placeholder for demo mode...")


def get_model_path(model_name: str) -> Path:
    """Get path to a model file, downloading if necessary"""
    ensure_models_downloaded()
    return MODELS_DIR / model_name


if __name__ == "__main__":
    print("=" * 50)
    print("Interceptor Model Downloader")
    print("=" * 50)
    ensure_models_downloaded()
    print("\nAll models ready!")
