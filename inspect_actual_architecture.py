"""
E-Raksha Architecture Inspector

Downloads and analyzes actual model architectures from Hugging Face Hub.
Provides detailed inspection of model structures for development and debugging.

Author: E-Raksha Team
"""

import requests
import torch
from pathlib import Path
from tqdm import tqdm

# Hugging Face repository configuration
HF_REPO = "Pran-ay-22077/interceptor-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

def download_model(filename):
    """
    Download a model file from Hugging Face repository.
    
    Args:
        filename (str): Name of the model file to download
        
    Returns:
        Path: Path to downloaded model file
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / filename
    
    if model_path.exists():
        print(f"✅ {filename} already exists")
        return model_path
    
    url = f"{HF_BASE_URL}/{filename}"
    print(f"⬇️  Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(model_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return model_path

def inspect_architecture(model_path):
    """Inspect model architecture in detail"""
    
    print(f"\n{'='*80}")
    print(f"🔍 INSPECTING: {model_path.name}")
    print(f"{'='*80}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Find specialist module parameters
    specialist_params = {}
    for key in state_dict.keys():
        if 'specialist_module' in key and 'weight' in key:
            specialist_params[key] = state_dict[key].shape
    
    print(f"\n📊 SPECIALIST MODULE PARAMETERS:")
    for key, shape in sorted(specialist_params.items()):
        print(f"   {key}: {shape}")
    
    # Determine channel counts
    print(f"\n🔢 CHANNEL ANALYSIS:")
    
    # Check each sub-module
    for module_name in ['face_boundary', 'lighting_analyzer', 'shadow_detector', 'color_temp',
                        'bg_texture', 'lighting_detector', 'shadow_checker',
                        'lip_sync', 'av_correlation', 'expression_analyzer', 'speech_detector',
                        'dct_analyzer', 'quantization_detector', 'block_detector', 'compression_estimator',
                        'resolution_scales', 'upscaling_detector']:
        
        # Find first conv layer of this module
        for key, shape in specialist_params.items():
            if module_name in key and '.0.weight' in key:
                out_channels = shape[0]
                in_channels = shape[1]
                print(f"   {module_name}: {out_channels} output channels (input: {in_channels})")
                break
    
    # Check attention layer
    for key, shape in specialist_params.items():
        if 'attention' in key and '.2.weight' in key:
            total_channels = shape[0]
            print(f"\n   ✅ TOTAL OUTPUT CHANNELS: {total_channels}")
            break
    
    return state_dict

# Inspect each model
models_to_inspect = [
    "ll_model_student.pt",  # LL model (should show us the architecture)
    "cm_model_student.pt",  # CM model (this one works)
]

for model_file in models_to_inspect:
    try:
        model_path = download_model(model_file)
        inspect_architecture(model_path)
    except Exception as e:
        print(f"❌ Failed to inspect {model_file}: {e}")

print(f"\n{'='*80}")
print("✅ INSPECTION COMPLETE")
print("="*80)
