"""
E-Raksha Local Model Inspector

Comprehensive inspection tool for analyzing locally trained model architectures,
parameters, and metadata. Provides detailed insights into model structure and training history.

Author: E-Raksha Team
"""

import torch
from pathlib import Path

def inspect_model(model_path):
    """
    Inspect a model's architecture and metadata.
    
    Args:
        model_path (Path): Path to the model file to inspect
    """
    print(f"\n{'='*80}")
    print(f"🔍 INSPECTING: {model_path.name}")
    print(f"{'='*80}")
    
    if not model_path.exists():
        print(f"❌ File not found!")
        return
    
    # File size analysis
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"📊 File Size: {size_mb:.1f} MB")
    
    try:
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"✅ Format: Training checkpoint with metadata")
            if 'epoch' in checkpoint:
                print(f"   Training Epoch: {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                print(f"   Metrics: {checkpoint['metrics']}")
        else:
            state_dict = checkpoint
            print(f"✅ Format: Direct state dict")
        
        print(f"📊 Total parameters: {len(state_dict)}")
        
        # Find specialist module
        specialist_keys = [k for k in state_dict.keys() if 'specialist_module' in k and 'weight' in k]
        
        if specialist_keys:
            print(f"\n🔧 SPECIALIST MODULE: YES ({len(specialist_keys)} weight parameters)")
            
            # Show first 15 parameters
            print(f"\n📋 First 15 specialist parameters:")
            for i, key in enumerate(specialist_keys[:15]):
                shape = state_dict[key].shape
                print(f"   {i+1}. {key}: {shape}")
            
            # Determine total output channels
            for key in state_dict.keys():
                if 'specialist_module.attention.2.weight' in key:
                    total_channels = state_dict[key].shape[0]
                    print(f"\n   ✅ TOTAL OUTPUT CHANNELS: {total_channels}")
                    break
        else:
            print(f"\n🔧 SPECIALIST MODULE: NO (might be baseline/student model)")
        
        # Check backbone
        if any('efficientnet' in k.lower() or 'backbone.features' in k for k in state_dict.keys()):
            print(f"\n🏗️  BACKBONE: EfficientNet-B4")
        elif any('resnet' in k.lower() for k in state_dict.keys()):
            print(f"\n🏗️  BACKBONE: ResNet")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Check your local models
local_models = [
    "rr_stage2_final_20260106_182540.pt",
    "AV_Stage2_CelebDF_Model.pt",
    "bg_stage2_final_20260106_143158.pt",
    "cm_stage2_final.pt",
    "stage4_dfdc_chunk9_complete_20260106_115234.pt"
]

print("🔍 INSPECTING YOUR LOCAL MODELS")
print("="*80)

for model_file in local_models:
    model_path = Path(model_file)
    inspect_model(model_path)

print(f"\n{'='*80}")
print("✅ INSPECTION COMPLETE")
print("="*80)
print("\nNow I'll create the correct architecture based on these models!")
