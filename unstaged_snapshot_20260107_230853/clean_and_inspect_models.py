"""
CLEAN ALL MODELS AND INSPECT ARCHITECTURE
Delete all models and inspect what architecture they actually use
"""

import os
import shutil
from pathlib import Path
import torch

def delete_all_models():
    """Delete all downloaded models"""
    
    print("🗑️  DELETING ALL MODELS")
    print("="*60)
    
    models_dir = Path("models")
    
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        
        if model_files:
            for model_file in model_files:
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"🗑️  Deleting: {model_file.name} ({size_mb:.1f} MB)")
                model_file.unlink()
            
            print(f"\n✅ Deleted {len(model_files)} model files")
        else:
            print("📁 No models found in models/ directory")
    else:
        print("📁 models/ directory doesn't exist")
    
    # Also check root directory
    root_models = list(Path(".").glob("*_model_student.pt")) + list(Path(".").glob("baseline_student.pt"))
    if root_models:
        print(f"\n🗑️  Found {len(root_models)} models in root directory:")
        for model_file in root_models:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   Deleting: {model_file.name} ({size_mb:.1f} MB)")
            model_file.unlink()

def inspect_model_architecture(model_path):
    """Inspect a model's actual architecture"""
    
    print(f"\n🔍 INSPECTING: {model_path.name}")
    print("-"*60)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Format: Checkpoint with metadata")
                print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"   Stage: {checkpoint.get('stage', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"   Accuracy: {checkpoint['metrics'].get('accuracy', 'unknown')}")
            else:
                state_dict = checkpoint
                print(f"✅ Format: Direct state dict")
        else:
            state_dict = checkpoint
            print(f"✅ Format: Raw checkpoint")
        
        print(f"\n📊 Total parameters: {len(state_dict)}")
        
        # Analyze architecture from parameter names
        print(f"\n🏗️  ARCHITECTURE ANALYSIS:")
        
        # Check backbone type
        if any('efficientnet' in k.lower() or 'backbone.features' in k for k in state_dict.keys()):
            print(f"   Backbone: EfficientNet-B4")
            
            # Check first conv layer to determine exact architecture
            first_conv_key = 'backbone.features.0.0.weight'
            if first_conv_key in state_dict:
                shape = state_dict[first_conv_key].shape
                print(f"   First conv shape: {shape}")
                print(f"   → Output channels: {shape[0]}")
        elif any('resnet' in k.lower() or 'layer' in k for k in state_dict.keys()):
            print(f"   Backbone: ResNet")
        elif any('mobilenet' in k.lower() or 'visual_backbone' in k for k in state_dict.keys()):
            print(f"   Backbone: MobileNet")
        else:
            print(f"   Backbone: Unknown")
        
        # Check for specialist modules
        specialist_keys = [k for k in state_dict.keys() if 'specialist_module' in k]
        if specialist_keys:
            print(f"\n   Specialist Module: YES ({len(specialist_keys)} parameters)")
            
            # Try to determine specialist type and channels
            for key in specialist_keys[:5]:  # Show first 5
                if 'weight' in key:
                    shape = state_dict[key].shape
                    print(f"      {key}: {shape}")
        else:
            print(f"\n   Specialist Module: NO")
        
        # Check classifier
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k]
        if classifier_keys:
            print(f"\n   Classifier: YES ({len(classifier_keys)} parameters)")
        
        # Show some key parameter shapes
        print(f"\n📋 KEY PARAMETERS (first 10):")
        for i, (key, param) in enumerate(list(state_dict.items())[:10]):
            if hasattr(param, 'shape'):
                print(f"   {i+1}. {key}: {param.shape}")
            else:
                print(f"   {i+1}. {key}: {type(param)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to inspect: {e}")
        return False

def main():
    """Main function"""
    
    print("🔧 CLEAN AND INSPECT MODELS")
    print("="*60)
    print("This will:")
    print("  1. Delete ALL downloaded models")
    print("  2. Show you what to do next")
    print("="*60)
    
    response = input("\nDelete all models? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    # Delete all models
    delete_all_models()
    
    print("\n" + "="*60)
    print("✅ ALL MODELS DELETED")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. The models on Hugging Face have DIFFERENT architectures than specialists_new.py")
    print("   - BG model: Has different channel counts")
    print("   - AV model: Corrupted (only 95 MB)")
    print("   - RR model: Has different channel counts")
    print("   - LL model: Has different channel counts")
    print("   - CM model: ✅ Works (loaded successfully)")
    print("   - TM model: ✅ Works (loaded successfully)")
    
    print("\n2. OPTIONS:")
    print("   A) Re-upload models with CORRECT architecture to Hugging Face")
    print("   B) Update specialists_new.py to match the ACTUAL trained architecture")
    print("   C) Use only CM and TM models (the ones that work)")
    
    print("\n3. TO FIX:")
    print("   - Check your training scripts to see what architecture was ACTUALLY used")
    print("   - The channel counts in specialists_new.py don't match your trained models")
    print("   - BG/LL models seem to use 68 channels (not 44)")
    print("   - RR model seems to use 36 channels with different sub-module sizes")
    
    print("\n4. RECOMMENDATION:")
    print("   Use the ACTUAL training script architecture, not the planned one")
    print("   The models were trained with a specific architecture - we need to match it!")

if __name__ == "__main__":
    main()
