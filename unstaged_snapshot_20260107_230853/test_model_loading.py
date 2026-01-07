"""
TEST MODEL LOADING
Simple test to see if we can load both models
"""

import torch
from pathlib import Path
import sys

def test_load_model(model_path):
    """Test loading a single model"""
    
    print(f"\n📦 Testing: {Path(model_path).name}")
    
    if not Path(model_path).exists():
        print(f"❌ File not found: {model_path}")
        return False
    
    file_size = Path(model_path).stat().st_size / (1024**2)
    print(f"📊 File size: {file_size:.1f} MB")
    
    try:
        # Try multiple loading methods
        methods = [
            ("weights_only=False", lambda: torch.load(model_path, map_location='cpu', weights_only=False)),
            ("default", lambda: torch.load(model_path, map_location='cpu')),
            ("no map_location", lambda: torch.load(model_path)),
        ]
        
        for method_name, load_func in methods:
            try:
                print(f"🔄 Trying method: {method_name}")
                checkpoint = load_func()
                
                if isinstance(checkpoint, dict):
                    print(f"✅ Loaded successfully with {method_name}")
                    print(f"📋 Keys in checkpoint: {list(checkpoint.keys())}")
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print(f"📊 Model state dict has {len(state_dict)} parameters")
                        
                        # Show some parameter info
                        param_count = 0
                        for key, value in list(state_dict.items())[:5]:
                            if hasattr(value, 'shape'):
                                param_count += value.numel()
                                print(f"   {key}: {value.shape}")
                        print(f"📊 Sample parameters: {param_count:,}")
                        
                        # Check for training info
                        if 'epoch' in checkpoint:
                            print(f"📈 Epoch: {checkpoint['epoch']}")
                        if 'metrics' in checkpoint:
                            print(f"📈 Metrics: {checkpoint['metrics']}")
                    else:
                        print(f"📊 Direct state dict with {len(checkpoint)} parameters")
                    
                    return True
                else:
                    print(f"⚠️ Unexpected checkpoint type: {type(checkpoint)}")
                    
            except Exception as e:
                print(f"❌ Method {method_name} failed: {e}")
                continue
        
        print(f"❌ All loading methods failed")
        return False
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return False

def main():
    """Test both models"""
    
    print("🧪 MODEL LOADING TEST")
    print("="*50)
    
    # Model paths
    old_model = "ll_model_student (1).pt"
    new_model = "stage2_full_celebdf_best_epoch3.pt"
    
    print(f"🎯 Testing model loading capabilities...")
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    
    # Test old model
    old_success = test_load_model(old_model)
    
    # Test new model
    new_success = test_load_model(new_model)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Old model ({old_model}): {'✅ Success' if old_success else '❌ Failed'}")
    print(f"   New model ({new_model}): {'✅ Success' if new_success else '❌ Failed'}")
    
    if old_success and new_success:
        print(f"\n🎉 Both models can be loaded! Ready for comparison.")
        return True
    elif old_success or new_success:
        print(f"\n⚠️ Only one model loaded successfully. Partial comparison possible.")
        return False
    else:
        print(f"\n❌ Neither model could be loaded. Check file paths and compatibility.")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n💡 Next step: Run the full comparison script")
    else:
        print(f"\n💡 Fix loading issues first, then try comparison")