"""
TEST OLD MODEL BIAS
Quick test to see if old model predicts everything as fake
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import torch.nn.functional as F

def test_old_model_predictions():
    """Test what the old model actually predicts"""
    
    print("🔍 TESTING OLD MODEL BIAS")
    print("="*50)
    
    # Load old model
    try:
        checkpoint = torch.load("ll_model_student (1).pt", map_location='cpu', weights_only=False)
        print(f"✅ Loaded old model checkpoint")
        print(f"📊 Keys: {list(checkpoint.keys())}")
        print(f"📈 Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"📈 Best accuracy: {checkpoint.get('best_acc', 'unknown')}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"📊 Model has {len(state_dict)} parameters")
            
            # Show some parameter names to understand architecture
            param_names = list(state_dict.keys())[:10]
            print(f"📋 Sample parameters: {param_names}")
            
        else:
            print("⚠️ No model_state_dict found")
            return
            
    except Exception as e:
        print(f"❌ Failed to load old model: {e}")
        return
    
    # Test on a few sample videos
    TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
    REAL_VIDEOS_DIR = TEST_DATA_ROOT / "real"
    FAKE_VIDEOS_DIR = TEST_DATA_ROOT / "fake"
    
    # Get a few sample videos
    real_videos = list(REAL_VIDEOS_DIR.glob("*.mp4"))[:5]
    fake_videos = list(FAKE_VIDEOS_DIR.glob("*.mp4"))[:5]
    
    print(f"\n📹 Testing on {len(real_videos)} real and {len(fake_videos)} fake videos")
    
    # Simple prediction test (without loading the actual model)
    print(f"\n🎯 EXPECTED BEHAVIOR FOR OLD MODEL:")
    print(f"   If trained only on fake data:")
    print(f"   - Should predict FAKE for all videos")
    print(f"   - Real detection: ~0%")
    print(f"   - Fake detection: ~100%")
    print(f"   - Overall accuracy: ~50%")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Run the full comparison to see actual predictions")
    print(f"   If old model shows >10% real detection, something's wrong")
    
    return True

if __name__ == "__main__":
    test_old_model_predictions()