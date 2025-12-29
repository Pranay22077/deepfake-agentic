#!/usr/bin/env python3
"""
Test Fixed Specialist Model Loading
Verify that the fixed architectures can load the actual trained weights
"""

import torch
import os
from src.models.specialists_fixed import load_specialist_model_fixed

def test_model_loading(model_path, model_type, model_name):
    """Test loading a specific model"""
    print(f"\n🧪 Testing {model_name}")
    print("-" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load the model
        model = load_specialist_model_fixed(model_path, model_type, device='cpu')
        
        # Test inference
        if model_type == 'tm':
            # Temporal model expects sequence input
            test_input = torch.randn(1, 8, 3, 224, 224)
        elif model_type == 'av':
            # AV model expects video + audio
            test_video = torch.randn(1, 8, 3, 224, 224)
            test_audio = torch.randn(1, 48000)
            
            with torch.no_grad():
                output = model(test_video, test_audio)
            
            print(f"✅ Model loaded and tested successfully")
            print(f"   Output shape: {output.shape}")
            print(f"   Sample output: {output[0].tolist()}")
            return True
        else:
            # Regular models expect single frame
            test_input = torch.randn(1, 3, 224, 224)
        
        # Run inference for non-AV models
        if model_type != 'av':
            with torch.no_grad():
                output = model(test_input)
            
            print(f"✅ Model loaded and tested successfully")
            print(f"   Output shape: {output.shape}")
            print(f"   Sample output: {output[0].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Test all specialist models"""
    print("🔧 TESTING FIXED SPECIALIST MODEL LOADING")
    print("=" * 60)
    
    models_to_test = [
        ("av_model_student.pt", "av", "AV-Model (Audio-Visual Specialist)"),
        ("cm_model_student.pt", "cm", "CM-Model (Compression Specialist)"),
        ("rr_model_student.pt", "rr", "RR-Model (Re-recording Specialist)"),
        ("ll_model_student.pt", "ll", "LL-Model (Low-light Specialist)"),
        ("tm_model_student.pt", "tm", "TM-Model (Temporal Specialist)")
    ]
    
    results = []
    
    for model_path, model_type, model_name in models_to_test:
        success = test_model_loading(model_path, model_type, model_name)
        results.append((model_name, success))
    
    # Summary
    print(f"\n📋 LOADING TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {model_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} models loaded successfully")
    
    if passed == total:
        print("🎉 All specialist models can be loaded with fixed architectures!")
        print("✅ Ready to integrate into the agentic system")
    elif passed > 0:
        print("⚠️ Some models loaded successfully, others need attention")
    else:
        print("❌ No models loaded successfully, architectures need more work")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)