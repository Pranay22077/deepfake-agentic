#!/usr/bin/env python3
"""
Inspect Specialist Model Files
Analyze the actual architecture of trained models to fix loading issues
"""

import torch
import os
from collections import OrderedDict

def inspect_model(model_path, model_name):
    """Inspect a model file and print its architecture"""
    print(f"\n🔍 Inspecting {model_name}: {model_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check if it's a dict with metadata
        if isinstance(checkpoint, dict):
            print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"✅ Found model_state_dict")
                
                # Print training info
                if 'best_acc' in checkpoint:
                    print(f"🎯 Best accuracy: {checkpoint['best_acc']:.2f}%")
                if 'epoch' in checkpoint:
                    print(f"📊 Epochs trained: {checkpoint['epoch'] + 1}")
                if 'config' in checkpoint:
                    print(f"⚙️ Config: {checkpoint['config']}")
            else:
                state_dict = checkpoint
                print(f"⚠️ Direct state dict (no metadata)")
        else:
            state_dict = checkpoint
            print(f"⚠️ Direct state dict format")
        
        # Analyze architecture from state dict
        print(f"\n🏗️ Model Architecture Analysis:")
        
        # Group layers by prefix
        layer_groups = {}
        for key in state_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in layer_groups:
                layer_groups[prefix] = []
            layer_groups[prefix].append(key)
        
        for prefix, layers in layer_groups.items():
            print(f"   {prefix}: {len(layers)} parameters")
            
            # Show some example layers
            if len(layers) <= 5:
                for layer in layers:
                    shape = state_dict[layer].shape
                    print(f"     - {layer}: {shape}")
            else:
                for layer in layers[:3]:
                    shape = state_dict[layer].shape
                    print(f"     - {layer}: {shape}")
                print(f"     ... and {len(layers)-3} more")
        
        # Check for specific architectures
        print(f"\n🔍 Architecture Detection:")
        
        # Check for ResNet backbone
        if any('layer1' in key for key in state_dict.keys()):
            print("   ✅ ResNet backbone detected")
        
        # Check for custom layers
        if any('compression_adapter' in key for key in state_dict.keys()):
            print("   ✅ Compression adapter detected")
        
        if any('moire_detector' in key for key in state_dict.keys()):
            print("   ✅ Moiré detector detected")
        
        if any('brightness_enhancer' in key for key in state_dict.keys()):
            print("   ✅ Brightness enhancer detected")
        
        if any('temporal_conv' in key for key in state_dict.keys()):
            print("   ✅ Temporal convolution detected")
        
        if any('lstm' in key for key in state_dict.keys()):
            print("   ✅ LSTM detected")
        
        if any('audio' in key for key in state_dict.keys()):
            print("   ✅ Audio processing detected")
        
        if any('lip_sync' in key for key in state_dict.keys()):
            print("   ✅ Lip sync detector detected")
        
        # Check classifier dimensions
        classifier_keys = [k for k in state_dict.keys() if 'classifier' in k or 'fc' in k]
        if classifier_keys:
            print(f"\n🎯 Classifier layers:")
            for key in classifier_keys:
                shape = state_dict[key].shape
                print(f"   - {key}: {shape}")
        
        # Estimate model size
        total_params = sum(p.numel() for p in state_dict.values())
        file_size = os.path.getsize(model_path) / (1024 * 1024)
        
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   File size: {file_size:.1f} MB")
        
        return state_dict
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return None

def main():
    """Inspect all specialist models"""
    print("🔍 E-RAKSHA SPECIALIST MODEL INSPECTION")
    print("=" * 70)
    
    models_to_inspect = [
        ("av_model_student.pt", "AV-Model (Audio-Visual Specialist)"),
        ("cm_model_student.pt", "CM-Model (Compression Specialist)"),
        ("rr_model_student.pt", "RR-Model (Re-recording Specialist)"),
        ("ll_model_student.pt", "LL-Model (Low-light Specialist)"),
        ("tm_model_student.pt", "TM-Model (Temporal Specialist)"),
        ("baseline_student.pkl", "BG-Model (Baseline Generalist)")
    ]
    
    model_info = {}
    
    for model_path, model_name in models_to_inspect:
        state_dict = inspect_model(model_path, model_name)
        if state_dict is not None:
            model_info[model_path] = state_dict
    
    # Summary
    print(f"\n📋 INSPECTION SUMMARY")
    print("=" * 70)
    
    for model_path, model_name in models_to_inspect:
        if model_path in model_info:
            print(f"✅ {model_name}: Architecture analyzed")
        else:
            print(f"❌ {model_name}: Failed to analyze")
    
    print(f"\n🔧 Next Steps:")
    print("1. Use this analysis to fix model architectures")
    print("2. Create compatible loading functions")
    print("3. Test all models in the agentic system")
    
    return model_info

if __name__ == "__main__":
    model_info = main()