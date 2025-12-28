#!/usr/bin/env python3
"""
Inspect the Kaggle-trained model to understand its architecture
"""

import torch

def inspect_kaggle_model():
    """Inspect the model architecture from Kaggle"""
    model_path = "fixed_deepfake_model.pt"
    
    print("🔍 Inspecting Kaggle Model Architecture")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            print(f"\nModel has {len(state_dict)} parameters:")
            
            # Group parameters by layer
            layers = {}
            for name, param in state_dict.items():
                layer_name = '.'.join(name.split('.')[:-1])  # Remove .weight/.bias
                if layer_name not in layers:
                    layers[layer_name] = []
                layers[layer_name].append((name.split('.')[-1], param.shape))
            
            # Print architecture
            for layer_name, params in sorted(layers.items()):
                print(f"\n{layer_name}:")
                for param_type, shape in params:
                    print(f"  {param_type}: {shape}")
        
        # Check other info
        if 'config' in checkpoint:
            print(f"\nConfig: {checkpoint['config']}")
        
        if 'best_acc' in checkpoint:
            print(f"Best Accuracy: {checkpoint['best_acc']}")
        
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    inspect_kaggle_model()