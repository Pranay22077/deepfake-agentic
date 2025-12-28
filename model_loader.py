#!/usr/bin/env python3
"""
Model Loader for Step 2
Handles both .pt and .pkl model formats from Step 1
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pickle
import os

class StudentModel(nn.Module):
    """ResNet18-based student model (matches Step 1 training)"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Create ResNet18 backbone
        self.backbone = models.resnet18(weights=None)  # No pretrained weights needed
        
        # Replace final layer (matches Step 1 architecture)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model_from_step1(model_path):
    """Load model from Step 1 outputs (handles both .pt and .pkl)"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model instance
    model = StudentModel(num_classes=2)
    
    try:
        if model_path.endswith('.pt'):
            # Load PyTorch format
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"Loaded PyTorch model with metadata from {model_path}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"Loaded PyTorch state dict from {model_path}")
                
        elif model_path.endswith('.pkl'):
            # Load pickle format
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                # Convert numpy arrays back to tensors and load
                state_dict = {}
                for name, param_array in model_data.items():
                    state_dict[name] = torch.from_numpy(param_array)
                
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pickle model from {model_path} (strict=False for missing BatchNorm stats)")
            else:
                raise ValueError("Unexpected pickle format")
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        model.eval()  # Set to evaluation mode
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def test_model_loading(model_path):
    """Test if model loads and works correctly"""
    try:
        model = load_model_from_step1(model_path)
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Model test successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Sample prediction: {torch.softmax(output, dim=1)}")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

if __name__ == "__main__":
    # Test with your model file
    model_paths = [
        "./kaggle_outputs/baseline_student.pt",
        "./kaggle_outputs/baseline_student.pkl"
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Testing model: {model_path}")
            test_model_loading(model_path)
            break
    else:
        print("No model file found for testing")