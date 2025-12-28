#!/usr/bin/env python3
"""
Debug the model to identify why it's performing poorly
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
import os

class StudentModel(nn.Module):
    """ResNet18-based student model"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_model_debug():
    """Load model and inspect its state"""
    model_path = "./kaggle_outputs_20251228_043850/baseline_student.pkl"
    
    print("Loading model for debugging...")
    print(f"Model path: {model_path}")
    
    # Load pickle data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Pickle data type: {type(model_data)}")
    print(f"Number of parameters: {len(model_data)}")
    
    # Create model
    model = StudentModel(num_classes=2)
    
    # Convert and load state dict
    state_dict = {}
    missing_keys = []
    unexpected_keys = []
    
    for name, param_array in model_data.items():
        if isinstance(param_array, np.ndarray):
            state_dict[name] = torch.from_numpy(param_array)
        else:
            print(f"WARNING: Parameter {name} is not numpy array: {type(param_array)}")
    
    # Check what's missing
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys
    
    print(f"\nModel state analysis:")
    print(f"Model expects {len(model_keys)} parameters")
    print(f"Loaded {len(loaded_keys)} parameters")
    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"\nMissing parameters (first 10):")
        for key in list(missing_keys)[:10]:
            print(f"  - {key}")
    
    if unexpected_keys:
        print(f"\nUnexpected parameters (first 10):")
        for key in list(unexpected_keys)[:10]:
            print(f"  - {key}")
    
    # Load with strict=False
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

def test_model_behavior(model):
    """Test model behavior with different inputs"""
    print("\n" + "="*50)
    print("TESTING MODEL BEHAVIOR")
    print("="*50)
    
    # Test with different input patterns
    test_cases = [
        ("Random noise", torch.randn(1, 3, 224, 224)),
        ("All zeros", torch.zeros(1, 3, 224, 224)),
        ("All ones", torch.ones(1, 3, 224, 224)),
        ("Normal range", torch.rand(1, 3, 224, 224)),
        ("ImageNet normalized", torch.randn(1, 3, 224, 224) * 0.5 + 0.5)
    ]
    
    with torch.no_grad():
        for name, input_tensor in test_cases:
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            print(f"{name:20} | Raw: [{output[0][0].item():8.3f}, {output[0][1].item():8.3f}] | "
                  f"Prob: [{probabilities[0][0].item():.3f}, {probabilities[0][1].item():.3f}] | "
                  f"Pred: {'Real' if pred_class == 0 else 'Fake'} ({confidence:.3f})")

def analyze_model_weights(model):
    """Analyze model weights for issues"""
    print("\n" + "="*50)
    print("ANALYZING MODEL WEIGHTS")
    print("="*50)
    
    total_params = 0
    zero_params = 0
    nan_params = 0
    inf_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Check for problematic values
        zeros = (param == 0).sum().item()
        nans = torch.isnan(param).sum().item()
        infs = torch.isinf(param).sum().item()
        
        zero_params += zeros
        nan_params += nans
        inf_params += infs
        
        if zeros > param_count * 0.9:  # More than 90% zeros
            print(f"WARNING: {name} has {zeros}/{param_count} ({zeros/param_count*100:.1f}%) zero values")
        
        if nans > 0:
            print(f"ERROR: {name} has {nans} NaN values")
        
        if infs > 0:
            print(f"ERROR: {name} has {infs} infinite values")
        
        # Check weight statistics
        mean_val = param.mean().item()
        std_val = param.std().item()
        min_val = param.min().item()
        max_val = param.max().item()
        
        if abs(mean_val) > 10 or std_val > 10:
            print(f"WARNING: {name} has unusual statistics - mean: {mean_val:.3f}, std: {std_val:.3f}")
    
    print(f"\nOverall statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Zero parameters: {zero_params:,} ({zero_params/total_params*100:.2f}%)")
    print(f"NaN parameters: {nan_params:,}")
    print(f"Infinite parameters: {inf_params:,}")

def test_preprocessing_pipeline():
    """Test if preprocessing matches training"""
    print("\n" + "="*50)
    print("TESTING PREPROCESSING PIPELINE")
    print("="*50)
    
    # Create a test image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Different preprocessing approaches
    transforms_list = [
        ("No normalization", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])),
        ("ImageNet normalization", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])),
        ("Zero-centered", transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]))
    ]
    
    model = load_model_debug()
    
    with torch.no_grad():
        for name, transform in transforms_list:
            input_tensor = transform(pil_image).unsqueeze(0)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            print(f"{name:20} | Input range: [{input_tensor.min().item():.3f}, {input_tensor.max().item():.3f}] | "
                  f"Pred: {'Real' if pred_class == 0 else 'Fake'} ({confidence:.3f})")

def check_batchnorm_stats(model):
    """Check BatchNorm layer statistics"""
    print("\n" + "="*50)
    print("CHECKING BATCHNORM STATISTICS")
    print("="*50)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            print(f"\n{name}:")
            print(f"  running_mean: {module.running_mean is not None}")
            print(f"  running_var: {module.running_var is not None}")
            
            if module.running_mean is not None:
                print(f"  mean range: [{module.running_mean.min().item():.3f}, {module.running_mean.max().item():.3f}]")
            if module.running_var is not None:
                print(f"  var range: [{module.running_var.min().item():.3f}, {module.running_var.max().item():.3f}]")
            
            # Check if they're initialized properly
            if module.running_mean is not None and torch.allclose(module.running_mean, torch.zeros_like(module.running_mean)):
                print(f"  WARNING: running_mean is all zeros!")
            if module.running_var is not None and torch.allclose(module.running_var, torch.ones_like(module.running_var)):
                print(f"  WARNING: running_var is all ones (not updated)!")

def main():
    print("E-Raksha Model Debugging")
    print("="*50)
    
    try:
        # Load and analyze model
        model = load_model_debug()
        
        # Test model behavior
        test_model_behavior(model)
        
        # Analyze weights
        analyze_model_weights(model)
        
        # Check BatchNorm
        check_batchnorm_stats(model)
        
        # Test preprocessing
        test_preprocessing_pipeline()
        
        print("\n" + "="*50)
        print("DEBUGGING COMPLETE")
        print("="*50)
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()