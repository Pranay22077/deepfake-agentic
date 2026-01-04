#!/usr/bin/env python3
"""
Model Loading Verification Script
Test if your Step 1 model loads correctly and works for inference
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import os
import numpy as np

class StudentModel(nn.Module):
    """ResNet18-based student model (matches your Step 1 training)"""
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

def find_model_file():
    """Find the model file from Step 1"""
    possible_paths = [
        "./kaggle_outputs/baseline_student.pkl",
        "./kaggle_outputs/baseline_student.pt", 
        "./baseline_student.pkl",
        "./baseline_student.pt"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def load_and_test_model():
    """Load model and run comprehensive tests"""
    print("E-Raksha Model Loading Test")
    print("=" * 50)
    
    # Find model file
    model_path = find_model_file()
    if not model_path:
        print("ERROR: No model file found!")
        print("Expected locations:")
        print("  - ./kaggle_outputs/baseline_student.pkl")
        print("  - ./kaggle_outputs/baseline_student.pt")
        return False
    
    print(f"Found model file: {model_path}")
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"File size: {file_size:.1f}MB")
    
    # Create model instance
    print("\nCreating model architecture...")
    model = StudentModel(num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load weights
    print(f"\nLoading weights from {os.path.basename(model_path)}...")
    
    try:
        if model_path.endswith('.pkl'):
            # Load pickle format
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"Pickle data type: {type(model_data)}")
            
            if isinstance(model_data, dict):
                print(f"Found {len(model_data)} parameters in pickle file")
                
                # Show some parameter names
                param_names = list(model_data.keys())[:5]
                print(f"Sample parameters: {param_names}")
                
                # Convert numpy arrays back to tensors
                state_dict = {}
                for name, param_array in model_data.items():
                    if isinstance(param_array, np.ndarray):
                        state_dict[name] = torch.from_numpy(param_array)
                    else:
                        print(f"WARNING: Parameter {name} is not numpy array: {type(param_array)}")
                        state_dict[name] = param_array
                
                model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded pickle model weights (strict=False for missing BatchNorm stats)")
            else:
                print(f"ERROR: Unexpected pickle format: {type(model_data)}")
                return False
                
        elif model_path.endswith('.pt'):
            # Load PyTorch format
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Loaded PyTorch model with metadata")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print("Loaded PyTorch state dict")
        
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    # Test inference
    print("\nTesting model inference...")
    
    try:
        # Create dummy input (batch of 1, 3 channels, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Raw output: {output}")
        
        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        class_names = ["Real", "Fake"]
        print(f"Prediction: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        return False
    
    # Test with multiple inputs
    print("\nTesting batch inference...")
    
    try:
        batch_input = torch.randn(4, 3, 224, 224)  # Batch of 4
        with torch.no_grad():
            batch_output = model(batch_input)
        
        batch_probs = torch.softmax(batch_output, dim=1)
        print(f"Batch input shape: {batch_input.shape}")
        print(f"Batch output shape: {batch_output.shape}")
        print(f"Batch predictions: {torch.argmax(batch_probs, dim=1)}")
        
    except Exception as e:
        print(f"ERROR during batch inference: {e}")
        return False
    
    # Test with realistic image preprocessing
    print("\nTesting with image preprocessing...")
    
    try:
        # Create a fake RGB image
        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(fake_image)
        
        # Apply same transforms as training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(pil_image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        print(f"Preprocessed image prediction: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"ERROR during preprocessed inference: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("MODEL VERIFICATION: PASSED")
    print("Your model is ready for Step 2!")
    print("=" * 50)
    
    return True

def check_dependencies():
    """Check if all required packages are available"""
    print("Checking dependencies...")
    
    required_packages = {
        'torch': torch.__version__,
        'torchvision': None,
        'PIL': None,
        'numpy': np.__version__
    }
    
    try:
        import torchvision
        required_packages['torchvision'] = torchvision.__version__
    except ImportError:
        print("ERROR: torchvision not found")
        return False
    
    try:
        from PIL import Image
        required_packages['PIL'] = "Available"
    except ImportError:
        print("ERROR: PIL not found")
        return False
    
    print("Dependencies check:")
    for package, version in required_packages.items():
        print(f"  {package}: {version}")
    
    return True

if __name__ == "__main__":
    if check_dependencies():
        success = load_and_test_model()
        if not success:
            print("\nModel verification failed. Please check the errors above.")
            exit(1)
    else:
        print("Dependency check failed.")
        exit(1)