#!/usr/bin/env python3
"""
Debug the FIXED model that's currently being used by the backend
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class StudentModel(nn.Module):
    """Fixed ResNet18-based model (matches backend)"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Use pretrained ResNet18 for better performance
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace final layer (matches fixed training)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def debug_current_model():
    """Debug the model currently being used by the backend"""
    print("🔍 Debugging CURRENT Fixed Model")
    print("=" * 60)
    
    model_path = "fixed_deepfake_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Fixed model not found: {model_path}")
        return
    
    # Load model
    model = StudentModel(num_classes=2)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best Accuracy: {checkpoint.get('best_acc', 'Unknown')}")
    else:
        print("❌ No model_state_dict found")
        return
    
    model.eval()
    
    # Test with the EXACT same preprocessing as the backend
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n🧪 Testing Model Behavior (Same as Backend)")
    print("=" * 60)
    
    test_cases = [
        ("Random Video Frame", np.random.randint(0, 255, (224, 224, 3))),
        ("Natural Face", create_natural_face()),
        ("Artificial Face", create_artificial_face()),
        ("All Black", np.zeros((224, 224, 3))),
        ("All White", np.ones((224, 224, 3)) * 255),
        ("Gradient", create_gradient_image())
    ]
    
    with torch.no_grad():
        for name, image_array in test_cases:
            # Convert to PIL and apply transform (EXACT backend process)
            pil_image = Image.fromarray(image_array.astype(np.uint8))
            input_tensor = transform(pil_image).unsqueeze(0)
            
            # Get prediction (EXACT backend process)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            class_name = "Real" if pred_class == 0 else "Fake"
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
            
            print(f"{name:18} | {class_name:4} | Conf: {confidence:.3f} | "
                  f"Real: {real_prob:.3f} | Fake: {fake_prob:.3f}")
            
            # Check if this is the problematic behavior
            if confidence >= 0.99:
                print(f"  ⚠️  HIGH CONFIDENCE DETECTED: {confidence:.3f}")
            if fake_prob > 0.9:
                print(f"  ⚠️  STRONG FAKE BIAS: {fake_prob:.3f}")

def create_natural_face():
    """Create a natural-looking face"""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Natural skin color
    image[:, :] = [200, 170, 140]
    
    # Add natural variations
    for y in range(224):
        for x in range(224):
            noise = np.random.normal(0, 10)
            image[y, x] = np.clip(image[y, x] + noise, 0, 255)
    
    return image

def create_artificial_face():
    """Create an artificial-looking face"""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Artificial patterns
    for y in range(224):
        for x in range(224):
            r = int(128 + 127 * np.sin(x * 0.1))
            g = int(128 + 127 * np.cos(y * 0.1))
            b = int(128 + 127 * np.sin((x + y) * 0.05))
            image[y, x] = [r, g, b]
    
    return image

def create_gradient_image():
    """Create a gradient image"""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    for y in range(224):
        for x in range(224):
            image[y, x] = [x, y, (x + y) // 2]
    
    return image

def analyze_training_data_bias():
    """Check if the training data was biased"""
    print("\n🔍 Analyzing Training Data Bias")
    print("=" * 60)
    
    info_file = "fixed_model_info.json"
    if os.path.exists(info_file):
        import json
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        if 'training_history' in info:
            history = info['training_history']
            if history:
                last_epoch = history[-1]
                train_acc = last_epoch.get('train_acc', 0)
                val_acc = last_epoch.get('val_acc', 0)
                
                print(f"Training Accuracy: {train_acc:.2f}%")
                print(f"Validation Accuracy: {val_acc:.2f}%")
                
                if train_acc > 95 and val_acc > 95:
                    print("⚠️  OVERFITTING DETECTED: Both accuracies too high")
                    print("   This suggests the synthetic data was too easy to distinguish")
                    print("   The model learned to classify based on artificial patterns")
                
                if abs(train_acc - val_acc) > 10:
                    print("⚠️  OVERFITTING: Large gap between train and validation")
    else:
        print("No training info available")

def check_model_weights():
    """Check if model weights are reasonable"""
    print("\n🔍 Checking Model Weights")
    print("=" * 60)
    
    model_path = "fixed_deepfake_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        
        # Check final layer weights
        final_layer_weight = None
        final_layer_bias = None
        
        for name, param in state_dict.items():
            if 'fc.5.weight' in name:  # Final linear layer
                final_layer_weight = param
            elif 'fc.5.bias' in name:
                final_layer_bias = param
        
        if final_layer_weight is not None and final_layer_bias is not None:
            print(f"Final layer weight shape: {final_layer_weight.shape}")
            print(f"Final layer bias: {final_layer_bias}")
            
            # Check if biased toward fake
            bias_diff = final_layer_bias[1] - final_layer_bias[0]  # fake - real
            print(f"Bias difference (fake - real): {bias_diff:.3f}")
            
            if bias_diff > 2:
                print("⚠️  STRONG FAKE BIAS: Model is biased toward predicting fake")
            elif bias_diff < -2:
                print("⚠️  STRONG REAL BIAS: Model is biased toward predicting real")
            else:
                print("✅ Bias seems reasonable")

if __name__ == "__main__":
    debug_current_model()
    analyze_training_data_bias()
    check_model_weights()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("The model is likely overfitted on synthetic training data.")
    print("It learned to distinguish artificial patterns rather than real deepfakes.")
    print("Solution: Train on REAL deepfake data (DFDC dataset on Kaggle).")