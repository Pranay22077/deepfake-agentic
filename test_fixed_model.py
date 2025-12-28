#!/usr/bin/env python3
"""
Test the newly trained fixed model to see if it's working properly
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class StudentModel(nn.Module):
    """Fixed ResNet18-based model"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Use pretrained ResNet18
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace final layer
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

def load_fixed_model():
    """Load the newly trained fixed model"""
    model_path = "fixed_deepfake_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"Loading fixed model from: {model_path}")
    
    # Create model
    model = StudentModel(num_classes=2)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded successfully!")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best Accuracy: {checkpoint.get('best_acc', 'Unknown')}")
    else:
        print("❌ No model_state_dict found in checkpoint")
        return None
    
    model.eval()
    return model, checkpoint

def test_model_behavior(model):
    """Test model behavior with different inputs"""
    print("\n" + "="*60)
    print("TESTING FIXED MODEL BEHAVIOR")
    print("="*60)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test cases
    test_cases = [
        ("Random Noise", np.random.randint(0, 255, (224, 224, 3))),
        ("All Zeros", np.zeros((224, 224, 3))),
        ("All Ones", np.ones((224, 224, 3)) * 255),
        ("Gradient", np.array([[[i, j, (i+j)//2] for j in range(224)] for i in range(224)])),
        ("Face-like Pattern", create_face_pattern()),
        ("Artificial Pattern", create_artificial_pattern())
    ]
    
    with torch.no_grad():
        for name, image_array in test_cases:
            # Convert to PIL and apply transform
            pil_image = Image.fromarray(image_array.astype(np.uint8))
            input_tensor = transform(pil_image).unsqueeze(0)
            
            # Get prediction
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            class_name = "Real" if pred_class == 0 else "Fake"
            real_prob = probabilities[0][0].item()
            fake_prob = probabilities[0][1].item()
            
            print(f"{name:18} | {class_name:4} | Conf: {confidence:.3f} | "
                  f"Real: {real_prob:.3f} | Fake: {fake_prob:.3f} | "
                  f"Raw: [{output[0][0].item():6.2f}, {output[0][1].item():6.2f}]")

def create_face_pattern():
    """Create a face-like pattern"""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Skin color background
    image[:, :] = [180, 150, 120]
    
    # Face shape (oval)
    center_x, center_y = 112, 112
    for y in range(224):
        for x in range(224):
            # Oval equation
            if ((x - center_x) / 80) ** 2 + ((y - center_y) / 100) ** 2 < 1:
                image[y, x] = [200, 170, 140]
    
    # Eyes
    cv2.circle(image, (center_x - 30, center_y - 20), 8, (50, 50, 50), -1)
    cv2.circle(image, (center_x + 30, center_y - 20), 8, (50, 50, 50), -1)
    
    # Nose
    cv2.line(image, (center_x, center_y - 10), (center_x, center_y + 10), (160, 130, 100), 2)
    
    # Mouth
    cv2.ellipse(image, (center_x, center_y + 30), (20, 8), 0, 0, 180, (120, 80, 80), -1)
    
    return image

def create_artificial_pattern():
    """Create an artificial/fake-looking pattern"""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Artificial gradient background
    for y in range(224):
        for x in range(224):
            r = int(128 + 127 * np.sin(x * 0.05))
            g = int(128 + 127 * np.cos(y * 0.05))
            b = int(128 + 127 * np.sin((x + y) * 0.03))
            image[y, x] = [r, g, b]
    
    # Add some artificial features
    center_x, center_y = 112, 112
    
    # Artificial "eyes" with different sizes
    cv2.circle(image, (center_x - 25, center_y - 15), 12, (255, 255, 255), -1)
    cv2.circle(image, (center_x + 35, center_y - 25), 8, (255, 255, 255), -1)
    
    # Artificial "mouth"
    cv2.rectangle(image, (center_x - 15, center_y + 25), (center_x + 15, center_y + 35), (0, 0, 0), -1)
    
    return image

def compare_with_broken_model():
    """Compare with the broken model if available"""
    print("\n" + "="*60)
    print("COMPARING WITH BROKEN MODEL")
    print("="*60)
    
    broken_model_path = "./kaggle_outputs_20251228_043850/baseline_student.pkl"
    
    if not os.path.exists(broken_model_path):
        print("❌ Broken model not found for comparison")
        return
    
    # Load broken model
    import pickle
    
    class BrokenStudentModel(nn.Module):
        def __init__(self, num_classes=2):
            super(BrokenStudentModel, self).__init__()
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
    
    broken_model = BrokenStudentModel(num_classes=2)
    
    # Load broken model data
    with open(broken_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    state_dict = {}
    for name, param_array in model_data.items():
        if isinstance(param_array, np.ndarray):
            state_dict[name] = torch.from_numpy(param_array)
    
    broken_model.load_state_dict(state_dict, strict=False)
    broken_model.eval()
    
    # Test both models
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load fixed model
    fixed_model, _ = load_fixed_model()
    if fixed_model is None:
        return
    
    test_image = np.random.randint(0, 255, (224, 224, 3))
    pil_image = Image.fromarray(test_image.astype(np.uint8))
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        # Broken model prediction
        broken_output = broken_model(input_tensor)
        broken_probs = torch.softmax(broken_output, dim=1)
        broken_pred = torch.argmax(broken_probs, dim=1).item()
        broken_conf = broken_probs[0][broken_pred].item()
        
        # Fixed model prediction
        fixed_output = fixed_model(input_tensor)
        fixed_probs = torch.softmax(fixed_output, dim=1)
        fixed_pred = torch.argmax(fixed_probs, dim=1).item()
        fixed_conf = fixed_probs[0][fixed_pred].item()
        
        print(f"Broken Model | {'Real' if broken_pred == 0 else 'Fake':4} | Conf: {broken_conf:.3f} | Raw: [{broken_output[0][0].item():6.2f}, {broken_output[0][1].item():6.2f}]")
        print(f"Fixed Model  | {'Real' if fixed_pred == 0 else 'Fake':4} | Conf: {fixed_conf:.3f} | Raw: [{fixed_output[0][0].item():6.2f}, {fixed_output[0][1].item():6.2f}]")
        
        if broken_conf == 1.0 and fixed_conf < 1.0:
            print("✅ IMPROVEMENT: Fixed model shows variable confidence (not always 100%)")
        else:
            print("⚠️  Both models showing similar behavior")

def check_training_progress():
    """Check if training info is available"""
    info_file = "fixed_model_info.json"
    if os.path.exists(info_file):
        import json
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print("\n" + "="*60)
        print("TRAINING PROGRESS")
        print("="*60)
        print(f"Best Accuracy: {info.get('best_accuracy', 'Unknown'):.2f}%")
        print(f"Architecture: {info.get('architecture', 'Unknown')}")
        print(f"Parameters: {info.get('parameters', 'Unknown'):,}")
        
        if 'training_history' in info:
            history = info['training_history']
            print(f"Epochs completed: {len(history)}")
            if history:
                last_epoch = history[-1]
                print(f"Last epoch - Train Acc: {last_epoch.get('train_acc', 0):.2f}%, Val Acc: {last_epoch.get('val_acc', 0):.2f}%")

def main():
    print("🔧 Testing Fixed Deepfake Detection Model")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists("fixed_deepfake_model.pt"):
        print("❌ Fixed model not found. Make sure training is running.")
        return
    
    # Load and test model
    result = load_fixed_model()
    if result is None:
        return
    
    model, checkpoint = result
    
    # Test model behavior
    test_model_behavior(model)
    
    # Compare with broken model
    compare_with_broken_model()
    
    # Check training progress
    check_training_progress()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Quick test to see if model is working
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_image = np.random.randint(0, 255, (224, 224, 3))
    pil_image = Image.fromarray(test_image.astype(np.uint8))
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence = probabilities[0].max().item()
    
    if confidence < 1.0:
        print("✅ Model appears to be working (confidence < 100%)")
        print("✅ Ready to replace broken model in backend")
    else:
        print("⚠️  Model still showing 100% confidence - may need more training")
    
    print(f"Model file size: {os.path.getsize('fixed_deepfake_model.pt') / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    try:
        import cv2
    except ImportError:
        print("Note: OpenCV not available, skipping some visual tests")
    
    main()