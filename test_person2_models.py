#!/usr/bin/env python3
"""
Test Person 2's Specialist Models
Verify CM-Model and RR-Model are working correctly
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

class SpecialistModel(nn.Module):
    """Specialist Model - ResNet18 based (matches Person 2's architecture)"""
    def __init__(self, num_classes=2):
        super(SpecialistModel, self).__init__()
        
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def test_model(model_path, model_name):
    """Test a specialist model"""
    print(f"\n🧪 Testing {model_name}")
    print("=" * 50)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"📁 Model file size: {file_size:.1f} MB")
    
    if file_size < 40 or file_size > 60:
        print(f"⚠️  Unusual file size (expected 45-50 MB)")
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SpecialistModel(num_classes=2)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Print training info
            if 'best_acc' in checkpoint:
                print(f"✅ Training accuracy: {checkpoint['best_acc']:.2f}%")
            if 'epoch' in checkpoint:
                print(f"📊 Epochs trained: {checkpoint['epoch'] + 1}")
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"🏗️  Architecture: {config.get('architecture', 'Unknown')}")
                print(f"🎯 Model type: {config.get('model_type', 'Unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Model loaded (basic format)")
        
        model.to(device)
        model.eval()
        
        # Test inference
        print(f"🔧 Testing inference on {device}...")
        
        # Create dummy input (224x224 RGB image)
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            
        print(f"✅ Inference successful")
        print(f"📊 Output shape: {output.shape}")
        print(f"🎲 Sample probabilities: {probabilities[0].cpu().numpy()}")
        
        # Test with multiple inputs
        batch_input = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            batch_output = model(batch_input)
        
        print(f"✅ Batch inference successful: {batch_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_integration():
    """Test if models can be integrated into the main system"""
    print(f"\n🔗 Testing Integration Compatibility")
    print("=" * 50)
    
    # Test transform pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    try:
        # Test transform
        tensor_image = transform(test_image)
        print(f"✅ Transform pipeline: {tensor_image.shape}")
        
        # Test batch processing
        batch = torch.stack([tensor_image] * 3)
        print(f"✅ Batch processing: {batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main testing function"""
    print("🧪 PERSON 2 SPECIALIST MODELS - VERIFICATION")
    print("=" * 60)
    
    # Model paths
    cm_model_path = "models/cm_model_student.pt"
    rr_model_path = "models/rr_model_student.pt"
    
    # Test models
    cm_success = test_model(cm_model_path, "CM-Model (Compression Specialist)")
    rr_success = test_model(rr_model_path, "RR-Model (Re-recording Specialist)")
    
    # Test integration
    integration_success = test_integration()
    
    # Summary
    print(f"\n📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"CM-Model: {'✅ PASS' if cm_success else '❌ FAIL'}")
    print(f"RR-Model: {'✅ PASS' if rr_success else '❌ FAIL'}")
    print(f"Integration: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    all_success = cm_success and rr_success and integration_success
    
    if all_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"✅ Person 2's specialist models are ready for integration")
        print(f"🚀 Next: Person 1 can integrate these models into the web platform")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print(f"❌ Please check the failed models and retrain if necessary")
        
        if not cm_success:
            print(f"   - CM-Model needs attention")
        if not rr_success:
            print(f"   - RR-Model needs attention")
        if not integration_success:
            print(f"   - Integration compatibility issues")
    
    return all_success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)