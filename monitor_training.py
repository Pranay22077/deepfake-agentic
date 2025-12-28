#!/usr/bin/env python3
"""
Monitor the training progress and test models as they're saved
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import time
import json
from datetime import datetime

class StudentModel(nn.Module):
    """ResNet18-based student model (matches training script)"""
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
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

def load_and_test_model(model_path):
    """Load model and test its behavior"""
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = StudentModel(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get training info
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
        
        print(f"\n📊 Model from Epoch {epoch} (Best Acc: {best_acc:.2f}%)")
        
        # Test with different inputs
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_cases = [
            ("Random Noise", np.random.randint(0, 255, (224, 224, 3))),
            ("All Black", np.zeros((224, 224, 3))),
            ("All White", np.ones((224, 224, 3)) * 255),
            ("Gradient", np.array([[[i, j, (i+j)//2] for j in range(224)] for i in range(224)])),
            ("Face-like", create_face_pattern())
        ]
        
        print("Testing model behavior:")
        diverse_predictions = False
        confidence_varies = False
        
        with torch.no_grad():
            for name, image_array in test_cases:
                pil_image = Image.fromarray(image_array.astype(np.uint8))
                input_tensor = transform(pil_image).unsqueeze(0)
                
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()
                
                class_name = "Real" if pred_class == 0 else "Fake"
                
                print(f"  {name:12} | {class_name:4} | Conf: {confidence:.3f} | Raw: [{output[0][0].item():6.2f}, {output[0][1].item():6.2f}]")
                
                # Check if model is learning diversity
                if pred_class == 1:  # Found a fake prediction
                    diverse_predictions = True
                if confidence < 0.95:  # Found non-100% confidence
                    confidence_varies = True
        
        # Assessment
        status = "🟢 GOOD" if diverse_predictions and confidence_varies else "🟡 LEARNING" if diverse_predictions or confidence_varies else "🔴 BROKEN"
        print(f"Status: {status}")
        
        if not diverse_predictions:
            print("  ⚠️  Still predicting everything as Real")
        if not confidence_varies:
            print("  ⚠️  Confidence too high (overfitting)")
        
        return {
            'epoch': epoch,
            'best_acc': best_acc,
            'diverse_predictions': diverse_predictions,
            'confidence_varies': confidence_varies,
            'status': status
        }
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def create_face_pattern():
    """Create a simple face-like pattern"""
    image = np.ones((224, 224, 3), dtype=np.uint8) * 180  # Skin color
    
    # Eyes
    cv2_available = True
    try:
        import cv2
        cv2.circle(image, (80, 80), 10, (50, 50, 50), -1)
        cv2.circle(image, (144, 80), 10, (50, 50, 50), -1)
        # Mouth
        cv2.ellipse(image, (112, 150), (20, 10), 0, 0, 180, (100, 50, 50), -1)
    except ImportError:
        # Fallback without cv2
        pass
    
    return image

def monitor_training():
    """Monitor training progress"""
    print("🔍 Training Monitor Started")
    print("=" * 50)
    
    model_file = "fixed_deepfake_model.pt"
    info_file = "fixed_model_info.json"
    
    last_modified = 0
    epoch_results = []
    
    print(f"Watching for: {model_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            if os.path.exists(model_file):
                current_modified = os.path.getmtime(model_file)
                
                if current_modified > last_modified:
                    last_modified = current_modified
                    
                    print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Model updated!")
                    
                    # Test the model
                    result = load_and_test_model(model_file)
                    if result:
                        epoch_results.append(result)
                        
                        # Show progress
                        if len(epoch_results) > 1:
                            prev = epoch_results[-2]
                            curr = epoch_results[-1]
                            
                            acc_change = curr['best_acc'] - prev['best_acc']
                            if acc_change > 0:
                                print(f"📈 Accuracy improved by {acc_change:.2f}%")
                            elif acc_change < 0:
                                print(f"📉 Accuracy dropped by {abs(acc_change):.2f}%")
                            else:
                                print("📊 Accuracy unchanged")
                    
                    print("-" * 50)
            
            else:
                print(f"⏳ Waiting for {model_file} to be created...")
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped")
        
        if epoch_results:
            print(f"\n📊 Training Summary:")
            print(f"Epochs monitored: {len(epoch_results)}")
            
            best_result = max(epoch_results, key=lambda x: x['best_acc'])
            print(f"Best accuracy: {best_result['best_acc']:.2f}% (Epoch {best_result['epoch']})")
            
            final_result = epoch_results[-1]
            print(f"Final status: {final_result['status']}")
            
            if final_result['diverse_predictions'] and final_result['confidence_varies']:
                print("✅ Model appears to be learning properly!")
            else:
                print("⚠️  Model may need more training or adjustments")

def quick_test_existing():
    """Test if there's already a model file"""
    model_file = "fixed_deepfake_model.pt"
    
    if os.path.exists(model_file):
        print("Found existing model file, testing...")
        result = load_and_test_model(model_file)
        return result
    else:
        print("No existing model file found")
        return None

if __name__ == "__main__":
    print("E-Raksha Training Monitor")
    print("=" * 50)
    
    # Check for existing model first
    existing_result = quick_test_existing()
    
    if existing_result:
        print(f"\nCurrent model status: {existing_result['status']}")
        
        if input("\nContinue monitoring for updates? (y/n): ").lower().startswith('y'):
            monitor_training()
    else:
        monitor_training()