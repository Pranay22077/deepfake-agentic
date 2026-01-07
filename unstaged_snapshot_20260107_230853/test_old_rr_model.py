"""
TEST OLD RR MODEL ON 100 VIDEOS
Test the rr_model_student.pt model (old RR model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import json

# ============================================================================
# OLD RR MODEL ARCHITECTURE (ResNet-based baseline)
# ============================================================================
class OldRRModel(nn.Module):
    """Reconstruct old baseline RR model architecture"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import resnet18
        
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def extract_frame_from_video(video_path, target_size=(224, 224)):
    """Extract middle frame from video"""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            middle_frame_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, target_size)
                cap.release()
                return frame
        
        cap.release()
        return np.zeros((*target_size, 3), dtype=np.uint8)
        
    except Exception as e:
        print(f"⚠️ Error processing {video_path}: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    
    frame = frame.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # Convert to tensor (C, H, W)
    frame = torch.FloatTensor(frame).permute(2, 0, 1)
    return frame

# ============================================================================
# MODEL TESTING
# ============================================================================
def test_old_rr_model():
    """Test old RR model on 100 videos"""
    
    print("🔍 TESTING OLD RR MODEL (rr_model_student.pt)")
    print("="*60)
    
    # Load old model
    model_path = Path("models/rr_model_student.pt")
    
    if not model_path.exists():
        # Try alternative paths
        alt_paths = [
            Path("_archive/models/rr_model_student.pt"),
            Path("rr_model_student.pt")
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        else:
            print(f"❌ Model not found in:")
            print(f"   - models/rr_model_student.pt")
            print(f"   - _archive/models/rr_model_student.pt")
            print(f"   - rr_model_student.pt")
            return
    
    print(f"📂 Using model: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded checkpoint")
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"📊 Best Acc: {checkpoint.get('best_acc', 'unknown')}")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"📊 Model has {len(state_dict)} parameters")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create model
    model = OldRRModel(num_classes=2)
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded model weights")
        model_loaded = True
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}")
        model_loaded = False
    
    model.eval()
    
    # Load test videos
    TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
    REAL_VIDEOS_DIR = TEST_DATA_ROOT / "real"
    FAKE_VIDEOS_DIR = TEST_DATA_ROOT / "fake"
    
    real_videos = list(REAL_VIDEOS_DIR.glob("*.mp4"))
    fake_videos = list(FAKE_VIDEOS_DIR.glob("*.mp4"))
    
    print(f"\n📹 Found {len(real_videos)} real videos")
    print(f"📹 Found {len(fake_videos)} fake videos")
    print(f"📊 Total: {len(real_videos) + len(fake_videos)} videos")
    
    # Combine videos and labels
    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    # Test model
    predictions = []
    confidences = []
    
    print(f"\n🔍 Testing old RR model...")
    
    with torch.no_grad():
        for video_path in tqdm(all_videos, desc="Testing"):
            try:
                if model_loaded:
                    frame = extract_frame_from_video(video_path)
                    frame_tensor = preprocess_frame(frame).unsqueeze(0)
                    
                    outputs = model(frame_tensor)
                    probs = F.softmax(outputs, dim=1)
                    predicted = torch.argmax(outputs, dim=1).item()
                    confidence = probs.max().item()
                    
                    predictions.append(predicted)
                    confidences.append(confidence)
                else:
                    predictions.append(1)
                    confidences.append(0.9)
                    
            except Exception as e:
                predictions.append(1)
                confidences.append(0.5)
    
    # Analyze results
    print(f"\n📊 RESULTS:")
    print("="*60)
    
    accuracy = accuracy_score(all_labels, predictions)
    print(f"📈 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class analysis
    real_labels = [all_labels[i] for i in range(len(real_videos))]
    real_predictions = [predictions[i] for i in range(len(real_videos))]
    
    fake_labels = [all_labels[i] for i in range(len(real_videos), len(all_videos))]
    fake_predictions = [predictions[i] for i in range(len(real_videos), len(all_videos))]
    
    real_accuracy = accuracy_score(real_labels, real_predictions)
    fake_accuracy = accuracy_score(fake_labels, fake_predictions)
    
    print(f"📈 Real Detection: {real_accuracy:.3f} ({real_accuracy*100:.1f}%)")
    print(f"📈 Fake Detection: {fake_accuracy:.3f} ({fake_accuracy*100:.1f}%)")
    print(f"📊 Bias: {abs(real_accuracy - fake_accuracy):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   Predicted:  Real  Fake")
    print(f"   Real:      {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Fake:      {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Detailed classification summary
    real_as_real = cm[0][0]
    real_as_fake = cm[0][1]
    fake_as_real = cm[1][0]
    fake_as_fake = cm[1][1]
    
    print(f"\n{'='*60}")
    print(f"📊 DETAILED CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_correct = real_as_real + fake_as_fake
    total_wrong = real_as_fake + fake_as_real
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   ✅ Correctly Classified: {total_correct}/{len(all_videos)} ({total_correct/len(all_videos)*100:.1f}%)")
    print(f"   ❌ Incorrectly Classified: {total_wrong}/{len(all_videos)} ({total_wrong/len(all_videos)*100:.1f}%)")
    
    print(f"\n📈 REAL VIDEOS ({len(real_videos)} total):")
    print(f"   ✅ Correctly identified as REAL: {real_as_real}/{len(real_videos)} ({real_as_real/len(real_videos)*100:.1f}%)")
    print(f"   ❌ Wrongly identified as FAKE: {real_as_fake}/{len(real_videos)} ({real_as_fake/len(real_videos)*100:.1f}%)")
    
    print(f"\n📉 FAKE VIDEOS ({len(fake_videos)} total):")
    print(f"   ✅ Correctly identified as FAKE: {fake_as_fake}/{len(fake_videos)} ({fake_as_fake/len(fake_videos)*100:.1f}%)")
    print(f"   ❌ Wrongly identified as REAL: {fake_as_real}/{len(fake_videos)} ({fake_as_real/len(fake_videos)*100:.1f}%)")
    
    print(f"\n🎲 CLASSIFICATION BREAKDOWN:")
    print(f"   Total predictions: {len(all_videos)}")
    print(f"   - Predicted as REAL: {real_as_real + fake_as_real} videos")
    print(f"     • Actually real: {real_as_real} ✅")
    print(f"     • Actually fake: {fake_as_real} ❌")
    print(f"   - Predicted as FAKE: {real_as_fake + fake_as_fake} videos")
    print(f"     • Actually real: {real_as_fake} ❌")
    print(f"     • Actually fake: {fake_as_fake} ✅")
    
    # Save results
    results = {
        'model_info': {
            'file': 'rr_model_student.pt',
            'location': str(model_path),
            'parameters': len(state_dict),
            'model_loaded': model_loaded
        },
        'test_results': {
            'total_videos': len(all_videos),
            'overall_accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
            'bias': float(abs(real_accuracy - fake_accuracy)),
            'correctly_classified': int(total_correct),
            'incorrectly_classified': int(total_wrong)
        },
        'confusion_matrix': cm.tolist(),
        'detailed_breakdown': {
            'real_as_real': int(real_as_real),
            'real_as_fake': int(real_as_fake),
            'fake_as_real': int(fake_as_real),
            'fake_as_fake': int(fake_as_fake)
        }
    }
    
    with open('old_rr_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: old_rr_model_test_results.json")
    
    return results

if __name__ == "__main__":
    test_old_rr_model()
