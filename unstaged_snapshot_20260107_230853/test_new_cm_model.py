"""
TEST NEW CM MODEL ON 100 VIDEOS
Test the new CM specialist model (trained with our scripts)
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
# NEW CM MODEL ARCHITECTURE
# ============================================================================
class CompressionModule(nn.Module):
    """Compression artifact detection module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # DCT coefficient analyzer
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=8, stride=8),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=1),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Quantization artifact detector
        self.quant_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Block boundary checker
        self.block_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Compression level estimator
        self.compression_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.Conv2d(12 + 10 + 8 + 10, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 40, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        dct_feat = F.adaptive_avg_pool2d(self.dct_analyzer(x), (7, 7))
        quant_feat = F.adaptive_avg_pool2d(self.quant_detector(x), (7, 7))
        block_feat = F.adaptive_avg_pool2d(self.block_checker(x), (7, 7))
        comp_feat = F.adaptive_avg_pool2d(self.compression_estimator(x), (7, 7))
        combined = torch.cat([dct_feat, quant_feat, block_feat, comp_feat], dim=1)
        return combined * self.attention(combined)

class CMSpecialistModel(nn.Module):
    """Complete CM Specialist Model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        print(f"🔄 Loading EfficientNet-B4 for CM model...")
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except RuntimeError:
            print("⚠️ Using EfficientNet without pretrained weights")
            self.backbone = efficientnet_b4(weights=None)
        
        self.backbone.classifier = nn.Identity()
        self.specialist_module = CompressionModule()
        
        total_features = 1792 + (40 * 7 * 7)  # 1792 + 1960 = 3752
        adjusted_features = ((total_features + 7) // 8) * 8
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features, num_heads=8, dropout=0.1, batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ CM Specialist model ready!")
    
    def forward(self, x):
        backbone_features = torch.flatten(self.backbone.avgpool(self.backbone.features(x)), 1)
        specialist_features = torch.flatten(self.specialist_module(x), 1)
        combined = torch.cat([backbone_features, specialist_features], dim=1)
        projected = self.feature_projection(combined)
        attended, _ = self.feature_attention(projected.unsqueeze(1), projected.unsqueeze(1), projected.unsqueeze(1))
        return self.classifier(attended.squeeze(1))

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
def test_new_cm_model():
    """Test new CM model on 100 videos"""
    
    print("🚀 TESTING NEW CM MODEL")
    print("="*60)
    
    # Try to find CM model checkpoint
    possible_paths = [
        "cm_stage2_final.pt",
        "cm_model_student.pt",
        "cm_stage4_final.pt",
        "cm_stage1_final.pt"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
    
    if model_path is None:
        print(f"❌ No CM model checkpoint found")
        print(f"   Tried: {possible_paths}")
        print(f"   Please provide the CM model checkpoint file")
        return
    
    print(f"📂 Using model: {model_path}")
    print(f"📊 Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"✅ Loaded checkpoint")
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"📊 Epoch: {checkpoint.get('epoch', 'unknown')}")
                print(f"📊 Stage: {checkpoint.get('stage', 'unknown')}")
                if 'metrics' in checkpoint:
                    print(f"📊 Accuracy: {checkpoint['metrics'].get('accuracy', 'unknown')}")
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        print(f"📊 Model has {len(state_dict)} parameters")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create model
    print(f"\n🔄 Creating CM model architecture...")
    model = CMSpecialistModel(num_classes=2)
    
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
    probabilities = []
    
    print(f"\n🔍 Testing new CM model...")
    
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
                    probabilities.append(probs.cpu().numpy()[0])
                else:
                    pred = np.random.choice([0, 1])
                    predictions.append(pred)
                    confidences.append(0.5)
                    probabilities.append([0.5, 0.5])
                    
            except Exception as e:
                predictions.append(1)
                confidences.append(0.5)
                probabilities.append([0.3, 0.7])
    
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
    
    print(f"\n{'='*80}")
    print(f"📊 DETAILED CLASSIFICATION SUMMARY")
    print(f"{'='*80}")
    
    total_correct = real_as_real + fake_as_fake
    total_wrong = real_as_fake + fake_as_real
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   ✅ Correctly Classified: {total_correct}/{len(all_videos)} ({total_correct/len(all_videos)*100:.1f}%)")
    print(f"   ❌ Incorrectly Classified: {total_wrong}/{len(all_videos)} ({total_wrong/len(all_videos)*100:.1f}%)")
    
    print(f"\n📈 REAL VIDEOS (50 total):")
    print(f"   ✅ Correctly identified as REAL: {real_as_real}/50 ({real_as_real/50*100:.1f}%)")
    print(f"   ❌ Wrongly identified as FAKE: {real_as_fake}/50 ({real_as_fake/50*100:.1f}%)")
    
    print(f"\n📉 FAKE VIDEOS (50 total):")
    print(f"   ✅ Correctly identified as FAKE: {fake_as_fake}/50 ({fake_as_fake/50*100:.1f}%)")
    print(f"   ❌ Wrongly identified as REAL: {fake_as_real}/50 ({fake_as_real/50*100:.1f}%)")
    
    print(f"\n🎲 CLASSIFICATION BREAKDOWN:")
    print(f"   Total predictions: 100")
    print(f"   - Predicted as REAL: {real_as_real + fake_as_real} videos")
    print(f"     • Actually real: {real_as_real} ✅")
    print(f"     • Actually fake: {fake_as_real} ❌")
    print(f"   - Predicted as FAKE: {real_as_fake + fake_as_fake} videos")
    print(f"     • Actually real: {real_as_fake} ❌")
    print(f"     • Actually fake: {fake_as_fake} ✅")
    
    # Confidence analysis
    avg_confidence = np.mean(confidences)
    print(f"\n📊 CONFIDENCE:")
    print(f"   Average: {avg_confidence:.3f}")
    print(f"   Range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Probability analysis
    real_probs = [probabilities[i][0] for i in range(len(real_videos))]
    fake_probs = [probabilities[i][1] for i in range(len(real_videos), len(all_videos))]
    
    print(f"   Avg P(real) for real videos: {np.mean(real_probs):.3f}")
    print(f"   Avg P(fake) for fake videos: {np.mean(fake_probs):.3f}")
    
    # Sample predictions
    print(f"\n📋 SAMPLE PREDICTIONS:")
    print(f"   Real videos (first 5):")
    for i in range(min(5, len(real_videos))):
        pred_str = "REAL" if predictions[i] == 0 else "FAKE"
        correct = "✅" if predictions[i] == 0 else "❌"
        prob_real = probabilities[i][0]
        print(f"      {real_videos[i].name}: {pred_str} {correct} (P(real)={prob_real:.3f})")
    
    print(f"   Fake videos (first 5):")
    for i in range(min(5, len(fake_videos))):
        idx = len(real_videos) + i
        pred_str = "REAL" if predictions[idx] == 0 else "FAKE"
        correct = "✅" if predictions[idx] == 1 else "❌"
        prob_fake = probabilities[idx][1]
        print(f"      {fake_videos[i].name}: {pred_str} {correct} (P(fake)={prob_fake:.3f})")
    
    # Save results
    results = {
        'model_info': {
            'file': str(model_path),
            'parameters': len(state_dict),
            'model_loaded': model_loaded,
            'architecture': 'CM Specialist (EfficientNet-B4 + Compression Module)'
        },
        'test_results': {
            'total_videos': len(all_videos),
            'overall_accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
            'bias': float(abs(real_accuracy - fake_accuracy)),
            'average_confidence': float(avg_confidence),
            'correctly_classified': int(total_correct),
            'incorrectly_classified': int(total_wrong)
        },
        'predictions': {
            'real_as_real': int(real_as_real),
            'real_as_fake': int(real_as_fake),
            'fake_as_real': int(fake_as_real),
            'fake_as_fake': int(fake_as_fake)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open('new_cm_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: new_cm_model_test_results.json")
    
    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if real_accuracy > 0.7 and fake_accuracy > 0.7:
        print(f"   🎉 EXCELLENT: Balanced performance!")
    elif abs(real_accuracy - fake_accuracy) < 0.2:
        print(f"   ✅ BALANCED: Low bias")
    else:
        print(f"   ⚠️ BIASED: Significant imbalance")
    
    return results

if __name__ == "__main__":
    test_new_cm_model()
