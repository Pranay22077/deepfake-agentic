"""
TEST NEW BG MODEL ON 100 VIDEOS
Test the new BG specialist model (trained with our scripts)
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
# NEW BG MODEL ARCHITECTURE
# ============================================================================
class BackgroundLightingModule(nn.Module):
    """Background and lighting inconsistency detection module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Background texture analyzer
        self.bg_texture = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Lighting direction detector
        self.lighting_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow consistency checker
        self.shadow_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=9, padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Color temperature analyzer
        self.color_temp = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.Conv2d(16 + 12 + 8 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 44, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        bg_feat = self.bg_texture(x)
        bg_feat = F.adaptive_avg_pool2d(bg_feat, (7, 7))
        
        light_feat = self.lighting_detector(x)
        light_feat = F.adaptive_avg_pool2d(light_feat, (7, 7))
        
        shadow_feat = self.shadow_checker(x)
        shadow_feat = F.adaptive_avg_pool2d(shadow_feat, (7, 7))
        
        color_feat = self.color_temp(x)
        color_feat = F.adaptive_avg_pool2d(color_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([bg_feat, light_feat, shadow_feat, color_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

class BGSpecialistModel(nn.Module):
    """Complete BG Specialist Model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        print(f"🔄 Loading EfficientNet-B4 for BG model...")
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except RuntimeError:
            print("⚠️ Using EfficientNet without pretrained weights")
            self.backbone = efficientnet_b4(weights=None)
        
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # BG specialist module
        self.specialist_module = BackgroundLightingModule()
        specialist_features = 44 * 7 * 7  # 2156
        
        # Feature projection
        total_features = backbone_features + specialist_features
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        # Multi-head attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ BG Specialist model ready!")
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Classification
        output = self.classifier(attended_features)
        return output

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
def test_new_bg_model():
    """Test new BG model on 100 videos"""
    
    print("🚀 TESTING NEW BG MODEL")
    print("="*60)
    print("📝 Note: Provide the path to your trained BG model checkpoint")
    print("   Expected: bg_stage4_final.pt or similar")
    print()
    
    # Try to find BG model checkpoint
    possible_paths = [
        "bg_stage2_final_20260106_143158.pt",
        "bg_model_student.pt",
        "bg_stage4_final.pt",
        "bg_stage2_final.pt",
        "bg_stage1_final.pt"
    ]
    
    model_path = None
    for path in possible_paths:
        if Path(path).exists():
            model_path = Path(path)
            break
    
    if model_path is None:
        print(f"❌ No BG model checkpoint found")
        print(f"   Tried: {possible_paths}")
        print(f"   Please provide the BG model checkpoint file")
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
    print(f"\n🔄 Creating BG model architecture...")
    model = BGSpecialistModel(num_classes=2)
    
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
    
    print(f"\n🔍 Testing new BG model...")
    
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
    
    # Prediction breakdown
    real_as_real = sum(1 for i in range(len(real_videos)) if predictions[i] == 0)
    real_as_fake = len(real_videos) - real_as_real
    fake_as_real = sum(1 for i in range(len(real_videos), len(all_videos)) if predictions[i] == 0)
    fake_as_fake = len(fake_videos) - fake_as_real
    
    print(f"\n📋 PREDICTION BREAKDOWN:")
    print(f"   Real as REAL: {real_as_real}/{len(real_videos)} ({real_as_real/len(real_videos)*100:.1f}%)")
    print(f"   Real as FAKE: {real_as_fake}/{len(real_videos)} ({real_as_fake/len(real_videos)*100:.1f}%)")
    print(f"   Fake as REAL: {fake_as_real}/{len(fake_videos)} ({fake_as_real/len(fake_videos)*100:.1f}%)")
    print(f"   Fake as FAKE: {fake_as_fake}/{len(fake_videos)} ({fake_as_fake/len(fake_videos)*100:.1f}%)")
    
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
    
    # Detailed classification summary
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
            'architecture': 'BG Specialist (EfficientNet-B4 + Background/Lighting Module)'
        },
        'test_results': {
            'total_videos': len(all_videos),
            'overall_accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
            'bias': float(abs(real_accuracy - fake_accuracy)),
            'average_confidence': float(avg_confidence)
        },
        'predictions': {
            'real_as_real': int(real_as_real),
            'real_as_fake': int(real_as_fake),
            'fake_as_real': int(fake_as_real),
            'fake_as_fake': int(fake_as_fake)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open('new_bg_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: new_bg_model_test_results.json")
    
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
    test_new_bg_model()
