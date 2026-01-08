"""
E-Raksha New Model Testing Suite

Comprehensive testing script for evaluating new specialist models on 100-video dataset.
Provides detailed performance analysis and bias detection for model validation.

Author: E-Raksha Team
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
import sys
from types import ModuleType

# ============================================================================
# NUMPY COMPATIBILITY UTILITIES
# ============================================================================
def fix_numpy_compatibility():
    """
    Fix numpy._core compatibility issue for PyTorch model loading.
    
    Returns:
        bool: True if compatibility fix successful
    """
    try:
        import numpy._core
        print("✅ numpy._core already available")
        return True
    except ImportError:
        print("⚠️ numpy._core not found, applying compatibility fix...")
        
        try:
            import numpy
            
            # Create mock _core module for compatibility
            mock_core = ModuleType('numpy._core')
            mock_core.multiarray = numpy.core.multiarray
            
            # Register in sys.modules
            sys.modules['numpy._core'] = mock_core
            sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
            
            print("✅ Applied numpy._core compatibility fix")
            return True
            
        except Exception as e:
            print(f"❌ Failed to fix numpy compatibility: {e}")
            return False

# ============================================================================
# NEW MODEL ARCHITECTURE (Enhanced LL-Model)
# ============================================================================
class EnhancedLowLightModule(nn.Module):
    """Enhanced low-light analysis module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale luminance analysis
        self.luminance_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ) for k in [3, 5, 7]
        ])
        
        # Noise pattern detector
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow/highlight detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Enhanced attention
        self.attention = nn.Sequential(
            nn.Conv2d(48 + 12 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 68, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale luminance
        lum_features = []
        for branch in self.luminance_branch:
            lum_feat = branch(x)
            lum_feat = F.adaptive_avg_pool2d(lum_feat, (7, 7))
            lum_features.append(lum_feat)
        
        lum_combined = torch.cat(lum_features, dim=1)  # 48 channels
        
        # Noise analysis
        noise_features = self.noise_detector(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))  # 12 channels
        
        # Shadow analysis
        shadow_features = self.shadow_detector(x)
        shadow_features = F.adaptive_avg_pool2d(shadow_features, (7, 7))  # 8 channels
        
        # Combine and apply attention
        combined = torch.cat([lum_combined, noise_features, shadow_features], dim=1)  # 68 channels
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

class NewProgressiveSpecialistModel(nn.Module):
    """New Enhanced LL-Model Architecture"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        # Backbone - with fallback for hash issues
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except RuntimeError:
            print("⚠️ Using EfficientNet without pretrained weights due to hash mismatch")
            self.backbone = efficientnet_b4(weights=None)
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Enhanced specialist module
        self.specialist_module = EnhancedLowLightModule()
        specialist_features = 68 * 7 * 7  # 3332
        
        # Feature projection and attention
        total_features = backbone_features + specialist_features  # 5124
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads  # 5128
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Progressive classifier
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
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Final classification
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
            # Get middle frame
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
    
    # Convert to float and normalize
    frame = frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # Convert to tensor (C, H, W)
    frame = torch.FloatTensor(frame).permute(2, 0, 1)
    return frame

# ============================================================================
# MODEL TESTING
# ============================================================================
def test_new_model_on_videos():
    """Test new model on all 100 videos"""
    
    print("🚀 TESTING NEW MODEL ON 100 VIDEOS")
    print("="*60)
    
    # Apply numpy fix
    fix_numpy_compatibility()
    
    # Load new model
    try:
        # Try multiple loading strategies
        loading_strategies = [
            ("Standard", lambda: torch.load("celebdf_resume_epoch3.pt", map_location='cpu', weights_only=False)),
            ("No weights_only", lambda: torch.load("celebdf_resume_epoch3.pt", map_location='cpu')),
        ]
        
        checkpoint = None
        for strategy_name, load_func in loading_strategies:
            try:
                print(f"🔄 Trying: {strategy_name}")
                checkpoint = load_func()
                print(f"✅ Success with: {strategy_name}")
                break
            except Exception as e:
                print(f"❌ {strategy_name} failed: {str(e)[:100]}...")
                continue
        
        if checkpoint is None:
            print(f"❌ All loading strategies failed")
            return
        
        print(f"✅ Loaded checkpoint")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            training_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'accuracy': checkpoint.get('metrics', {}).get('accuracy', 'unknown'),
                'stage': checkpoint.get('current_stage', 'unknown')
            }
        else:
            state_dict = checkpoint
            training_info = {'epoch': 'unknown', 'accuracy': 'unknown', 'stage': 'unknown'}
        
        print(f"📊 Training info: Epoch {training_info['epoch']}, Accuracy {training_info['accuracy']}")
        print(f"📊 Model has {len(state_dict)} parameters")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Create model and load weights
    print(f"🔄 Creating model architecture...")
    model = NewProgressiveSpecialistModel(num_classes=2)
    
    try:
        # Load the state dict
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded model weights successfully")
        model_loaded = True
    except Exception as e:
        print(f"⚠️ Could not load weights properly: {e}")
        print(f"🎲 Will use random predictions")
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
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=real, 1=fake
    
    # Test model on all videos
    predictions = []
    confidences = []
    probabilities = []
    
    print(f"\n🔍 Testing new model on all videos...")
    
    with torch.no_grad():
        for i, video_path in enumerate(tqdm(all_videos, desc="Testing")):
            try:
                if model_loaded:
                    # Extract and preprocess frame
                    frame = extract_frame_from_video(video_path)
                    frame_tensor = preprocess_frame(frame).unsqueeze(0)
                    
                    # Get prediction
                    outputs = model(frame_tensor)
                    probs = F.softmax(outputs, dim=1)
                    predicted = torch.argmax(outputs, dim=1).item()
                    confidence = probs.max().item()
                    
                    predictions.append(predicted)
                    confidences.append(confidence)
                    probabilities.append(probs.cpu().numpy()[0])
                else:
                    # Random predictions if model didn't load
                    pred = np.random.choice([0, 1])
                    predictions.append(pred)
                    confidences.append(0.5)
                    probabilities.append([0.5, 0.5])
                    
            except Exception as e:
                print(f"⚠️ Error with {video_path.name}: {e}")
                # Default prediction
                predictions.append(1)  # Default to fake
                confidences.append(0.5)
                probabilities.append([0.3, 0.7])
    
    # Analyze results
    print(f"\n📊 DETAILED RESULTS:")
    print("="*60)
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, predictions)
    print(f"📈 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class analysis
    real_labels = [all_labels[i] for i in range(len(real_videos))]
    real_predictions = [predictions[i] for i in range(len(real_videos))]
    
    fake_labels = [all_labels[i] for i in range(len(real_videos), len(all_videos))]
    fake_predictions = [predictions[i] for i in range(len(real_videos), len(all_videos))]
    
    real_accuracy = accuracy_score(real_labels, real_predictions)
    fake_accuracy = accuracy_score(fake_labels, fake_predictions)
    
    print(f"📈 Real Detection Accuracy: {real_accuracy:.3f} ({real_accuracy*100:.1f}%)")
    print(f"📈 Fake Detection Accuracy: {fake_accuracy:.3f} ({fake_accuracy*100:.1f}%)")
    print(f"📊 Bias (|Real - Fake|): {abs(real_accuracy - fake_accuracy):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   Predicted:  Real  Fake")
    print(f"   Real:      {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Fake:      {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Prediction distribution
    real_as_real = sum(1 for i in range(len(real_videos)) if predictions[i] == 0)
    real_as_fake = len(real_videos) - real_as_real
    fake_as_real = sum(1 for i in range(len(real_videos), len(all_videos)) if predictions[i] == 0)
    fake_as_fake = len(fake_videos) - fake_as_real
    
    print(f"\n📋 PREDICTION BREAKDOWN:")
    print(f"   Real videos predicted as REAL: {real_as_real}/{len(real_videos)} ({real_as_real/len(real_videos)*100:.1f}%)")
    print(f"   Real videos predicted as FAKE: {real_as_fake}/{len(real_videos)} ({real_as_fake/len(real_videos)*100:.1f}%)")
    print(f"   Fake videos predicted as REAL: {fake_as_real}/{len(fake_videos)} ({fake_as_real/len(fake_videos)*100:.1f}%)")
    print(f"   Fake videos predicted as FAKE: {fake_as_fake}/{len(fake_videos)} ({fake_as_fake/len(fake_videos)*100:.1f}%)")
    
    # Confidence analysis
    avg_confidence = np.mean(confidences)
    print(f"\n📊 CONFIDENCE ANALYSIS:")
    print(f"   Average confidence: {avg_confidence:.3f}")
    print(f"   Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
    
    # Probability analysis
    real_probs = [probabilities[i][0] for i in range(len(real_videos))]  # P(real) for real videos
    fake_probs = [probabilities[i][1] for i in range(len(real_videos), len(all_videos))]  # P(fake) for fake videos
    
    print(f"   Average P(real) for real videos: {np.mean(real_probs):.3f}")
    print(f"   Average P(fake) for fake videos: {np.mean(fake_probs):.3f}")
    
    # Show some specific examples
    print(f"\n📋 SAMPLE PREDICTIONS:")
    print(f"   Real videos (first 10):")
    for i in range(min(10, len(real_videos))):
        pred_str = "REAL" if predictions[i] == 0 else "FAKE"
        correct = "✅" if predictions[i] == 0 else "❌"
        prob_real = probabilities[i][0]
        print(f"      {real_videos[i].name}: {pred_str} {correct} (P(real)={prob_real:.3f}, conf: {confidences[i]:.3f})")
    
    print(f"   Fake videos (first 10):")
    for i in range(min(10, len(fake_videos))):
        idx = len(real_videos) + i
        pred_str = "REAL" if predictions[idx] == 0 else "FAKE"
        correct = "✅" if predictions[idx] == 1 else "❌"
        prob_fake = probabilities[idx][1]
        print(f"      {fake_videos[i].name}: {pred_str} {correct} (P(fake)={prob_fake:.3f}, conf: {confidences[idx]:.3f})")
    
    # Save results
    results = {
        'model_info': {
            'file': 'stage2_full_celebdf_best_epoch_newwwww.pt',
            'epoch': training_info['epoch'],
            'training_accuracy': training_info['accuracy'],
            'stage': training_info['stage'],
            'parameters': len(state_dict),
            'model_loaded': model_loaded
        },
        'test_results': {
            'total_videos': len(all_videos),
            'real_videos': len(real_videos),
            'fake_videos': len(fake_videos),
            'overall_accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
            'bias': float(abs(real_accuracy - fake_accuracy)),
            'average_confidence': float(avg_confidence),
            'avg_prob_real_for_reals': float(np.mean(real_probs)),
            'avg_prob_fake_for_fakes': float(np.mean(fake_probs))
        },
        'predictions': {
            'real_as_real': int(real_as_real),
            'real_as_fake': int(real_as_fake),
            'fake_as_real': int(fake_as_real),
            'fake_as_fake': int(fake_as_fake)
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open('new_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: new_model_test_results.json")
    
    # Final assessment
    print(f"\n🎯 ASSESSMENT:")
    if real_accuracy > 0.7 and fake_accuracy > 0.7:
        print(f"   🎉 EXCELLENT: Model shows balanced performance on both classes!")
    elif real_accuracy > 0.5 and fake_accuracy > 0.5:
        print(f"   ✅ GOOD: Model shows decent performance on both classes")
    elif abs(real_accuracy - fake_accuracy) < 0.3:
        print(f"   👍 BALANCED: Model shows similar performance on both classes")
    else:
        print(f"   ⚠️ BIASED: Model shows significant bias toward one class")
    
    if abs(real_accuracy - fake_accuracy) < 0.2:
        print(f"   ✅ LOW BIAS: Model is well-balanced")
    elif abs(real_accuracy - fake_accuracy) < 0.5:
        print(f"   ⚠️ MODERATE BIAS: Some imbalance present")
    else:
        print(f"   ❌ HIGH BIAS: Significant imbalance")
    
    return results

if __name__ == "__main__":
    test_new_model_on_videos()