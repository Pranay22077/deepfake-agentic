"""
INTERCEPTOR MODEL COMPARISON: OLD vs NEW
Compare the old LL-Model with the new enhanced Stage 2 model
Test on real and fake videos from _archive/test-files/
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

# OLD MODEL ARCHITECTURE (Original LL-Model)
class OldLowLightModule(nn.Module):
    """Original Low-Light Analysis Module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Simple luminance analysis
        self.luminance_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Basic noise detector
        self.noise_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        # Simple attention
        self.attention = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Luminance features
        lum_features = self.luminance_conv(x)
        lum_features = F.adaptive_avg_pool2d(lum_features, (7, 7))
        
        # Noise features
        noise_features = self.noise_conv(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([lum_features, noise_features], dim=1)  # 24 channels
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

class OldLLModel(nn.Module):
    """Original LL-Model Architecture"""
    
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
        
        # Old specialist module
        self.specialist_module = OldLowLightModule()
        specialist_features = 24 * 7 * 7  # 1176
        
        # Simple classifier
        total_features = backbone_features + specialist_features  # 2968
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
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
        
        # Combine and classify
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        output = self.classifier(combined_features)
        return output

# NEW MODEL ARCHITECTURE (Enhanced LL-Model)
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
# VIDEO PROCESSING UTILITIES
# ============================================================================

def extract_frames_from_video(video_path, num_frames=8, target_size=(224, 224)):
    """Extract frames from video for model input"""
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            # Select frames uniformly
            indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, target_size)
                    frames.append(frame)
        
        cap.release()
        
        # Ensure we have enough frames
        while len(frames) < num_frames:
            frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.uint8))
        
        return np.array(frames[:num_frames])
        
    except Exception as e:
        print(f"⚠️ Error processing {video_path}: {e}")
        return np.zeros((num_frames, *target_size, 3), dtype=np.uint8)

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = (frame - mean) / std
    
    # Convert to tensor format (C, H, W)
    frame = torch.FloatTensor(frame).permute(2, 0, 1)
    
    return frame

# ============================================================================
# MODEL LOADING AND EVALUATION
# ============================================================================

def load_model_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint"""
    
    print(f"📦 Loading checkpoint: {Path(checkpoint_path).name}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict")
            
            # Extract training info if available
            training_info = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'accuracy': checkpoint.get('metrics', {}).get('accuracy', 'unknown'),
                'stage': checkpoint.get('current_stage', 'unknown')
            }
            return training_info
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
            return {'epoch': 'unknown', 'accuracy': 'unknown', 'stage': 'unknown'}
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None

def evaluate_model_on_videos(model, video_paths, labels, device, model_name):
    """Evaluate model on a set of videos"""
    
    model.eval()
    predictions = []
    probabilities = []
    
    print(f"\n🔍 Evaluating {model_name} on {len(video_paths)} videos...")
    
    with torch.no_grad():
        for video_path in tqdm(video_paths, desc=f"Testing {model_name}"):
            # Extract frames
            frames = extract_frames_from_video(video_path)
            
            # Use middle frame for prediction
            middle_frame = frames[len(frames)//2]
            frame_tensor = preprocess_frame(middle_frame).unsqueeze(0).to(device)
            
            # Get prediction
            try:
                outputs = model(frame_tensor)
                probabilities_batch = F.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, dim=1)
                
                predictions.append(predicted.cpu().item())
                probabilities.append(probabilities_batch.cpu().numpy()[0])
                
            except Exception as e:
                print(f"⚠️ Error predicting {video_path}: {e}")
                predictions.append(1)  # Default to fake
                probabilities.append([0.0, 1.0])
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    # Per-class accuracy
    real_mask = np.array(labels) == 0
    fake_mask = np.array(labels) == 1
    
    real_accuracy = accuracy_score(np.array(labels)[real_mask], np.array(predictions)[real_mask]) if real_mask.sum() > 0 else 0
    fake_accuracy = accuracy_score(np.array(labels)[fake_mask], np.array(predictions)[fake_mask]) if fake_mask.sum() > 0 else 0
    
    # AUC-ROC
    try:
        probs_fake = [p[1] for p in probabilities]  # Probability of fake
        auc_roc = roc_auc_score(labels, probs_fake)
    except:
        auc_roc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'predictions': predictions,
        'probabilities': probabilities,
        'confusion_matrix': cm,
        'bias': abs(real_accuracy - fake_accuracy)
    }
    
    return results

# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

def create_comparison_visualizations(old_results, new_results, video_paths, labels):
    """Create comprehensive comparison visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Overall Metrics Comparison
    ax1 = plt.subplot(2, 4, 1)
    metrics = ['Accuracy', 'Real Acc', 'Fake Acc', 'F1-Score', 'AUC-ROC']
    old_values = [old_results['accuracy'], old_results['real_accuracy'], 
                  old_results['fake_accuracy'], old_results['f1_score'], old_results['auc_roc']]
    new_values = [new_results['accuracy'], new_results['real_accuracy'], 
                  new_results['fake_accuracy'], new_results['f1_score'], new_results['auc_roc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, old_values, width, label='Old Model', alpha=0.8, color='lightcoral')
    bars2 = ax1.bar(x + width/2, new_values, width, label='New Model', alpha=0.8, color='lightblue')
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # 2. Confusion Matrix - Old Model
    ax2 = plt.subplot(2, 4, 2)
    sns.heatmap(old_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax2)
    ax2.set_title('Old Model - Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Confusion Matrix - New Model
    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(new_results['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'], ax=ax3)
    ax3.set_title('New Model - Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # 4. Bias Comparison
    ax4 = plt.subplot(2, 4, 4)
    bias_data = [old_results['bias'], new_results['bias']]
    colors = ['lightcoral', 'lightblue']
    bars = ax4.bar(['Old Model', 'New Model'], bias_data, color=colors, alpha=0.8)
    ax4.set_ylabel('Bias (|Real Acc - Fake Acc|)')
    ax4.set_title('Model Bias Comparison')
    ax4.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, bias_data):
        ax4.annotate(f'{value:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # 5. Prediction Confidence Distribution - Old Model
    ax5 = plt.subplot(2, 4, 5)
    old_fake_probs = [p[1] for p in old_results['probabilities']]
    ax5.hist(old_fake_probs, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    ax5.set_xlabel('Fake Probability')
    ax5.set_ylabel('Count')
    ax5.set_title('Old Model - Prediction Confidence')
    ax5.grid(True, alpha=0.3)
    
    # 6. Prediction Confidence Distribution - New Model
    ax6 = plt.subplot(2, 4, 6)
    new_fake_probs = [p[1] for p in new_results['probabilities']]
    ax6.hist(new_fake_probs, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax6.set_xlabel('Fake Probability')
    ax6.set_ylabel('Count')
    ax6.set_title('New Model - Prediction Confidence')
    ax6.grid(True, alpha=0.3)
    
    # 7. Per-Video Comparison (sample)
    ax7 = plt.subplot(2, 4, 7)
    sample_indices = np.random.choice(len(video_paths), min(20, len(video_paths)), replace=False)
    sample_old_probs = [old_fake_probs[i] for i in sample_indices]
    sample_new_probs = [new_fake_probs[i] for i in sample_indices]
    
    ax7.scatter(sample_old_probs, sample_new_probs, alpha=0.6)
    ax7.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax7.set_xlabel('Old Model Fake Probability')
    ax7.set_ylabel('New Model Fake Probability')
    ax7.set_title('Per-Video Prediction Comparison')
    ax7.grid(True, alpha=0.3)
    
    # 8. Improvement Summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Calculate improvements
    acc_improvement = (new_results['accuracy'] - old_results['accuracy']) * 100
    real_improvement = (new_results['real_accuracy'] - old_results['real_accuracy']) * 100
    fake_improvement = (new_results['fake_accuracy'] - old_results['fake_accuracy']) * 100
    bias_improvement = (old_results['bias'] - new_results['bias']) * 100
    
    summary_text = f"""
    IMPROVEMENT SUMMARY
    
    Overall Accuracy: {acc_improvement:+.2f}%
    Real Detection: {real_improvement:+.2f}%
    Fake Detection: {fake_improvement:+.2f}%
    Bias Reduction: {bias_improvement:+.2f}%
    
    OLD MODEL:
    • Accuracy: {old_results['accuracy']:.3f}
    • Real: {old_results['real_accuracy']:.3f}
    • Fake: {old_results['fake_accuracy']:.3f}
    • Bias: {old_results['bias']:.3f}
    
    NEW MODEL:
    • Accuracy: {new_results['accuracy']:.3f}
    • Real: {new_results['real_accuracy']:.3f}
    • Fake: {new_results['fake_accuracy']:.3f}
    • Bias: {new_results['bias']:.3f}
    """
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_detailed_report(old_results, new_results, video_paths, labels):
    """Generate detailed comparison report"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_dataset': {
            'total_videos': len(video_paths),
            'real_videos': sum(1 for l in labels if l == 0),
            'fake_videos': sum(1 for l in labels if l == 1),
            'test_path': '_archive/test-files/test-data/test-data/raw/'
        },
        'old_model': {
            'name': 'Original LL-Model',
            'checkpoint': 'll_model_student (1).pt',
            'architecture': 'EfficientNet-B4 + Simple Low-Light Module',
            'parameters': '~30M',
            'metrics': {
                'accuracy': float(old_results['accuracy']),
                'real_accuracy': float(old_results['real_accuracy']),
                'fake_accuracy': float(old_results['fake_accuracy']),
                'f1_score': float(old_results['f1_score']),
                'auc_roc': float(old_results['auc_roc']),
                'bias': float(old_results['bias'])
            }
        },
        'new_model': {
            'name': 'Enhanced LL-Model (Stage 2)',
            'checkpoint': 'stage2_full_celebdf_best_epoch3.pt',
            'architecture': 'EfficientNet-B4 + Enhanced Low-Light Module + Multi-Head Attention',
            'parameters': '~47M',
            'training': 'Progressive multi-dataset (FaceForensics++ + Full Celeb-DF)',
            'metrics': {
                'accuracy': float(new_results['accuracy']),
                'real_accuracy': float(new_results['real_accuracy']),
                'fake_accuracy': float(new_results['fake_accuracy']),
                'f1_score': float(new_results['f1_score']),
                'auc_roc': float(new_results['auc_roc']),
                'bias': float(new_results['bias'])
            }
        },
        'improvements': {
            'accuracy_improvement': float((new_results['accuracy'] - old_results['accuracy']) * 100),
            'real_detection_improvement': float((new_results['real_accuracy'] - old_results['real_accuracy']) * 100),
            'fake_detection_change': float((new_results['fake_accuracy'] - old_results['fake_accuracy']) * 100),
            'bias_reduction': float((old_results['bias'] - new_results['bias']) * 100),
            'f1_improvement': float((new_results['f1_score'] - old_results['f1_score']) * 100),
            'auc_improvement': float((new_results['auc_roc'] - old_results['auc_roc']) * 100)
        }
    }
    
    # Save detailed report
    with open('model_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create summary text report
    with open('model_comparison_summary.txt', 'w') as f:
        f.write("🚀 INTERCEPTOR MODEL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📊 TEST DATASET:\n")
        f.write(f"   Total Videos: {report['test_dataset']['total_videos']}\n")
        f.write(f"   Real Videos: {report['test_dataset']['real_videos']}\n")
        f.write(f"   Fake Videos: {report['test_dataset']['fake_videos']}\n\n")
        
        f.write(f"🔍 OLD MODEL RESULTS:\n")
        f.write(f"   Model: {report['old_model']['name']}\n")
        f.write(f"   Accuracy: {report['old_model']['metrics']['accuracy']:.3f} ({report['old_model']['metrics']['accuracy']*100:.1f}%)\n")
        f.write(f"   Real Detection: {report['old_model']['metrics']['real_accuracy']:.3f} ({report['old_model']['metrics']['real_accuracy']*100:.1f}%)\n")
        f.write(f"   Fake Detection: {report['old_model']['metrics']['fake_accuracy']:.3f} ({report['old_model']['metrics']['fake_accuracy']*100:.1f}%)\n")
        f.write(f"   F1-Score: {report['old_model']['metrics']['f1_score']:.3f}\n")
        f.write(f"   AUC-ROC: {report['old_model']['metrics']['auc_roc']:.3f}\n")
        f.write(f"   Bias: {report['old_model']['metrics']['bias']:.3f}\n\n")
        
        f.write(f"🚀 NEW MODEL RESULTS:\n")
        f.write(f"   Model: {report['new_model']['name']}\n")
        f.write(f"   Accuracy: {report['new_model']['metrics']['accuracy']:.3f} ({report['new_model']['metrics']['accuracy']*100:.1f}%)\n")
        f.write(f"   Real Detection: {report['new_model']['metrics']['real_accuracy']:.3f} ({report['new_model']['metrics']['real_accuracy']*100:.1f}%)\n")
        f.write(f"   Fake Detection: {report['new_model']['metrics']['fake_accuracy']:.3f} ({report['new_model']['metrics']['fake_accuracy']*100:.1f}%)\n")
        f.write(f"   F1-Score: {report['new_model']['metrics']['f1_score']:.3f}\n")
        f.write(f"   AUC-ROC: {report['new_model']['metrics']['auc_roc']:.3f}\n")
        f.write(f"   Bias: {report['new_model']['metrics']['bias']:.3f}\n\n")
        
        f.write(f"📈 IMPROVEMENTS:\n")
        f.write(f"   Overall Accuracy: {report['improvements']['accuracy_improvement']:+.2f}%\n")
        f.write(f"   Real Detection: {report['improvements']['real_detection_improvement']:+.2f}%\n")
        f.write(f"   Fake Detection: {report['improvements']['fake_detection_change']:+.2f}%\n")
        f.write(f"   Bias Reduction: {report['improvements']['bias_reduction']:+.2f}%\n")
        f.write(f"   F1-Score: {report['improvements']['f1_improvement']:+.2f}%\n")
        f.write(f"   AUC-ROC: {report['improvements']['auc_improvement']:+.2f}%\n\n")
        
        # Interpretation
        f.write(f"🎯 INTERPRETATION:\n")
        if report['improvements']['accuracy_improvement'] > 5:
            f.write(f"   ✅ Significant overall improvement\n")
        elif report['improvements']['accuracy_improvement'] > 0:
            f.write(f"   ✅ Moderate overall improvement\n")
        else:
            f.write(f"   ⚠️ No significant overall improvement\n")
            
        if report['improvements']['real_detection_improvement'] > 10:
            f.write(f"   ✅ Major improvement in real detection\n")
        elif report['improvements']['real_detection_improvement'] > 0:
            f.write(f"   ✅ Some improvement in real detection\n")
        else:
            f.write(f"   ❌ No improvement in real detection\n")
            
        if report['improvements']['bias_reduction'] > 10:
            f.write(f"   ✅ Significant bias reduction\n")
        elif report['improvements']['bias_reduction'] > 0:
            f.write(f"   ✅ Some bias reduction\n")
        else:
            f.write(f"   ⚠️ No bias reduction\n")
    
    return report

# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def main():
    """Main comparison function"""
    
    print("🚀 INTERCEPTOR MODEL COMPARISON: OLD vs NEW")
    print("="*80)
    
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Device: {DEVICE}")
    
    # Model paths
    OLD_MODEL_PATH = "ll_model_student (1).pt"
    NEW_MODEL_PATH = "stage2_full_celebdf_best_epoch3.pt"
    
    # Test data paths
    TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
    REAL_VIDEOS_DIR = TEST_DATA_ROOT / "real"
    FAKE_VIDEOS_DIR = TEST_DATA_ROOT / "fake"
    
    print(f"📂 Test data location: {TEST_DATA_ROOT}")
    
    # Load test videos
    real_videos = list(REAL_VIDEOS_DIR.glob("*.mp4"))
    fake_videos = list(FAKE_VIDEOS_DIR.glob("*.mp4"))
    
    print(f"📹 Found {len(real_videos)} real videos")
    print(f"📹 Found {len(fake_videos)} fake videos")
    
    if not real_videos or not fake_videos:
        print("❌ No test videos found! Check the path.")
        return
    
    # Combine videos and labels
    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=real, 1=fake
    
    print(f"📊 Total test videos: {len(all_videos)}")
    print(f"📊 Real: {len(real_videos)} ({len(real_videos)/len(all_videos)*100:.1f}%)")
    print(f"📊 Fake: {len(fake_videos)} ({len(fake_videos)/len(all_videos)*100:.1f}%)")
    
    # Initialize models
    print(f"\n🔄 Initializing models...")
    old_model = OldLLModel(num_classes=2).to(DEVICE)
    new_model = NewProgressiveSpecialistModel(num_classes=2).to(DEVICE)
    
    # Load checkpoints
    print(f"\n📦 Loading model checkpoints...")
    old_info = load_model_checkpoint(old_model, OLD_MODEL_PATH, DEVICE)
    new_info = load_model_checkpoint(new_model, NEW_MODEL_PATH, DEVICE)
    
    if old_info is None or new_info is None:
        print("❌ Failed to load one or both models!")
        return
    
    print(f"✅ Old model info: Epoch {old_info['epoch']}, Accuracy {old_info['accuracy']}")
    print(f"✅ New model info: Epoch {new_info['epoch']}, Accuracy {new_info['accuracy']}")
    
    # Evaluate both models
    print(f"\n🔍 Starting model evaluation...")
    
    old_results = evaluate_model_on_videos(
        old_model, all_videos, all_labels, DEVICE, "Old LL-Model"
    )
    
    new_results = evaluate_model_on_videos(
        new_model, all_videos, all_labels, DEVICE, "New Enhanced LL-Model"
    )
    
    # Print results
    print(f"\n📊 COMPARISON RESULTS:")
    print("="*60)
    
    print(f"\n🔍 OLD MODEL (Original LL-Model):")
    print(f"   Overall Accuracy: {old_results['accuracy']:.3f} ({old_results['accuracy']*100:.1f}%)")
    print(f"   Real Detection: {old_results['real_accuracy']:.3f} ({old_results['real_accuracy']*100:.1f}%)")
    print(f"   Fake Detection: {old_results['fake_accuracy']:.3f} ({old_results['fake_accuracy']*100:.1f}%)")
    print(f"   F1-Score: {old_results['f1_score']:.3f}")
    print(f"   AUC-ROC: {old_results['auc_roc']:.3f}")
    print(f"   Bias: {old_results['bias']:.3f}")
    
    print(f"\n🚀 NEW MODEL (Enhanced LL-Model):")
    print(f"   Overall Accuracy: {new_results['accuracy']:.3f} ({new_results['accuracy']*100:.1f}%)")
    print(f"   Real Detection: {new_results['real_accuracy']:.3f} ({new_results['real_accuracy']*100:.1f}%)")
    print(f"   Fake Detection: {new_results['fake_accuracy']:.3f} ({new_results['fake_accuracy']*100:.1f}%)")
    print(f"   F1-Score: {new_results['f1_score']:.3f}")
    print(f"   AUC-ROC: {new_results['auc_roc']:.3f}")
    print(f"   Bias: {new_results['bias']:.3f}")
    
    # Calculate improvements
    acc_improvement = (new_results['accuracy'] - old_results['accuracy']) * 100
    real_improvement = (new_results['real_accuracy'] - old_results['real_accuracy']) * 100
    fake_change = (new_results['fake_accuracy'] - old_results['fake_accuracy']) * 100
    bias_reduction = (old_results['bias'] - new_results['bias']) * 100
    
    print(f"\n📈 IMPROVEMENTS:")
    print(f"   Overall Accuracy: {acc_improvement:+.2f}%")
    print(f"   Real Detection: {real_improvement:+.2f}%")
    print(f"   Fake Detection: {fake_change:+.2f}%")
    print(f"   Bias Reduction: {bias_reduction:+.2f}%")
    
    # Generate visualizations and reports
    print(f"\n📊 Generating visualizations and reports...")
    
    create_comparison_visualizations(old_results, new_results, all_videos, all_labels)
    report = generate_detailed_report(old_results, new_results, all_videos, all_labels)
    
    print(f"\n✅ COMPARISON COMPLETE!")
    print(f"📁 Generated files:")
    print(f"   📊 model_comparison_analysis.png - Visual comparison")
    print(f"   📋 model_comparison_report.json - Detailed JSON report")
    print(f"   📄 model_comparison_summary.txt - Summary report")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if acc_improvement > 5 and real_improvement > 10:
        print(f"   🎉 EXCELLENT: Significant improvement in both overall and real detection!")
    elif acc_improvement > 0 and real_improvement > 5:
        print(f"   ✅ GOOD: Noticeable improvement, especially in real detection!")
    elif real_improvement > 0:
        print(f"   👍 MODERATE: Some improvement in real detection!")
    else:
        print(f"   ⚠️ LIMITED: No significant improvement detected!")
    
    return report

if __name__ == "__main__":
    main()