"""
E-Raksha BG Model Training - Stage 1: FaceForensics++

Background/Lighting Specialist Model Training Script
Detects inconsistencies in background textures, lighting direction, shadows, and color temperature.

Training Strategy:
- Stage 1: FaceForensics++ (Foundation training) ← THIS SCRIPT  
- Stage 2: CelebDF (Celebrity adaptation)
- Stage 4: DFDC (Large-scale validation)

Model Architecture:
- EfficientNet-B4 backbone with 44-channel specialist module
- Components: bg_texture (16ch), lighting_detector (12ch), shadow_checker (8ch), color_temp (8ch)

Author: E-Raksha Team
Created: Initial development phase
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import zipfile
from IPython.display import FileLink, display

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = '/kaggle/input/ff-c23'
OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

# Global settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 1e-4
EPOCHS = 6
SAVE_FREQUENCY_GB = 5.0
MAX_CHECKPOINTS = 3

# Mixed precision
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

print(f"🚀 BG MODEL - STAGE 1: FACEFORENSICS++ TRAINING")
print(f"📊 Specialist: Background/Lighting Inconsistency Detection")
print(f"💾 Output: {OUTPUT_DIR}")
print(f"🔥 Device: {DEVICE}")
print(f"⚡ Mixed Precision: {USE_MIXED_PRECISION}")
print("="*80)

# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================
class CheckpointManager:
    """Manages checkpoints with automatic cleanup"""
    
    def __init__(self, output_dir, max_checkpoints=3):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
        
        print(f"📦 Checkpoint Manager initialized")
        print(f"   Directory: {self.checkpoint_dir}")
        print(f"   Max checkpoints: {max_checkpoints}")
    
    def save_checkpoint(self, model_state, checkpoint_name, metrics):
        """Save checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"{checkpoint_name}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        checkpoint_data = {
            **model_state,
            'checkpoint_name': checkpoint_name,
            'save_timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Compress
        compressed_path = checkpoint_path.with_suffix('.zip')
        with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(checkpoint_path, checkpoint_path.name)
        
        if compressed_path.exists():
            os.remove(checkpoint_path)
            checkpoint_path = compressed_path
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        
        self.checkpoint_history.append({
            'path': checkpoint_path,
            'name': checkpoint_name,
            'timestamp': timestamp,
            'file_size_mb': file_size_mb,
            'metrics': metrics
        })
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name} ({file_size_mb:.1f} MB)")
        
        # Trigger download
        try:
            display(FileLink(str(checkpoint_path)))
        except:
            pass
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints"""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        self.checkpoint_history.sort(key=lambda x: x['timestamp'], reverse=True)
        checkpoints_to_remove = self.checkpoint_history[self.max_checkpoints:]
        
        for checkpoint_info in checkpoints_to_remove:
            try:
                if checkpoint_info['path'].exists():
                    os.remove(checkpoint_info['path'])
                    print(f"🗑️ Removed old checkpoint: {checkpoint_info['path'].name}")
            except Exception as e:
                print(f"⚠️ Failed to remove {checkpoint_info['path'].name}: {e}")
        
        self.checkpoint_history = self.checkpoint_history[:self.max_checkpoints]

# ============================================================================
# BG SPECIALIST MODULE
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

# ============================================================================
# BG SPECIALIST MODEL
# ============================================================================
class BGSpecialistModel(nn.Module):
    """Complete BG Specialist Model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        print(f"🔄 Loading EfficientNet-B4 for BG model...")
        self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
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
# DATASET
# ============================================================================
class FaceForensicsDataset(Dataset):
    """FaceForensics++ dataset"""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"📊 Dataset created:")
        print(f"   Total: {len(samples)} samples")
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        
        self.real_count = real_count
        self.fake_count = fake_count
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frame from video
        frame = self._extract_frame(sample['video_path'])
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        return frame, label
    
    def _extract_frame(self, video_path):
        """Extract a random frame from video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                frame_idx = np.random.randint(0, total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            
            return frame
        except Exception as e:
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_class_weights(self):
        """Get class weights for balanced sampling"""
        if self.real_count == 0 or self.fake_count == 0:
            return [1.0, 1.0]
        total = len(self.samples)
        real_weight = total / (2 * self.real_count)
        fake_weight = total / (2 * self.fake_count)
        return [real_weight, fake_weight]

# ============================================================================
# DATA LOADING
# ============================================================================
def load_faceforensics_samples():
    """Load FaceForensics++ samples"""
    print(f"📂 Loading FaceForensics++ from {DATASET_PATH}")
    
    samples = []
    base_dir = Path(DATASET_PATH) / 'FaceForensics++_C23'
    
    if not base_dir.exists():
        print(f"❌ Dataset not found: {base_dir}")
        return samples
    
    # Fake methods
    fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    # Load fake videos
    for method in fake_methods:
        method_dir = base_dir / method
        if method_dir.exists():
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(method_dir.rglob(f'*{ext}')))
            
            for video_file in video_files:
                samples.append({
                    'video_path': str(video_file),
                    'label': 1,  # Fake
                    'method': method
                })
            print(f"   {method}: {len(video_files)} videos")
    
    # Load real videos
    original_dir = base_dir / 'original'
    if original_dir.exists():
        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(list(original_dir.rglob(f'*{ext}')))
        
        for video_file in video_files:
            samples.append({
                'video_path': str(video_file),
                'label': 0,  # Real
                'method': 'original'
            })
        print(f"   Original: {len(video_files)} videos")
    
    if samples:
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        print(f"✅ Loaded {len(samples)} samples")
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
    
    return samples

def get_transforms():
    """Get training transforms"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloader(dataset):
    """Create balanced dataloader"""
    class_weights = dataset.get_class_weights()
    sample_weights = [class_weights[s['label']] for s in dataset.samples]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for frames, labels in pbar:
        frames = frames.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)
        
        optimizer.zero_grad()
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(frames)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i in range(len(labels)):
            if labels[i] == 0:
                real_total += 1
                if predicted[i] == 0:
                    real_correct += 1
            else:
                fake_total += 1
                if predicted[i] == 1:
                    fake_correct += 1
        
        # Update progress
        acc = 100 * correct / total
        real_acc = 100 * real_correct / real_total if real_total > 0 else 0
        fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc:.1f}%',
            'Real': f'{real_acc:.1f}%',
            'Fake': f'{fake_acc:.1f}%'
        })
    
    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct / total
    real_accuracy = real_correct / real_total if real_total > 0 else 0
    fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'bias_difference': abs(real_accuracy - fake_accuracy)
    }

def evaluate_model(model, dataloader):
    """Evaluate model"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Evaluating"):
            frames = frames.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)
            
            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    cm = confusion_matrix(all_labels, all_predictions)
    real_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    fake_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'bias_difference': abs(real_accuracy - fake_accuracy)
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main training function"""
    print("\n🚀 STARTING BG MODEL STAGE 1 TRAINING")
    print("="*80)
    
    # Initialize
    checkpoint_manager = CheckpointManager(OUTPUT_DIR, MAX_CHECKPOINTS)
    
    # Load data
    samples = load_faceforensics_samples()
    if not samples:
        print("❌ No samples found!")
        return
    
    # Create dataset and dataloader
    transform = get_transforms()
    dataset = FaceForensicsDataset(samples, transform)
    dataloader = create_dataloader(dataset)
    
    # Create model
    model = BGSpecialistModel(num_classes=2)
    model.to(DEVICE)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Device: {DEVICE}")
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, dataloader, criterion, optimizer, epoch)
        scheduler.step()
        
        print(f"📊 Epoch {epoch} Results:")
        print(f"   Loss: {train_metrics['loss']:.4f}")
        print(f"   Accuracy: {train_metrics['accuracy']*100:.2f}%")
        print(f"   Real: {train_metrics['real_accuracy']*100:.2f}%")
        print(f"   Fake: {train_metrics['fake_accuracy']*100:.2f}%")
        print(f"   Bias: {train_metrics['bias_difference']*100:.1f}%")
        
        # Save checkpoint
        if train_metrics['accuracy'] > best_accuracy:
            best_accuracy = train_metrics['accuracy']
            
            model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'model_type': 'bg_specialist'
            }
            
            checkpoint_manager.save_checkpoint(
                model_state,
                f"bg_stage1_epoch{epoch}",
                train_metrics
            )
    
    # Final evaluation
    print(f"\n📊 Final Evaluation")
    eval_metrics = evaluate_model(model, dataloader)
    print(f"   Accuracy: {eval_metrics['accuracy']*100:.2f}%")
    print(f"   F1-Score: {eval_metrics['f1_score']*100:.2f}%")
    print(f"   Real: {eval_metrics['real_accuracy']*100:.2f}%")
    print(f"   Fake: {eval_metrics['fake_accuracy']*100:.2f}%")
    
    # Save final model
    model_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'model_type': 'bg_specialist',
        'final_metrics': eval_metrics
    }
    
    final_path = checkpoint_manager.save_checkpoint(
        model_state,
        "bg_stage1_final",
        eval_metrics
    )
    
    print(f"\n🎉 STAGE 1 TRAINING COMPLETED!")
    print(f"💾 Final model: {final_path.name}")
    print(f"📊 Best accuracy: {best_accuracy*100:.2f}%")
    print(f"\n🚀 Next: Use this model for Stage 2 (Celeb-DF)")

if __name__ == "__main__":
    main()
