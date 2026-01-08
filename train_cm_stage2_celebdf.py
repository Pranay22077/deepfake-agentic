"""
E-Raksha CM Model Training - Stage 2: CelebDF

Compression Specialist Model for detecting compression artifacts and encoding inconsistencies.
Focuses on DCT analysis, quantization detection, and compression pattern recognition.

Progressive Training Strategy:
- Stage 1: FaceForensics++ (Foundation) ✓
- Stage 2: CelebDF (Celebrity adaptation) ← THIS SCRIPT  
- Stage 4: DFDC (Large-scale validation)

Model Architecture:
- EfficientNet-B4 backbone with 40-channel specialist module
- Components: dct_analyzer (12ch), quant_detector (10ch), block_checker (10ch), compression_estimator (8ch)

Input: Stage 1 checkpoint from FaceForensics++ training
Output: Stage 2 checkpoint ready for DFDC training

Author: E-Raksha Team
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
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
DATASET_PATH = '/kaggle/input/celeb-df-v2'
STAGE1_CHECKPOINT = '/kaggle/input/cm-stage1-model/cm_stage1_final.pt'  # UPDATE THIS PATH
OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

# Global settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 5e-5  # Lower LR for fine-tuning
EPOCHS = 5
SAVE_FREQUENCY_GB = 5.0
MAX_CHECKPOINTS = 3

# Mixed precision
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

print(f"🚀 CM MODEL - STAGE 2: CELEB-DF TRAINING")
print(f"📊 Specialist: Compression Artifact Detection")
print(f"📥 Loading from: {STAGE1_CHECKPOINT}")
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
# CM SPECIALIST MODULE
# ============================================================================
class CompressionModule(nn.Module):
    """Compression artifact detection module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # DCT coefficient analyzer (8x8 blocks like JPEG)
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=8, stride=8),  # 8x8 blocks
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Quantization artifact detector
        self.quant_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Block boundary checker
        self.block_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Compression level estimator
        self.compression_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion
        self.attention = nn.Sequential(
            nn.Conv2d(12 + 10 + 8 + 10, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 40, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        dct_feat = self.dct_analyzer(x)
        dct_feat = F.adaptive_avg_pool2d(dct_feat, (7, 7))
        
        quant_feat = self.quant_detector(x)
        quant_feat = F.adaptive_avg_pool2d(quant_feat, (7, 7))
        
        block_feat = self.block_checker(x)
        block_feat = F.adaptive_avg_pool2d(block_feat, (7, 7))
        
        comp_feat = self.compression_estimator(x)
        comp_feat = F.adaptive_avg_pool2d(comp_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([dct_feat, quant_feat, block_feat, comp_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

# ============================================================================
# CM SPECIALIST MODEL
# ============================================================================
class CMSpecialistModel(nn.Module):
    """Complete CM Specialist Model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        print(f"🔄 Loading EfficientNet-B4 for CM model...")
        self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # CM specialist module
        self.specialist_module = CompressionModule()
        specialist_features = 40 * 7 * 7  # 1960
        
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
        
        print(f"✅ CM Specialist model ready!")
    
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
class CelebDFDataset(Dataset):
    """Celeb-DF v2 dataset"""
    
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
        """Extract a random frame from video with enhanced preprocessing"""
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
                
                # Enhanced contrast for Celeb-DF
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply(l)
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                
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
def load_celebdf_samples():
    """Load Celeb-DF v2 samples"""
    print(f"📂 Loading Celeb-DF v2 from {DATASET_PATH}")
    
    samples = []
    base_dir = Path(DATASET_PATH)
    
    # Try loading from label file first
    label_file = base_dir / 'List_of_testing_videos.txt'
    
    if label_file.exists():
        print(f"📋 Found label file: {label_file}")
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label = int(parts[0])  # 0 = real, 1 = fake
                    video_path_rel = parts[1]
                    
                    full_path = base_dir / video_path_rel
                    
                    if full_path.exists():
                        samples.append({
                            'video_path': str(full_path),
                            'label': label,
                            'source': Path(video_path_rel).parent.name
                        })
        except Exception as e:
            print(f"⚠️ Error reading label file: {e}")
    
    # If no samples from label file, use folder structure
    if not samples:
        print(f"📁 Loading from folder structure...")
        
        folder_mappings = [
            ('Celeb-real', 0),      # Real
            ('YouTube-real', 0),    # Real
            ('Celeb-synthesis', 1)  # Fake
        ]
        
        for folder_name, label in folder_mappings:
            folder_path = base_dir / folder_name
            
            if folder_path.exists():
                print(f"📂 Processing {folder_name} directory")
                
                video_files = []
                for ext in ['.mp4', '.avi', '.mov']:
                    video_files.extend(list(folder_path.glob(f'*{ext}')))
                
                print(f"   Found {len(video_files)} videos")
                
                for video_file in video_files:
                    samples.append({
                        'video_path': str(video_file),
                        'label': label,
                        'source': folder_name
                    })
    
    if samples:
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        print(f"✅ Loaded {len(samples)} samples")
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
    
    return samples

def get_transforms():
    """Get training transforms for Stage 2"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
# FOCAL LOSS
# ============================================================================
def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, epoch):
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
                loss = focal_loss(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = focal_loss(outputs, labels)
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
    print("\n🚀 STARTING CM MODEL STAGE 2 TRAINING")
    print("="*80)
    
    # Initialize
    checkpoint_manager = CheckpointManager(OUTPUT_DIR, MAX_CHECKPOINTS)
    
    # Load data
    samples = load_celebdf_samples()
    if not samples:
        print("❌ No samples found!")
        return
    
    # Create dataset and dataloader
    transform = get_transforms()
    dataset = CelebDFDataset(samples, transform)
    dataloader = create_dataloader(dataset)
    
    # Create model
    model = CMSpecialistModel(num_classes=2)
    model.to(DEVICE)
    
    # Load Stage 1 checkpoint
    print(f"\n📥 Loading Stage 1 checkpoint...")
    try:
        checkpoint = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded Stage 1 weights successfully")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint: {e}")
        print(f"🔄 Starting with fresh weights")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    print(f"\n📊 Training Configuration:")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Loss: Focal Loss")
    print(f"   Device: {DEVICE}")
    
    # Training loop
    best_accuracy = 0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{EPOCHS}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, dataloader, optimizer, epoch)
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
                'model_type': 'cm_specialist',
                'stage': 2
            }
            
            checkpoint_manager.save_checkpoint(
                model_state,
                f"cm_stage2_best_epoch{epoch}",
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
        'model_type': 'cm_specialist',
        'stage': 2,
        'final_metrics': eval_metrics
    }
    
    final_path = checkpoint_manager.save_checkpoint(
        model_state,
        "cm_stage2_final",
        eval_metrics
    )
    
    print(f"\n🎉 STAGE 2 TRAINING COMPLETED!")
    print(f"💾 Final model: {final_path.name}")
    print(f"📊 Best accuracy: {best_accuracy*100:.2f}%")
    print(f"\n🚀 Next: Use this model for Stage 4 (DFDC)")

if __name__ == "__main__":
    main()
