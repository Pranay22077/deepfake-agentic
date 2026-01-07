"""
RESUME STAGE 2 TRAINING ON FULL CELEB-DF DATASET
Continue training from Stage 2 checkpoint on the complete Celeb-DF dataset
(not just the 518 test videos, but all videos in all folders)

🔧 SETUP INSTRUCTIONS:
1. Upload your trained model checkpoint (.zip file) to Kaggle as a dataset
2. Update the paths below:
   - Replace "YOUR-DATASET-NAME" with your actual Kaggle dataset name
   - Update CELEBDF_DATASET_PATH to your Celeb-DF dataset path
3. Run the script and choose your checkpoint when prompted

📦 RECOMMENDED CHECKPOINT: Stage2_CelebDF_stage2_celebdf_complete_20260105_211721.zip
"""

import os
import sys
import time
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm
import torch.nn.functional as F
import warnings
import pandas as pd
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import zipfile
from IPython.display import FileLink, display
import subprocess

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - 🔧 UPDATE THESE PATHS
# ============================================================================
# 🔧 UPDATE THIS: Replace with your actual Celeb-DF dataset path in Kaggle
CELEBDF_DATASET_PATH = '/kaggle/input/celeb-df-v2'  # ← UPDATE THIS PATH

OUTPUT_DIR = "/kaggle/working"

# Ask user for checkpoint file
print("🚀 RESUME STAGE 2 TRAINING ON FULL CELEB-DF DATASET")
print("="*80)
print("📋 Available checkpoint options:")
print("1. Stage2_CelebDF_stage2_celebdf_complete_20260105_211721 (RECOMMENDED)")
print("2. Stage2_CelebDF_stage2_celebdf_epoch1_20260105_211125")
print("3. Custom path")
print()

checkpoint_choice = input("Enter choice (1/2/3): ").strip()

if checkpoint_choice == "1":
    # 🔧 UPDATE THIS PATH TO YOUR ACTUAL KAGGLE INPUT PATH
    CHECKPOINT_PATH = "/kaggle/input/stage-2-model-inc-resume/stage2_celebdf_complete_20260105_211721.pt"
elif checkpoint_choice == "2":
    # 🔧 UPDATE THIS PATH TO YOUR ACTUAL KAGGLE INPUT PATH  
    CHECKPOINT_PATH = "/kaggle/input/stage-2-model-inc-resume/Stage2_CelebDF_stage2_celebdf_epoch1_20260105_211125.zip"
elif checkpoint_choice == "3":
    CHECKPOINT_PATH = input("Enter full path to checkpoint file: ").strip()
else:
    print("❌ Invalid choice. Using recommended checkpoint.")
    # 🔧 UPDATE THIS PATH TO YOUR ACTUAL KAGGLE INPUT PATH
    CHECKPOINT_PATH = "/kaggle/input/stage-2-model-inc-resume/stage2_celebdf_complete_20260105_211721.pt"

print(f"📦 Using checkpoint: {CHECKPOINT_PATH}")

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# Mixed precision training
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

print(f"🔥 Device: {DEVICE}")
print(f"⚡ Mixed Precision: {USE_MIXED_PRECISION}")
print(f"📊 Batch Size: {BATCH_SIZE}")

# ============================================================================
# MODEL ARCHITECTURE (Same as original)
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
        
        # Shadow/highlight inconsistency detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Conv2d(48 + 12 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 68, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale luminance analysis
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
        
        # Combine all features
        combined = torch.cat([lum_combined, noise_features, shadow_features], dim=1)  # 68 channels
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

class ProgressiveSpecialistModel(nn.Module):
    """Progressive specialist model"""
    
    def __init__(self, num_classes=2, model_type='progressive'):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        self.model_type = model_type
        
        # Enhanced specialist module
        self.specialist_module = EnhancedLowLightModule()
        specialist_features = 68 * 7 * 7  # Enhanced features
        
        # Progressive classifier with stage-aware design
        total_features = backbone_features + specialist_features
        
        # Make total_features divisible by num_heads for MultiheadAttention
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        # Feature projection to make it divisible by num_heads
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        # Multi-head attention for feature importance
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features, 
            num_heads=num_heads, 
            dropout=0.1,
            batch_first=True
        )
        
        # Stage-adaptive classifier
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
        
        # Stage tracking for adaptive behavior
        self.current_stage = 2  # Starting from Stage 2
        self.stage_history = []
    
    def set_stage(self, stage, stage_info=None):
        """Set current training stage for adaptive behavior"""
        self.current_stage = stage
        if stage_info:
            self.stage_history.append(stage_info)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        
        # Project to attention-compatible dimensions
        projected_features = self.feature_projection(combined_features)
        
        # Apply multi-head attention (reshape for attention)
        projected_reshaped = projected_features.unsqueeze(1)  # [batch, 1, features]
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)  # [batch, features]
        
        # Final classification
        output = self.classifier(attended_features)
        return output

# ============================================================================
# FULL CELEB-DF DATASET LOADER
# ============================================================================
class FullCelebDFDatasetLoader:
    """Load the COMPLETE Celeb-DF dataset from all folders"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_all_samples(self):
        """Load ALL samples from all folders in Celeb-DF dataset"""
        
        print(f"📂 Loading FULL Celeb-DF dataset from {self.data_dir}")
        print(f"🎯 This will load ALL videos, not just the 518 test samples!")
        
        samples = []
        
        # Folder mappings based on the dataset structure
        folder_mappings = [
            ('Celeb-real', 0),      # Real celebrity videos
            ('YouTube-real', 0),    # Real YouTube videos  
            ('Celeb-synthesis', 1)  # Fake celebrity videos
        ]
        
        total_videos_found = 0
        
        for folder_name, label in folder_mappings:
            folder_path = self.data_dir / folder_name
            
            if folder_path.exists():
                print(f"📁 Processing {folder_name} directory...")
                
                # Find all video files
                video_files = []
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    video_files.extend(list(folder_path.glob(f'*{ext}')))
                    # Also search recursively in case there are subdirectories
                    video_files.extend(list(folder_path.rglob(f'*{ext}')))
                
                # Remove duplicates
                video_files = list(set(video_files))
                
                print(f"   📹 Found {len(video_files)} videos")
                total_videos_found += len(video_files)
                
                # Add to samples
                for video_file in video_files:
                    samples.append({
                        'video_path': str(video_file),
                        'label': label,
                        'source': folder_name,
                        'dataset': 'celebdf_full'
                    })
                
                # Show some sample filenames
                if video_files:
                    print(f"   📋 Sample files:")
                    for i, vf in enumerate(video_files[:3]):
                        print(f"      {i+1}. {vf.name}")
                    if len(video_files) > 3:
                        print(f"      ... and {len(video_files)-3} more")
            else:
                print(f"⚠️ Directory not found: {folder_path}")
        
        # Remove any samples that were in the original 518 test set to avoid duplication
        # (Optional: you can comment this out if you want to include test samples too)
        samples = self._remove_test_samples(samples)
        
        # Final analysis
        if samples:
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            
            print(f"\n✅ FULL CELEB-DF DATASET LOADED:")
            print(f"   📊 Total videos found in folders: {total_videos_found}")
            print(f"   📊 Training samples (after filtering): {len(samples)}")
            print(f"   📈 Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
            print(f"   📈 Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
            
            # Show distribution by source
            sources = {}
            for sample in samples:
                source = sample['source']
                if source not in sources:
                    sources[source] = {'real': 0, 'fake': 0}
                if sample['label'] == 0:
                    sources[source]['real'] += 1
                else:
                    sources[source]['fake'] += 1
            
            print(f"\n📋 Distribution by source:")
            for source, counts in sources.items():
                total = counts['real'] + counts['fake']
                print(f"   {source}: {total} videos ({counts['real']} real, {counts['fake']} fake)")
        else:
            print(f"❌ No samples found in Celeb-DF dataset!")
        
        return samples
    
    def _remove_test_samples(self, samples):
        """Remove samples that were in the original test set to avoid duplication"""
        
        test_file = self.data_dir / 'List_of_testing_videos.txt'
        if not test_file.exists():
            print(f"📋 No test file found, keeping all samples")
            return samples
        
        # Load test video names
        test_videos = set()
        try:
            with open(test_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ' ' in line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            video_path = parts[1]
                            video_name = Path(video_path).name
                            test_videos.add(video_name)
            
            print(f"📋 Found {len(test_videos)} test videos to exclude")
        except Exception as e:
            print(f"⚠️ Error reading test file: {e}")
            return samples
        
        # Filter out test samples
        filtered_samples = []
        excluded_count = 0
        
        for sample in samples:
            video_name = Path(sample['video_path']).name
            if video_name not in test_videos:
                filtered_samples.append(sample)
            else:
                excluded_count += 1
        
        print(f"📊 Excluded {excluded_count} test samples from training")
        return filtered_samples

# ============================================================================
# DATASET CLASS
# ============================================================================
class CelebDFDataset(Dataset):
    """Dataset class for Celeb-DF videos"""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
        # Analyze distribution
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"📊 Dataset created:")
        print(f"   Total: {len(samples)} samples")
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        
        # Store for balanced sampling
        self.real_count = real_count
        self.fake_count = fake_count
    
    def _extract_frames(self, video_path, num_frames=8):
        """Extract frames from video"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # Smart frame selection
                indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
            
            cap.release()
            
            # Ensure we have enough frames
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
            
            return np.array(frames[:num_frames])
            
        except Exception as e:
            print(f"⚠️ Error extracting frames from {video_path}: {e}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    
    def get_class_weights(self):
        """Get class weights for balanced sampling"""
        
        if self.real_count == 0 or self.fake_count == 0:
            return [1.0, 1.0]
        
        total = len(self.samples)
        real_weight = total / (2 * self.real_count)
        fake_weight = total / (2 * self.fake_count)
        
        return [real_weight, fake_weight]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        return frame, label

# ============================================================================
# CHECKPOINT LOADING
# ============================================================================
def load_checkpoint(checkpoint_path, model, device):
    """Load checkpoint and return model state"""
    
    print(f"📦 Loading checkpoint from: {checkpoint_path}")
    
    try:
        # Handle .zip files
        if checkpoint_path.endswith('.zip'):
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find .pt file inside
                pt_files = list(Path(temp_dir).rglob("*.pt"))
                if not pt_files:
                    raise ValueError("No .pt file found in zip archive")
                
                checkpoint = torch.load(pt_files[0], map_location=device, weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
        
        # Extract training info
        training_info = {
            'stage': checkpoint.get('current_stage', 2),
            'epoch': checkpoint.get('epoch', 0),
            'training_history': checkpoint.get('training_history', []),
            'stage_metrics': checkpoint.get('stage_metrics', {}),
            'total_data_processed_gb': checkpoint.get('total_data_processed_gb', 0.0)
        }
        
        print(f"📊 Checkpoint info:")
        print(f"   Stage: {training_info['stage']}")
        print(f"   Epoch: {training_info['epoch']}")
        print(f"   Data processed: {training_info['total_data_processed_gb']:.1f} GB")
        
        if training_info['stage_metrics']:
            print(f"   Previous metrics:")
            for stage, metrics in training_info['stage_metrics'].items():
                if 'accuracy' in metrics:
                    print(f"     {stage}: {metrics['accuracy']*100:.2f}% accuracy")
        
        return training_info
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print(f"🔄 Starting with fresh model weights")
        return {
            'stage': 2,
            'epoch': 0,
            'training_history': [],
            'stage_metrics': {},
            'total_data_processed_gb': 0.0
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def create_balanced_dataloader(dataset, batch_size=8):
    """Create balanced dataloader with weighted sampling"""
    
    # Get class weights
    class_weights = dataset.get_class_weights()
    print(f"📊 Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
    
    # Create sample weights
    sample_weights = []
    for sample in dataset.samples:
        sample_weights.append(class_weights[sample['label']])
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR
    )

def get_transforms():
    """Get data augmentation transforms"""
    
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

def train_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler=None):
    """Train one epoch"""
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.squeeze().to(device)
        
        optimizer.zero_grad()
        
        if scaler and USE_MIXED_PRECISION:
            with autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        
        # Per-class accuracy
        for i in range(len(targets)):
            if targets[i] == 0:  # Real
                real_total += 1
                if predicted[i] == targets[i]:
                    real_correct += 1
            else:  # Fake
                fake_total += 1
                if predicted[i] == targets[i]:
                    fake_correct += 1
        
        # Update progress bar
        if batch_idx % 10 == 0:
            acc = 100. * correct / total
            real_acc = 100. * real_correct / real_total if real_total > 0 else 0
            fake_acc = 100. * fake_correct / fake_total if fake_total > 0 else 0
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{acc:.1f}%',
                'Real': f'{real_acc:.1f}%',
                'Fake': f'{fake_acc:.1f}%'
            })
    
    # Final metrics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    real_acc = real_correct / real_total if real_total > 0 else 0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'real_samples': real_total,
        'fake_samples': fake_total
    }

def evaluate_model(model, dataloader, device):
    """Evaluate model"""
    
    model.eval()
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(dataloader, desc="Evaluating"):
            data, targets = data.to(device), targets.squeeze().to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Store for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Per-class accuracy
            for i in range(len(targets)):
                if targets[i] == 0:  # Real
                    real_total += 1
                    if predicted[i] == targets[i]:
                        real_correct += 1
                else:  # Fake
                    fake_total += 1
                    if predicted[i] == targets[i]:
                        fake_correct += 1
    
    # Calculate metrics
    accuracy = correct / total
    real_acc = real_correct / real_total if real_total > 0 else 0
    fake_acc = fake_correct / fake_total if fake_total > 0 else 0
    
    # Additional metrics
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    try:
        auc_roc = roc_auc_score(all_targets, all_predictions)
    except:
        auc_roc = 0.0
    
    return {
        'accuracy': accuracy,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'real_samples': real_total,
        'fake_samples': fake_total
    }

def save_checkpoint(model, optimizer, epoch, metrics, output_dir, name):
    """Save training checkpoint"""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'model_type': 'progressive_specialist',
        'current_stage': 2,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = Path(output_dir) / f"{name}_epoch{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def main():
    """Main training function"""
    
    print("🚀 RESUME STAGE 2 TRAINING ON FULL CELEB-DF DATASET")
    print("="*80)
    print("🎯 Goal: Train on ALL Celeb-DF videos, not just 518 test samples")
    print("📊 Expected: Thousands of videos instead of 518")
    print("⚡ This will significantly improve model performance!")
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize model
    print("🔄 Initializing model...")
    model = ProgressiveSpecialistModel(num_classes=2, model_type='progressive')
    model.to(DEVICE)
    
    # Load checkpoint
    training_info = load_checkpoint(CHECKPOINT_PATH, model, DEVICE)
    
    # Load FULL Celeb-DF dataset
    print("\n📂 Loading FULL Celeb-DF dataset...")
    loader = FullCelebDFDatasetLoader(CELEBDF_DATASET_PATH)
    samples = loader.load_all_samples()
    
    if not samples:
        print("❌ No samples found! Check dataset path.")
        return
    
    print(f"\n🎉 SUCCESS! Found {len(samples)} videos (vs. previous 518)")
    print(f"📈 This is {len(samples)/518:.1f}x more data!")
    
    # Create dataset and dataloader
    transform = get_transforms()
    dataset = CelebDFDataset(samples, transform)
    dataloader = create_balanced_dataloader(dataset, BATCH_SIZE)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    criterion = focal_loss  # Use focal loss for class imbalance
    
    if USE_MIXED_PRECISION:
        scaler = GradScaler()
    else:
        scaler = None
    
    # Training loop
    print(f"\n🚀 Starting training on {len(samples)} samples...")
    print(f"📊 Batches per epoch: {len(dataloader)}")
    
    num_epochs = 10  # Train for more epochs since we have more data
    best_accuracy = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n📚 Epoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, dataloader, optimizer, criterion, DEVICE, epoch, scaler)
        
        # Evaluate
        eval_metrics = evaluate_model(model, dataloader, DEVICE)
        
        # Print results
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Training Loss: {train_metrics['loss']:.4f}")
        print(f"   Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        print(f"   Real Accuracy: {eval_metrics['real_accuracy']*100:.2f}%")
        print(f"   Fake Accuracy: {eval_metrics['fake_accuracy']*100:.2f}%")
        print(f"   F1-Score: {eval_metrics['f1_score']*100:.2f}%")
        print(f"   AUC-ROC: {eval_metrics['auc_roc']:.3f}")
        print(f"   Bias: {abs(eval_metrics['real_accuracy'] - eval_metrics['fake_accuracy'])*100:.1f}%")
        
        # Save checkpoint if improved
        if eval_metrics['accuracy'] > best_accuracy:
            best_accuracy = eval_metrics['accuracy']
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, eval_metrics, 
                output_dir, "stage2_full_celebdf_best"
            )
            print(f"🎯 New best accuracy: {best_accuracy*100:.2f}%")
        
        # Save regular checkpoint every 3 epochs
        if epoch % 3 == 0:
            save_checkpoint(
                model, optimizer, epoch, eval_metrics,
                output_dir, f"stage2_full_celebdf_epoch{epoch}"
            )
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"🏆 Best accuracy achieved: {best_accuracy*100:.2f}%")
    print(f"📁 Checkpoints saved in: {output_dir}")
    print(f"🚀 Model is now trained on the FULL Celeb-DF dataset!")

if __name__ == "__main__":
    main()