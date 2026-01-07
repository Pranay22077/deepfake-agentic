"""
COMPRESSION (CM) MODEL - STAGE 4: DFDC TRAINING
Specialist model for detecting compression artifacts and encoding inconsistencies

This is Stage 4 of 3-stage progressive training:
- Stage 1: FaceForensics++ (Foundation) ✓
- Stage 2: Celeb-DF (Realism adaptation) ✓
- Stage 4: DFDC (Large-scale training) ← THIS SCRIPT

INPUT: Stage 2 checkpoint (.pt file from Celeb-DF training)
OUTPUT: Final CM specialist model
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
DATASET_PATH = '/kaggle/input/dfdc-10'
STAGE2_CHECKPOINT = '/kaggle/input/cm-stage2-model/cm_stage2_final.pt'  # UPDATE THIS PATH
OUTPUT_DIR = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"

# DFDC settings
CHUNKS_TO_TRAIN = [9, 8, 3, 5, 7, 2, 6, 4, 1, 0]  # Most to least balanced
EPOCHS_PER_CHUNK = 2

# Global settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 1e-5  # Very low LR for large-scale fine-tuning
SAVE_FREQUENCY_GB = 5.0
MAX_CHECKPOINTS = 3

# Mixed precision
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

print(f"🚀 CM MODEL - STAGE 4: DFDC TRAINING")
print(f"📊 Specialist: Compression Artifact Detection")
print(f"📥 Loading from: {STAGE2_CHECKPOINT}")
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
            nn.Dropout(0.4),
            nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
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
class DFDCDataset(Dataset):
    """DFDC dataset"""
    
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
        """Extract a random frame from video with full enhancement"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # Focus on middle frames for DFDC
                start_frame = max(0, total_frames // 4)
                end_frame = min(total_frames, 3 * total_frames // 4)
                frame_idx = np.random.randint(start_frame, end_frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Full enhancement for DFDC
                lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l_enhanced = clahe.apply(l)
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                frame = cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)
                
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
def load_dfdc_chunk(chunk_idx):
    """Load DFDC chunk samples"""
    print(f"📂 Loading DFDC chunk {chunk_idx}")
    
    chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
    chunk_dir = Path(DATASET_PATH) / chunk_name
    
    if not chunk_dir.exists():
        print(f"❌ Chunk directory not found: {chunk_dir}")
        return []
    
    # Check for subdirectory structure
    subdir_name = f"dfdc_train_part_{chunk_idx}"
    subdir_path = chunk_dir / subdir_name
    chunk_path = subdir_path if subdir_path.exists() else chunk_dir
    
    print(f"📁 Using chunk path: {chunk_path}")
    
    # Load metadata.json
    metadata_file = chunk_path / "metadata.json"
    if not metadata_file.exists():
        print(f"❌ No metadata.json in {chunk_path}")
        return []
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"📋 Loaded metadata with {len(metadata)} entries")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return []
    
    # Find video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(chunk_path.glob(f"*{ext}")))
    
    print(f"🎬 Found {len(video_files)} video files")
    
    # Create samples
    samples = []
    matched_count = 0
    
    for video_file in video_files:
        video_name = video_file.stem
        
        # Try different matching strategies
        metadata_key = None
        if video_name in metadata:
            metadata_key = video_name
        elif f"{video_name}.mp4" in metadata:
            metadata_key = f"{video_name}.mp4"
        elif video_file.name in metadata:
            metadata_key = video_file.name
        
        if metadata_key:
            entry = metadata[metadata_key]
            label_value = entry.get('label', 'UNKNOWN')
            
            if label_value == 'REAL':
                label = 0
                matched_count += 1
            elif label_value == 'FAKE':
                label = 1
                matched_count += 1
            else:
                continue
            
            samples.append({
                'video_path': str(video_file),
                'label': label,
                'chunk': chunk_idx
            })
    
    if samples:
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"✅ Loaded {len(samples)} samples from chunk {chunk_idx}")
        print(f"   📊 Matched: {matched_count}/{len(video_files)} videos")
        print(f"   📈 Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   📈 Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
    
    return samples

def get_transforms():
    """Get training transforms for Stage 4"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
# WEIGHTED LOSS
# ============================================================================
def weighted_loss(outputs, targets):
    """Weighted cross-entropy for severe imbalance"""
    unique, counts = torch.unique(targets, return_counts=True)
    weights = torch.ones(2, device=DEVICE)
    
    for i, count in zip(unique, counts):
        weights[i] = len(targets) / (2 * count)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    return criterion(outputs, targets)

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, epoch, chunk_idx):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    pbar = tqdm(dataloader, desc=f"Chunk {chunk_idx} Epoch {epoch}")
    
    for frames, labels in pbar:
        frames = frames.to(DEVICE)
        labels = labels.squeeze().to(DEVICE)
        
        optimizer.zero_grad()
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(frames)
                loss = weighted_loss(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(frames)
            loss = weighted_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # Memory cleanup
        if total % 400 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
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
    print("\n🚀 STARTING CM MODEL STAGE 4 TRAINING")
    print("="*80)
    
    # Initialize
    checkpoint_manager = CheckpointManager(OUTPUT_DIR, MAX_CHECKPOINTS)
    
    # Create model
    model = CMSpecialistModel(num_classes=2)
    model.to(DEVICE)
    
    # Load Stage 2 checkpoint
    print(f"\n📥 Loading Stage 2 checkpoint...")
    try:
        checkpoint = torch.load(STAGE2_CHECKPOINT, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded Stage 2 weights successfully")
    except Exception as e:
        print(f"⚠️ Could not load checkpoint: {e}")
        print(f"🔄 Starting with fresh weights")
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    print(f"\n📊 Training Configuration:")
    print(f"   Chunks: {len(CHUNKS_TO_TRAIN)}")
    print(f"   Epochs per chunk: {EPOCHS_PER_CHUNK}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Loss: Weighted CrossEntropy")
    print(f"   Device: {DEVICE}")
    
    # Training loop - process each chunk
    best_accuracy = 0
    transform = get_transforms()
    
    for chunk_idx in CHUNKS_TO_TRAIN:
        print(f"\n{'='*80}")
        print(f"📦 PROCESSING CHUNK {chunk_idx}")
        print(f"{'='*80}")
        
        # Load chunk data
        samples = load_dfdc_chunk(chunk_idx)
        if not samples:
            print(f"⏭️ Skipping chunk {chunk_idx} (no samples)")
            continue
        
        # Create dataset and dataloader
        dataset = DFDCDataset(samples, transform)
        dataloader = create_dataloader(dataset)
        
        # Train on this chunk
        for epoch in range(1, EPOCHS_PER_CHUNK + 1):
            print(f"\n📚 Chunk {chunk_idx} - Epoch {epoch}/{EPOCHS_PER_CHUNK}")
            print("-" * 50)
            
            # Train
            train_metrics = train_epoch(model, dataloader, optimizer, epoch, chunk_idx)
            scheduler.step()
            
            print(f"📊 Results:")
            print(f"   Loss: {train_metrics['loss']:.4f}")
            print(f"   Accuracy: {train_metrics['accuracy']*100:.2f}%")
            print(f"   Real: {train_metrics['real_accuracy']*100:.2f}%")
            print(f"   Fake: {train_metrics['fake_accuracy']*100:.2f}%")
            print(f"   Bias: {train_metrics['bias_difference']*100:.1f}%")
            
            # Save checkpoint if improved
            if train_metrics['accuracy'] > best_accuracy:
                best_accuracy = train_metrics['accuracy']
                
                model_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'chunk': chunk_idx,
                    'epoch': epoch,
                    'model_type': 'cm_specialist',
                    'stage': 4
                }
                
                checkpoint_manager.save_checkpoint(
                    model_state,
                    f"cm_stage4_chunk{chunk_idx}_epoch{epoch}",
                    train_metrics
                )
        
        # Evaluate on chunk
        print(f"\n📊 Chunk {chunk_idx} Final Evaluation")
        eval_metrics = evaluate_model(model, dataloader)
        print(f"   Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        print(f"   F1-Score: {eval_metrics['f1_score']*100:.2f}%")
        print(f"   Real: {eval_metrics['real_accuracy']*100:.2f}%")
        print(f"   Fake: {eval_metrics['fake_accuracy']*100:.2f}%")
        
        # Save chunk completion checkpoint
        model_state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'chunk': chunk_idx,
            'model_type': 'cm_specialist',
            'stage': 4
        }
        
        checkpoint_manager.save_checkpoint(
            model_state,
            f"cm_stage4_chunk{chunk_idx}_complete",
            eval_metrics
        )
        
        # Memory cleanup
        del dataset, dataloader
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final model
    print(f"\n{'='*80}")
    print(f"💾 SAVING FINAL MODEL")
    print(f"{'='*80}")
    
    model_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': 'cm_specialist',
        'stage': 4,
        'chunks_trained': CHUNKS_TO_TRAIN
    }
    
    final_path = checkpoint_manager.save_checkpoint(
        model_state,
        "cm_stage4_final",
        {'best_accuracy': best_accuracy}
    )
    
    print(f"\n🎉 STAGE 4 TRAINING COMPLETED!")
    print(f"💾 Final model: {final_path.name}")
    print(f"📊 Best accuracy: {best_accuracy*100:.2f}%")
    print(f"\n✅ CM Specialist Model Training Complete!")
    print(f"🚀 Ready for integration into agent framework")

if __name__ == "__main__":
    main()
