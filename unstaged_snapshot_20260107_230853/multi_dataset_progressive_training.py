"""
MULTI-DATASET PROGRESSIVE DEEPFAKE TRAINING
Complete pipeline for training on 4 datasets sequentially:
1. FaceForensics++ (CSV files)
2. Celeb-DF v2 (TXT files with 0/1 labels)  
3. Wild Deepfake (Separate folders)
4. DFDC (JSON metadata files)

Progressive training strategy to eliminate bias and scale to 100GB+
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
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================
DATASET_CONFIGS = {
    'faceforensics': {
        'path': '/kaggle/input/ff-c23',
        'enabled': True,
        'stage': 1,
        'epochs_per_method': 3,
        'description': 'Balanced foundation training'
    },
    'celebdf': {
        'path': '/kaggle/input/celeb-df-v2',
        'enabled': True,
        'stage': 2,
        'epochs': 5,
        'description': 'High-quality realism adaptation'
    },
    'wilddeepfake': {
        'path': '/kaggle/input/wild-deepfake',
        'enabled': True,
        'stage': 3,
        'epochs': 3,
        'description': 'Real-world noise adaptation'
    },
    'dfdc': {
        'path': '/kaggle/input/dfdc-10',
        'enabled': True,
        'stage': 4,
        'chunks': 10,  # Chunks 00-09 (10 total)
        'epochs_per_chunk': 2,
        'description': 'Large-scale diversity training'
    }
}

CHECKPOINT_PATH = None  # Set if resuming from checkpoint
OUTPUT_DIR = "/kaggle/working"

# ============================================================================
# GLOBAL SETTINGS
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8  # Adjusted for multi-dataset training
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2

# Mixed precision training
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION:
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

# ============================================================================
# CHECKPOINT MANAGEMENT SETTINGS
# ============================================================================
CHECKPOINT_MANAGEMENT = {
    'save_frequency_gb': 5.0,  # Save checkpoint every 5GB of data processed
    'max_checkpoints': 3,      # Keep only 3 latest checkpoints
    'auto_download': True,     # Auto-trigger download in Kaggle
    'compress_checkpoints': True,  # Compress checkpoints to save space
    'backup_to_dataset': False,    # Set to True if you want to backup to Kaggle dataset
}

# Track data processed for checkpoint frequency
class DataTracker:
    def __init__(self):
        self.total_data_processed_gb = 0.0
        self.last_checkpoint_gb = 0.0
        self.checkpoint_counter = 0
    
    def add_data(self, data_size_gb):
        self.total_data_processed_gb += data_size_gb
    
    def should_save_checkpoint(self):
        return (self.total_data_processed_gb - self.last_checkpoint_gb) >= CHECKPOINT_MANAGEMENT['save_frequency_gb']
    
    def checkpoint_saved(self):
        self.last_checkpoint_gb = self.total_data_processed_gb
        self.checkpoint_counter += 1

data_tracker = DataTracker()

print(f"🚀 MULTI-DATASET PROGRESSIVE DEEPFAKE TRAINING")
print(f"📊 Training on 4 datasets sequentially for bias elimination")
print(f"💾 Output: {OUTPUT_DIR}")
print(f"🔥 Device: {DEVICE}")
print(f"⚡ Mixed Precision: {USE_MIXED_PRECISION}")
print(f"💾 Checkpoint Management:")
print(f"   📦 Save every: {CHECKPOINT_MANAGEMENT['save_frequency_gb']} GB")
print(f"   🗂️ Keep latest: {CHECKPOINT_MANAGEMENT['max_checkpoints']} checkpoints")
print(f"   📥 Auto download: {CHECKPOINT_MANAGEMENT['auto_download']}")
print("="*80)

# ============================================================================
# CHECKPOINT MANAGEMENT SYSTEM
# ============================================================================
class CheckpointManager:
    """Manages checkpoints with automatic downloads and storage cleanup"""
    
    def __init__(self, output_dir, max_checkpoints=3, save_frequency_gb=5.0):
        self.output_dir = Path(output_dir)
        self.max_checkpoints = max_checkpoints
        self.save_frequency_gb = save_frequency_gb
        self.checkpoint_history = []
        
        # Create checkpoints directory
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"📦 Checkpoint Manager initialized:")
        print(f"   Directory: {self.checkpoint_dir}")
        print(f"   Max checkpoints: {max_checkpoints}")
        print(f"   Save frequency: {save_frequency_gb} GB")
    
    def estimate_data_size_gb(self, num_samples, avg_video_size_mb=50):
        """Estimate data size in GB"""
        return (num_samples * avg_video_size_mb) / 1024
    
    def get_checkpoint_size_mb(self, checkpoint_path):
        """Get checkpoint file size in MB"""
        try:
            size_bytes = os.path.getsize(checkpoint_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0
    
    def compress_checkpoint(self, checkpoint_path):
        """Compress checkpoint to save space"""
        if not CHECKPOINT_MANAGEMENT['compress_checkpoints']:
            return checkpoint_path
        
        compressed_path = checkpoint_path.with_suffix('.zip')
        
        try:
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                zipf.write(checkpoint_path, checkpoint_path.name)
            
            # Remove original if compression successful
            if compressed_path.exists():
                os.remove(checkpoint_path)
                print(f"🗜️ Compressed checkpoint: {compressed_path.name}")
                return compressed_path
        except Exception as e:
            print(f"⚠️ Compression failed: {e}")
        
        return checkpoint_path
    
    def save_checkpoint(self, model_state, checkpoint_name, metrics, data_processed_gb):
        """Save checkpoint with automatic management"""
        
        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"{checkpoint_name}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        checkpoint_data = {
            **model_state,
            'checkpoint_name': checkpoint_name,
            'data_processed_gb': data_processed_gb,
            'save_timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Compress if enabled
        if CHECKPOINT_MANAGEMENT['compress_checkpoints']:
            checkpoint_path = self.compress_checkpoint(checkpoint_path)
        
        # Get file size
        file_size_mb = self.get_checkpoint_size_mb(checkpoint_path)
        
        # Add to history
        checkpoint_info = {
            'path': checkpoint_path,
            'name': checkpoint_name,
            'timestamp': timestamp,
            'data_processed_gb': data_processed_gb,
            'file_size_mb': file_size_mb,
            'metrics': metrics
        }
        
        self.checkpoint_history.append(checkpoint_info)
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name} ({file_size_mb:.1f} MB)")
        
        # Trigger download if enabled
        if CHECKPOINT_MANAGEMENT['auto_download']:
            self.trigger_download(checkpoint_path)
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def trigger_download(self, checkpoint_path):
        """Trigger automatic download in Kaggle environment"""
        
        try:
            # Check if running in Kaggle
            if '/kaggle/' in str(checkpoint_path):
                print(f"📥 Triggering download for: {checkpoint_path.name}")
                
                # Create download link (works in Kaggle notebooks)
                display(FileLink(str(checkpoint_path)))
                
                # Also try to copy to a download-friendly location
                download_dir = self.output_dir / "downloads"
                download_dir.mkdir(exist_ok=True)
                
                download_path = download_dir / checkpoint_path.name
                shutil.copy2(checkpoint_path, download_path)
                
                print(f"📁 Copy available at: {download_path}")
                
        except Exception as e:
            print(f"⚠️ Download trigger failed: {e}")
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space"""
        
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by timestamp (newest first)
        self.checkpoint_history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Remove old checkpoints
        checkpoints_to_remove = self.checkpoint_history[self.max_checkpoints:]
        
        for checkpoint_info in checkpoints_to_remove:
            try:
                if checkpoint_info['path'].exists():
                    os.remove(checkpoint_info['path'])
                    print(f"🗑️ Removed old checkpoint: {checkpoint_info['path'].name}")
            except Exception as e:
                print(f"⚠️ Failed to remove {checkpoint_info['path'].name}: {e}")
        
        # Keep only recent checkpoints in history
        self.checkpoint_history = self.checkpoint_history[:self.max_checkpoints]
        
        # Print storage summary
        self.print_storage_summary()
    
    def print_storage_summary(self):
        """Print current storage usage"""
        
        total_size_mb = sum(info['file_size_mb'] for info in self.checkpoint_history)
        
        print(f"📊 Storage Summary:")
        print(f"   Active checkpoints: {len(self.checkpoint_history)}")
        print(f"   Total size: {total_size_mb:.1f} MB")
        print(f"   Available space: ~{20*1024 - total_size_mb:.0f} MB")
        
        if total_size_mb > 15*1024:  # Warn if using >15GB
            print(f"⚠️ WARNING: High storage usage!")
    
    def should_save_checkpoint(self, data_processed_gb):
        """Check if it's time to save a checkpoint"""
        
        if not self.checkpoint_history:
            return True  # Always save first checkpoint
        
        last_checkpoint_gb = self.checkpoint_history[-1]['data_processed_gb']
        return (data_processed_gb - last_checkpoint_gb) >= self.save_frequency_gb
    
    def get_latest_checkpoint(self):
        """Get path to latest checkpoint"""
        
        if not self.checkpoint_history:
            return None
        
        return self.checkpoint_history[0]['path']  # Most recent first
    
    def list_checkpoints(self):
        """List all available checkpoints"""
        
        print(f"\n📋 Available Checkpoints:")
        print("-" * 80)
        
        for i, info in enumerate(self.checkpoint_history):
            print(f"{i+1}. {info['name']} ({info['timestamp']})")
            print(f"   Data: {info['data_processed_gb']:.1f} GB")
            print(f"   Size: {info['file_size_mb']:.1f} MB")
            if 'accuracy' in info['metrics']:
                print(f"   Accuracy: {info['metrics']['accuracy']*100:.2f}%")
            print()

# ============================================================================
# ENHANCED LOW-LIGHT ANALYSIS MODULE
# ============================================================================
class EnhancedLowLightModule(nn.Module):
    """Enhanced low-light analysis with multi-scale features"""
    
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
            nn.Conv2d(48 + 12 + 8, 32, kernel_size=1),  # 48 from luminance, 12 from noise, 8 from shadow
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

# ============================================================================
# PROGRESSIVE SPECIALIST MODEL
# ============================================================================
class ProgressiveSpecialistModel(nn.Module):
    """Progressive specialist model that adapts through training stages"""
    
    def __init__(self, num_classes=2, model_type='ll'):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        print(f"🔄 Loading EfficientNet-B4 for Progressive {model_type.upper()} model...")
        
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
        self.current_stage = 1
        self.stage_history = []
        
        print(f"✅ Progressive {model_type.upper()} model ready!")
    
    def set_stage(self, stage, stage_info=None):
        """Set current training stage for adaptive behavior"""
        self.current_stage = stage
        if stage_info:
            self.stage_history.append(stage_info)
        
        # Adjust dropout based on stage
        stage_dropout = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        dropout_rate = stage_dropout.get(stage, 0.3)
        
        # Update dropout layers
        for module in self.classifier.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
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
# DATASET LOADERS FOR EACH FORMAT
# ============================================================================

class FaceForensicsDatasetLoader:
    """Loader for FaceForensics++ dataset with CSV metadata"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def load_samples(self):
        """Load FaceForensics++ samples from CSV files"""
        
        print(f"📂 Loading FaceForensics++ from {self.data_dir}")
        
        samples = []
        
        # Look for CSV directory: /kaggle/input/ff-c23/FaceForensics++_C23/csv/
        csv_dir = self.data_dir / 'FaceForensics++_C23' / 'csv'
        
        if not csv_dir.exists():
            print(f"❌ CSV directory not found: {csv_dir}")
            return samples
        
        print(f"📋 Found CSV directory: {csv_dir}")
        csv_files = list(csv_dir.glob('*.csv'))
        print(f"📄 Found {len(csv_files)} CSV files")
        
        for csv_file in csv_files:
            print(f"📊 Processing: {csv_file.name}")
            try:
                df = pd.read_csv(csv_file)
                print(f"   Columns: {list(df.columns)}")
                print(f"   Rows: {len(df)}")
                
                # Show sample to understand structure
                if len(df) > 0:
                    print(f"   Sample row: {dict(df.iloc[0])}")
                
                # Determine method from CSV filename
                method = csv_file.stem
                
                # Process each row in CSV
                for _, row in df.iterrows():
                    # Try different column name patterns for video filename
                    video_name = None
                    label = None
                    
                    # Common column names for video files
                    for col in ['filename', 'video', 'name', 'file', 'video_name']:
                        if col in row and pd.notna(row[col]):
                            video_name = row[col]
                            break
                    
                    # Common column names for labels
                    for col in ['label', 'class', 'target', 'fake', 'real']:
                        if col in row and pd.notna(row[col]):
                            label_value = row[col]
                            # Convert to binary
                            if label_value in ['REAL', 'real', 0, '0', 'original']:
                                label = 0  # Real
                            elif label_value in ['FAKE', 'fake', 1, '1', 'manipulated']:
                                label = 1  # Fake
                            break
                    
                    # If no explicit label, infer from method
                    if label is None:
                        if method.lower() in ['original', 'real']:
                            label = 0  # Real
                        else:
                            label = 1  # Fake
                    
                    if video_name:
                        # Try to find the actual video file
                        video_path = self._find_video_file(video_name, method)
                        if video_path:
                            samples.append({
                                'video_path': str(video_path),
                                'label': label,
                                'method': method,
                                'dataset': 'faceforensics'
                            })
            
            except Exception as e:
                print(f"⚠️ Error reading {csv_file}: {e}")
        
        # If no samples from CSV, try folder structure
        if not samples:
            print(f"📁 No CSV data found, trying folder structure...")
            samples = self._load_from_folders()
        
        # Analyze distribution
        if samples:
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            print(f"✅ Loaded {len(samples)} FaceForensics++ samples")
            print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
            print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        else:
            print(f"❌ No FaceForensics++ samples found")
        
        return samples
    
    def _find_video_file(self, video_name, method):
        """Find video file in the dataset directory"""
        
        # Remove extension if present
        base_name = Path(video_name).stem
        
        # Search in FaceForensics++ structure
        search_dirs = [
            self.data_dir / 'FaceForensics++_C23' / 'manipulated_sequences' / method,
            self.data_dir / 'FaceForensics++_C23' / 'original_sequences' / method,
            self.data_dir / 'FaceForensics++_C23' / method,
            self.data_dir / method,
        ]
        
        # Search for the video file
        for search_dir in search_dirs:
            if search_dir.exists():
                for ext in ['.mp4', '.avi', '.mov']:
                    # Try exact match
                    video_path = search_dir / f"{base_name}{ext}"
                    if video_path.exists():
                        return video_path
                    
                    # Try with original name
                    video_path = search_dir / video_name
                    if video_path.exists():
                        return video_path
                    
                    # Search recursively
                    for video_file in search_dir.rglob(f"{base_name}{ext}"):
                        return video_file
                    
                    for video_file in search_dir.rglob(video_name):
                        return video_file
        
        return None
    
    def _load_from_folders(self):
        """Fallback: Load from folder structure"""
        
        samples = []
        
        # Standard FaceForensics++ directory structure from screenshots
        base_dir = self.data_dir / 'FaceForensics++_C23'
        
        if not base_dir.exists():
            return samples
        
        print(f"📁 Exploring FaceForensics++ structure in {base_dir}")
        
        # Method directories for fake videos
        fake_methods = ['Deepfakes', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures', 'DeepFakeDetection']
        
        # Load fake videos from method directories
        for method in fake_methods:
            method_dir = base_dir / method
            if method_dir.exists():
                print(f"📂 Processing {method} directory")
                video_files = []
                for ext in ['.mp4', '.avi', '.mov']:
                    video_files.extend(list(method_dir.rglob(f'*{ext}')))
                
                print(f"   Found {len(video_files)} videos")
                
                for video_file in video_files:
                    samples.append({
                        'video_path': str(video_file),
                        'label': 1,  # Fake
                        'method': method,
                        'dataset': 'faceforensics'
                    })
        
        # Load real videos from original directory
        original_dir = base_dir / 'original'
        if original_dir.exists():
            print(f"📂 Processing original directory")
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(original_dir.rglob(f'*{ext}')))
            
            print(f"   Found {len(video_files)} original videos")
            
            for video_file in video_files:
                samples.append({
                    'video_path': str(video_file),
                    'label': 0,  # Real
                    'method': 'original',
                    'dataset': 'faceforensics'
                })
        
        # Also check for manipulated_sequences and original_sequences (standard FF++ structure)
        manipulated_dir = base_dir / 'manipulated_sequences'
        original_sequences_dir = base_dir / 'original_sequences'
        
        if manipulated_dir.exists():
            print(f"📂 Processing manipulated_sequences directory")
            for method in fake_methods:
                method_dir = manipulated_dir / method
                if method_dir.exists():
                    video_files = []
                    for ext in ['.mp4', '.avi', '.mov']:
                        video_files.extend(list(method_dir.rglob(f'*{ext}')))
                    
                    for video_file in video_files:
                        samples.append({
                            'video_path': str(video_file),
                            'label': 1,  # Fake
                            'method': method,
                            'dataset': 'faceforensics'
                        })
        
        if original_sequences_dir.exists():
            print(f"📂 Processing original_sequences directory")
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(original_sequences_dir.rglob(f'*{ext}')))
            
            for video_file in video_files:
                samples.append({
                    'video_path': str(video_file),
                    'label': 0,  # Real
                    'method': 'original',
                    'dataset': 'faceforensics'
                })
        
        return samples

class CelebDFDatasetLoader:
    """Loader for Celeb-DF v2 dataset with TXT labels and folder structure"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_samples(self):
        """Load Celeb-DF samples from TXT file and folder structure"""
        
        print(f"📂 Loading Celeb-DF v2 from {self.data_dir}")
        
        samples = []
        
        # Method 1: Try to use List_of_testing_videos.txt if it exists
        label_file = self.data_dir / 'List_of_testing_videos.txt'
        
        if label_file.exists():
            print(f"📋 Found label file: {label_file}")
            samples = self._load_from_txt_file(label_file)
        
        # Method 2: If no samples from TXT or TXT doesn't exist, use folder structure
        if not samples:
            print(f"📁 Loading from folder structure...")
            samples = self._load_from_folders()
        
        # Analyze distribution
        if samples:
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            print(f"✅ Loaded {len(samples)} Celeb-DF samples")
            print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
            print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        else:
            print(f"❌ No Celeb-DF samples found")
        
        return samples
    
    def _load_from_txt_file(self, label_file):
        """Load samples from TXT label file"""
        
        samples = []
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            print(f"📊 Found {len(lines)} entries in label file")
            
            # Show sample lines to understand format
            sample_lines = [line.strip() for line in lines[:3] if line.strip()]
            print(f"📋 Sample entries:")
            for line in sample_lines:
                print(f"   {line}")
            
            for line_num, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Parse line format: "1 YouTube-real/00170.mp4" or "0 Celeb-synthesis/id0_id2_0001.mp4"
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    label = int(parts[0])  # 0 = real, 1 = fake
                    video_path_rel = parts[1]
                    
                    # Find full path to video
                    full_path = self.data_dir / video_path_rel
                    
                    if full_path.exists():
                        samples.append({
                            'video_path': str(full_path),
                            'label': label,
                            'source': Path(video_path_rel).parent.name,
                            'dataset': 'celebdf'
                        })
                    else:
                        # Try to find video in the actual folder structure
                        video_filename = Path(video_path_rel).name
                        found = False
                        
                        # Search in the three main directories
                        search_dirs = ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']
                        for search_dir in search_dirs:
                            search_path = self.data_dir / search_dir
                            if search_path.exists():
                                video_path = search_path / video_filename
                                if video_path.exists():
                                    samples.append({
                                        'video_path': str(video_path),
                                        'label': label,
                                        'source': search_dir,
                                        'dataset': 'celebdf'
                                    })
                                    found = True
                                    break
                        
                        if not found:
                            print(f"⚠️ Video not found: {video_path_rel}")
        
        except Exception as e:
            print(f"❌ Error reading label file: {e}")
        
        return samples
    
    def _load_from_folders(self):
        """Load samples directly from folder structure"""
        
        samples = []
        
        # Based on the folder structure shown:
        # Celeb-real/ -> Real videos (label = 0)
        # Celeb-synthesis/ -> Fake videos (label = 1) 
        # YouTube-real/ -> Real videos (label = 0)
        
        folder_mappings = [
            ('Celeb-real', 0),      # Real
            ('YouTube-real', 0),    # Real
            ('Celeb-synthesis', 1)  # Fake
        ]
        
        for folder_name, label in folder_mappings:
            folder_path = self.data_dir / folder_name
            
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
                        'source': folder_name,
                        'dataset': 'celebdf'
                    })
            else:
                print(f"⚠️ Directory not found: {folder_path}")
        
        return samples

class WildDeepfakeDatasetLoader:
    """Loader for Wild Deepfake dataset with separate folders"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_samples(self):
        """Load Wild Deepfake samples from separate real/fake folders"""
        
        print(f"📂 Loading Wild Deepfake from {self.data_dir}")
        
        samples = []
        
        # Based on folder structure: test/, train/, valid/
        # Each should contain real/ and fake/ subdirectories
        splits = ['test', 'train', 'valid']
        
        for split in splits:
            split_dir = self.data_dir / split
            if not split_dir.exists():
                print(f"⚠️ Split directory not found: {split}")
                continue
                
            print(f"📁 Processing {split} split")
            
            # Check what's inside the split directory
            subdirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
            print(f"   Subdirectories in {split}: {subdirs}")
            
            # Real videos folder
            real_dir = split_dir / 'real'
            if real_dir.exists():
                real_videos = []
                # Use rglob for recursive search
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
                    real_videos.extend(list(real_dir.rglob(f'*{ext}')))
                
                for video_path in real_videos:
                    samples.append({
                        'video_path': str(video_path),
                        'label': 0,  # Real
                        'split': split,
                        'dataset': 'wilddeepfake'
                    })
                
                print(f"   📹 Real videos: {len(real_videos)}")
            else:
                print(f"   ⚠️ No 'real' directory in {split}")
            
            # Fake videos folder
            fake_dir = split_dir / 'fake'
            if fake_dir.exists():
                fake_videos = []
                # Use rglob for recursive search
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
                    fake_videos.extend(list(fake_dir.rglob(f'*{ext}')))
                
                for video_path in fake_videos:
                    samples.append({
                        'video_path': str(video_path),
                        'label': 1,  # Fake
                        'split': split,
                        'dataset': 'wilddeepfake'
                    })
                
                print(f"   📹 Fake videos: {len(fake_videos)}")
            else:
                print(f"   ⚠️ No 'fake' directory in {split}")
        
        # If no samples found in splits, check if videos are directly in real/fake at root
        if not samples:
            print(f"📁 No split structure found, checking root level...")
            
            # Real videos at root
            real_dir = self.data_dir / 'real'
            if real_dir.exists():
                real_videos = []
                # Use rglob for recursive search
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
                    real_videos.extend(list(real_dir.rglob(f'*{ext}')))
                
                for video_path in real_videos:
                    samples.append({
                        'video_path': str(video_path),
                        'label': 0,  # Real
                        'dataset': 'wilddeepfake'
                    })
                
                print(f"📹 Real videos at root: {len(real_videos)}")
            
            # Fake videos at root
            fake_dir = self.data_dir / 'fake'
            if fake_dir.exists():
                fake_videos = []
                # Use rglob for recursive search
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
                    fake_videos.extend(list(fake_dir.rglob(f'*{ext}')))
                
                for video_path in fake_videos:
                    samples.append({
                        'video_path': str(video_path),
                        'label': 1,  # Fake
                        'dataset': 'wilddeepfake'
                    })
                
                print(f"📹 Fake videos at root: {len(fake_videos)}")
        
        # Analyze distribution
        if samples:
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            print(f"✅ Loaded {len(samples)} Wild Deepfake samples")
            print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
            print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
            
            # Show distribution by split
            splits_found = set(s.get('split', 'root') for s in samples)
            for split in splits_found:
                split_samples = [s for s in samples if s.get('split', 'root') == split]
                split_real = sum(1 for s in split_samples if s['label'] == 0)
                split_fake = sum(1 for s in split_samples if s['label'] == 1)
                print(f"   {split}: {len(split_samples)} total ({split_real} real, {split_fake} fake)")
        else:
            print(f"❌ No Wild Deepfake samples found")
        
        return samples

class DFDCDatasetLoader:
    """Loader for DFDC dataset with JSON metadata"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    
    def load_chunk_samples(self, chunk_idx):
        """Load samples from a specific DFDC chunk"""
        
        print(f"📂 Loading DFDC chunk {chunk_idx}")
        
        # Find chunk directory: /kaggle/input/dfdc-10-deepfake-detection/dfdc_train_part_XX/
        chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
        chunk_dir = self.data_dir / chunk_name
        
        if not chunk_dir.exists():
            print(f"❌ Chunk directory not found: {chunk_dir}")
            return []
        
        # Check for subdirectory structure: dfdc_train_part_XX/dfdc_train_part_X/
        subdir_name = f"dfdc_train_part_{chunk_idx}"
        subdir_path = chunk_dir / subdir_name
        
        # Use subdirectory if it exists, otherwise use main directory
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
        
        # Find video files in the chunk directory
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(list(chunk_path.glob(f"*{ext}")))
        
        print(f"🎬 Found {len(video_files)} video files")
        
        # Create samples by matching video files with metadata
        samples = []
        matched_count = 0
        
        for video_file in video_files:
            video_name = video_file.stem
            
            # Try different matching strategies for metadata keys
            metadata_key = None
            
            # Strategy 1: Direct stem match
            if video_name in metadata:
                metadata_key = video_name
            # Strategy 2: Stem + .mp4
            elif f"{video_name}.mp4" in metadata:
                metadata_key = f"{video_name}.mp4"
            # Strategy 3: Full filename
            elif video_file.name in metadata:
                metadata_key = video_file.name
            
            if metadata_key:
                entry = metadata[metadata_key]
                label_value = entry.get('label', 'UNKNOWN')
                
                if label_value == 'REAL':
                    label = 0  # Real
                    matched_count += 1
                elif label_value == 'FAKE':
                    label = 1  # Fake
                    matched_count += 1
                else:
                    print(f"⚠️ Unknown label '{label_value}' for {video_name}")
                    continue
                
                samples.append({
                    'video_path': str(video_file),
                    'label': label,
                    'chunk': chunk_idx,
                    'original': entry.get('original', None),
                    'dataset': 'dfdc'
                })
            else:
                print(f"⚠️ No metadata found for video: {video_name}")
        
        # Analyze distribution
        if samples:
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            
            print(f"✅ Loaded {len(samples)} samples from DFDC chunk {chunk_idx}")
            print(f"   📊 Matched: {matched_count}/{len(video_files)} videos")
            print(f"   📈 Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
            print(f"   📈 Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        else:
            print(f"❌ No samples loaded from DFDC chunk {chunk_idx}")
        
        return samples

# ============================================================================
# UNIFIED DATASET CLASS
# ============================================================================
class UnifiedDeepfakeDataset(Dataset):
    """Unified dataset class for all deepfake datasets"""
    
    def __init__(self, samples, transform=None, stage=1):
        self.samples = samples
        self.transform = transform
        self.stage = stage
        
        # Analyze distribution
        real_count = sum(1 for s in samples if s['label'] == 0)
        fake_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"📊 Dataset Stage {stage}:")
        print(f"   Total: {len(samples)} samples")
        print(f"   Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
        print(f"   Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
        
        # Store for balanced sampling
        self.real_count = real_count
        self.fake_count = fake_count
    
    def _extract_frames_enhanced(self, video_path, num_frames=8):
        """Enhanced frame extraction with stage-specific preprocessing"""
        
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # Smart frame selection based on stage
                if self.stage <= 2:  # Early stages: uniform sampling
                    indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
                else:  # Later stages: focus on middle frames
                    start_frame = max(0, total_frames // 4)
                    end_frame = min(total_frames, 3 * total_frames // 4)
                    indices = np.linspace(start_frame, end_frame-1, min(num_frames, end_frame-start_frame), dtype=int)
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self._preprocess_for_stage(frame)
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
    
    def _preprocess_for_stage(self, frame):
        """Stage-specific preprocessing"""
        
        if self.stage == 1:  # FaceForensics: Clean preprocessing
            return frame
        
        elif self.stage == 2:  # Celeb-DF: Enhanced contrast
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        elif self.stage == 3:  # Wild Deepfake: Noise reduction
            return cv2.bilateralFilter(frame, 9, 75, 75)
        
        else:  # DFDC: Full enhancement
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            return cv2.bilateralFilter(enhanced_rgb, 5, 50, 50)
    
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
        frames = self._extract_frames_enhanced(sample['video_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        return frame, label

# ============================================================================
# PROGRESSIVE TRAINER
# ============================================================================
class ProgressiveDeepfakeTrainer:
    """Progressive trainer for multi-dataset deepfake detection"""
    
    def __init__(self, output_dir, checkpoint_path=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Initializing Progressive Deepfake Trainer...")
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            max_checkpoints=CHECKPOINT_MANAGEMENT['max_checkpoints'],
            save_frequency_gb=CHECKPOINT_MANAGEMENT['save_frequency_gb']
        )
        
        # Initialize model
        self.model = ProgressiveSpecialistModel(num_classes=2, model_type='progressive')
        self.model.to(DEVICE)
        
        # Initialize dataset loaders
        self.loaders = {
            'faceforensics': FaceForensicsDatasetLoader(DATASET_CONFIGS['faceforensics']['path']),
            'celebdf': CelebDFDatasetLoader(DATASET_CONFIGS['celebdf']['path']),
            'wilddeepfake': WildDeepfakeDatasetLoader(DATASET_CONFIGS['wilddeepfake']['path']),
            'dfdc': DFDCDatasetLoader(DATASET_CONFIGS['dfdc']['path'])
        }
        
        # Training state
        self.current_stage = 1
        self.training_history = []
        self.stage_metrics = {}
        self.total_data_processed_gb = 0.0
        
        # Initialize optimizer and scheduler (will be reset per stage)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        print(f"✅ Progressive trainer ready!")
    
    def _estimate_dataset_size(self, samples):
        """Estimate dataset size in GB"""
        return self.checkpoint_manager.estimate_data_size_gb(len(samples))
    
    def _update_data_processed(self, samples):
        """Update total data processed and check for checkpoint"""
        data_size_gb = self._estimate_dataset_size(samples)
        self.total_data_processed_gb += data_size_gb
        
        print(f"📊 Data processed: {self.total_data_processed_gb:.2f} GB total (+{data_size_gb:.2f} GB)")
        
        return data_size_gb
    
    def _setup_stage_training(self, stage, dataset_name):
        """Setup training components for specific stage"""
        
        print(f"\n🔧 Setting up training for Stage {stage}: {dataset_name}")
        
        # Set model stage
        stage_info = {
            'stage': stage,
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat()
        }
        self.model.set_stage(stage, stage_info)
        
        # Stage-specific learning rates and loss functions
        stage_configs = {
            1: {'lr': 1e-4, 'weight_decay': 1e-5, 'loss': 'ce'},  # FaceForensics: Standard
            2: {'lr': 5e-5, 'weight_decay': 1e-4, 'loss': 'focal'},  # Celeb-DF: Lower LR
            3: {'lr': 3e-5, 'weight_decay': 1e-4, 'loss': 'focal'},  # Wild: Even lower
            4: {'lr': 1e-5, 'weight_decay': 1e-3, 'loss': 'weighted'}  # DFDC: Minimal LR
        }
        
        config = stage_configs[stage]
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2
        )
        
        # Setup loss function
        if config['loss'] == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif config['loss'] == 'focal':
            self.criterion = self._focal_loss
        else:  # weighted
            self.criterion = self._weighted_loss
        
        print(f"   Learning Rate: {config['lr']}")
        print(f"   Loss Function: {config['loss']}")
    
    def _focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def _weighted_loss(self, outputs, targets):
        """Weighted cross-entropy for severe imbalance"""
        
        # Dynamic weights based on current batch
        unique, counts = torch.unique(targets, return_counts=True)
        weights = torch.ones(2, device=DEVICE)
        
        for i, count in zip(unique, counts):
            weights[i] = len(targets) / (2 * count)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        return criterion(outputs, targets)
    
    def _create_balanced_dataloader(self, dataset):
        """Create balanced dataloader with weighted sampling"""
        
        # Get class weights
        class_weights = dataset.get_class_weights()
        
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
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR
        )
    
    def _get_stage_transforms(self, stage):
        """Get stage-specific data augmentations"""
        
        base_transforms = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
        ]
        
        if stage == 1:  # FaceForensics: Minimal augmentation
            stage_transforms = [
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ]
        elif stage == 2:  # Celeb-DF: Moderate augmentation
            stage_transforms = [
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
            ]
        elif stage == 3:  # Wild: Strong augmentation
            stage_transforms = [
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3)
            ]
        else:  # DFDC: Maximum augmentation
            stage_transforms = [
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ]
        
        final_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        return transforms.Compose(base_transforms + stage_transforms + final_transforms)
    
    def train_stage_1_faceforensics(self):
        """Stage 1: Foundation training on FaceForensics++"""
        
        if not DATASET_CONFIGS['faceforensics']['enabled']:
            print("⏭️ Skipping FaceForensics++ (disabled)")
            return
        
        print(f"\n🎯 STAGE 1: FACEFORENSICS++ FOUNDATION TRAINING")
        print("="*60)
        
        self._setup_stage_training(1, 'faceforensics')
        
        # Load samples
        samples = self.loaders['faceforensics'].load_samples()
        if not samples:
            print("❌ No FaceForensics++ samples found")
            return
        
        # Update data processed
        stage_data_size = self._update_data_processed(samples)
        
        # Create dataset and dataloader
        transform = self._get_stage_transforms(1)
        dataset = UnifiedDeepfakeDataset(samples, transform, stage=1)
        dataloader = self._create_balanced_dataloader(dataset)
        
        # Training loop - BALANCED APPROACH
        # Instead of training on each method separately, train on balanced batches
        # that include both real and fake videos for proper learning
        
        print(f"\n📚 Training on balanced FaceForensics++ dataset")
        print(f"   Total samples: {len(samples)}")
        
        # Analyze overall distribution
        real_samples = [s for s in samples if s['label'] == 0]
        fake_samples = [s for s in samples if s['label'] == 1]
        
        print(f"   Real samples: {len(real_samples)} ({len(real_samples)/len(samples)*100:.1f}%)")
        print(f"   Fake samples: {len(fake_samples)} ({len(fake_samples)/len(samples)*100:.1f}%)")
        
        # Show distribution by method
        methods_found = set(s.get('method', 'unknown') for s in samples)
        for method in methods_found:
            method_samples = [s for s in samples if s.get('method') == method]
            method_real = sum(1 for s in method_samples if s['label'] == 0)
            method_fake = sum(1 for s in method_samples if s['label'] == 1)
            print(f"   {method}: {len(method_samples)} total ({method_real} real, {method_fake} fake)")
        
        # Train on the full balanced dataset
        epochs_total = DATASET_CONFIGS['faceforensics']['epochs_per_method'] * 2  # More epochs since we're not splitting
        
        for epoch in range(epochs_total):
            metrics = self._train_epoch(dataloader, f"Stage1-Balanced-E{epoch+1}")
            self.training_history.append(metrics)
            
            # Check for checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0 and self.checkpoint_manager.should_save_checkpoint(self.total_data_processed_gb):
                checkpoint_name = f"stage1_balanced_epoch{epoch+1}"
                self._save_smart_checkpoint(checkpoint_name, metrics)
        
        # Final evaluation on all FaceForensics++ data
        final_metrics = self._evaluate_model(dataloader, "Stage1-Final")
        self.stage_metrics[1] = final_metrics
        
        # Always save stage completion checkpoint
        self._save_smart_checkpoint("stage1_faceforensics_complete", final_metrics)
        
        print(f"✅ Stage 1 completed: {final_metrics['accuracy']*100:.2f}% accuracy")
    
    def train_stage_2_celebdf(self):
        """Stage 2: Realism adaptation on Celeb-DF v2"""
        
        if not DATASET_CONFIGS['celebdf']['enabled']:
            print("⏭️ Skipping Celeb-DF v2 (disabled)")
            return
        
        print(f"\n🎯 STAGE 2: CELEB-DF V2 REALISM ADAPTATION")
        print("="*60)
        
        self._setup_stage_training(2, 'celebdf')
        
        # Load samples
        samples = self.loaders['celebdf'].load_samples()
        if not samples:
            print("❌ No Celeb-DF samples found")
            return
        
        # Update data processed
        stage_data_size = self._update_data_processed(samples)
        
        # Create dataset and dataloader
        transform = self._get_stage_transforms(2)
        dataset = UnifiedDeepfakeDataset(samples, transform, stage=2)
        dataloader = self._create_balanced_dataloader(dataset)
        
        # Training loop
        epochs = DATASET_CONFIGS['celebdf']['epochs']
        
        for epoch in range(epochs):
            metrics = self._train_epoch(dataloader, f"Stage2-E{epoch+1}")
            self.training_history.append(metrics)
            
            # Check for checkpoint
            if self.checkpoint_manager.should_save_checkpoint(self.total_data_processed_gb):
                checkpoint_name = f"stage2_celebdf_epoch{epoch+1}"
                self._save_smart_checkpoint(checkpoint_name, metrics)
        
        # Final evaluation
        final_metrics = self._evaluate_model(dataloader, "Stage2-Final")
        self.stage_metrics[2] = final_metrics
        
        # Save stage completion checkpoint
        self._save_smart_checkpoint("stage2_celebdf_complete", final_metrics)
        
        print(f"✅ Stage 2 completed: {final_metrics['accuracy']*100:.2f}% accuracy")
    
    def train_stage_3_wilddeepfake(self):
        """Stage 3: Real-world adaptation on Wild Deepfake"""
        
        if not DATASET_CONFIGS['wilddeepfake']['enabled']:
            print("⏭️ Skipping Wild Deepfake (disabled)")
            return
        
        print(f"\n🎯 STAGE 3: WILD DEEPFAKE REAL-WORLD ADAPTATION")
        print("="*60)
        
        self._setup_stage_training(3, 'wilddeepfake')
        
        # Load samples
        samples = self.loaders['wilddeepfake'].load_samples()
        if not samples:
            print("❌ No Wild Deepfake samples found")
            return
        
        # Update data processed
        stage_data_size = self._update_data_processed(samples)
        
        # Create dataset and dataloader
        transform = self._get_stage_transforms(3)
        dataset = UnifiedDeepfakeDataset(samples, transform, stage=3)
        dataloader = self._create_balanced_dataloader(dataset)
        
        # Training loop
        epochs = DATASET_CONFIGS['wilddeepfake']['epochs']
        
        for epoch in range(epochs):
            metrics = self._train_epoch(dataloader, f"Stage3-E{epoch+1}")
            self.training_history.append(metrics)
            
            # Check for checkpoint
            if self.checkpoint_manager.should_save_checkpoint(self.total_data_processed_gb):
                checkpoint_name = f"stage3_wilddeepfake_epoch{epoch+1}"
                self._save_smart_checkpoint(checkpoint_name, metrics)
        
        # Final evaluation
        final_metrics = self._evaluate_model(dataloader, "Stage3-Final")
        self.stage_metrics[3] = final_metrics
        
        # Save stage completion checkpoint
        self._save_smart_checkpoint("stage3_wilddeepfake_complete", final_metrics)
        
        print(f"✅ Stage 3 completed: {final_metrics['accuracy']*100:.2f}% accuracy")
    
    def train_stage_4_dfdc(self):
        """Stage 4: Large-scale training on DFDC"""
        
        if not DATASET_CONFIGS['dfdc']['enabled']:
            print("⏭️ Skipping DFDC (disabled)")
            return
        
        print(f"\n🎯 STAGE 4: DFDC LARGE-SCALE TRAINING")
        print("="*60)
        
        self._setup_stage_training(4, 'dfdc')
        
        # Train on chunks progressively (most balanced first)
        chunk_order = [9, 8, 3, 5, 7, 2, 6, 4, 1, 0]  # From most to least balanced
        epochs_per_chunk = DATASET_CONFIGS['dfdc']['epochs_per_chunk']
        
        for chunk_idx in chunk_order:
            print(f"\n📊 Training on DFDC chunk {chunk_idx}")
            
            # Load chunk samples
            samples = self.loaders['dfdc'].load_chunk_samples(chunk_idx)
            if not samples:
                continue
            
            # Update data processed
            chunk_data_size = self._update_data_processed(samples)
            
            # Create dataset and dataloader
            transform = self._get_stage_transforms(4)
            dataset = UnifiedDeepfakeDataset(samples, transform, stage=4)
            dataloader = self._create_balanced_dataloader(dataset)
            
            # Train on this chunk
            for epoch in range(epochs_per_chunk):
                metrics = self._train_epoch(dataloader, f"Stage4-Chunk{chunk_idx}-E{epoch+1}")
                self.training_history.append(metrics)
                
                # Check for checkpoint after each epoch (DFDC is large)
                if self.checkpoint_manager.should_save_checkpoint(self.total_data_processed_gb):
                    checkpoint_name = f"stage4_dfdc_chunk{chunk_idx}_epoch{epoch+1}"
                    self._save_smart_checkpoint(checkpoint_name, metrics)
            
            # Save chunk completion checkpoint
            chunk_metrics = self._evaluate_model(dataloader, f"Stage4-Chunk{chunk_idx}")
            chunk_checkpoint_name = f"stage4_dfdc_chunk{chunk_idx}_complete"
            self._save_smart_checkpoint(chunk_checkpoint_name, chunk_metrics)
        
        # Final evaluation on all DFDC data (sample from all chunks)
        print(f"\n📊 Final DFDC evaluation...")
        all_samples = []
        for chunk_idx in range(10):
            chunk_samples = self.loaders['dfdc'].load_chunk_samples(chunk_idx)
            # Sample 10% from each chunk for final evaluation
            sample_size = max(1, len(chunk_samples) // 10)
            all_samples.extend(random.sample(chunk_samples, min(sample_size, len(chunk_samples))))
        
        if all_samples:
            transform = self._get_stage_transforms(4)
            final_dataset = UnifiedDeepfakeDataset(all_samples, transform, stage=4)
            final_dataloader = self._create_balanced_dataloader(final_dataset)
            
            final_metrics = self._evaluate_model(final_dataloader, "Stage4-Final")
            self.stage_metrics[4] = final_metrics
            
            # Save final stage checkpoint
            self._save_smart_checkpoint("stage4_dfdc_final", final_metrics)
            
            print(f"✅ Stage 4 completed: {final_metrics['accuracy']*100:.2f}% accuracy")
    
    def _train_epoch(self, dataloader, epoch_name):
        """Train for one epoch"""
        
        self.model.train()
        
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        pbar = tqdm(dataloader, desc=f"Training {epoch_name}")
        
        for batch_idx, (frames, labels) in enumerate(pbar):
            try:
                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.squeeze().to(DEVICE, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(frames)
                        loss = self.criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(frames)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
                # Class-specific accuracy
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pred_label = predicted[i].item()
                    
                    if true_label == 0:  # Real
                        real_total += 1
                        if pred_label == true_label:
                            real_correct += 1
                    else:  # Fake
                        fake_total += 1
                        if pred_label == true_label:
                            fake_correct += 1
                
                # Update progress
                current_acc = correct_predictions / total_predictions * 100
                real_acc = (real_correct / real_total * 100) if real_total > 0 else 0
                fake_acc = (fake_correct / fake_total * 100) if fake_total > 0 else 0
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%',
                    'Real': f'{real_acc:.1f}%',
                    'Fake': f'{fake_acc:.1f}%',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"⚠️ Batch error: {e}")
                continue
        
        self.scheduler.step()
        
        # Calculate final metrics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
        balanced_accuracy = (real_accuracy + fake_accuracy) / 2
        
        metrics = {
            'epoch_name': epoch_name,
            'loss': avg_loss,
            'accuracy': accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'bias_difference': abs(real_accuracy - fake_accuracy),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ {epoch_name}: Acc={accuracy*100:.2f}%, Real={real_accuracy*100:.2f}%, Fake={fake_accuracy*100:.2f}%, Bias={abs(real_accuracy - fake_accuracy)*100:.1f}%")
        
        return metrics
    
    def _evaluate_model(self, dataloader, eval_name):
        """Evaluate model on given dataloader"""
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for frames, labels in tqdm(dataloader, desc=f"Evaluating {eval_name}"):
                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.squeeze().to(DEVICE, non_blocking=True)
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(frames)
                else:
                    outputs = self.model(frames)
                
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Class-specific metrics
        cm = confusion_matrix(all_labels, all_predictions)
        real_accuracy = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        fake_accuracy = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        # AUC-ROC
        try:
            probabilities_positive = [p[1] for p in all_probabilities]
            auc_roc = roc_auc_score(all_labels, probabilities_positive)
        except:
            auc_roc = 0.0
        
        metrics = {
            'eval_name': eval_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'balanced_accuracy': (real_accuracy + fake_accuracy) / 2,
            'bias_difference': abs(real_accuracy - fake_accuracy),
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"📊 {eval_name} Evaluation:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   F1-Score: {f1*100:.2f}%")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        print(f"   Real Accuracy: {real_accuracy*100:.2f}%")
        print(f"   Fake Accuracy: {fake_accuracy*100:.2f}%")
        print(f"   Bias Difference: {abs(real_accuracy - fake_accuracy)*100:.1f}%")
        
        return metrics
    
    def _save_smart_checkpoint(self, checkpoint_name, metrics):
        """Save checkpoint using the smart checkpoint manager"""
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'current_stage': self.current_stage,
            'training_history': self.training_history,
            'stage_metrics': self.stage_metrics,
            'total_data_processed_gb': self.total_data_processed_gb,
            'model_type': 'progressive_specialist',
            'dataset_configs': DATASET_CONFIGS
        }
        
        return self.checkpoint_manager.save_checkpoint(
            model_state=model_state,
            checkpoint_name=checkpoint_name,
            metrics=metrics,
            data_processed_gb=self.total_data_processed_gb
        )
    
    def _save_checkpoint(self, stage, metrics, checkpoint_name):
        """Legacy checkpoint method - redirects to smart checkpoint manager"""
        return self._save_smart_checkpoint(checkpoint_name, metrics)
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"📥 Loading checkpoint: {checkpoint_path}")
        
        try:
            # Try with weights_only=False for compatibility with older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model state dict")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"✅ Loaded model weights directly")
            
            # Load training state if available
            self.current_stage = checkpoint.get('current_stage', 1)
            self.training_history = checkpoint.get('training_history', [])
            self.stage_metrics = checkpoint.get('stage_metrics', {})
            
            if 'epoch' in checkpoint:
                print(f"📊 Previous epoch: {checkpoint['epoch']}")
            if 'accuracy' in checkpoint:
                print(f"📊 Previous accuracy: {checkpoint['accuracy']*100:.2f}%")
                
            print(f"✅ Loaded checkpoint from stage {self.current_stage}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"🔄 Starting with fresh model weights")
    
    def configure_stage3_balanced_training(self):
        """Configure trainer for balanced Stage 3 training"""
        
        print(f"🔧 CONFIGURING STAGE 3 BALANCED TRAINING")
        print("="*60)
        
        # Set current stage
        self.current_stage = 3
        
        # Setup enhanced optimizer for balance
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=STAGE3_ENHANCED_CONFIG['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup scheduler with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=3, T_mult=2, eta_min=1e-6
        )
        
        # Setup balanced focal loss
        self.criterion = lambda outputs, targets: balanced_focal_loss(
            outputs, targets,
            alpha=STAGE3_ENHANCED_CONFIG['focal_alpha'],
            gamma=STAGE3_ENHANCED_CONFIG['focal_gamma'],
            label_smoothing=STAGE3_ENHANCED_CONFIG['label_smoothing']
        )
        
        print(f"✅ Configured for balanced training:")
        print(f"   📈 Learning Rate: {STAGE3_ENHANCED_CONFIG['learning_rate']}")
        print(f"   📈 Loss: Balanced Focal Loss")
        print(f"   📈 Target Real Accuracy: {STAGE3_ENHANCED_CONFIG['target_real_accuracy']*100:.0f}%")
        print(f"   📈 Target Fake Accuracy: {STAGE3_ENHANCED_CONFIG['target_fake_accuracy']*100:.0f}%")
        print(f"   📈 Max Bias Tolerance: {STAGE3_ENHANCED_CONFIG['max_bias_tolerance']*100:.0f}%")
    
    def run_stage3_balanced_training(self):
        """Run Stage 3 training with balanced real/fake detection"""
        
        print(f"\n🎯 STAGE 3: WILD DEEPFAKE BALANCED TRAINING")
        print("="*60)
        
        # Load Wild Deepfake dataset
        print(f"📂 Loading Wild Deepfake dataset...")
        samples = self.loaders['wilddeepfake'].load_samples()
        
        if not samples:
            print("❌ No Wild Deepfake samples found! Skipping Stage 3.")
            return
        
        # Analyze dataset balance
        real_samples = [s for s in samples if s['label'] == 0]
        fake_samples = [s for s in samples if s['label'] == 1]
        
        print(f"📊 Wild Deepfake Dataset:")
        print(f"   Total: {len(samples)} samples")
        print(f"   Real: {len(real_samples)} ({len(real_samples)/len(samples)*100:.1f}%)")
        print(f"   Fake: {len(fake_samples)} ({len(fake_samples)/len(samples)*100:.1f}%)")
        
        # Create balanced dataset with enhanced sampling
        transform = self._get_stage_transforms(3)
        dataset = UnifiedDeepfakeDataset(samples, transform, stage=3)
        
        # Create balanced dataloader with enhanced sampling
        dataloader = self._create_enhanced_balanced_dataloader(dataset)
        
        # Training loop with balance monitoring
        best_balance_score = 0.0
        patience_counter = 0
        
        for epoch in range(1, STAGE3_ENHANCED_CONFIG['epochs'] + 1):
            print(f"\n📚 Stage 3 - Epoch {epoch}/{STAGE3_ENHANCED_CONFIG['epochs']}")
            print("-" * 50)
            
            # Train one epoch
            train_metrics = self._train_balanced_epoch(dataloader, epoch)
            
            # Evaluate balance
            eval_metrics = self._evaluate_balance(dataloader)
            
            # Calculate balance score (higher is better)
            real_acc = eval_metrics['real_accuracy']
            fake_acc = eval_metrics['fake_accuracy']
            bias = abs(real_acc - fake_acc)
            
            # Balance score: reward high accuracy and low bias
            balance_score = (real_acc + fake_acc) / 2 - bias
            
            print(f"\n📊 Epoch {epoch} Results:")
            print(f"   Overall Accuracy: {eval_metrics['accuracy']*100:.2f}%")
            print(f"   Real Accuracy: {real_acc*100:.2f}%")
            print(f"   Fake Accuracy: {fake_acc*100:.2f}%")
            print(f"   Bias: {bias*100:.2f}%")
            print(f"   Balance Score: {balance_score:.3f}")
            
            # Check if targets are met
            targets_met = (
                real_acc >= STAGE3_ENHANCED_CONFIG['target_real_accuracy'] and
                fake_acc >= STAGE3_ENHANCED_CONFIG['target_fake_accuracy'] and
                bias <= STAGE3_ENHANCED_CONFIG['max_bias_tolerance']
            )
            
            if targets_met:
                print(f"🎉 TARGETS ACHIEVED!")
                print(f"   ✅ Real accuracy: {real_acc*100:.1f}% >= {STAGE3_ENHANCED_CONFIG['target_real_accuracy']*100:.0f}%")
                print(f"   ✅ Fake accuracy: {fake_acc*100:.1f}% >= {STAGE3_ENHANCED_CONFIG['target_fake_accuracy']*100:.0f}%")
                print(f"   ✅ Bias: {bias*100:.1f}% <= {STAGE3_ENHANCED_CONFIG['max_bias_tolerance']*100:.0f}%")
            
            # Save checkpoint if improved
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                patience_counter = 0
                
                checkpoint_name = f"stage3_wilddeepfake_balanced_epoch{epoch}"
                self._save_checkpoint(checkpoint_name, eval_metrics, epoch)
                print(f"💾 Saved best balanced model (score: {balance_score:.3f})")
                
                if targets_met:
                    print(f"🎯 Targets achieved! Stopping early.")
                    break
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= STAGE3_ENHANCED_CONFIG['early_stopping_patience']:
                print(f"⏹️ Early stopping: No improvement for {patience_counter} epochs")
                break
            
            # Update learning rate
            self.scheduler.step()
        
        # Final evaluation
        print(f"\n📊 STAGE 3 FINAL RESULTS:")
        final_metrics = self._evaluate_balance(dataloader)
        
        print(f"   🎯 Final Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"   🎯 Final Real Detection: {final_metrics['real_accuracy']*100:.2f}%")
        print(f"   🎯 Final Fake Detection: {final_metrics['fake_accuracy']*100:.2f}%")
        print(f"   🎯 Final Bias: {abs(final_metrics['real_accuracy'] - final_metrics['fake_accuracy'])*100:.2f}%")
        
        # Save final checkpoint
        final_checkpoint_name = f"stage3_wilddeepfake_final"
        self._save_checkpoint(final_checkpoint_name, final_metrics, STAGE3_ENHANCED_CONFIG['epochs'])
        
        print(f"✅ Stage 3 completed!")
        return final_metrics
    
    def _create_enhanced_balanced_dataloader(self, dataset):
        """Create enhanced balanced dataloader for Stage 3"""
        
        # Get class weights with enhanced balancing
        class_weights = dataset.get_class_weights()
        
        # Apply additional balancing for Stage 3
        enhanced_weights = [
            class_weights[0] * STAGE3_ENHANCED_CONFIG['class_weights'][0],
            class_weights[1] * STAGE3_ENHANCED_CONFIG['class_weights'][1]
        ]
        
        print(f"📊 Enhanced class weights: Real={enhanced_weights[0]:.3f}, Fake={enhanced_weights[1]:.3f}")
        
        # Create sample weights
        sample_weights = []
        for sample in dataset.samples:
            sample_weights.append(enhanced_weights[sample['label']])
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return DataLoader(
            dataset,
            batch_size=STAGE3_ENHANCED_CONFIG['batch_size'],
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR
        )
    
    def _train_balanced_epoch(self, dataloader, epoch):
        """Train one epoch with balance-focused techniques"""
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(DEVICE), targets.squeeze().to(DEVICE)
            
            # Apply mixup augmentation occasionally
            if np.random.random() < STAGE3_ENHANCED_CONFIG['mixup_alpha']:
                mixed_data, targets_a, targets_b, lam = mixup_data(data, targets, STAGE3_ENHANCED_CONFIG['mixup_alpha'])
                
                self.optimizer.zero_grad()
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(mixed_data)
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(mixed_data)
                    loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                    loss.backward()
                    self.optimizer.step()
            else:
                # Regular training
                self.optimizer.zero_grad()
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
            
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
                bias = abs(real_acc - fake_acc)
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{acc:.1f}%',
                    'Real': f'{real_acc:.1f}%',
                    'Fake': f'{fake_acc:.1f}%',
                    'Bias': f'{bias:.1f}%'
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
    
    def _evaluate_balance(self, dataloader):
        """Evaluate model with focus on balance metrics"""
        
        self.model.eval()
        correct = 0
        total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(DEVICE), targets.squeeze().to(DEVICE)
                outputs = self.model(data)
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
    
    def _save_checkpoint(self, name, metrics, epoch):
        """Save checkpoint with current training state"""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'current_stage': self.current_stage,
            'metrics': metrics,
            'training_history': self.training_history,
            'stage_metrics': self.stage_metrics,
            'total_data_processed_gb': self.total_data_processed_gb,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'progressive_specialist',
            'stage3_config': STAGE3_ENHANCED_CONFIG
        }
        
        checkpoint_path = self.output_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path.name}")
        return checkpoint_path

    def configure_stage3_training(self):
        """Configure trainer for Stage 3 natural training"""
        
        print("🔧 CONFIGURING STAGE 3 TRAINING")
        print("="*50)
        
        # Set current stage
        self.current_stage = 3
        
        # Setup optimizer for natural training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=STAGE3_CONFIG['learning_rate'],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Standard scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=3, T_mult=2, eta_min=1e-6
        )
        
        # Set standard focal loss as criterion
        self.criterion = lambda outputs, targets: standard_focal_loss(
            outputs, targets,
            alpha=STAGE3_CONFIG['focal_alpha'],
            gamma=STAGE3_CONFIG['focal_gamma']
        )
        
        print(f"✅ Learning Rate: {STAGE3_CONFIG['learning_rate']}")
        print(f"✅ Loss Function: Standard Focal Loss")
        print(f"✅ Focal Alpha: {STAGE3_CONFIG['focal_alpha']}")
        print(f"✅ Focal Gamma: {STAGE3_CONFIG['focal_gamma']}")
        print(f"✅ Training Strategy: Natural (no aggressive balancing)")
        print(f"✅ Epochs: {STAGE3_CONFIG['epochs']}")
        print(f"💡 Stage 4 DFDC will compensate bias with 100GB+ fake data")
    
    def run_stage3_training(self):
        """Run Stage 3 training with natural approach"""
        
        print("\n🚀 STARTING STAGE 3: WILD DEEPFAKE TRAINING")
        print("="*60)
        print("🎯 Strategy:")
        print("   📚 Natural training on Wild Deepfake dataset")
        print("   🔄 Equal real/fake data (11.25 GB)")
        print("   💡 Let model learn naturally without forcing balance")
        print("   🎯 Stage 4 DFDC will compensate with fake-heavy data")
        print()
        
        # Load Wild Deepfake dataset
        print("📂 Loading Wild Deepfake dataset...")
        try:
            samples = self.loaders['wilddeepfake'].load_samples()
            print(f"✅ Loaded {len(samples)} Wild Deepfake samples")
            
            # Check dataset balance
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            print(f"📊 Dataset balance: {real_count} real, {fake_count} fake")
            print(f"📊 Ratio: {real_count/(real_count+fake_count)*100:.1f}% real, {fake_count/(real_count+fake_count)*100:.1f}% fake")
            
        except Exception as e:
            print(f"❌ Error loading Wild Deepfake dataset: {e}")
            print("⚠️ Skipping Stage 3 training")
            return
        
        # Create dataset with natural sampling
        dataset = UnifiedDeepfakeDataset(samples, transform=self._get_stage_transforms(3))
        
        # Create data loader with natural sampling (no weighted sampling)
        dataloader = DataLoader(
            dataset,
            batch_size=STAGE3_CONFIG['batch_size'],
            shuffle=True,  # Natural shuffling
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=True
        )
        
        print(f"📊 Batches per epoch: {len(dataloader)}")
        print(f"📊 Total samples per epoch: {len(dataloader) * STAGE3_CONFIG['batch_size']}")
        
        # Training loop
        best_accuracy = 0
        patience_counter = 0
        
        for epoch in range(STAGE3_CONFIG['epochs']):
            print(f"\n📚 Epoch {epoch+1}/{STAGE3_CONFIG['epochs']}")
            print("-" * 50)
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            real_correct = 0
            real_total = 0
            fake_correct = 0
            fake_total = 0
            
            progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                self.optimizer.zero_grad()
                
                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()
                epoch_loss += loss.item()
                
                # Track real/fake accuracy
                for i in range(targets.size(0)):
                    if targets[i] == 0:  # Real
                        real_total += 1
                        if predicted[i] == 0:
                            real_correct += 1
                    else:  # Fake
                        fake_total += 1
                        if predicted[i] == 1:
                            fake_correct += 1
                
                # Update progress bar
                current_acc = 100 * epoch_correct / epoch_total
                current_real_acc = 100 * real_correct / real_total if real_total > 0 else 0
                current_fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%',
                    'Real': f'{current_real_acc:.1f}%',
                    'Fake': f'{current_fake_acc:.1f}%'
                })
            
            # Calculate epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            real_accuracy = real_correct / real_total if real_total > 0 else 0
            fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
            bias_difference = abs(real_accuracy - fake_accuracy)
            avg_loss = epoch_loss / len(dataloader)
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"   Loss: {avg_loss:.4f}")
            print(f"   Overall Accuracy: {epoch_accuracy*100:.2f}%")
            print(f"   Real Detection: {real_accuracy*100:.2f}%")
            print(f"   Fake Detection: {fake_accuracy*100:.2f}%")
            print(f"   Bias Difference: {bias_difference*100:.1f}%")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model based on overall accuracy
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                patience_counter = 0
                
                # Save best model
                metrics = {
                    'stage': 3,
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': epoch_accuracy,
                    'real_accuracy': real_accuracy,
                    'fake_accuracy': fake_accuracy,
                    'bias_difference': bias_difference,
                    'balanced_accuracy': (real_accuracy + fake_accuracy) / 2,
                    'f1_score': 2 * (real_accuracy * fake_accuracy) / (real_accuracy + fake_accuracy) if (real_accuracy + fake_accuracy) > 0 else 0,
                }
                
                checkpoint_path = self._save_smart_checkpoint(
                    f"stage3_wilddeepfake_epoch{epoch+1}", metrics
                )
                print(f"💾 Best model saved: {checkpoint_path.name}")
                
            else:
                patience_counter += 1
                if patience_counter >= STAGE3_CONFIG['early_stopping_patience']:
                    print(f"⏹️ Early stopping: No improvement for {patience_counter} epochs")
                    break
            
            # Update data processed
            data_size_gb = self._estimate_dataset_size(samples)
            self.total_data_processed_gb += data_size_gb
            print(f"📊 Total data processed: {self.total_data_processed_gb:.2f} GB")
        
        # Update stage metrics
        final_metrics = {
            'stage': 3,
            'accuracy': epoch_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'bias_difference': bias_difference,
            'balanced_accuracy': (real_accuracy + fake_accuracy) / 2,
            'f1_score': 2 * (real_accuracy * fake_accuracy) / (real_accuracy + fake_accuracy) if (real_accuracy + fake_accuracy) > 0 else 0,
        }
        
        self.stage_metrics[3] = final_metrics
        
        print(f"\n🎯 STAGE 3 COMPLETED!")
        print("="*60)
        print(f"📊 Final Results:")
        print(f"   Overall Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"   Real Detection: {final_metrics['real_accuracy']*100:.2f}%")
        print(f"   Fake Detection: {final_metrics['fake_accuracy']*100:.2f}%")
        print(f"   Bias Difference: {final_metrics['bias_difference']*100:.1f}%")
        print(f"   Balanced Accuracy: {final_metrics['balanced_accuracy']*100:.2f}%")
        print()
        print("💡 Analysis:")
        
        # Analyze improvement
        if final_metrics['fake_accuracy'] > 0.4:
            print(f"   ✅ Fake detection improved significantly!")
        elif final_metrics['fake_accuracy'] > 0.3:
            print(f"   ✅ Fake detection showing improvement")
        else:
            print(f"   ⚠️ Fake detection still needs work - Stage 4 will help")
        
        if final_metrics['bias_difference'] < 0.3:
            print(f"   ✅ Bias is reasonable")
        else:
            print(f"   💡 Bias still present - Stage 4 DFDC will compensate")
        
        print()
        print("🚀 Next Steps:")
        print("   1. Test this model on your test videos")
        print("   2. Proceed to Stage 4 DFDC training (100GB+ mostly fake)")
        print("   3. Stage 4 will naturally improve fake detection")
        print("   4. Final model will be balanced after Stage 4")
        
    def run_stage3_balanced_training(self):
        """Run Stage 3 training with balanced real/fake detection focus"""
        
        print("\n🚀 STARTING STAGE 3: BALANCED WILD DEEPFAKE TRAINING")
        print("="*60)
        print("🎯 Goals:")
        print("   📈 Improve fake detection: 32% → 70%")
        print("   📈 Maintain real detection: 84% → 80%+")
        print("   📈 Reduce bias: 52% → <20%")
        print("   📈 Overall accuracy: 58% → 75%+")
        print()
        
        # Load Wild Deepfake dataset
        print("📂 Loading Wild Deepfake dataset...")
        try:
            samples = self.loaders['wilddeepfake'].load_samples()
            print(f"✅ Loaded {len(samples)} Wild Deepfake samples")
            
            # Check dataset balance
            real_count = sum(1 for s in samples if s['label'] == 0)
            fake_count = sum(1 for s in samples if s['label'] == 1)
            print(f"📊 Dataset balance: {real_count} real, {fake_count} fake")
            
        except Exception as e:
            print(f"❌ Error loading Wild Deepfake dataset: {e}")
            print("⚠️ Skipping Stage 3 training")
            return
        
        # Create balanced dataset with weighted sampling
        dataset = VideoDataset(samples, transform=get_train_transforms())
        
        # Calculate class weights for balanced sampling
        class_counts = [real_count, fake_count]
        total_samples = len(samples)
        class_weights = [total_samples / (2 * count) for count in class_counts]
        
        # Create sample weights for balanced sampling
        sample_weights = [class_weights[sample['label']] for sample in samples]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(samples),
            replacement=True
        )
        
        # Create data loader with balanced sampling
        dataloader = DataLoader(
            dataset,
            batch_size=STAGE3_ENHANCED_CONFIG['batch_size'],
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=True
        )
        
        print(f"📊 Class weights: Real={class_weights[0]:.3f}, Fake={class_weights[1]:.3f}")
        print(f"📊 Batches per epoch: {len(dataloader)}")
        
        # Training loop with balance monitoring
        best_balance_score = 0
        patience_counter = 0
        
        for epoch in range(STAGE3_ENHANCED_CONFIG['epochs']):
            print(f"\n📚 Epoch {epoch+1}/{STAGE3_ENHANCED_CONFIG['epochs']}")
            print("-" * 50)
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            real_correct = 0
            real_total = 0
            fake_correct = 0
            fake_total = 0
            
            progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # Apply mixup augmentation
                if STAGE3_ENHANCED_CONFIG['mixup_alpha'] > 0:
                    inputs, targets_a, targets_b, lam = mixup_data(
                        inputs, targets, STAGE3_ENHANCED_CONFIG['mixup_alpha']
                    )
                    
                    self.optimizer.zero_grad()
                    
                    if USE_MIXED_PRECISION:
                        with autocast():
                            outputs = self.model(inputs)
                            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        outputs = self.model(inputs)
                        loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
                        loss.backward()
                        self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    
                    if USE_MIXED_PRECISION:
                        with autocast():
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        loss.backward()
                        self.optimizer.step()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                epoch_total += targets.size(0)
                epoch_correct += (predicted == targets).sum().item()
                epoch_loss += loss.item()
                
                # Track real/fake accuracy
                for i in range(targets.size(0)):
                    if targets[i] == 0:  # Real
                        real_total += 1
                        if predicted[i] == 0:
                            real_correct += 1
                    else:  # Fake
                        fake_total += 1
                        if predicted[i] == 1:
                            fake_correct += 1
                
                # Update progress bar
                current_acc = 100 * epoch_correct / epoch_total
                current_real_acc = 100 * real_correct / real_total if real_total > 0 else 0
                current_fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%',
                    'Real': f'{current_real_acc:.1f}%',
                    'Fake': f'{current_fake_acc:.1f}%'
                })
            
            # Calculate epoch metrics
            epoch_accuracy = epoch_correct / epoch_total
            real_accuracy = real_correct / real_total if real_total > 0 else 0
            fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
            bias_difference = abs(real_accuracy - fake_accuracy)
            balance_score = 1 - bias_difference  # Higher is better
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"📊 Epoch {epoch+1} Results:")
            print(f"   Overall Accuracy: {epoch_accuracy*100:.2f}%")
            print(f"   Real Detection: {real_accuracy*100:.2f}%")
            print(f"   Fake Detection: {fake_accuracy*100:.2f}%")
            print(f"   Bias Difference: {bias_difference*100:.1f}%")
            print(f"   Balance Score: {balance_score*100:.1f}%")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Check if we've achieved target balance
            target_real = STAGE3_ENHANCED_CONFIG['target_real_accuracy']
            target_fake = STAGE3_ENHANCED_CONFIG['target_fake_accuracy']
            max_bias = STAGE3_ENHANCED_CONFIG['max_bias_tolerance']
            
            if (real_accuracy >= target_real and 
                fake_accuracy >= target_fake and 
                bias_difference <= max_bias):
                print(f"🎉 TARGET BALANCE ACHIEVED!")
                print(f"   ✅ Real: {real_accuracy*100:.1f}% >= {target_real*100:.1f}%")
                print(f"   ✅ Fake: {fake_accuracy*100:.1f}% >= {target_fake*100:.1f}%")
                print(f"   ✅ Bias: {bias_difference*100:.1f}% <= {max_bias*100:.1f}%")
                break
            
            # Early stopping based on balance improvement
            if balance_score > best_balance_score:
                best_balance_score = balance_score
                patience_counter = 0
                
                # Save best balanced model
                metrics = {
                    'stage': 3,
                    'epoch': epoch + 1,
                    'accuracy': epoch_accuracy,
                    'real_accuracy': real_accuracy,
                    'fake_accuracy': fake_accuracy,
                    'bias_difference': bias_difference,
                    'balance_score': balance_score,
                    'f1_score': 2 * (real_accuracy * fake_accuracy) / (real_accuracy + fake_accuracy) if (real_accuracy + fake_accuracy) > 0 else 0,
                    'auc_roc': 0.5 + abs(real_accuracy - 0.5) + abs(fake_accuracy - 0.5)  # Approximation
                }
                
                checkpoint_path = self._save_smart_checkpoint(
                    f"stage3_wilddeepfake_balanced_epoch{epoch+1}", metrics
                )
                print(f"💾 Best balanced model saved: {checkpoint_path.name}")
                
            else:
                patience_counter += 1
                if patience_counter >= STAGE3_ENHANCED_CONFIG['early_stopping_patience']:
                    print(f"⏹️ Early stopping: No balance improvement for {patience_counter} epochs")
                    break
        
        # Update stage metrics
        final_metrics = {
            'stage': 3,
            'accuracy': epoch_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'bias_difference': bias_difference,
            'balance_score': balance_score,
            'balanced_accuracy': (real_accuracy + fake_accuracy) / 2,
            'f1_score': 2 * (real_accuracy * fake_accuracy) / (real_accuracy + fake_accuracy) if (real_accuracy + fake_accuracy) > 0 else 0,
            'auc_roc': 0.5 + abs(real_accuracy - 0.5) + abs(fake_accuracy - 0.5)
        }
        
        self.stage_metrics[3] = final_metrics
        
        print(f"\n🎯 STAGE 3 COMPLETED!")
        print(f"📊 Final Results:")
        print(f"   Overall Accuracy: {final_metrics['accuracy']*100:.2f}%")
        print(f"   Real Detection: {final_metrics['real_accuracy']*100:.2f}%")
        print(f"   Fake Detection: {final_metrics['fake_accuracy']*100:.2f}%")
        print(f"   Bias Difference: {final_metrics['bias_difference']*100:.1f}%")
        print(f"   Balance Score: {final_metrics['balance_score']*100:.1f}%")
        
        # Check improvement
        if final_metrics['fake_accuracy'] > 0.6:
            print(f"✅ EXCELLENT: Fake detection significantly improved!")
        elif final_metrics['fake_accuracy'] > 0.5:
            print(f"✅ GOOD: Fake detection improved!")
        else:
            print(f"⚠️ NEEDS MORE TRAINING: Fake detection still low")
        
        if final_metrics['bias_difference'] < 0.2:
            print(f"✅ EXCELLENT: Model is well-balanced!")
        elif final_metrics['bias_difference'] < 0.3:
            print(f"✅ GOOD: Model shows improved balance")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: Model still shows bias")

    def run_progressive_training(self):
        """Run the complete progressive training pipeline"""
        
        print(f"\n🚀 STARTING PROGRESSIVE DEEPFAKE TRAINING")
        print(f"📊 Training on 4 datasets sequentially")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Stage 1: FaceForensics++ (Foundation)
            self.train_stage_1_faceforensics()
            
            # Stage 2: Celeb-DF v2 (Realism)
            self.train_stage_2_celebdf()
            
            # Stage 3: Wild Deepfake (Real-world)
            self.train_stage_3_wilddeepfake()
            
            # Stage 4: DFDC (Scale)
            self.train_stage_4_dfdc()
            
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Save final comprehensive model
        final_metrics = self.stage_metrics.get(4, self.stage_metrics.get(max(self.stage_metrics.keys()), {}))
        final_checkpoint_path = self._save_smart_checkpoint("progressive_deepfake_model_FINAL", final_metrics)
        
        print(f"\n🎉 PROGRESSIVE TRAINING COMPLETED!")
        print(f"⏱️ Total time: {total_time/3600:.2f} hours")
        print(f"📊 Total data processed: {self.total_data_processed_gb:.2f} GB")
        print(f"💾 Final model saved: {final_checkpoint_path.name}")
        print(f"📊 Stages completed: {len(self.stage_metrics)}/4")
        
        # List all available checkpoints
        self.checkpoint_manager.list_checkpoints()
        
        # Print final summary
        self._print_training_summary()
    
    def _print_training_summary(self):
        """Print comprehensive training summary"""
        
        print(f"\n📈 TRAINING SUMMARY")
        print("="*60)
        
        for stage, metrics in self.stage_metrics.items():
            stage_names = {1: 'FaceForensics++', 2: 'Celeb-DF v2', 3: 'Wild Deepfake', 4: 'DFDC'}
            stage_name = stage_names.get(stage, f'Stage {stage}')
            
            print(f"\n🎯 {stage_name}:")
            print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"   Real Accuracy: {metrics['real_accuracy']*100:.2f}%")
            print(f"   Fake Accuracy: {metrics['fake_accuracy']*100:.2f}%")
            print(f"   Bias Difference: {metrics['bias_difference']*100:.1f}%")
            print(f"   F1-Score: {metrics['f1_score']*100:.2f}%")
            print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
        
        if self.stage_metrics:
            final_stage = max(self.stage_metrics.keys())
            final_metrics = self.stage_metrics[final_stage]
            
            print(f"\n🏆 FINAL PERFORMANCE:")
            print(f"   Overall Accuracy: {final_metrics['accuracy']*100:.2f}%")
            print(f"   Balanced Accuracy: {final_metrics['balanced_accuracy']*100:.2f}%")
            print(f"   Bias Reduction: {(1 - final_metrics['bias_difference'])*100:.1f}%")
            
            if final_metrics['bias_difference'] < 0.1:
                print(f"   ✅ EXCELLENT: Model is well-balanced!")
            elif final_metrics['bias_difference'] < 0.2:
                print(f"   ✅ GOOD: Model shows good balance")
            else:
                print(f"   ⚠️ NEEDS IMPROVEMENT: Model still shows bias")
        
        # Show download instructions
        self._print_download_instructions()
    
    def _print_download_instructions(self):
        """Print instructions for downloading checkpoints"""
        
        print(f"\n📥 CHECKPOINT DOWNLOAD INSTRUCTIONS")
        print("="*60)
        print(f"💾 Checkpoints are automatically saved every {CHECKPOINT_MANAGEMENT['save_frequency_gb']} GB")
        print(f"🗂️ Only the latest {CHECKPOINT_MANAGEMENT['max_checkpoints']} checkpoints are kept")
        print(f"📁 Check the 'downloads' folder in output directory")
        print(f"🔗 Download links are displayed automatically during training")
        print()
        print(f"📋 Available checkpoints:")
        
        if hasattr(self, 'checkpoint_manager'):
            for i, info in enumerate(self.checkpoint_manager.checkpoint_history):
                print(f"   {i+1}. {info['name']} - {info['file_size_mb']:.1f} MB")
                print(f"      Data: {info['data_processed_gb']:.1f} GB, Accuracy: {info['metrics'].get('accuracy', 0)*100:.1f}%")
        
        print(f"\n💡 Tips:")
        print(f"   - Download checkpoints immediately when links appear")
        print(f"   - Compressed checkpoints save ~50% space")
        print(f"   - Resume training from any checkpoint if needed")
        print(f"   - Final model contains all training history")

# ============================================================================
# OUTPUT DIRECTORY MANAGEMENT
# ============================================================================
def clean_output_directory_keep_latest():
    """Clean output directory but keep the latest checkpoint"""
    
    print("🧹 CLEANING OUTPUT DIRECTORY...")
    print("="*60)
    
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        print(f"📁 Output directory doesn't exist: {output_path}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = []
    for pattern in ["*.pt", "*.zip"]:
        checkpoint_files.extend(list(output_path.rglob(pattern)))
    
    if not checkpoint_files:
        print(f"📦 No checkpoint files found")
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    latest_checkpoint = checkpoint_files[0]
    print(f"🎯 Latest checkpoint: {latest_checkpoint.name}")
    print(f"📊 Size: {latest_checkpoint.stat().st_size / (1024**2):.1f} MB")
    print(f"📅 Modified: {datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)}")
    
    # Create backup of latest checkpoint in root
    backup_path = output_path / f"LATEST_CHECKPOINT_{latest_checkpoint.name}"
    if not backup_path.exists():
        import shutil
        shutil.copy2(latest_checkpoint, backup_path)
        print(f"💾 Backed up to: {backup_path.name}")
    
    # Clean directories but keep essential files
    dirs_to_clean = ["checkpoints", "downloads", "temp"]
    files_cleaned = 0
    space_freed = 0
    
    for dir_name in dirs_to_clean:
        dir_path = output_path / dir_name
        if dir_path.exists():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and file_path != latest_checkpoint:
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        files_cleaned += 1
                        space_freed += file_size
                    except Exception as e:
                        print(f"⚠️ Could not delete {file_path}: {e}")
    
    print(f"✅ Cleaned {files_cleaned} files")
    print(f"💾 Freed {space_freed / (1024**2):.1f} MB space")
    print(f"🎯 Kept latest checkpoint: {latest_checkpoint}")
    
    return latest_checkpoint

def find_latest_checkpoint():
    """Find the latest checkpoint in output directory"""
    
    output_path = Path(OUTPUT_DIR)
    if not output_path.exists():
        return None
    
    # Look for checkpoint files
    checkpoint_files = []
    for pattern in ["*.pt", "*.zip"]:
        checkpoint_files.extend(list(output_path.rglob(pattern)))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return checkpoint_files[0]

# ============================================================================
# STAGE 3 TRAINING CONFIGURATION (NATURAL TRAINING)
# ============================================================================
STAGE3_CONFIG = {
    'learning_rate': 3e-5,  # Lower LR for fine-tuning
    'epochs': 5,  # Standard epochs
    'batch_size': 8,
    'loss_function': 'focal',  # Standard focal loss
    'focal_alpha': 0.25,  # Standard focal loss alpha
    'focal_gamma': 2.0,  # Standard focal loss gamma
    'label_smoothing': 0.0,  # No label smoothing
    'mixup_alpha': 0.0,  # No mixup augmentation
    'weighted_sampling': False,  # Natural sampling
    'early_stopping_patience': 3,
    'save_best_only': True,
}

def standard_focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Standard focal loss for deepfake detection"""
    
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    
    return focal_loss.mean()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main function for progressive training - Stage 4 DFDC continuation"""
    
    print("🚀 INTERCEPTOR STAGE 4: DFDC TRAINING")
    print("="*80)
    print("📊 Current Model Status:")
    print("   ✅ Real Detection: 92% (excellent)")
    print("   ⚠️  Fake Detection: 28% (needs improvement)")
    print("   📊 Bias: 64% (real > fake)")
    print()
    print("🎯 Stage 4 Strategy:")
    print("   📚 Train on DFDC (100GB+, mostly fake videos)")
    print("   🔄 Natural training - let fake-heavy data compensate bias")
    print("   💡 DFDC has 10 chunks with ~83% fake videos")
    print("   🎯 Goal: Improve fake detection from 28% to 70%+")
    print()
    print("⏭️  SKIPPING STAGE 3:")
    print("   ⚠️  Wild Deepfake dataset contains images only, not videos")
    print("   ✅ Moving directly to Stage 4 DFDC for maximum impact")
    print("="*80)
    
    # Step 1: Use hardcoded checkpoint path
    print("\n📋 STEP 1: LOAD STAGE 2 CHECKPOINT")
    print("-" * 50)
    
    checkpoint_path = Path("/kaggle/input/celeb-df-model/stage2_full_celebdf_best_epoch8.pt")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("� Mak{e sure the dataset is properly mounted in Kaggle")
        return
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    print(f"📦 Size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    
    # Step 2: Initialize trainer with checkpoint
    print(f"\n📋 STEP 2: INITIALIZE TRAINER")
    print("-" * 50)
    trainer = ProgressiveDeepfakeTrainer(OUTPUT_DIR, str(checkpoint_path))
    
    # Step 3: Configure for Stage 4 training
    print(f"\n📋 STEP 3: CONFIGURE STAGE 4 DFDC TRAINING")
    print("-" * 50)
    trainer._setup_stage_training(4, 'dfdc')
    
    # Step 4: Run Stage 4 training
    print(f"\n📋 STEP 4: START STAGE 4 DFDC TRAINING")
    print("-" * 50)
    trainer.train_stage_4_dfdc()
    
    print("\n🎉 STAGE 4 DFDC TRAINING COMPLETED!")
    print("="*80)
    print("📊 Results:")
    print("   🎯 Model trained on 100GB+ DFDC data")
    print("   🎯 Fake detection should be significantly improved")
    print("   🎯 Bias should be reduced due to fake-heavy dataset")
    print()
    print("📊 Next Steps:")
    print("   1. Test model on your test videos")
    print("   2. Compare fake detection: 28% → target 70%+")
    print("   3. Check if bias reduced: 64% → target <30%")
    print("   4. Deploy final model if performance is satisfactory")

if __name__ == "__main__":
    main()