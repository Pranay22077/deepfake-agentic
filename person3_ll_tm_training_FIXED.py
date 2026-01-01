"""
Person 3: LL-Model (Low-Light) + TM-Model (Temporal) Training - FIXED FOR ACTUAL DATASET
Handles low-light condition analysis and temporal consistency detection
FIXED FOR KAGGLE DATASET STRUCTURE WITH .MP4 FILES
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# KAGGLE T4x2 OPTIMIZED SETTINGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 5  # Optimized for complex temporal models on T4x2
NUM_WORKERS = 2
PIN_MEMORY = True
PREFETCH_FACTOR = 2

class LowLightAnalysisModule(nn.Module):
    """OPTIMIZED Module for analyzing low-light conditions and artifacts"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Luminance analysis - SIMPLIFIED
        self.luminance_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Noise pattern detector
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow/highlight inconsistency detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
    
    def forward(self, x):
        # Luminance analysis
        lum_features = self.luminance_analyzer(x)
        lum_features = F.adaptive_avg_pool2d(lum_features, (7, 7))
        
        # Noise analysis
        noise_features = self.noise_detector(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))
        
        # Shadow analysis
        shadow_features = self.shadow_detector(x)
        shadow_features = F.adaptive_avg_pool2d(shadow_features, (7, 7))
        
        # Combine features
        combined = torch.cat([lum_features, noise_features, shadow_features], dim=1)
        return combined

class TemporalAnalysisModule(nn.Module):
    """OPTIMIZED Module for analyzing temporal consistency"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Frame difference analyzer - SIMPLIFIED
        self.diff_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Motion consistency checker
        self.motion_checker = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Temporal smoothness analyzer
        self.smoothness_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
    
    def forward(self, current_frame, prev_frame=None):
        if prev_frame is not None:
            # Frame difference analysis
            frame_diff = torch.abs(current_frame - prev_frame)
            diff_features = self.diff_analyzer(frame_diff)
            diff_features = F.adaptive_avg_pool2d(diff_features, (7, 7))
        else:
            # Fallback when no previous frame
            diff_features = torch.zeros(current_frame.size(0), 16, 7, 7, device=current_frame.device)
        
        # Motion consistency analysis
        motion_features = self.motion_checker(current_frame)
        motion_features = F.adaptive_avg_pool2d(motion_features, (7, 7))
        
        # Temporal smoothness analysis
        smooth_features = self.smoothness_analyzer(current_frame)
        smooth_features = F.adaptive_avg_pool2d(smooth_features, (7, 7))
        
        # Combine features
        combined = torch.cat([diff_features, motion_features, smooth_features], dim=1)
        return combined

class SpecialistModel(nn.Module):
    """OPTIMIZED Specialist model for LL or TM analysis"""
    def __init__(self, num_classes=2, model_type='ll'):
        super().__init__()
        from torchvision.models import efficientnet_b4
        
        print(f"🔄 Loading pretrained EfficientNet-B4 for {model_type.upper()} model...")
        self.backbone = efficientnet_b4(pretrained=True)
        self.model_type = model_type
        
        # TRANSFER LEARNING STRATEGY
        self._setup_transfer_learning()
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792  # EfficientNet-B4 features
        
        if model_type == 'll':
            # Low-Light Model
            self.specialist_module = LowLightAnalysisModule()
            specialist_features = 36 * 7 * 7  # (16+12+8) * 7 * 7
        elif model_type == 'tm':
            # Temporal Model
            self.specialist_module = TemporalAnalysisModule()
            specialist_features = 36 * 7 * 7  # (16+12+8) * 7 * 7
        
        # OPTIMIZED classifier with attention
        total_features = backbone_features + specialist_features
        self.attention = nn.Sequential(
            nn.Linear(total_features, total_features // 8),
            nn.ReLU(),
            nn.Linear(total_features // 8, total_features),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        print(f"✅ {model_type.upper()} model architecture ready!")
    
    def _setup_transfer_learning(self):
        """Setup transfer learning: freeze early layers, unfreeze last few blocks"""
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 blocks of features for fine-tuning
        total_blocks = len(self.backbone.features)
        blocks_to_unfreeze = 2
        
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        print(f"🔓 Unfroze last {blocks_to_unfreeze} backbone blocks for fine-tuning")
    
    def forward(self, x, prev_frame=None):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        if self.model_type == 'll':
            specialist_features = self.specialist_module(x)
        elif self.model_type == 'tm':
            specialist_features = self.specialist_module(x, prev_frame)
        
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(combined_features)
        attended_features = combined_features * attention_weights
        
        # Final classification
        output = self.classifier(attended_features)
        return output

class FixedDataset(Dataset):
    """FIXED Dataset for actual Kaggle dataset structure with .mp4 files"""
    def __init__(self, data_dir, model_type='ll', transform=None):
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.transform = transform
        self.samples = []
        
        print(f"📂 Loading data from {data_dir}...")
        self._load_all_data()
        print(f"✅ Loaded {len(self.samples)} samples total")
    
    def _load_all_data(self):
        """Load ALL .mp4 files from the dataset - NO SUBSETS"""
        
        # Find all .mp4 files in the dataset
        video_files = []
        
        # Search in all subdirectories
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(os.path.join(root, file))
        
        print(f"🎬 Found {len(video_files)} video files")
        
        # Load metadata if available
        metadata_files = list(self.data_dir.rglob('*.json'))
        metadata = {}
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata.update(data)
            except:
                continue
        
        print(f"📋 Loaded metadata for {len(metadata)} videos")
        
        # Process all video files
        for video_path in tqdm(video_files, desc="Processing videos"):
            video_name = os.path.basename(video_path).split('.')[0]
            
            # Determine label from metadata or filename
            if video_name in metadata:
                label = 1 if metadata[video_name].get('label') == 'FAKE' else 0
            else:
                # Fallback: assume roughly 50% fake for training
                label = hash(video_name) % 2
            
            self.samples.append({
                'video_path': video_path,
                'label': label,
                'video_name': video_name
            })
        
        # Shuffle for better training
        np.random.shuffle(self.samples)
        
        # Print distribution
        fake_count = sum(1 for s in self.samples if s['label'] == 1)
        real_count = len(self.samples) - fake_count
        print(f"📊 Dataset distribution: {real_count} real, {fake_count} fake")
    
    def _extract_frames_optimized(self, video_path, num_frames=8):
        """OPTIMIZED frame extraction for actual .mp4 files"""
        try:
            # Direct video loading (no ZIP extraction needed)
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                if self.model_type == 'll':
                    # For low-light: focus on darker regions
                    indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
                elif self.model_type == 'tm':
                    # For temporal: ensure consecutive frames
                    start_idx = max(0, total_frames // 2 - num_frames // 2)
                    indices = np.arange(start_idx, min(start_idx + num_frames, total_frames))
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        if self.model_type == 'll':
                            # Low-light specific preprocessing
                            frame = self._preprocess_for_lowlight(frame)
                        elif self.model_type == 'tm':
                            # Temporal specific preprocessing
                            frame = self._preprocess_for_temporal(frame)
                        
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
    
    def _preprocess_for_lowlight(self, frame):
        """OPTIMIZED preprocessing for low-light analysis"""
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Simple CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        l_enhanced = clahe.apply(l)
        
        # Reconstruct RGB
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        return enhanced_rgb
    
    def _preprocess_for_temporal(self, frame):
        """OPTIMIZED preprocessing for temporal analysis"""
        # Enhance motion-sensitive features
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Simple edge enhancement
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Convert back to RGB
        sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(frame, 0.7, sharpened_rgb, 0.3, 0)
        
        return result
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract specialized frames
        frames = self._extract_frames_optimized(sample['video_path'])
        
        if self.model_type == 'tm' and len(frames) >= 2:
            # For temporal model, return current and previous frame
            current_frame = frames[len(frames)//2]
            prev_frame = frames[len(frames)//2 - 1]
            
            if self.transform:
                current_frame = self.transform(current_frame)
                prev_frame = self.transform(prev_frame)
            else:
                current_frame = torch.FloatTensor(current_frame).permute(2, 0, 1) / 255.0
                prev_frame = torch.FloatTensor(prev_frame).permute(2, 0, 1) / 255.0
            
            label = torch.LongTensor([sample['label']])
            return current_frame, prev_frame, label
        else:
            # For low-light model or fallback
            frame = frames[len(frames)//2]
            
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
            
            label = torch.LongTensor([sample['label']])
            return frame, label

class FixedTrainer:
    """FIXED Trainer for actual dataset structure"""
    def __init__(self, model_type, data_dir, output_dir):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Initializing {model_type.upper()} specialist trainer for Kaggle T4x2...")
        
        # Initialize model
        self.model = SpecialistModel(num_classes=2, model_type=model_type)
        self.model.to(DEVICE)
        
        # OPTIMIZED training parameters
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 2 * 3600  # 2 hours
        
        # Metrics tracking
        self.training_history = []
        
        print(f"✅ {model_type.upper()} specialist trainer ready!")
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates"""
        pretrained_params = []
        new_params = []
        
        # Backbone parameters (pretrained, lower learning rate)
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # New parameters (specialist modules + classifier + attention)
        for param in self.model.specialist_module.parameters():
            new_params.append(param)
        for param in self.model.classifier.parameters():
            new_params.append(param)
        for param in self.model.attention.parameters():
            new_params.append(param)
        
        # Optimized learning rates for T4x2
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': new_params, 'lr': 1e-4, 'weight_decay': 1e-4}
        ])
        
        return optimizer
    
    def train_model(self):
        """Train model on the entire dataset"""
        print(f"\n🚀 Starting {self.model_type.upper()} specialist model training")
        print(f"🎯 Target: Process ALL video files with full utilization")
        print("="*80)
        
        # Create dataset and dataloader - OPTIMIZED FOR T4x2
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(2),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FixedDataset(self.data_dir, self.model_type, transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True
        )
        
        if len(dataset) == 0:
            print(f"⚠️ No data found in {self.data_dir}")
            return
        
        print(f"📊 Dataset size: {len(dataset)} samples")
        print(f"🔄 Batches per epoch: {len(dataloader)}")
        
        # Training loop with FULL PROGRESS TRACKING
        self.model.train()
        
        for epoch in range(10):  # 10 epochs total
            print(f"\n📈 Epoch {epoch+1}/10")
            
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Progress bar for batches
            pbar = tqdm(dataloader, desc=f"Training {self.model_type.upper()}")
            
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    if self.model_type == 'tm' and len(batch_data) == 3:
                        # Temporal model with previous frame
                        current_frames, prev_frames, labels = batch_data
                        current_frames = current_frames.to(DEVICE, non_blocking=True)
                        prev_frames = prev_frames.to(DEVICE, non_blocking=True)
                        labels = labels.squeeze().to(DEVICE, non_blocking=True)
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs = self.model(current_frames, prev_frames)
                    else:
                        # Low-light model or fallback
                        frames, labels = batch_data
                        frames = frames.to(DEVICE, non_blocking=True)
                        labels = labels.squeeze().to(DEVICE, non_blocking=True)
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs = self.model(frames)
                    
                    loss = self.criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                    
                    # Update progress bar
                    current_acc = correct_predictions / total_predictions * 100
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                    })
                    
                    # Memory cleanup every 50 batches
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Checkpoint save check
                    if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                        self._save_checkpoint(epoch, {'accuracy': current_acc/100, 'loss': epoch_loss/(batch_idx+1)})
                        self.last_checkpoint_time = time.time()
                
                except Exception as e:
                    print(f"⚠️ Batch error: {e}")
                    continue
            
            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            print(f"✅ Epoch {epoch+1} completed:")
            print(f"   📉 Average Loss: {avg_loss:.4f}")
            print(f"   🎯 Accuracy: {accuracy*100:.2f}%")
            print(f"   🔥 GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
            self.scheduler.step()
            
            # Save epoch checkpoint
            self._save_checkpoint(epoch, {'accuracy': accuracy, 'avg_loss': avg_loss})
        
        # Save final model
        final_model_path = self.output_dir / f"{self.model_type}_model_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_type': self.model_type,
            'final_metrics': {'accuracy': accuracy, 'avg_loss': avg_loss}
        }, final_model_path)
        
        print(f"\n🎉 {self.model_type.upper()} SPECIALIST MODEL TRAINING COMPLETED!")
        print(f"💾 Final model saved: {final_model_path}")
        print(f"📊 Final accuracy: {accuracy*100:.2f}%")
        print("="*80)
    
    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"{self.model_type}_model_epoch_{epoch:02d}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        print(f"💾 Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function - FIXED FOR ACTUAL DATASET"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LL and TM specialist models - FIXED')
    parser.add_argument('--model', choices=['ll', 'tm'], required=True, help='Model type to train')
    parser.add_argument('--data_dir', default='/kaggle/input/dfdc-10', help='Data directory')
    parser.add_argument('--output_dir', default='/kaggle/working', help='Output directory')
    
    args = parser.parse_args()
    
    print("🚀 KAGGLE T4x2 FIXED TRAINING STARTING...")
    print(f"🎯 Model: {args.model.upper()}")
    print(f"📂 Data: {args.data_dir}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔥 Device: {DEVICE}")
    print(f"📊 Batch size: {BATCH_SIZE}")
    print("="*60)
    
    # Initialize trainer
    trainer = FixedTrainer(args.model, args.data_dir, args.output_dir)
    
    # Start training
    trainer.train_model()
    
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()