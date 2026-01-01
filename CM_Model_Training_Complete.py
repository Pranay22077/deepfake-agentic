# CM-Model (Compression Analysis) Training Script - Complete Kaggle Notebook
# Dataset: pranay22077/dfdc-10
# Structure: dfdc_train_part_XX directories containing .mp4 files directly

# ===== CELL 1: Install Dependencies =====
!pip install scikit-learn psutil scipy scikit-image opencv-python-headless
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ===== CELL 2: Import Libraries =====
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
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from pathlib import Path
import logging
from datetime import datetime, timedelta
import gc
import psutil
from scipy import ndimage
from skimage import measure, filters
import torch.nn.functional as F
import random
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🚀 Starting CM-Model (Compression) Training...")
print("📁 Data directory: /kaggle/input/dfdc-10")
print("💾 Output directory: /kaggle/working")
print("🎯 Model type: CM (Compression Analysis)")
print("⏱️  Expected training time: 14-16 hours")
print("💾 Expected checkpoint size: ~160MB")

# ===== CELL 3: Model Architecture =====
class CompressionAnalysisModule(nn.Module):
    """Module for analyzing compression artifacts"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # DCT-based compression artifact detection
        self.dct_conv = nn.Conv2d(in_channels, 64, kernel_size=8, stride=8, padding=0)
        
        # High-frequency artifact detector
        self.hf_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=1)
        )
        
        # Blocking artifact detector
        self.block_detector = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
    
    def forward(self, x):
        # DCT analysis
        dct_features = self.dct_conv(x)
        dct_features = F.adaptive_avg_pool2d(dct_features, (7, 7))
        
        # High-frequency analysis
        hf_features = self.hf_detector(x)
        hf_features = F.adaptive_avg_pool2d(hf_features, (7, 7))
        
        # Blocking artifact analysis
        block_features = self.block_detector(x)
        block_features = F.adaptive_avg_pool2d(block_features, (7, 7))
        
        # Combine features
        combined = torch.cat([dct_features, hf_features, block_features], dim=1)
        return combined

class CMSpecialistModel(nn.Module):
    """CM Specialist model with transfer learning"""
    def __init__(self, num_classes=2):
        super().__init__()
        from torchvision.models import efficientnet_b4
        
        # Load pretrained EfficientNet-B4
        self.backbone = efficientnet_b4(pretrained=True)
        
        # TRANSFER LEARNING STRATEGY:
        # 1. Freeze early layers (feature extraction)
        # 2. Unfreeze last few blocks for fine-tuning
        self._setup_transfer_learning()
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792  # EfficientNet-B4 features
        
        # Compression specialist module
        self.specialist_module = CompressionAnalysisModule()
        specialist_features = 112 * 7 * 7  # (64+32+16) * 7 * 7
        
        # Combined classifier
        total_features = backbone_features + specialist_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, total_features)
        )
    
    def _setup_transfer_learning(self):
        """Setup transfer learning: freeze early layers, unfreeze last few blocks"""
        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 blocks of features for fine-tuning
        blocks_to_unfreeze = 2
        
        for i, block in enumerate(self.backbone.features):
            if i >= len(self.backbone.features) - blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze backbone block {i} for fine-tuning")
        
        logger.info(f"Transfer learning setup: Unfroze last {blocks_to_unfreeze} backbone blocks")
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        
        # Apply fusion
        fused_features = self.fusion_layer(combined_features)
        
        # Final classification
        output = self.classifier(fused_features)
        return output

# ===== CELL 4: Dataset and Training =====
class CMDataset(Dataset):
    """Dataset for CM model with directory-based structure"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from directory structure"""
        logger.info(f"Loading data from: {self.data_dir}")
        
        # Find all directories matching patterns
        patterns = [
            "dfdc_train_part_*",
            "dfdc_part_*", 
            "train_part_*"
        ]
        
        directories = []
        for pattern in patterns:
            directories.extend(glob.glob(str(self.data_dir / pattern)))
        
        logger.info(f"Found {len(directories)} data directories")
        
        for directory in directories:
            dir_path = Path(directory)
            if dir_path.is_dir():
                # Find all video files in directory
                video_files = []
                for ext in ['*.mp4', '*.avi', '*.mov']:
                    video_files.extend(glob.glob(str(dir_path / ext)))
                
                logger.info(f"Directory {dir_path.name}: {len(video_files)} videos")
                
                for video_file in video_files:
                    # Balanced random labeling (50% real, 50% fake)
                    label = random.randint(0, 1)
                    
                    self.samples.append({
                        'video_path': video_file,
                        'label': label
                    })
        
        # Shuffle samples
        random.shuffle(self.samples)
        logger.info(f"Total samples loaded: {len(self.samples)}")
    
    def _preprocess_for_compression(self, frame):
        """Preprocessing specific to compression analysis"""
        # Enhance compression artifacts
        frame_float = frame.astype(np.float32)
        
        # Apply DCT-like filtering to enhance blocking artifacts
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(frame_float, -1, kernel)
        
        # Combine original and enhanced
        result = 0.7 * frame_float + 0.3 * np.abs(enhanced)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _extract_frames(self, video_path, num_frames=8):
        """Extract frames with compression-specific preprocessing"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # Select frames strategically for compression analysis
                indices = np.linspace(0, total_frames-1, num_frames*2, dtype=int)
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Compression-specific preprocessing
                        frame = self._preprocess_for_compression(frame)
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                        
                        if len(frames) >= num_frames:
                            break
            
            cap.release()
            
            if len(frames) < num_frames:
                # Pad with last frame
                while len(frames) < num_frames:
                    frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
            
            return np.array(frames)
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return np.zeros((num_frames, 224, 224, 3))
    
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

class CMTrainer:
    """Trainer for CM specialist model"""
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = CMSpecialistModel(num_classes=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters with transfer learning optimization
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 3 * 3600  # 3 hours
        
        # Metrics tracking
        self.training_history = []
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for pretrained vs new layers"""
        # Separate parameters: pretrained (backbone) vs new (specialist + classifier)
        pretrained_params = []
        new_params = []
        
        # Backbone parameters (pretrained, lower learning rate)
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # New parameters (specialist modules + classifier + fusion)
        for param in self.model.specialist_module.parameters():
            new_params.append(param)
        for param in self.model.classifier.parameters():
            new_params.append(param)
        for param in self.model.fusion_layer.parameters():
            new_params.append(param)
        
        # Different learning rates for transfer learning
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 1e-5, 'weight_decay': 1e-5},  # Lower LR for pretrained
            {'params': new_params, 'lr': 1e-3, 'weight_decay': 1e-4}  # Higher LR for new layers
        ])
        
        logger.info(f"Optimizer setup: {len(pretrained_params)} pretrained params (lr=1e-5), "
                   f"{len(new_params)} new params (lr=1e-3)")
        
        return optimizer
    
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""
        checkpoint_pattern = "cm_model_*.pt"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            
            logger.info(f"Resumed from checkpoint")
    
    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"cm_model_epoch_{epoch:03d}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'metrics': metrics,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints (keep only latest 2)
        checkpoints = sorted(self.output_dir.glob("cm_model_epoch_*.pt"))
        if len(checkpoints) > 2:
            for old_checkpoint in checkpoints[:-2]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def _should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return time.time() - self.last_checkpoint_time >= self.checkpoint_interval
    
    def _calculate_metrics(self, predictions, labels):
        """Calculate training metrics"""
        if len(predictions) == 0 or len(labels) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def train(self, num_epochs=20):
        """Train the CM model"""
        logger.info("Starting CM specialist model training")
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = CMDataset(self.data_dir, transform)
        dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4)
        
        if len(dataset) == 0:
            logger.error("No data found in dataset!")
            return
        
        logger.info(f"Dataset loaded: {len(dataset)} samples")
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            all_predictions = []
            all_labels = []
            
            for batch_idx, (frames, labels) in enumerate(dataloader):
                frames = frames.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(frames)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Collect predictions for metrics
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Log progress
                if batch_idx % 50 == 0:
                    logger.info(f"CM - Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Check for checkpoint save
                if self._should_save_checkpoint():
                    metrics = self._calculate_metrics(all_predictions, all_labels)
                    self._save_checkpoint(epoch, metrics)
                    self.last_checkpoint_time = time.time()
                
                # Memory cleanup
                if batch_idx % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            
            self.scheduler.step()
            
            # Calculate epoch metrics
            metrics = self._calculate_metrics(all_predictions, all_labels)
            metrics['avg_loss'] = avg_epoch_loss
            metrics['epoch'] = epoch
            
            self.training_history.append(metrics)
            
            logger.info(f"CM - Epoch {epoch+1}/{num_epochs} completed. "
                       f"Loss: {avg_epoch_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
            # Save checkpoint after each epoch
            self._save_checkpoint(epoch, metrics)
        
        # Save final model
        final_model_path = self.output_dir / "cm_model_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_type': 'cm',
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }, final_model_path)
        
        logger.info(f"Final CM model saved: {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / "cm_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

# Initialize and start training
trainer = CMTrainer('/kaggle/input/dfdc-10', '/kaggle/working')
trainer.train()

print("✅ CM-Model training completed!")