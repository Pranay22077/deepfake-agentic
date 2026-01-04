#!/usr/bin/env python3
"""
Fixed Kaggle Teacher Model Training Script
Corrected version that works with the actual dataset implementation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import pickle
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FIXED DATASET CLASS
# ============================================================================

class MultimodalDataset(Dataset):
    """Fixed dataset class for Kaggle training"""
    def __init__(self, data_dir, split='train', num_frames=8, audio_duration=3.0, 
                 sample_rate=16000, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.num_frames = num_frames
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        self.augment = augment
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Load samples
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        """Load image samples from directory"""
        samples = []
        
        # Look for standard structure first
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            # Standard train/val split
            split_dir = train_dir if self.split == 'train' else val_dir
            
            # Look for real/fake subdirectories
            real_dir = os.path.join(split_dir, 'real')
            fake_dir = os.path.join(split_dir, 'fake')
            
            if os.path.exists(real_dir):
                for img_file in os.listdir(real_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        samples.append((os.path.join(real_dir, img_file), 0))
            
            if os.path.exists(fake_dir):
                for img_file in os.listdir(fake_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        samples.append((os.path.join(fake_dir, img_file), 1))
        
        else:
            # Fallback: scan all directories
            all_samples = []
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # Simple heuristic for labeling
                        label = 1 if 'fake' in root.lower() else 0
                        all_samples.append((os.path.join(root, file), label))
            
            # Split 80/20
            split_idx = int(0.8 * len(all_samples))
            if self.split == 'train':
                samples = all_samples[:split_idx]
            else:
                samples = all_samples[split_idx:]
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Create video sequence (repeat frame)
            video_frames = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)  # [T, C, H, W]
            
            # Create dummy audio (random noise)
            audio_waveform = torch.randn(self.audio_samples) * 0.1
            
            return video_frames, audio_waveform, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy data
            video_frames = torch.zeros(self.num_frames, 3, 224, 224)
            audio_waveform = torch.zeros(self.audio_samples)
            return video_frames, audio_waveform, label

# ============================================================================
# SIMPLIFIED TEACHER MODEL
# ============================================================================

class SimpleTeacher(nn.Module):
    """Simplified teacher model for Kaggle"""
    def __init__(self, num_classes=2, visual_frames=8):
        super().__init__()
        
        # Visual backbone
        import torchvision.models as models
        self.visual_backbone = models.efficientnet_b0(weights='DEFAULT')
        visual_features = self.visual_backbone.classifier.in_features
        self.visual_backbone.classifier = nn.Identity()
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(visual_features, 256, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Audio processing (simplified)
        self.audio_net = nn.Sequential(
            nn.Linear(48000, 1024),  # 3 seconds at 16kHz
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),  # visual + audio
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, video_frames, audio_waveform, return_features=False):
        """
        Args:
            video_frames: [B, T, C, H, W]
            audio_waveform: [B, audio_length]
        """
        B, T, C, H, W = video_frames.shape
        
        # Process video frames
        frames = video_frames.view(B*T, C, H, W)
        frame_features = self.visual_backbone(frames)  # [B*T, features]
        frame_features = frame_features.view(B, T, -1)  # [B, T, features]
        
        # Temporal processing
        temp_features = frame_features.transpose(1, 2)  # [B, features, T]
        temp_features = self.temporal_conv(temp_features)  # [B, 256, T]
        visual_feat = self.temporal_pool(temp_features).squeeze(-1)  # [B, 256]
        
        # Process audio
        audio_feat = self.audio_net(audio_waveform)  # [B, 128]
        
        # Fusion
        combined = torch.cat([visual_feat, audio_feat], dim=1)  # [B, 384]
        output = self.fusion(combined)
        
        if return_features:
            features = {
                'visual_feat': visual_feat,
                'audio_feat': audio_feat,
                'fused_feat': combined
            }
            return output, features
        
        return output

def create_teacher_model(num_classes=2, visual_frames=8):
    """Factory function to create teacher model"""
    return SimpleTeacher(num_classes=num_classes, visual_frames=visual_frames)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def setup_kaggle_environment():
    """Setup Kaggle environment and paths"""
    os.makedirs('/kaggle/working/models', exist_ok=True)
    os.makedirs('/kaggle/working/logs', exist_ok=True)
    os.makedirs('/kaggle/working/teacher_predictions', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/kaggle/working/logs/teacher_training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def find_data_directory():
    """Find the best data directory from Kaggle inputs"""
    print("[SEARCH] Looking for data directories...")
    
    possible_dirs = []
    for item in os.listdir('/kaggle/input/'):
        item_path = f'/kaggle/input/{item}'
        if os.path.isdir(item_path):
            # Count image files
            img_count = 0
            for root, dirs, files in os.walk(item_path):
                img_count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
            
            if img_count > 50:  # At least 50 images
                possible_dirs.append((item_path, img_count))
                print(f"  [FOLDER] {item}: {img_count} images")
    
    if possible_dirs:
        # Return directory with most images
        best_dir = max(possible_dirs, key=lambda x: x[1])
        print(f"[OK] Using: {best_dir[0]} ({best_dir[1]} images)")
        return best_dir[0]
    else:
        print("[ERROR] No suitable data directory found!")
        print("Available inputs:", os.listdir('/kaggle/input/'))
        return None

def train_teacher_model():
    """Main teacher training function"""
    logger = setup_kaggle_environment()
    
    # Configuration
    config = {
        'epochs': 15,  # Reduced for faster training
        'batch_size': 8,   # Small batch for memory
        'lr': 1e-4,
        'num_frames': 8,
        'audio_duration': 3.0,
        'sample_rate': 16000,
        'save_predictions': True
    }
    
    logger.info("Starting teacher model training on Kaggle")
    logger.info(f"Configuration: {config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Find data directory
    data_dir = find_data_directory()
    if not data_dir:
        logger.error("No data directory found!")
        return None, 0
    
    # Create datasets
    try:
        train_dataset = MultimodalDataset(
            data_dir=data_dir,
            split='train',
            num_frames=config['num_frames'],
            audio_duration=config['audio_duration'],
            sample_rate=config['sample_rate'],
            augment=True
        )
        
        val_dataset = MultimodalDataset(
            data_dir=data_dir,
            split='val',
            num_frames=config['num_frames'],
            audio_duration=config['audio_duration'],
            sample_rate=config['sample_rate'],
            augment=False
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        return None, 0
    
    if len(train_dataset) == 0:
        logger.error("No training samples found!")
        return None, 0
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create teacher model
    model = create_teacher_model(num_classes=2, visual_frames=config['num_frames'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Teacher model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_auc = 0
    train_history = []
    
    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, (frames, audio, labels) in enumerate(pbar):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(frames, audio)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Collect predictions
            probs = torch.softmax(logits, dim=1)
            train_preds.extend(probs[:, 1].cpu().detach().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for frames, audio, labels in tqdm(val_loader, desc='Validation'):
                frames = frames.to(device)
                audio = audio.to(device)
                labels = labels.to(device)
                
                logits = model(frames, audio)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                val_preds.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if len(set(train_labels)) > 1:  # Check if we have both classes
            train_auc = roc_auc_score(train_labels, train_preds)
            val_auc = roc_auc_score(val_labels, val_preds)
        else:
            train_auc = val_auc = 0.5
        
        train_acc = accuracy_score(train_labels, np.array(train_preds) > 0.5)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_auc': train_auc,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_auc': best_auc,
                'config': config
            }, '/kaggle/working/models/teacher_model.pt')
            
            logger.info(f"New best teacher model saved! AUC: {val_auc:.4f}")
    
    # Save training history
    with open('/kaggle/working/models/teacher_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logger.info(f"Teacher training completed! Best AUC: {best_auc:.4f}")
    
    return model, best_auc

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("[RUN] Starting Fixed Teacher Model Training")
    print("="*50)
    
    # Run teacher training
    model, best_auc = train_teacher_model()
    
    if model is not None:
        print(f"\n[TARGET] Teacher Training Complete!")
        print(f"[STATS] Best AUC: {best_auc:.4f}")
        print(f"[SAVE] Model saved to: /kaggle/working/models/teacher_model.pt")
        print(f"[PROGRESS] History saved to: /kaggle/working/models/teacher_history.json")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[STATS] Model size: {total_params:,} parameters")
        
        # Check model file size
        model_path = '/kaggle/working/models/teacher_model.pt'
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"[SAVE] File size: {size_mb:.1f} MB")
    else:
        print("[ERROR] Teacher training failed!")
        print("Check the logs for details.")