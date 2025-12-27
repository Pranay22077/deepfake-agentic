#!/usr/bin/env python3
"""
Teacher Model Training Script
Trains the heavy multimodal teacher model for knowledge distillation
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
from tqdm import tqdm
import logging
from pathlib import Path
import cv2
import torchaudio
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.teacher import create_teacher_model
from preprocess.augmentation import create_teacher_augmentation
from preprocess.extract_faces import FaceExtractor
from preprocess.extract_audio import AudioExtractor

class MultimodalDataset(Dataset):
    """Dataset for multimodal teacher training"""
    
    def __init__(self, data_dir, split='train', num_frames=8, audio_duration=3.0, 
                 sample_rate=16000, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        
        # Setup augmentation
        self.augment = augment and split == 'train'
        if self.augment:
            self.augmentation = create_teacher_augmentation()
        
        # Load samples
        self.samples = self._load_samples()
        
        # Initialize extractors
        self.face_extractor = FaceExtractor()
        self.audio_extractor = AudioExtractor(sample_rate=sample_rate)
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        """Load video file paths and labels"""
        samples = []
        
        # Real videos
        real_dir = self.data_dir / 'raw' / 'real'
        if real_dir.exists():
            for video_file in real_dir.glob('*.mp4'):
                samples.append((str(video_file), 0))  # 0 = real
        
        # Fake videos  
        fake_dir = self.data_dir / 'raw' / 'fake'
        if fake_dir.exists():
            for video_file in fake_dir.glob('*.mp4'):
                samples.append((str(video_file), 1))  # 1 = fake
        
        # Split data
        np.random.seed(42)
        np.random.shuffle(samples)
        
        if self.split == 'train':
            samples = samples[:int(0.8 * len(samples))]
        else:  # val
            samples = samples[int(0.8 * len(samples)):]
        
        return samples
    
    def _extract_video_frames(self, video_path):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        
        cap.release()
        
        if len(frames) != self.num_frames:
            return None
        
        # Convert to tensor [T, H, W, C] -> [T, C, H, W]
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        return frames
    
    def _extract_audio(self, video_path):
        """Extract audio from video"""
        try:
            # Load audio using torchaudio
            waveform, sr = torchaudio.load(video_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to desired length
            if waveform.shape[1] > self.audio_samples:
                # Trim from center
                start = (waveform.shape[1] - self.audio_samples) // 2
                waveform = waveform[:, start:start + self.audio_samples]
            else:
                # Pad with zeros
                padding = self.audio_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            # Return silence
            return torch.zeros(self.audio_samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Extract video frames
        frames = self._extract_video_frames(video_path)
        if frames is None:
            # Return dummy data if extraction fails
            frames = torch.zeros(self.num_frames, 3, 224, 224)
        
        # Extract audio
        audio = self._extract_audio(video_path)
        
        # Apply augmentations
        if self.augment:
            frames_batch = frames.unsqueeze(0)  # Add batch dim
            audio_batch = audio.unsqueeze(0)
            frames_batch, audio_batch = self.augmentation(frames_batch, audio_batch, self.sample_rate)
            frames = frames_batch.squeeze(0)
            audio = audio_batch.squeeze(0)
        
        return frames, audio, label

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (frames, audio, labels) in enumerate(pbar):
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, features = model(frames, audio, return_features=True)
        
        # Main classification loss
        ce_loss = criterion(logits, labels)
        
        # Auxiliary lip-sync loss (optional)
        lip_sync_scores = features['lip_sync_score'].squeeze()
        # For real videos, lip-sync should be high; for fake, it varies
        lip_sync_targets = (labels == 0).float()  # 1 for real, 0 for fake
        lip_sync_loss = nn.BCELoss()(lip_sync_scores, lip_sync_targets)
        
        # Combined loss
        total_loss_batch = ce_loss + 0.1 * lip_sync_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
        # Collect predictions
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(probs[:, 1].cpu().detach().numpy())  # Fake probability
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'CE': f'{ce_loss.item():.4f}',
            'Sync': f'{lip_sync_loss.item():.4f}'
        })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, auc, acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, labels in tqdm(dataloader, desc='Validation'):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            logits, features = model(frames, audio, return_features=True)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, auc, acc

def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', default='models/teacher_full.pt', help='Model save path')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration in seconds')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        args.data_dir, 
        split='train',
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        augment=True
    )
    
    val_dataset = MultimodalDataset(
        args.data_dir,
        split='val', 
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = create_teacher_model(num_classes=2, visual_frames=args.num_frames)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Teacher model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_auc = 0
    train_history = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_auc, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_auc, val_acc = validate(model, val_loader, criterion, device)
        
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
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_auc': best_auc,
                'args': args
            }, args.save_path)
            logger.info(f"New best model saved! AUC: {val_auc:.4f}")
    
    # Save training history
    history_path = args.save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logger.info(f"Training completed! Best AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()