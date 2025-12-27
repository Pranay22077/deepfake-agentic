#!/usr/bin/env python3
"""
Knowledge Distillation Training Script
Trains compact student model using teacher soft labels
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
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model
from preprocess.augmentation import create_student_augmentation
from train.train_teacher import MultimodalDataset

class DistillationDataset(Dataset):
    """Dataset that loads teacher predictions for distillation"""
    
    def __init__(self, data_dir, teacher_preds_dir, split='train', num_frames=8, 
                 audio_duration=3.0, sample_rate=16000, augment=True):
        self.data_dir = Path(data_dir)
        self.teacher_preds_dir = Path(teacher_preds_dir)
        self.split = split
        self.num_frames = num_frames
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        
        # Setup augmentation (lighter for student)
        self.augment = augment and split == 'train'
        if self.augment:
            self.augmentation = create_student_augmentation()
        
        # Load teacher predictions
        self.teacher_preds = self._load_teacher_predictions()
        
        # Load sample paths
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} with teacher predictions")
    
    def _load_teacher_predictions(self):
        """Load teacher predictions from pickle file"""
        pred_file = self.teacher_preds_dir / self.split / 'teacher_predictions.pkl'
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Teacher predictions not found: {pred_file}")
        
        with open(pred_file, 'rb') as f:
            predictions = pickle.load(f)
        
        return predictions
    
    def _load_samples(self):
        """Load sample paths"""
        samples_file = self.teacher_preds_dir / 'sample_paths.json'
        
        if not samples_file.exists():
            raise FileNotFoundError(f"Sample paths not found: {samples_file}")
        
        with open(samples_file, 'r') as f:
            samples_info = json.load(f)
        
        if self.split == 'train':
            return samples_info['train_samples']
        else:
            return samples_info['val_samples']
    
    def _extract_video_frames(self, video_path):
        """Extract frames from video (same as teacher dataset)"""
        import cv2
        
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
        """Extract audio from video (same as teacher dataset)"""
        import torchaudio
        
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
        video_path, hard_label = self.samples[idx]
        
        # Get teacher predictions for this sample
        teacher_pred = self.teacher_preds[idx]
        soft_probs = torch.from_numpy(teacher_pred['soft_probs']).float()
        
        # Extract video frames
        frames = self._extract_video_frames(video_path)
        if frames is None:
            frames = torch.zeros(self.num_frames, 3, 224, 224)
        
        # Extract audio
        audio = self._extract_audio(video_path)
        
        # Apply augmentations (lighter than teacher)
        if self.augment:
            frames_batch = frames.unsqueeze(0)  # Add batch dim
            audio_batch = audio.unsqueeze(0)
            frames_batch, audio_batch = self.augmentation(frames_batch, audio_batch, self.sample_rate)
            frames = frames_batch.squeeze(0)
            audio = audio_batch.squeeze(0)
        
        return frames, audio, hard_label, soft_probs

class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets"""
    
    def __init__(self, alpha=0.5, temperature=3.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, hard_labels, teacher_soft_probs):
        """
        Args:
            student_logits: [B, num_classes]
            hard_labels: [B]
            teacher_soft_probs: [B, num_classes]
        """
        # Hard label loss
        hard_loss = self.ce_loss(student_logits, hard_labels)
        
        # Soft label loss (KL divergence)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft_probs) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch with distillation"""
    model.train()
    total_loss = 0
    total_hard_loss = 0
    total_soft_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (frames, audio, hard_labels, soft_probs) in enumerate(pbar):
        frames = frames.to(device)
        audio = audio.to(device)
        hard_labels = hard_labels.to(device)
        soft_probs = soft_probs.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(frames, audio)
        
        # Distillation loss
        total_loss_batch, hard_loss, soft_loss = criterion(logits, hard_labels, soft_probs)
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()
        
        # Collect predictions
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(probs[:, 1].cpu().detach().numpy())  # Fake probability
        all_labels.extend(hard_labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Hard': f'{hard_loss.item():.4f}',
            'Soft': f'{soft_loss.item():.4f}'
        })
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    avg_hard_loss = total_hard_loss / len(dataloader)
    avg_soft_loss = total_soft_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, avg_hard_loss, avg_soft_loss, auc, acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_hard_loss = 0
    total_soft_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, hard_labels, soft_probs in tqdm(dataloader, desc='Validation'):
            frames = frames.to(device)
            audio = audio.to(device)
            hard_labels = hard_labels.to(device)
            soft_probs = soft_probs.to(device)
            
            logits = model(frames, audio)
            total_loss_batch, hard_loss, soft_loss = criterion(logits, hard_labels, soft_probs)
            
            total_loss += total_loss_batch.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    avg_hard_loss = total_hard_loss / len(dataloader)
    avg_soft_loss = total_soft_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, avg_hard_loss, avg_soft_loss, auc, acc

def main():
    parser = argparse.ArgumentParser(description='Distill Student Model')
    parser.add_argument('--teacher_preds', required=True, help='Teacher predictions directory')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--save_path', default='models/student_distilled.pt', help='Model save path')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.5, help='Distillation loss weight')
    parser.add_argument('--temp', type=float, default=3.0, help='Distillation temperature')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = DistillationDataset(
        args.data_dir,
        args.teacher_preds,
        split='train',
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        augment=True
    )
    
    val_dataset = DistillationDataset(
        args.data_dir,
        args.teacher_preds,
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
    
    # Create student model
    model = create_student_model(num_classes=2, visual_frames=args.num_frames)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Student model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temp)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_auc = 0
    train_history = []
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_hard, train_soft, train_auc, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_hard, val_soft, val_auc, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f} (Hard: {train_hard:.4f}, Soft: {train_soft:.4f}), AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f} (Hard: {val_hard:.4f}, Soft: {val_soft:.4f}), AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_hard_loss': train_hard,
            'train_soft_loss': train_soft,
            'train_auc': train_auc,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_hard_loss': val_hard,
            'val_soft_loss': val_soft,
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
    
    logger.info(f"Distillation completed! Best AUC: {best_auc:.4f}")
    logger.info(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()