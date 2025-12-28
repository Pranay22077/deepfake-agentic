#!/usr/bin/env python3
"""
Fine-tune Pruned Model
Recovers accuracy after pruning through fine-tuning
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import logging
from tqdm import tqdm
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model
from train.train_baseline import MultimodalDataset

def create_lightweight_dataset(data_dir, split='train', num_frames=8, 
                              audio_duration=3.0, sample_rate=16000):
    """Create dataset for fine-tuning (lighter augmentation)"""
    return MultimodalDataset(
        data_dir=data_dir,
        split=split,
        num_frames=num_frames,
        audio_duration=audio_duration,
        sample_rate=sample_rate,
        augment=(split == 'train')  # Light augmentation only for training
    )

def fine_tune_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Fine-tune for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Fine-tune Epoch {epoch}')
    for batch_idx, (frames, audio, labels) in enumerate(pbar):
        frames = frames.to(device)
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(frames, audio)
        loss = criterion(logits, labels)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Collect predictions
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(probs[:, 1].cpu().detach().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, auc, acc

def validate_model(model, dataloader, criterion, device):
    """Validate pruned model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, audio, labels in tqdm(dataloader, desc='Validation'):
            frames = frames.to(device)
            audio = audio.to(device)
            labels = labels.to(device)
            
            logits = model(frames, audio)
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
    parser = argparse.ArgumentParser(description='Fine-tune Pruned Model')
    parser.add_argument('--pruned_model', required=True, help='Path to pruned model')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--save_path', required=True, help='Path to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=5, help='Fine-tuning epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (low for fine-tuning)')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load pruned model
    logger.info(f"Loading pruned model from {args.pruned_model}")
    
    model = create_student_model()
    checkpoint = torch.load(args.pruned_model, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        pruning_info = checkpoint.get('pruning_info', {})
    else:
        # Direct model save
        model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        pruning_info = checkpoint.get('pruning_info', {}) if isinstance(checkpoint, dict) else {}
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Create datasets
    train_dataset = create_lightweight_dataset(
        args.data_dir, 
        split='train',
        num_frames=args.num_frames,
        audio_duration=args.audio_duration
    )
    
    val_dataset = create_lightweight_dataset(
        args.data_dir,
        split='val', 
        num_frames=args.num_frames,
        audio_duration=args.audio_duration
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Loss and optimizer (low learning rate for fine-tuning)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Evaluate before fine-tuning
    logger.info("Evaluating pruned model before fine-tuning...")
    initial_loss, initial_auc, initial_acc = validate_model(model, val_loader, criterion, device)
    logger.info(f"Initial - Loss: {initial_loss:.4f}, AUC: {initial_auc:.4f}, Acc: {initial_acc:.4f}")
    
    # Fine-tuning loop
    best_auc = initial_auc
    history = []
    
    for epoch in range(args.epochs):
        logger.info(f"Fine-tuning epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_auc, train_acc = fine_tune_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_auc, val_acc = validate_model(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}")
        
        # Save history
        history.append({
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
            
            # Create save directory
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            
            # Save fine-tuned model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_auc': best_auc,
                'initial_metrics': {
                    'loss': initial_loss,
                    'auc': initial_auc,
                    'acc': initial_acc
                },
                'final_metrics': {
                    'loss': val_loss,
                    'auc': val_auc,
                    'acc': val_acc
                },
                'pruning_info': pruning_info,
                'fine_tuning_info': {
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'batch_size': args.batch_size
                },
                'args': args
            }, args.save_path)
            
            logger.info(f"New best model saved! AUC: {val_auc:.4f}")
    
    # Save training history
    history_path = args.save_path.replace('.pt', '_fine_tune_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    improvement = best_auc - initial_auc
    logger.info(f"\nFine-tuning completed!")
    logger.info(f"Initial AUC: {initial_auc:.4f}")
    logger.info(f"Final AUC: {best_auc:.4f}")
    logger.info(f"Improvement: {improvement:.4f} ({improvement/initial_auc*100:.1f}%)")
    logger.info(f"Fine-tuned model saved to: {args.save_path}")

if __name__ == "__main__":
    main()