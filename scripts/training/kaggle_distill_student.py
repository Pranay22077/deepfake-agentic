#!/usr/bin/env python3
"""
Kaggle Student Distillation Script
Trains student model using teacher predictions on Kaggle
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle
from tqdm import tqdm
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Kaggle environment setup
sys.path.append('/kaggle/working')

# Import our models
from models.student import create_student_model
from train.distill_student import DistillationLoss, DistillationDataset

def setup_kaggle_environment():
    """Setup Kaggle environment"""
    os.makedirs('/kaggle/working/models', exist_ok=True)
    os.makedirs('/kaggle/working/logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/kaggle/working/logs/student_distillation.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def distill_student_model():
    """Main student distillation function"""
    logger = setup_kaggle_environment()
    
    # Configuration
    config = {
        'epochs': 15,
        'batch_size': 32,  # Larger batch for lighter student
        'lr': 1e-4,
        'alpha': 0.5,  # Distillation weight
        'temperature': 3.0,
        'num_frames': 8,
        'audio_duration': 3.0
    }
    
    logger.info("Starting student distillation on Kaggle")
    logger.info(f"Configuration: {config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data paths
    data_dir = '/kaggle/input/deepfake-faces-dataset'
    teacher_preds_dir = '/kaggle/working/teacher_predictions'
    
    # Check if teacher predictions exist
    if not os.path.exists(teacher_preds_dir):
        logger.error("Teacher predictions not found! Run teacher training first.")
        return None, 0
    
    # Create distillation datasets
    train_dataset = DistillationDataset(
        data_dir=data_dir,
        teacher_preds_dir=teacher_preds_dir,
        split='train',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        augment=True
    )
    
    val_dataset = DistillationDataset(
        data_dir=data_dir,
        teacher_preds_dir=teacher_preds_dir,
        split='val',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        augment=False
    )
    
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
    
    # Create student model
    model = create_student_model(num_classes=2, visual_frames=config['num_frames'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Student model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DistillationLoss(alpha=config['alpha'], temperature=config['temperature'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    best_auc = 0
    train_history = []
    
    for epoch in range(config['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        model.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f'Distillation Epoch {epoch+1}')
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
            all_preds.extend(probs[:, 1].cpu().detach().numpy())
            all_labels.extend(hard_labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Hard': f'{hard_loss.item():.4f}',
                'Soft': f'{soft_loss.item():.4f}'
            })
        
        # Validate
        model.eval()
        val_total_loss = 0
        val_hard_loss = 0
        val_soft_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for frames, audio, hard_labels, soft_probs in tqdm(val_loader, desc='Validation'):
                frames = frames.to(device)
                audio = audio.to(device)
                hard_labels = hard_labels.to(device)
                soft_probs = soft_probs.to(device)
                
                logits = model(frames, audio)
                total_loss_batch, hard_loss_batch, soft_loss_batch = criterion(logits, hard_labels, soft_probs)
                
                val_total_loss += total_loss_batch.item()
                val_hard_loss += hard_loss_batch.item()
                val_soft_loss += soft_loss_batch.item()
                
                probs = torch.softmax(logits, dim=1)
                val_preds.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(hard_labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = total_loss / len(train_loader)
        train_hard = total_hard_loss / len(train_loader)
        train_soft = total_soft_loss / len(train_loader)
        train_auc = roc_auc_score(all_labels, all_preds)
        train_acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
        
        val_loss = val_total_loss / len(val_loader)
        val_hard = val_hard_loss / len(val_loader)
        val_soft = val_soft_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)
        
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
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_auc': best_auc,
                'config': config
            }, '/kaggle/working/models/student_distilled.pt')
            
            logger.info(f"New best student model saved! AUC: {val_auc:.4f}")
    
    # Save training history
    with open('/kaggle/working/models/student_distillation_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    logger.info(f"Student distillation completed! Best AUC: {best_auc:.4f}")
    
    return model, best_auc

if __name__ == "__main__":
    # Run student distillation
    model, best_auc = distill_student_model()
    
    if model is not None:
        print(f"\n[TARGET] Student Distillation Complete!")
        print(f"Best AUC: {best_auc:.4f}")
        print(f"Model saved to: /kaggle/working/models/student_distilled.pt")
    else:
        print("[ERROR] Student distillation failed!")