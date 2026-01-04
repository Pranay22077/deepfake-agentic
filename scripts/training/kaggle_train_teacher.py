#!/usr/bin/env python3
"""
Kaggle Teacher Model Training Script
Trains the heavy multimodal teacher model on Kaggle GPUs
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

# Import our models (copy from src/)
from models.teacher import create_teacher_model
from preprocess.augmentation import create_teacher_augmentation
from train.train_teacher import MultimodalDataset

def setup_kaggle_environment():
    """Setup Kaggle environment and paths"""
    # Create necessary directories
    os.makedirs('/kaggle/working/models', exist_ok=True)
    os.makedirs('/kaggle/working/logs', exist_ok=True)
    os.makedirs('/kaggle/working/teacher_predictions', exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/kaggle/working/logs/teacher_training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def train_teacher_model():
    """Main teacher training function"""
    logger = setup_kaggle_environment()
    
    # Configuration
    config = {
        'epochs': 25,
        'batch_size': 8,  # Smaller batch for heavy model
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
    
    # Data paths (adjust for your Kaggle dataset)
    data_dir = '/kaggle/input/deepfake-faces-dataset'  # Your dataset path
    
    # Create datasets with heavy augmentation
    train_dataset = MultimodalDataset(
        data_dir=data_dir,
        split='train',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        sample_rate=config['sample_rate'],
        augment=True,
        heavy_augment=True  # Use heavy augmentation for teacher
    )
    
    val_dataset = MultimodalDataset(
        data_dir=data_dir,
        split='val',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        sample_rate=config['sample_rate'],
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
        train_auc = roc_auc_score(train_labels, train_preds)
        val_auc = roc_auc_score(val_labels, val_preds)
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
    
    # Generate teacher predictions for distillation
    if config['save_predictions']:
        logger.info("Generating teacher predictions for distillation...")
        generate_teacher_predictions(model, train_loader, val_loader, device)
    
    return model, best_auc

def generate_teacher_predictions(model, train_loader, val_loader, device):
    """Generate teacher predictions for knowledge distillation"""
    model.eval()
    
    # Generate predictions for training set
    train_predictions = []
    train_paths = []
    
    with torch.no_grad():
        for batch_idx, (frames, audio, labels) in enumerate(tqdm(train_loader, desc='Teacher predictions (train)')):
            frames = frames.to(device)
            audio = audio.to(device)
            
            logits, features = model(frames, audio, return_features=True)
            probs = torch.softmax(logits, dim=1)
            
            # Store predictions
            for i in range(len(labels)):
                train_predictions.append({
                    'soft_probs': probs[i].cpu().numpy(),
                    'features': {k: v[i].cpu().numpy() for k, v in features.items()},
                    'hard_label': labels[i].item()
                })
    
    # Generate predictions for validation set
    val_predictions = []
    
    with torch.no_grad():
        for batch_idx, (frames, audio, labels) in enumerate(tqdm(val_loader, desc='Teacher predictions (val)')):
            frames = frames.to(device)
            audio = audio.to(device)
            
            logits, features = model(frames, audio, return_features=True)
            probs = torch.softmax(logits, dim=1)
            
            # Store predictions
            for i in range(len(labels)):
                val_predictions.append({
                    'soft_probs': probs[i].cpu().numpy(),
                    'features': {k: v[i].cpu().numpy() for k, v in features.items()},
                    'hard_label': labels[i].item()
                })
    
    # Save predictions
    os.makedirs('/kaggle/working/teacher_predictions/train', exist_ok=True)
    os.makedirs('/kaggle/working/teacher_predictions/val', exist_ok=True)
    
    with open('/kaggle/working/teacher_predictions/train/teacher_predictions.pkl', 'wb') as f:
        pickle.dump(train_predictions, f)
    
    with open('/kaggle/working/teacher_predictions/val/teacher_predictions.pkl', 'wb') as f:
        pickle.dump(val_predictions, f)
    
    # Save sample paths info
    sample_info = {
        'train_samples': len(train_predictions),
        'val_samples': len(val_predictions),
        'generated_at': str(torch.utils.data.get_worker_info())
    }
    
    with open('/kaggle/working/teacher_predictions/sample_paths.json', 'w') as f:
        json.dump(sample_info, f, indent=2)
    
    print(f"Teacher predictions saved: {len(train_predictions)} train, {len(val_predictions)} val")

if __name__ == "__main__":
    # Run teacher training
    model, best_auc = train_teacher_model()
    
    print(f"\n[DONE] Teacher Training Complete!")
    print(f"Best AUC: {best_auc:.4f}")
    print(f"Model saved to: /kaggle/working/models/teacher_model.pt")
    print(f"Predictions saved to: /kaggle/working/teacher_predictions/")