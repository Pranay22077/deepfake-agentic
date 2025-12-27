#!/usr/bin/env python3
"""
Save Teacher Predictions for Knowledge Distillation
Generates soft labels from trained teacher model
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import pickle

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.teacher import create_teacher_model
from train.train_teacher import MultimodalDataset

def save_teacher_predictions(model, dataloader, output_dir, device, temperature=1.0):
    """Generate and save teacher predictions"""
    model.eval()
    
    predictions = {}
    features_dict = {}
    
    with torch.no_grad():
        for batch_idx, (frames, audio, labels) in enumerate(tqdm(dataloader, desc='Generating predictions')):
            frames = frames.to(device)
            audio = audio.to(device)
            
            # Get teacher predictions
            logits, features = model(frames, audio, return_features=True)
            
            # Apply temperature scaling for soft labels
            soft_logits = logits / temperature
            soft_probs = torch.softmax(soft_logits, dim=1)
            
            # Store predictions for each sample in batch
            batch_size = frames.shape[0]
            for i in range(batch_size):
                sample_idx = batch_idx * dataloader.batch_size + i
                
                predictions[sample_idx] = {
                    'logits': logits[i].cpu().numpy(),
                    'soft_probs': soft_probs[i].cpu().numpy(),
                    'hard_label': labels[i].item(),
                    'visual_feat': features['visual_feat'][i].cpu().numpy(),
                    'audio_feat': features['audio_feat'][i].cpu().numpy(),
                    'fused_feat': features['fused_feat'][i].cpu().numpy(),
                    'lip_sync_score': features['lip_sync_score'][i].cpu().numpy()
                }
    
    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as pickle for efficiency
    pred_file = os.path.join(output_dir, 'teacher_predictions.pkl')
    with open(pred_file, 'wb') as f:
        pickle.dump(predictions, f)
    
    # Save metadata
    metadata = {
        'num_samples': len(predictions),
        'temperature': temperature,
        'feature_dims': {
            'visual_feat': predictions[0]['visual_feat'].shape[0],
            'audio_feat': predictions[0]['audio_feat'].shape[0],
            'fused_feat': predictions[0]['fused_feat'].shape[0]
        }
    }
    
    meta_file = os.path.join(output_dir, 'metadata.json')
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(predictions)} teacher predictions to {output_dir}")
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Save Teacher Predictions')
    parser.add_argument('--model', required=True, help='Path to trained teacher model')
    parser.add_argument('--data_dir', default='data', help='Data directory')
    parser.add_argument('--output', default='teacher_preds', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft labels')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load teacher model
    print(f"Loading teacher model from {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    model = create_teacher_model(num_classes=2, visual_frames=args.num_frames)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Teacher model loaded successfully")
    
    # Create dataset (both train and val for distillation)
    print("Creating datasets...")
    
    # Training set
    train_dataset = MultimodalDataset(
        args.data_dir,
        split='train',
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        augment=False  # No augmentation for prediction generation
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep order for indexing
        num_workers=4,
        pin_memory=True
    )
    
    # Validation set
    val_dataset = MultimodalDataset(
        args.data_dir,
        split='val',
        num_frames=args.num_frames,
        audio_duration=args.audio_duration,
        augment=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Generate predictions for training set
    print("Generating predictions for training set...")
    train_output_dir = os.path.join(args.output, 'train')
    train_preds = save_teacher_predictions(
        model, train_loader, train_output_dir, device, args.temperature
    )
    
    # Generate predictions for validation set
    print("Generating predictions for validation set...")
    val_output_dir = os.path.join(args.output, 'val')
    val_preds = save_teacher_predictions(
        model, val_loader, val_output_dir, device, args.temperature
    )
    
    # Save sample file paths for reference
    train_samples = [(path, label) for path, label in train_dataset.samples]
    val_samples = [(path, label) for path, label in val_dataset.samples]
    
    samples_info = {
        'train_samples': train_samples,
        'val_samples': val_samples
    }
    
    samples_file = os.path.join(args.output, 'sample_paths.json')
    with open(samples_file, 'w') as f:
        json.dump(samples_info, f, indent=2)
    
    print(f"Teacher predictions saved successfully!")
    print(f"Train samples: {len(train_preds)}")
    print(f"Val samples: {len(val_preds)}")
    print(f"Output directory: {args.output}")

if __name__ == "__main__":
    main()