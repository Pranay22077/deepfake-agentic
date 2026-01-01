"""
Person 1: BG-Model (Background/Baseline) + AV-Model (Audio-Visual) Training
Handles general deepfake detection and audio-visual synchronization analysis
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
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import librosa
import zipfile
from pathlib import Path
import logging
from datetime import datetime, timedelta
import gc
import psutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EfficientNetBackbone(nn.Module):
    """EfficientNet-B4 based backbone with transfer learning for both models"""
    def __init__(self, num_classes=2, model_type='bg'):
        super().__init__()
        from torchvision.models import efficientnet_b4
        
        # Load pretrained EfficientNet-B4
        self.backbone = efficientnet_b4(pretrained=True)
        self.model_type = model_type
        
        # TRANSFER LEARNING STRATEGY:
        # 1. Freeze early layers (feature extraction)
        # 2. Unfreeze last few blocks for fine-tuning
        self._setup_transfer_learning()
        
        # Modify classifier
        in_features = self.backbone.classifier[1].in_features
        
        if model_type == 'bg':
            # BG-Model: Simple binary classification
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        elif model_type == 'av':
            # AV-Model: Audio-visual features fusion
            self.audio_encoder = nn.Sequential(
                nn.Linear(128, 256),  # MFCC features
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 512)
            )
            
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features + 512, 1024),  # Visual + Audio
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
    
    def _setup_transfer_learning(self):
        """Setup transfer learning: freeze early layers, unfreeze last few blocks"""
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last 3 blocks of features for fine-tuning
        # EfficientNet-B4 has 8 blocks, we unfreeze blocks 6, 7, 8
        blocks_to_unfreeze = [6, 7]  # Last 2 blocks + classifier
        
        for i, block in enumerate(self.backbone.features):
            if i >= len(self.backbone.features) - len(blocks_to_unfreeze):
                for param in block.parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze block {i} for fine-tuning")
        
        # Always unfreeze classifier (will be replaced anyway)
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        logger.info(f"Transfer learning setup: Unfroze last {len(blocks_to_unfreeze)} blocks + classifier")
    
    def forward(self, x, audio_features=None):
        if self.model_type == 'bg':
            return self.backbone(x)
        elif self.model_type == 'av':
            # Extract visual features
            visual_features = self.backbone.features(x)
            visual_features = self.backbone.avgpool(visual_features)
            visual_features = torch.flatten(visual_features, 1)
            
            # Process audio features
            if audio_features is not None:
                audio_encoded = self.audio_encoder(audio_features)
                combined_features = torch.cat([visual_features, audio_encoded], dim=1)
            else:
                # Fallback if no audio
                combined_features = torch.cat([visual_features, torch.zeros_like(visual_features[:, :512])], dim=1)
            
            return self.backbone.classifier(combined_features)

class DeepfakeDataset(Dataset):
    """Dataset for loading video frames and audio features"""
    def __init__(self, data_dir, chunk_file, model_type='bg', transform=None):
        self.data_dir = Path(data_dir)
        self.chunk_file = chunk_file
        self.model_type = model_type
        self.transform = transform
        self.samples = []
        
        self._load_chunk_data()
    
    def _load_chunk_data(self):
        """Load data from current chunk"""
        chunk_path = self.data_dir / self.chunk_file
        
        if not chunk_path.exists():
            logger.warning(f"Chunk file {self.chunk_file} not found, waiting...")
            return
        
        logger.info(f"Loading chunk: {self.chunk_file}")
        
        with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Filter video files
            video_files = [f for f in file_list if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in video_files:
                # Determine label from filename or directory structure
                label = 1 if 'fake' in video_file.lower() or 'manipulated' in video_file.lower() else 0
                
                self.samples.append({
                    'video_path': video_file,
                    'label': label,
                    'chunk_path': chunk_path
                })
        
        logger.info(f"Loaded {len(self.samples)} samples from {self.chunk_file}")
    
    def _extract_frames(self, video_path, chunk_path, num_frames=16):
        """Extract frames from video"""
        try:
            with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
                with zip_ref.open(video_path) as video_file:
                    # Save temporarily
                    temp_path = f"/tmp/{os.path.basename(video_path)}"
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    cap = cv2.VideoCapture(temp_path)
                    frames = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames > 0:
                        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
                        
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame = cv2.resize(frame, (224, 224))
                                frames.append(frame)
                    
                    cap.release()
                    os.remove(temp_path)
                    
                    if len(frames) < num_frames:
                        # Pad with last frame
                        while len(frames) < num_frames:
                            frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
                    
                    return np.array(frames)
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {e}")
            return np.zeros((num_frames, 224, 224, 3))
    
    def _extract_audio_features(self, video_path, chunk_path):
        """Extract MFCC features from audio"""
        try:
            with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
                with zip_ref.open(video_path) as video_file:
                    temp_path = f"/tmp/{os.path.basename(video_path)}"
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    # Extract audio
                    y, sr = librosa.load(temp_path, sr=16000, duration=10.0)
                    
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                    mfccs_mean = np.mean(mfccs, axis=1)
                    
                    # Pad or truncate to 128 features
                    if len(mfccs_mean) < 128:
                        mfccs_mean = np.pad(mfccs_mean, (0, 128 - len(mfccs_mean)))
                    else:
                        mfccs_mean = mfccs_mean[:128]
                    
                    os.remove(temp_path)
                    return mfccs_mean
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path}: {e}")
            return np.zeros(128)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames(sample['video_path'], sample['chunk_path'])
        
        # Use middle frame for image-based models
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        if self.model_type == 'av':
            # Extract audio features for AV model
            audio_features = self._extract_audio_features(sample['video_path'], sample['chunk_path'])
            audio_features = torch.FloatTensor(audio_features)
            return frame, audio_features, label
        else:
            return frame, label

class ModelTrainer:
    """Handles training for both BG and AV models"""
    def __init__(self, model_type, data_dir, output_dir):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = EfficientNetBackbone(num_classes=2, model_type=model_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters with different learning rates for transfer learning
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_chunk = 0
        self.total_chunks = 10  # Only first 10 chunks (00-09)
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 3 * 3600  # 3 hours
        
        # Metrics tracking
        self.training_history = []
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for pretrained vs new layers"""
        # Separate parameters: pretrained (backbone) vs new (classifier + specialist modules)
        pretrained_params = []
        new_params = []
        
        # Backbone parameters (pretrained, lower learning rate)
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # New parameters (classifier + audio encoder if AV model)
        if hasattr(self.model, 'audio_encoder'):
            for param in self.model.audio_encoder.parameters():
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
        checkpoint_pattern = f"{self.model_type}_model_chunk_*.pt"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_chunk = checkpoint.get('current_chunk', 0)
            self.training_history = checkpoint.get('training_history', [])
            
            logger.info(f"Resumed from chunk {self.current_chunk}")
    
    def _save_checkpoint(self, chunk_idx, metrics):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"{self.model_type}_model_chunk_{chunk_idx:02d}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_chunk': chunk_idx,
            'training_history': self.training_history,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints (keep only latest 2)
        checkpoints = sorted(self.output_dir.glob(f"{self.model_type}_model_chunk_*.pt"))
        if len(checkpoints) > 2:
            for old_checkpoint in checkpoints[:-2]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def _should_save_checkpoint(self):
        """Check if it's time to save a checkpoint"""
        return time.time() - self.last_checkpoint_time >= self.checkpoint_interval
    
    def train_on_chunk(self, chunk_idx):
        """Train model on a specific chunk"""
        chunk_file = f"{chunk_idx:02d}.zip"
        
        # Wait for chunk to be available
        chunk_path = self.data_dir / chunk_file
        while not chunk_path.exists():
            logger.info(f"Waiting for chunk {chunk_file} to be downloaded...")
            time.sleep(60)  # Wait 1 minute
        
        logger.info(f"Starting training on chunk {chunk_idx}: {chunk_file}")
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = DeepfakeDataset(self.data_dir, chunk_file, self.model_type, transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
        
        if len(dataset) == 0:
            logger.warning(f"No data found in chunk {chunk_file}")
            return
        
        # Training loop
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        for epoch in range(3):  # 3 epochs per chunk
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                if self.model_type == 'av':
                    frames, audio_features, labels = batch_data
                    audio_features = audio_features.to(self.device)
                else:
                    frames, labels = batch_data
                    audio_features = None
                
                frames = frames.to(self.device)
                labels = labels.squeeze().to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(frames, audio_features)
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
                    logger.info(f"Chunk {chunk_idx}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                # Check for checkpoint save
                if self._should_save_checkpoint():
                    metrics = self._calculate_metrics(all_predictions, all_labels)
                    self._save_checkpoint(chunk_idx, metrics)
                    self.last_checkpoint_time = time.time()
                
                # Memory cleanup
                if batch_idx % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            
            self.scheduler.step()
            
            logger.info(f"Chunk {chunk_idx}, Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Calculate final metrics for this chunk
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['avg_loss'] = np.mean(epoch_losses)
        metrics['chunk_idx'] = chunk_idx
        
        self.training_history.append(metrics)
        
        # Save checkpoint after chunk completion
        self._save_checkpoint(chunk_idx, metrics)
        
        logger.info(f"Chunk {chunk_idx} training completed. Metrics: {metrics}")
        
        # Cleanup
        del dataset, dataloader
        gc.collect()
        torch.cuda.empty_cache()
    
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
    
    def train_all_chunks(self):
        """Train on all available chunks sequentially"""
        logger.info(f"Starting {self.model_type.upper()} model training")
        
        for chunk_idx in range(self.current_chunk, self.total_chunks):
            try:
                self.train_on_chunk(chunk_idx)
                self.current_chunk = chunk_idx + 1
                
                # Log memory usage
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"Memory usage after chunk {chunk_idx}: {memory_usage:.1f}%")
                
            except Exception as e:
                logger.error(f"Error training on chunk {chunk_idx}: {e}")
                continue
        
        # Save final model
        final_model_path = self.output_dir / f"{self.model_type}_model_final.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_history': self.training_history,
            'model_type': self.model_type,
            'total_chunks_processed': self.current_chunk,
            'final_metrics': self.training_history[-1] if self.training_history else {}
        }, final_model_path)
        
        logger.info(f"Final {self.model_type.upper()} model saved: {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / f"{self.model_type}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BG and AV models')
    parser.add_argument('--model', choices=['bg', 'av'], required=True, help='Model type to train')
    parser.add_argument('--data_dir', default='/kaggle/input/dfdc-10', help='Data directory')
    parser.add_argument('--output_dir', default='/kaggle/working', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(args.model, args.data_dir, args.output_dir)
    
    # Start training
    trainer.train_all_chunks()

if __name__ == "__main__":
    main()