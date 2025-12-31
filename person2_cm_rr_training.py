"""
Person 2: CM-Model (Compression) + RR-Model (Resolution/Reconstruction) Training
Handles compression artifact detection and resolution inconsistency analysis
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
import zipfile
from pathlib import Path
import logging
from datetime import datetime, timedelta
import gc
import psutil
from scipy import ndimage
from skimage import measure, filters
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class ResolutionAnalysisModule(nn.Module):
    """Module for analyzing resolution inconsistencies"""
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale resolution analysis
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, padding=3),
            nn.ReLU()
        )
        
        # Edge consistency analyzer
        self.edge_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Texture consistency analyzer
        self.texture_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(48, 24, kernel_size=1)
        )
    
    def forward(self, x):
        # Multi-scale analysis
        scale1_features = self.scale1(x)
        scale1_features = F.adaptive_avg_pool2d(scale1_features, (7, 7))
        
        scale2_features = self.scale2(x)
        scale2_features = F.adaptive_avg_pool2d(scale2_features, (7, 7))
        
        scale3_features = self.scale3(x)
        scale3_features = F.adaptive_avg_pool2d(scale3_features, (7, 7))
        
        # Edge analysis
        edge_features = self.edge_analyzer(x)
        edge_features = F.adaptive_avg_pool2d(edge_features, (7, 7))
        
        # Texture analysis
        texture_features = self.texture_analyzer(x)
        texture_features = F.adaptive_avg_pool2d(texture_features, (7, 7))
        
        # Combine all features
        combined = torch.cat([scale1_features, scale2_features, scale3_features, 
                             edge_features, texture_features], dim=1)
        return combined

class SpecialistModel(nn.Module):
    """Specialist model for CM or RR analysis with transfer learning"""
    def __init__(self, num_classes=2, model_type='cm'):
        super().__init__()
        from torchvision.models import efficientnet_b4
        
        # Load pretrained EfficientNet-B4
        self.backbone = efficientnet_b4(pretrained=True)
        self.model_type = model_type
        
        # TRANSFER LEARNING STRATEGY:
        # 1. Freeze early layers (feature extraction)
        # 2. Unfreeze last few blocks for fine-tuning
        self._setup_transfer_learning()
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792  # EfficientNet-B4 features
        
        if model_type == 'cm':
            # Compression Model
            self.specialist_module = CompressionAnalysisModule()
            specialist_features = 112 * 7 * 7  # (64+32+16) * 7 * 7
        elif model_type == 'rr':
            # Resolution/Reconstruction Model
            self.specialist_module = ResolutionAnalysisModule()
            specialist_features = 208 * 7 * 7  # (64+64+64+16+24) * 7 * 7
        
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

class SpecialistDataset(Dataset):
    """Dataset with specialized preprocessing for CM/RR models"""
    def __init__(self, data_dir, chunk_file, model_type='cm', transform=None):
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
    
    def _extract_specialized_frames(self, video_path, chunk_path, num_frames=8):
        """Extract frames with specialized preprocessing"""
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
                        # Select frames strategically
                        if self.model_type == 'cm':
                            # For compression: focus on high-motion areas
                            indices = np.linspace(0, total_frames-1, num_frames*2, dtype=int)
                        elif self.model_type == 'rr':
                            # For resolution: focus on detailed areas
                            indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
                        
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                if self.model_type == 'cm':
                                    # Compression-specific preprocessing
                                    frame = self._preprocess_for_compression(frame)
                                elif self.model_type == 'rr':
                                    # Resolution-specific preprocessing
                                    frame = self._preprocess_for_resolution(frame)
                                
                                frame = cv2.resize(frame, (224, 224))
                                frames.append(frame)
                                
                                if len(frames) >= num_frames:
                                    break
                    
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
    
    def _preprocess_for_resolution(self, frame):
        """Preprocessing specific to resolution analysis"""
        # Enhance edge inconsistencies
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        
        # Combine edges
        combined_edges = cv2.bitwise_or(edges1, edges2)
        
        # Convert back to RGB and blend
        edges_rgb = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2RGB)
        result = cv2.addWeighted(frame, 0.8, edges_rgb, 0.2, 0)
        
        return result
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract specialized frames
        frames = self._extract_specialized_frames(sample['video_path'], sample['chunk_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        return frame, label

class SpecialistTrainer:
    """Trainer for CM and RR specialist models"""
    def __init__(self, model_type, data_dir, output_dir):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = SpecialistModel(num_classes=2, model_type=model_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters with transfer learning optimization
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
        
        logger.info(f"Starting {self.model_type.upper()} training on chunk {chunk_idx}: {chunk_file}")
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = SpecialistDataset(self.data_dir, chunk_file, self.model_type, transform)
        dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=4)
        
        if len(dataset) == 0:
            logger.warning(f"No data found in chunk {chunk_file}")
            return
        
        # Training loop
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        for epoch in range(4):  # 4 epochs per chunk for specialist models
            epoch_loss = 0.0
            batch_count = 0
            
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
                    logger.info(f"{self.model_type.upper()} - Chunk {chunk_idx}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
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
            
            logger.info(f"{self.model_type.upper()} - Chunk {chunk_idx}, Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Calculate final metrics for this chunk
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['avg_loss'] = np.mean(epoch_losses)
        metrics['chunk_idx'] = chunk_idx
        
        self.training_history.append(metrics)
        
        # Save checkpoint after chunk completion
        self._save_checkpoint(chunk_idx, metrics)
        
        logger.info(f"{self.model_type.upper()} - Chunk {chunk_idx} training completed. Metrics: {metrics}")
        
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
        logger.info(f"Starting {self.model_type.upper()} specialist model training")
        
        for chunk_idx in range(self.current_chunk, self.total_chunks):
            try:
                self.train_on_chunk(chunk_idx)
                self.current_chunk = chunk_idx + 1
                
                # Log memory usage
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"Memory usage after chunk {chunk_idx}: {memory_usage:.1f}%")
                
            except Exception as e:
                logger.error(f"Error training {self.model_type.upper()} on chunk {chunk_idx}: {e}")
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
    
    parser = argparse.ArgumentParser(description='Train CM and RR specialist models')
    parser.add_argument('--model', choices=['cm', 'rr'], required=True, help='Model type to train')
    parser.add_argument('--data_dir', default='/kaggle/input/dfdc-10', help='Data directory')
    parser.add_argument('--output_dir', default='/kaggle/working', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SpecialistTrainer(args.model, args.data_dir, args.output_dir)
    
    # Start training
    trainer.train_all_chunks()

if __name__ == "__main__":
    main()