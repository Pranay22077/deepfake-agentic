"""
FIXED RR Model Training - Resolves EfficientNet-B4 Download Issues
Uses cached weights or simple architecture to avoid network connectivity problems
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

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/kaggle/input/dfdc-10'
OUTPUT_DIR = '/kaggle/working'

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

class SpecialistModelSimple(nn.Module):
    """Simple RR model that avoids EfficientNet-B4 download issues"""
    def __init__(self, num_classes=2, model_type='rr'):
        super().__init__()
        self.model_type = model_type
        
        # Simple but effective backbone (no external downloads)
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        backbone_features = 512 * 7 * 7  # 25088
        
        # Resolution specialist module
        self.specialist_module = ResolutionAnalysisModule()
        specialist_features = 208 * 7 * 7  # (64+64+64+16+24) * 7 * 7 = 10192
        
        # Combined classifier
        total_features = backbone_features + specialist_features  # 35280
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(total_features, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, 4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, total_features)
        )
        
        logger.info(f"✅ Simple RR model initialized (no external downloads required)")
        logger.info(f"📊 Total features: {total_features} (backbone: {backbone_features}, specialist: {specialist_features})")
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone(x)
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
    """Dataset with specialized preprocessing for RR models"""
    def __init__(self, data_dir, chunk_folder, model_type='rr', transform=None):
        self.data_dir = Path(data_dir)
        self.chunk_folder = chunk_folder
        self.model_type = model_type
        self.transform = transform
        self.samples = []
        
        self._load_chunk_data()
    
    def _load_chunk_data(self):
        """Load data from current chunk folder"""
        chunk_path = self.data_dir / self.chunk_folder
        
        if not chunk_path.exists():
            logger.warning(f"Chunk folder {self.chunk_folder} not found, waiting...")
            return
        
        logger.info(f"📦 Processing folder {self.chunk_folder}...")
        
        # Find all video files in the folder
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(chunk_path.glob(f'*{ext}'))
        
        # Load metadata if available
        metadata_file = chunk_path / 'metadata.json'
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        real_count = 0
        fake_count = 0
        
        for video_file in video_files:
            video_name = video_file.name
            
            # Determine label from metadata or filename
            if video_name in metadata:
                label = 0 if metadata[video_name]['label'] == 'REAL' else 1
            else:
                # Fallback: use filename patterns
                label = 1 if any(pattern in video_name.lower() for pattern in ['fake', 'manipulated', 'deepfake']) else 0
            
            self.samples.append({
                'video_path': str(video_file),
                'label': label
            })
            
            if label == 0:
                real_count += 1
            else:
                fake_count += 1
        
        logger.info(f"🎬 Found {len(video_files)} video files")
        logger.info(f"📊 Label distribution: {real_count} real, {fake_count} fake")
    
    def _extract_specialized_frames(self, video_path, num_frames=8):
        """Extract frames with resolution-specific preprocessing"""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames > 0:
                # For resolution analysis: focus on detailed areas
                indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
                
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Resolution-specific preprocessing
                        frame = self._preprocess_for_resolution(frame)
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
        frames = self._extract_specialized_frames(sample['video_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        return frame, label

class OptimizedSpecialistTrainerFixed:
    """Fixed trainer for RR specialist model"""
    def __init__(self, model_type, data_dir, output_dir):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 Initializing {model_type.upper()} specialist trainer for Kaggle T4x2...")
        
        # Initialize SIMPLE model (no downloads)
        self.model = SpecialistModelSimple(num_classes=2, model_type=model_type)
        self.model.to(DEVICE)
        
        # Training parameters
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_chunk = 0
        self.total_chunks = 10  # Only first 10 chunks (00-09)
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 3 * 3600  # 3 hours
        
        # Metrics tracking
        self.training_history = []
        
        logger.info(f"✅ {model_type.upper()} specialist trainer ready!")
    
    def train_on_chunk(self, chunk_idx):
        """Train model on a specific chunk"""
        chunk_folder = f"dfdc_train_part_{chunk_idx:02d}"
        
        # Wait for chunk to be available
        chunk_path = self.data_dir / chunk_folder
        while not chunk_path.exists():
            logger.info(f"⏳ Waiting for chunk {chunk_folder}...")
            time.sleep(60)  # Wait 1 minute
        
        logger.info(f"🎯 Starting {self.model_type.upper()} specialist training on chunk {chunk_idx}")
        logger.info("=" * 60)
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = SpecialistDataset(self.data_dir, chunk_folder, self.model_type, transform)
        
        if len(dataset) == 0:
            logger.warning(f"No data found in chunk {chunk_folder}")
            return
        
        dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=2)
        
        logger.info(f"✅ Loaded {len(dataset)} samples from chunk {chunk_idx}")
        logger.info(f"📊 Dataset size: {len(dataset)} samples")
        logger.info(f"🔄 Batches per epoch: {len(dataloader)}")
        
        # Training loop
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        for epoch in range(4):  # 4 epochs per chunk
            logger.info(f"📈 Epoch {epoch+1}/4 for chunk {chunk_idx}")
            
            epoch_loss = 0.0
            batch_count = 0
            correct_predictions = 0
            total_samples = 0
            
            for batch_idx, (frames, labels) in enumerate(dataloader):
                frames = frames.to(DEVICE)
                labels = labels.squeeze().to(DEVICE)
                
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
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
                
                # Collect predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Log progress every 10 batches
                if batch_idx % 10 == 0:
                    current_acc = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    
                    print(f"Training {self.model_type.upper()}: {batch_idx:3d}/{len(dataloader)} "
                          f"[{100.*batch_idx/len(dataloader):5.1f}%] "
                          f"Loss={loss.item():.4f}, Acc={current_acc:.2f}%, GPU={gpu_memory:.1f}GB", 
                          end='\r')
                
                # Memory cleanup
                if batch_idx % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            epoch_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0
            epoch_losses.append(avg_epoch_loss)
            
            self.scheduler.step()
            
            logger.info(f"✅ {self.model_type.upper()} - Chunk {chunk_idx}, Epoch {epoch+1} completed. "
                       f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # Calculate final metrics for this chunk
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['avg_loss'] = np.mean(epoch_losses)
        metrics['chunk_idx'] = chunk_idx
        
        self.training_history.append(metrics)
        
        # Save checkpoint after chunk completion
        self._save_checkpoint(chunk_idx, metrics)
        
        logger.info(f"🎉 {self.model_type.upper()} - Chunk {chunk_idx} training completed!")
        logger.info(f"📊 Final metrics: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
        
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
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def train_all_chunks(self):
        """Train on all available chunks sequentially"""
        logger.info(f"🚀 Starting {self.model_type.upper()} specialist model training on ALL 10 chunks")
        logger.info("🎯 Target: Process 100GB of data with full utilization")
        logger.info("=" * 80)
        
        for chunk_idx in range(self.current_chunk, self.total_chunks):
            try:
                self.train_on_chunk(chunk_idx)
                self.current_chunk = chunk_idx + 1
                
                # Log memory usage
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"💾 Memory usage after chunk {chunk_idx}: {memory_usage:.1f}%")
                
            except Exception as e:
                logger.error(f"❌ Error training {self.model_type.upper()} on chunk {chunk_idx}: {e}")
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
        
        logger.info(f"🎉 Final {self.model_type.upper()} model saved: {final_model_path}")
        
        # Save training history
        history_path = self.output_dir / f"{self.model_type}_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

# Initialize FIXED RR trainer
rr_trainer = OptimizedSpecialistTrainerFixed('rr', DATA_DIR, OUTPUT_DIR)

# Start training
rr_trainer.train_all_chunks()