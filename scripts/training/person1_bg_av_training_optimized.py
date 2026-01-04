"""
Person 1: BG-Model (Background/Baseline) + AV-Model (Audio-Visual) Training - OPTIMIZED
Handles general deepfake detection and audio-visual synchronization analysis
OPTIMIZED FOR KAGGLE T4x2 GPU WITH FULL DATA UTILIZATION
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import librosa
import zipfile
from pathlib import Path
import logging
from datetime import datetime
import gc
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging with progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# KAGGLE T4x2 OPTIMIZED SETTINGS
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8  # Optimized for T4x2
NUM_WORKERS = 2  # Kaggle CPU cores
PIN_MEMORY = True
PREFETCH_FACTOR = 2

class EfficientNetBackbone(nn.Module):
    """EfficientNet-B4 based backbone with transfer learning - OPTIMIZED"""
    def __init__(self, num_classes=2, model_type='bg'):
        super().__init__()
        from torchvision.models import efficientnet_b4
        
        print(f"[LOOP] Loading pretrained EfficientNet-B4 for {model_type.upper()} model...")
        self.backbone = efficientnet_b4(pretrained=True)
        self.model_type = model_type
        
        # TRANSFER LEARNING STRATEGY
        self._setup_transfer_learning()
        
        # Get features
        in_features = self.backbone.classifier[1].in_features
        
        if model_type == 'bg':
            # BG-Model: Simple binary classification
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        elif model_type == 'av':
            # AV-Model: Audio-visual features fusion
            self.audio_encoder = nn.Sequential(
                nn.Linear(128, 128),  # MFCC features
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256)
            )
            
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features + 256, 512),  # Visual + Audio
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
        
        print(f"[OK] {model_type.upper()} model architecture ready!")
    
    def _setup_transfer_learning(self):
        """Setup transfer learning: freeze early layers, unfreeze last few blocks"""
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze last 2 blocks of features for fine-tuning
        total_blocks = len(self.backbone.features)
        blocks_to_unfreeze = 2
        
        for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True
        
        # Always unfreeze classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        
        print(f"🔓 Unfroze last {blocks_to_unfreeze} blocks + classifier for fine-tuning")
    
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
                combined_features = torch.cat([visual_features, torch.zeros_like(visual_features[:, :256])], dim=1)
            
            return self.backbone.classifier(combined_features)

class OptimizedDataset(Dataset):
    """OPTIMIZED Dataset for Kaggle T4x2 with full data utilization"""
    def __init__(self, data_dir, chunk_file, model_type='bg', transform=None):
        self.data_dir = Path(data_dir)
        self.chunk_file = chunk_file
        self.model_type = model_type
        self.transform = transform
        self.samples = []
        
        print(f"[DIR] Loading data from {chunk_file}...")
        self._load_chunk_data()
        print(f"[OK] Loaded {len(self.samples)} samples from {chunk_file}")
    
    def _load_chunk_data(self):
        """Load ALL data from current chunk - NO SUBSETS"""
        chunk_path = self.data_dir / self.chunk_file
        
        if not chunk_path.exists():
            print(f"⏳ Waiting for chunk {self.chunk_file}...")
            while not chunk_path.exists():
                time.sleep(30)
        
        print(f"📦 Extracting {self.chunk_file}...")
        
        with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            # Filter video files - USE ALL OF THEM
            video_files = [f for f in file_list if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            print(f"[VIDEO] Found {len(video_files)} video files")
            
            for video_file in tqdm(video_files, desc="Processing videos"):
                # Determine label from filename or directory structure
                label = 1 if any(keyword in video_file.lower() for keyword in ['fake', 'manipulated', 'deepfake', 'synthetic']) else 0
                
                self.samples.append({
                    'video_path': video_file,
                    'label': label,
                    'chunk_path': chunk_path
                })
        
        # Shuffle for better training
        np.random.shuffle(self.samples)
    
    def _extract_frames_optimized(self, video_path, chunk_path, num_frames=8):
        """OPTIMIZED frame extraction for T4x2"""
        try:
            with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
                with zip_ref.open(video_path) as video_file:
                    # Create temp file
                    temp_path = f"/tmp/{hash(video_path) % 10000}.mp4"
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    # Extract frames efficiently
                    cap = cv2.VideoCapture(temp_path)
                    frames = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    if total_frames > 0:
                        # Sample frames evenly
                        indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
                        
                        for idx in indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame = cv2.resize(frame, (224, 224))
                                frames.append(frame)
                    
                    cap.release()
                    os.remove(temp_path)
                    
                    # Ensure we have enough frames
                    while len(frames) < num_frames:
                        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
                    
                    return np.array(frames[:num_frames])
                    
        except Exception as e:
            print(f"[WARNING] Error extracting frames from {video_path}: {e}")
            return np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
    
    def _extract_audio_features_optimized(self, video_path, chunk_path):
        """OPTIMIZED audio feature extraction"""
        try:
            with zipfile.ZipFile(chunk_path, 'r') as zip_ref:
                with zip_ref.open(video_path) as video_file:
                    temp_path = f"/tmp/{hash(video_path) % 10000}_audio.mp4"
                    with open(temp_path, 'wb') as f:
                        f.write(video_file.read())
                    
                    # Extract audio with librosa
                    y, sr = librosa.load(temp_path, sr=16000, duration=5.0)
                    
                    # Extract MFCC features
                    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024, hop_length=512)
                    mfccs_mean = np.mean(mfccs, axis=1)
                    
                    # Pad or truncate to 128 features
                    if len(mfccs_mean) < 128:
                        mfccs_mean = np.pad(mfccs_mean, (0, 128 - len(mfccs_mean)))
                    else:
                        mfccs_mean = mfccs_mean[:128]
                    
                    os.remove(temp_path)
                    return mfccs_mean.astype(np.float32)
                    
        except Exception as e:
            return np.zeros(128, dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract frames
        frames = self._extract_frames_optimized(sample['video_path'], sample['chunk_path'])
        
        # Use middle frame
        frame = frames[len(frames)//2]
        
        if self.transform:
            frame = self.transform(frame)
        else:
            frame = torch.FloatTensor(frame).permute(2, 0, 1) / 255.0
        
        label = torch.LongTensor([sample['label']])
        
        if self.model_type == 'av':
            # Extract audio features for AV model
            audio_features = self._extract_audio_features_optimized(sample['video_path'], sample['chunk_path'])
            audio_features = torch.FloatTensor(audio_features)
            return frame, audio_features, label
        else:
            return frame, label

class OptimizedTrainer:
    """OPTIMIZED Trainer for Kaggle T4x2"""
    def __init__(self, model_type, data_dir, output_dir):
        self.model_type = model_type
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"[RUN] Initializing {model_type.upper()} trainer for Kaggle T4x2...")
        
        # Initialize model
        self.model = EfficientNetBackbone(num_classes=2, model_type=model_type)
        self.model.to(DEVICE)
        
        # OPTIMIZED training parameters
        self.optimizer = self._setup_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_chunk = 0
        self.total_chunks = 10
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 2 * 3600  # 2 hours for faster saves
        
        # Metrics tracking
        self.training_history = []
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
        
        print(f"[OK] {model_type.upper()} trainer ready!")
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates"""
        pretrained_params = []
        new_params = []
        
        # Separate parameters
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                pretrained_params.append(param)
        
        # New parameters
        if hasattr(self.model, 'audio_encoder'):
            for param in self.model.audio_encoder.parameters():
                new_params.append(param)
        
        # Optimized learning rates for T4x2
        optimizer = optim.AdamW([
            {'params': pretrained_params, 'lr': 5e-6, 'weight_decay': 1e-5},
            {'params': new_params, 'lr': 1e-4, 'weight_decay': 1e-4}
        ])
        
        return optimizer
    
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""
        checkpoint_pattern = f"{self.model_type}_model_chunk_*.pt"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"📥 Loading checkpoint: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_chunk = checkpoint.get('current_chunk', 0)
            self.training_history = checkpoint.get('training_history', [])
            
            print(f"[OK] Resumed from chunk {self.current_chunk}")
    
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
        
        print(f"[SAVE] Checkpoint saved: {checkpoint_path}")
        
        # Clean up old checkpoints (keep only latest 2)
        checkpoints = sorted(self.output_dir.glob(f"{self.model_type}_model_chunk_*.pt"))
        if len(checkpoints) > 2:
            for old_checkpoint in checkpoints[:-2]:
                old_checkpoint.unlink()
    
    def train_on_chunk(self, chunk_idx):
        """Train model on a specific chunk with FULL PROGRESS TRACKING"""
        chunk_file = f"{chunk_idx:02d}.zip"
        
        print(f"\n[TARGET] Starting {self.model_type.upper()} training on chunk {chunk_idx}: {chunk_file}")
        print("="*60)
        
        # Create dataset and dataloader - OPTIMIZED FOR T4x2
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = OptimizedDataset(self.data_dir, chunk_file, self.model_type, transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR,
            persistent_workers=True
        )
        
        if len(dataset) == 0:
            print(f"[WARNING] No data found in chunk {chunk_file}")
            return
        
        print(f"[STATS] Dataset size: {len(dataset)} samples")
        print(f"[LOOP] Batches per epoch: {len(dataloader)}")
        
        # Training loop with FULL PROGRESS TRACKING
        self.model.train()
        
        for epoch in range(3):  # 3 epochs per chunk
            print(f"\n[PROGRESS] Epoch {epoch+1}/3 for chunk {chunk_idx}")
            
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Progress bar for batches
            pbar = tqdm(dataloader, desc=f"Training {self.model_type.upper()}")
            
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    if self.model_type == 'av' and len(batch_data) == 3:
                        frames, audio_features, labels = batch_data
                        frames = frames.to(DEVICE, non_blocking=True)
                        audio_features = audio_features.to(DEVICE, non_blocking=True)
                        labels = labels.squeeze().to(DEVICE, non_blocking=True)
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs = self.model(frames, audio_features)
                    else:
                        frames, labels = batch_data
                        frames = frames.to(DEVICE, non_blocking=True)
                        labels = labels.squeeze().to(DEVICE, non_blocking=True)
                        
                        # Forward pass
                        self.optimizer.zero_grad()
                        outputs = self.model(frames)
                    
                    loss = self.criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    # Track metrics
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_predictions += labels.size(0)
                    
                    # Update progress bar
                    current_acc = correct_predictions / total_predictions * 100
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                    })
                    
                    # Memory cleanup every 50 batches
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Checkpoint save check
                    if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                        metrics = {'accuracy': current_acc/100, 'loss': epoch_loss/(batch_idx+1)}
                        self._save_checkpoint(chunk_idx, metrics)
                        self.last_checkpoint_time = time.time()
                
                except Exception as e:
                    print(f"[WARNING] Batch error: {e}")
                    continue
            
            # Epoch summary
            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct_predictions / total_predictions
            
            print(f"[OK] Epoch {epoch+1} completed:")
            print(f"   [LOSS] Average Loss: {avg_loss:.4f}")
            print(f"   [TARGET] Accuracy: {accuracy*100:.2f}%")
            print(f"   [GPU] GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
            
            self.scheduler.step()
        
        # Final metrics for this chunk
        final_metrics = {
            'accuracy': accuracy,
            'avg_loss': avg_loss,
            'chunk_idx': chunk_idx,
            'samples_processed': len(dataset)
        }
        
        self.training_history.append(final_metrics)
        self._save_checkpoint(chunk_idx, final_metrics)
        
        print(f"\n[DONE] Chunk {chunk_idx} training completed!")
        print(f"[STATS] Final metrics: Accuracy={accuracy*100:.2f}%, Loss={avg_loss:.4f}")
        print("="*60)
        
        # Cleanup
        del dataset, dataloader
        torch.cuda.empty_cache()
        gc.collect()
    
    def train_all_chunks(self):
        """Train on all chunks with FULL PROGRESS TRACKING"""
        print(f"\n[RUN] Starting {self.model_type.upper()} model training on ALL 10 chunks")
        print(f"[TARGET] Target: Process 100GB of data with full utilization")
        print("="*80)
        
        for chunk_idx in range(self.current_chunk, self.total_chunks):
            try:
                start_time = time.time()
                self.train_on_chunk(chunk_idx)
                end_time = time.time()
                
                self.current_chunk = chunk_idx + 1
                
                # Progress summary
                progress = (chunk_idx + 1) / self.total_chunks * 100
                time_taken = end_time - start_time
                
                print(f"\n[PROGRESS] OVERALL PROGRESS: {progress:.1f}% ({chunk_idx+1}/10 chunks)")
                print(f"[TIME]  Chunk {chunk_idx} took: {time_taken/3600:.2f} hours")
                print(f"[SAVE] Memory usage: {psutil.virtual_memory().percent:.1f}%")
                
                if chunk_idx < self.total_chunks - 1:
                    print(f"⏭️  Moving to chunk {chunk_idx + 1}...")
                
            except Exception as e:
                print(f"[ERROR] Error training on chunk {chunk_idx}: {e}")
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
        
        print(f"\n[DONE] {self.model_type.upper()} MODEL TRAINING COMPLETED!")
        print(f"[SAVE] Final model saved: {final_model_path}")
        print(f"[STATS] Total chunks processed: {self.current_chunk}/10")
        print("="*80)

def main():
    """Main training function - OPTIMIZED FOR KAGGLE"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BG and AV models - OPTIMIZED')
    parser.add_argument('--model', choices=['bg', 'av'], required=True, help='Model type to train')
    parser.add_argument('--data_dir', default='/kaggle/input/dfdc-10-deepfake-detection-challenge-first-10', help='Data directory')
    parser.add_argument('--output_dir', default='/kaggle/working', help='Output directory')
    
    args = parser.parse_args()
    
    print("[RUN] KAGGLE T4x2 OPTIMIZED TRAINING STARTING...")
    print(f"[TARGET] Model: {args.model.upper()}")
    print(f"[DIR] Data: {args.data_dir}")
    print(f"[SAVE] Output: {args.output_dir}")
    print(f"[GPU] Device: {DEVICE}")
    print(f"[STATS] Batch size: {BATCH_SIZE}")
    print("="*60)
    
    # Initialize trainer
    trainer = OptimizedTrainer(args.model, args.data_dir, args.output_dir)
    
    # Start training
    trainer.train_all_chunks()
    
    print("[DONE] TRAINING COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()