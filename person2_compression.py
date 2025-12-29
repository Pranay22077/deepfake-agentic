#!/usr/bin/env python3
"""
============================================
PERSON 2: CM-MODEL & RR-MODEL TRAINING
============================================
Compression Specialist & Re-Recording Specialist

Run this in Kaggle with:
- GPU: T4 x2 or P100 
- Dataset: deepfake-detection-challenge
- Runtime: 9+ hours for complete training

Output: cm_model_student.pt, rr_model_student.pt

This script creates augmented datasets and trains two specialist models:
1. CM-Model: Handles compressed videos (WhatsApp, Instagram, etc.)
2. RR-Model: Handles re-recorded/screen-captured videos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import numpy as np
from tqdm import tqdm
import json
import subprocess
import time
import random

# GPU monitoring function
def monitor_gpu_usage():
    """Monitor and print GPU usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        print(f"🔍 GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        return allocated > 0.1  # Return True if GPU is actually being used
    return False

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Force GPU usage and verify
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    # Clear GPU cache
    torch.cuda.empty_cache()
    # Test GPU with a simple operation
    test_tensor = torch.randn(100, 100).cuda()
    test_result = torch.mm(test_tensor, test_tensor)
    print(f"🚀 Person 2 Training Pipeline")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✅ GPU test successful: {test_result.shape}")
    del test_tensor, test_result
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print(f"⚠️  CUDA not available, using CPU")
    print(f"This will be very slow for training!")

# ============================================
# PART 1: CREATE COMPRESSION DATASET
# ============================================

def create_compressed_video(input_path, output_path, bitrate_kbps):
    """
    Re-encode video at specified bitrate using FFmpeg
    Simulates compression from social media platforms
    """
    try:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', 
            '-b:v', f'{bitrate_kbps}k',
            '-preset', 'fast',  # Faster encoding
            '-crf', '28',       # Add quality degradation
            '-maxrate', f'{int(bitrate_kbps * 1.2)}k',
            '-bufsize', f'{int(bitrate_kbps * 2)}k',
            '-an',              # Remove audio for speed
            '-t', '10',         # Limit to 10 seconds
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"FFmpeg error for {input_path}: {e}")
        return False

def create_compression_dataset(video_dir, output_dir, bitrates=[200, 400, 600], max_videos=150):
    """Create compressed versions of videos with realistic social media compression"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    # Limit videos for faster processing
    video_files = video_files[:max_videos]
    
    print(f"📹 Creating compression dataset from {len(video_files)} videos")
    print(f"Bitrates: {bitrates} kbps")
    
    successful = 0
    total_expected = len(video_files) * len(bitrates)
    
    for video_file in tqdm(video_files, desc="Compressing videos"):
        input_path = os.path.join(video_dir, video_file)
        
        for bitrate in bitrates:
            output_name = f"comp_{bitrate}_{video_file}"
            output_path = os.path.join(output_dir, output_name)
            
            if create_compressed_video(input_path, output_path, bitrate):
                successful += 1
    
    print(f"✅ Compression dataset: {successful}/{total_expected} videos created")
    return successful > 0

# ============================================
# PART 2: CREATE RE-RECORDING DATASET
# ============================================

def apply_rerecording_artifacts(frame):
    """
    Apply realistic re-recording artifacts:
    1. Moiré patterns from screen interference
    2. Pixel grid from LCD displays
    3. Slight blur from camera recording screen
    4. Color shift from display-camera mismatch
    """
    h, w = frame.shape[:2]
    
    # 1. Moiré pattern (interference between camera and display)
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Multiple frequency moiré patterns
    pattern1 = np.sin(0.08 * X) * np.sin(0.08 * Y) * 8
    pattern2 = np.sin(0.12 * X) * np.sin(0.06 * Y) * 5
    combined_pattern = pattern1 + pattern2
    
    # Apply to all channels
    pattern_3d = np.stack([combined_pattern]*3, axis=-1).astype(np.int16)
    frame = frame.astype(np.int16) + pattern_3d
    
    # 2. LCD pixel grid simulation
    grid_size = random.choice([2, 3, 4])  # Variable grid sizes
    for i in range(0, h, grid_size):
        if i < h:
            frame[i, :] = (frame[i, :] * 0.92).astype(np.int16)
    for j in range(0, w, grid_size):
        if j < w:
            frame[:, j] = (frame[:, j] * 0.92).astype(np.int16)
    
    # 3. Slight blur (camera focus on screen)
    frame = cv2.GaussianBlur(frame.astype(np.uint8), (3, 3), 0.5)
    
    # 4. Color shift (display-camera color space mismatch)
    frame = frame.astype(np.float32)
    frame[:, :, 0] *= 1.05  # Slight red boost
    frame[:, :, 2] *= 0.95  # Slight blue reduction
    
    return np.clip(frame, 0, 255).astype(np.uint8)

def create_rerecording_dataset(video_dir, output_dir, max_videos=150):
    """Create re-recorded versions of videos with realistic screen recording artifacts"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = video_files[:max_videos]
    
    print(f"📱 Creating re-recording dataset from {len(video_files)} videos")
    
    successful = 0
    
    for video_file in tqdm(video_files, desc="Creating re-recorded videos"):
        input_path = os.path.join(video_dir, video_file)
        output_path = os.path.join(output_dir, f"rerec_{video_file}")
        
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                continue
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            max_frames = int(fps * 10)  # Limit to 10 seconds
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply re-recording artifacts
                frame = apply_rerecording_artifacts(frame)
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            successful += 1
            
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            continue
    
    print(f"✅ Re-recording dataset: {successful}/{len(video_files)} videos created")
    return successful > 0


# ============================================
# MODEL ARCHITECTURE (Same as baseline)
# ============================================

class SpecialistModel(nn.Module):
    """Specialist Model - ResNet18 based"""
    def __init__(self, num_classes=2):
        super(SpecialistModel, self).__init__()
        
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================
# DATASET
# ============================================

class AugmentedVideoDataset(Dataset):
    """Enhanced dataset for augmented videos with better error handling"""
    def __init__(self, video_dir, original_metadata_path, transform=None, max_frames_per_video=8):
        self.video_dir = video_dir
        self.transform = transform
        self.max_frames_per_video = max_frames_per_video
        
        # Load original metadata for labels
        with open(original_metadata_path, 'r') as f:
            self.original_metadata = json.load(f)
        
        # Filter valid video files
        all_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.video_files = []
        
        # Validate files and extract labels
        for video_file in all_files:
            original_name = self.extract_original_name(video_file)
            if original_name in self.original_metadata:
                self.video_files.append(video_file)
        
        print(f"📊 Dataset loaded: {len(self.video_files)} valid videos")
        
        # Calculate class distribution
        fake_count = sum(1 for vf in self.video_files 
                        if self.original_metadata.get(self.extract_original_name(vf), {}).get('label') == 'FAKE')
        real_count = len(self.video_files) - fake_count
        print(f"   Real: {real_count}, Fake: {fake_count}")
    
    def extract_original_name(self, augmented_name):
        """Extract original filename from augmented filename"""
        # Handle: comp_300_original.mp4 -> original.mp4
        # Handle: rerec_original.mp4 -> original.mp4
        parts = augmented_name.split('_')
        if parts[0] == 'comp' and len(parts) >= 3:
            return '_'.join(parts[2:])
        elif parts[0] == 'rerec' and len(parts) >= 2:
            return '_'.join(parts[1:])
        else:
            return augmented_name
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        
        # Get label
        original_name = self.extract_original_name(video_name)
        label = 1 if self.original_metadata.get(original_name, {}).get('label') == 'FAKE' else 0
        
        # Extract multiple frames
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            # Return dummy data if extraction fails
            dummy_frame = torch.zeros(3, 224, 224)
            return dummy_frame, label
        
        # Return random frame from extracted frames
        frame = random.choice(frames)
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label
    
    def extract_frames(self, video_path):
        """Extract multiple frames from video with better error handling"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, 
                                      min(self.max_frames_per_video, total_frames), 
                                      dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize to 224x224
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    
                    # Convert to PIL Image for transforms
                    from PIL import Image
                    pil_frame = Image.fromarray(frame_resized)
                    frames.append(pil_frame)
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
        
        return frames

# ============================================
# TRAINING FUNCTION
# ============================================

def train_specialist(model_name, video_dir, metadata_path, output_name):
    """Enhanced training function with better monitoring and validation"""
    
    print(f"\n{'='*60}")
    print(f"🎯 Training {model_name}")
    print(f"{'='*60}")
    
    # Hyperparameters
    BATCH_SIZE = 16  # Reduced for stability
    EPOCHS = 12      # Reduced for faster training
    LR = 3e-5        # Lower learning rate
    WEIGHT_DECAY = 1e-4
    
    # Enhanced transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = AugmentedVideoDataset(video_dir, metadata_path, transform=None)
    
    if len(full_dataset) == 0:
        print(f"❌ No valid videos found in {video_dir}")
        return 0.0
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )
    
    # Create separate datasets with different transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices.indices)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Data loaders with proper GPU settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for GPU training in Kaggle
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for GPU training in Kaggle
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=False
    )
    
    print(f"📊 Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Model setup - ensure it's on GPU
    model = SpecialistModel(num_classes=2)
    model = model.to(device)
    
    # Verify model is on GPU
    if device.type == 'cuda':
        print(f"✅ Model moved to GPU: {next(model.parameters()).device}")
        # Warm up GPU
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            _ = model(dummy_input)
        del dummy_input
        torch.cuda.empty_cache()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training metrics
    best_acc = 0
    train_losses = []
    val_accuracies = []
    
    print(f"🚀 Starting training...")
    
    # Monitor GPU before training starts
    gpu_in_use = monitor_gpu_usage()
    if device.type == 'cuda' and not gpu_in_use:
        print("⚠️  WARNING: GPU detected but not showing memory usage!")
        print("   This might indicate the model/data isn't actually on GPU")
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (frames, labels) in enumerate(train_pbar):
            # Ensure data is on GPU with proper dtype
            frames = frames.to(device, dtype=torch.float32, non_blocking=False)
            labels = labels.to(device, dtype=torch.long, non_blocking=False)
            
            # Verify GPU usage
            if batch_idx == 0 and device.type == 'cuda':
                print(f"🔧 Batch 0 - Frames on: {frames.device}, Labels on: {labels.device}")
                print(f"🔧 GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{current_acc:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'gpu_mem': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if device.type == 'cuda' else 'N/A'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in tqdm(val_loader, desc='Validation', leave=False):
                frames = frames.to(device, dtype=torch.float32, non_blocking=False)
                labels = labels.to(device, dtype=torch.long, non_blocking=False)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        epoch_time = time.time() - epoch_start
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_accuracies.append(val_acc)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{EPOCHS} ({epoch_time:.1f}s):')
        print(f'  Train: Loss={avg_train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Val:   Loss={avg_val_loss:.4f}, Acc={val_acc:.2f}%')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            
            # Save comprehensive checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies,
                'config': {
                    'architecture': 'ResNet18',
                    'num_classes': 2,
                    'model_type': model_name,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LR,
                    'weight_decay': WEIGHT_DECAY,
                    'epochs_trained': epoch + 1,
                    'dataset_size': len(full_dataset),
                    'train_size': len(train_dataset),
                    'val_size': len(val_dataset)
                }
            }
            
            torch.save(checkpoint, output_name)
            print(f'  ✅ Saved best model: {best_acc:.2f}%')
        
        print()
    
    print(f'🎉 {model_name} training complete!')
    print(f'   Best validation accuracy: {best_acc:.2f}%')
    print(f'   Model saved as: {output_name}')
    
    return best_acc

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == '__main__':
    print("🎯 E-RAKSHA PERSON 2: SPECIALIST MODELS TRAINING")
    print("=" * 70)
    print("Training CM-Model (Compression) and RR-Model (Re-recording)")
    print("=" * 70)
    
    # Configuration
    ORIGINAL_VIDEO_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos'
    METADATA_PATH = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/metadata.json'
    COMPRESSED_DIR = '/kaggle/working/compressed_videos'
    RERECORDED_DIR = '/kaggle/working/rerecorded_videos'
    
    # Verify input data
    if not os.path.exists(ORIGINAL_VIDEO_DIR):
        print(f"❌ Video directory not found: {ORIGINAL_VIDEO_DIR}")
        print("Please ensure the DFDC dataset is properly loaded in Kaggle")
        exit(1)
    
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Metadata file not found: {METADATA_PATH}")
        exit(1)
    
    # Count original videos
    original_videos = [f for f in os.listdir(ORIGINAL_VIDEO_DIR) if f.endswith('.mp4')]
    print(f"📹 Found {len(original_videos)} original videos")
    
    # Step 1: Create augmented datasets
    print(f"\n{'='*50}")
    print("STEP 1: CREATING AUGMENTED DATASETS")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    print("Creating compression dataset...")
    compression_success = create_compression_dataset(
        ORIGINAL_VIDEO_DIR, 
        COMPRESSED_DIR,
        bitrates=[200, 400, 600],  # Low, medium, high compression
        max_videos=200  # Limit for reasonable training time
    )
    
    print("Creating re-recording dataset...")
    rerecording_success = create_rerecording_dataset(
        ORIGINAL_VIDEO_DIR, 
        RERECORDED_DIR,
        max_videos=200
    )
    
    dataset_time = time.time() - start_time
    print(f"\n⏱️ Dataset creation completed in {dataset_time/60:.1f} minutes")
    
    if not compression_success:
        print("❌ Compression dataset creation failed")
        exit(1)
    
    if not rerecording_success:
        print("❌ Re-recording dataset creation failed")
        exit(1)
    
    # Verify created datasets
    comp_videos = len([f for f in os.listdir(COMPRESSED_DIR) if f.endswith('.mp4')])
    rerec_videos = len([f for f in os.listdir(RERECORDED_DIR) if f.endswith('.mp4')])
    
    print(f"✅ Datasets created successfully:")
    print(f"   Compressed videos: {comp_videos}")
    print(f"   Re-recorded videos: {rerec_videos}")
    
    # Step 2: Train CM-Model
    print(f"\n{'='*50}")
    print("STEP 2: TRAINING CM-MODEL (COMPRESSION SPECIALIST)")
    print(f"{'='*50}")
    
    cm_start_time = time.time()
    cm_acc = train_specialist(
        'CM-Model (Compression Specialist)',
        COMPRESSED_DIR,
        METADATA_PATH,
        'cm_model_student.pt'
    )
    cm_time = time.time() - cm_start_time
    
    # Step 3: Train RR-Model
    print(f"\n{'='*50}")
    print("STEP 3: TRAINING RR-MODEL (RE-RECORDING SPECIALIST)")
    print(f"{'='*50}")
    
    rr_start_time = time.time()
    rr_acc = train_specialist(
        'RR-Model (Re-recording Specialist)',
        RERECORDED_DIR,
        METADATA_PATH,
        'rr_model_student.pt'
    )
    rr_time = time.time() - rr_start_time
    
    # Final summary
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("🎉 PERSON 2 TRAINING PIPELINE COMPLETE!")
    print(f"{'='*70}")
    
    print(f"📊 RESULTS SUMMARY:")
    print(f"   CM-Model accuracy: {cm_acc:.2f}% (trained in {cm_time/60:.1f} min)")
    print(f"   RR-Model accuracy: {rr_acc:.2f}% (trained in {rr_time/60:.1f} min)")
    print(f"   Total pipeline time: {total_time/60:.1f} minutes")
    
    print(f"\n📁 GENERATED FILES:")
    if os.path.exists('cm_model_student.pt'):
        cm_size = os.path.getsize('cm_model_student.pt') / (1024*1024)
        print(f"   ✅ cm_model_student.pt ({cm_size:.1f} MB)")
    else:
        print(f"   ❌ cm_model_student.pt (not found)")
    
    if os.path.exists('rr_model_student.pt'):
        rr_size = os.path.getsize('rr_model_student.pt') / (1024*1024)
        print(f"   ✅ rr_model_student.pt ({rr_size:.1f} MB)")
    else:
        print(f"   ❌ rr_model_student.pt (not found)")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Download both model files from Kaggle")
    print(f"   2. Place them in the main project's models/ directory")
    print(f"   3. Commit and push to your branch: person2/compression-models")
    print(f"   4. Create pull request for Person 1 (Pranay) to integrate")
    
    print(f"\n🎯 INTEGRATION READY!")
    print(f"   Person 2's specialist models are ready for the multi-agent system")
    
    # Save training report
    report = {
        'person': 'Person 2',
        'models_trained': ['CM-Model', 'RR-Model'],
        'cm_accuracy': cm_acc,
        'rr_accuracy': rr_acc,
        'training_time_minutes': total_time / 60,
        'dataset_sizes': {
            'compressed_videos': comp_videos,
            'rerecorded_videos': rerec_videos
        },
        'timestamp': time.time(),
        'status': 'complete'
    }
    
    with open('person2_training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Training report saved: person2_training_report.json")