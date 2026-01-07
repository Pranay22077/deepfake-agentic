"""
RESUME STAGE 3: WILD DEEPFAKE TRAINING
Continue training from Stage 2 checkpoint on Wild Deepfake dataset

Strategy:
- Natural training (no aggressive balancing)
- Let Stage 4 DFDC compensate bias with 100GB+ fake data
- Focus on learning from Wild Deepfake's equal real/fake distribution
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models import efficientnet_b4

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 3e-5
EPOCHS = 5
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Dataset path (update for your environment)
WILD_DEEPFAKE_PATH = "/kaggle/input/wild-deepfake"
OUTPUT_DIR = "/kaggle/working"

# Mixed precision
USE_MIXED_PRECISION = True
if USE_MIXED_PRECISION and torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class EnhancedLowLightModule(nn.Module):
    """Enhanced low-light analysis module"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale luminance analysis
        self.luminance_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ) for k in [3, 5, 7]
        ])
        
        # Noise pattern detector
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow/highlight detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Attention for feature fusion
        self.attention = nn.Sequential(
            nn.Conv2d(48 + 12 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 68, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale luminance
        lum_features = []
        for branch in self.luminance_branch:
            lum_feat = branch(x)
            lum_feat = F.adaptive_avg_pool2d(lum_feat, (7, 7))
            lum_features.append(lum_feat)
        
        lum_combined = torch.cat(lum_features, dim=1)  # 48 channels
        
        # Noise analysis
        noise_features = self.noise_detector(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))  # 12 channels
        
        # Shadow analysis
        shadow_features = self.shadow_detector(x)
        shadow_features = F.adaptive_avg_pool2d(shadow_features, (7, 7))  # 8 channels
        
        # Combine and apply attention
        combined = torch.cat([lum_combined, noise_features, shadow_features], dim=1)  # 68 channels
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features

class InterceptorLLModel(nn.Module):
    """Complete Interceptor LL-Model"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Backbone
        self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Specialist module
        self.specialist_module = EnhancedLowLightModule()
        specialist_features = 68 * 7 * 7  # 3332
        
        # Feature projection for attention
        total_features = backbone_features + specialist_features
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        # Multi-head attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Final classification
        output = self.classifier(attended_features)
        return output

# ============================================================================
# DATASET
# ============================================================================
class WildDeepfakeDataset(Dataset):
    """Wild Deepfake dataset loader"""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['path']
        label = sample['label']
        
        # Extract frame from video
        frame = self._extract_frame(video_path)
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label
    
    def _extract_frame(self, video_path):
        """Extract a random frame from video"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames > 0:
            # Select random frame
            frame_idx = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            # Return black frame if failed
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        
        return frame

def get_transforms():
    """Get training transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    ])

def load_wild_deepfake_samples():
    """Load Wild Deepfake dataset samples"""
    
    dataset_path = Path(WILD_DEEPFAKE_PATH)
    samples = []
    
    print(f"📂 Loading Wild Deepfake dataset from: {dataset_path}")
    
    # Load from train folder structure
    train_path = dataset_path / "train"
    
    if not train_path.exists():
        print(f"❌ Train folder not found: {train_path}")
        return []
    
    # Load real videos
    real_path = train_path / "real"
    if real_path.exists():
        real_videos = list(real_path.glob("*.mp4"))
        for video_path in real_videos:
            samples.append({'path': video_path, 'label': 0})
        print(f"✅ Found {len(real_videos)} real videos")
    
    # Load fake videos
    fake_path = train_path / "fake"
    if fake_path.exists():
        fake_videos = list(fake_path.glob("*.mp4"))
        for video_path in fake_videos:
            samples.append({'path': video_path, 'label': 1})
        print(f"✅ Found {len(fake_videos)} fake videos")
    
    return samples

# ============================================================================
# LOSS FUNCTION
# ============================================================================
def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Standard focal loss"""
    ce_loss = F.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()

# ============================================================================
# TRAINING
# ============================================================================
def train_stage3():
    """Train Stage 3 on Wild Deepfake dataset"""
    
    print("🚀 INTERCEPTOR STAGE 3: WILD DEEPFAKE TRAINING")
    print("="*80)
    print("📊 Current Model Status:")
    print("   ✅ Real Detection: 92% (excellent)")
    print("   ⚠️  Fake Detection: 28% (needs improvement)")
    print("   📊 Bias: 64% (real > fake)")
    print()
    print("🎯 Stage 3 Strategy:")
    print("   📚 Train on Wild Deepfake (11.25 GB, equal real/fake)")
    print("   🔄 Natural training - no aggressive bias correction")
    print("   💡 Let Stage 4 DFDC (100GB+ mostly fake) compensate bias")
    print("   🎯 Goal: Improve fake detection naturally")
    print("="*80)
    
    # Step 1: Get checkpoint path
    print("\n📋 STEP 1: SELECT STAGE 2 CHECKPOINT")
    print("-" * 50)
    
    checkpoint_path = input("Enter path to Stage 2 checkpoint (.pt file): ").strip()
    
    if not checkpoint_path:
        print("❌ No checkpoint path provided!")
        return
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"✅ Using checkpoint: {checkpoint_path}")
    print(f"📦 Size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")
    
    # Step 2: Load model
    print(f"\n📋 STEP 2: LOAD MODEL")
    print("-" * 50)
    
    model = InterceptorLLModel(num_classes=2)
    model.to(DEVICE)
    
    # Load checkpoint with weights_only=False for compatibility
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model state dict")
            if 'epoch' in checkpoint:
                print(f"📊 Previous epoch: {checkpoint['epoch']}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded model weights")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Step 3: Load dataset
    print(f"\n📋 STEP 3: LOAD WILD DEEPFAKE DATASET")
    print("-" * 50)
    
    samples = load_wild_deepfake_samples()
    
    if len(samples) == 0:
        print("❌ No samples found!")
        return
    
    real_count = sum(1 for s in samples if s['label'] == 0)
    fake_count = sum(1 for s in samples if s['label'] == 1)
    
    print(f"📊 Total samples: {len(samples)}")
    print(f"📊 Real: {real_count} ({real_count/len(samples)*100:.1f}%)")
    print(f"📊 Fake: {fake_count} ({fake_count/len(samples)*100:.1f}%)")
    
    # Create dataset and dataloader
    dataset = WildDeepfakeDataset(samples, transform=get_transforms())
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True
    )
    
    print(f"📊 Batches per epoch: {len(dataloader)}")
    
    # Step 4: Setup training
    print(f"\n📋 STEP 4: SETUP TRAINING")
    print("-" * 50)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)
    
    print(f"✅ Optimizer: AdamW")
    print(f"✅ Learning Rate: {LEARNING_RATE}")
    print(f"✅ Loss: Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    print(f"✅ Epochs: {EPOCHS}")
    print(f"✅ Device: {DEVICE}")
    print(f"✅ Mixed Precision: {USE_MIXED_PRECISION and torch.cuda.is_available()}")
    
    # Step 5: Training loop
    print(f"\n📋 STEP 5: START TRAINING")
    print("-" * 50)
    
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        real_correct = 0
        real_total = 0
        fake_correct = 0
        fake_total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            if USE_MIXED_PRECISION and torch.cuda.is_available():
                with autocast():
                    outputs = model(inputs)
                    loss = focal_loss(outputs, targets, FOCAL_ALPHA, FOCAL_GAMMA)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = focal_loss(outputs, targets, FOCAL_ALPHA, FOCAL_GAMMA)
                loss.backward()
                optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += targets.size(0)
            epoch_correct += (predicted == targets).sum().item()
            epoch_loss += loss.item()
            
            # Track real/fake accuracy
            for i in range(targets.size(0)):
                if targets[i] == 0:  # Real
                    real_total += 1
                    if predicted[i] == 0:
                        real_correct += 1
                else:  # Fake
                    fake_total += 1
                    if predicted[i] == 1:
                        fake_correct += 1
            
            # Update progress bar
            current_acc = 100 * epoch_correct / epoch_total
            current_real_acc = 100 * real_correct / real_total if real_total > 0 else 0
            current_fake_acc = 100 * fake_correct / fake_total if fake_total > 0 else 0
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.1f}%',
                'Real': f'{current_real_acc:.1f}%',
                'Fake': f'{current_fake_acc:.1f}%'
            })
        
        # Calculate epoch metrics
        epoch_accuracy = epoch_correct / epoch_total
        real_accuracy = real_correct / real_total if real_total > 0 else 0
        fake_accuracy = fake_correct / fake_total if fake_total > 0 else 0
        bias_difference = abs(real_accuracy - fake_accuracy)
        avg_loss = epoch_loss / len(dataloader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Overall Accuracy: {epoch_accuracy*100:.2f}%")
        print(f"   Real Detection: {real_accuracy*100:.2f}%")
        print(f"   Fake Detection: {fake_accuracy*100:.2f}%")
        print(f"   Bias Difference: {bias_difference*100:.1f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            
            save_path = Path(OUTPUT_DIR) / f"stage3_wilddeepfake_epoch{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'accuracy': epoch_accuracy,
                'real_accuracy': real_accuracy,
                'fake_accuracy': fake_accuracy,
                'bias_difference': bias_difference,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            
            print(f"💾 Best model saved: {save_path.name}")
    
    # Final summary
    print(f"\n🎯 STAGE 3 TRAINING COMPLETED!")
    print("="*80)
    print(f"📊 Final Results:")
    print(f"   Overall Accuracy: {epoch_accuracy*100:.2f}%")
    print(f"   Real Detection: {real_accuracy*100:.2f}%")
    print(f"   Fake Detection: {fake_accuracy*100:.2f}%")
    print(f"   Bias Difference: {bias_difference*100:.1f}%")
    print()
    print("💡 Analysis:")
    if fake_accuracy > 0.4:
        print(f"   ✅ Fake detection improved significantly!")
    elif fake_accuracy > 0.3:
        print(f"   ✅ Fake detection showing improvement")
    else:
        print(f"   ⚠️ Fake detection still needs work - Stage 4 will help")
    
    if bias_difference < 0.3:
        print(f"   ✅ Bias is reasonable")
    else:
        print(f"   💡 Bias still present - Stage 4 DFDC will compensate")
    
    print()
    print("🚀 Next Steps:")
    print("   1. Test this model on your test videos")
    print("   2. Proceed to Stage 4 DFDC training (100GB+ mostly fake)")
    print("   3. Stage 4 will naturally improve fake detection")
    print("   4. Final model will be balanced after Stage 4")

if __name__ == "__main__":
    train_stage3()
