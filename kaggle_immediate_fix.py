#!/usr/bin/env python3
"""
IMMEDIATE FIX FOR KAGGLE NOTEBOOK
Copy this entire code into a new cell and run it
"""

# ============================================================================
# IMMEDIATE FIX - COPY THIS ENTIRE CELL TO KAGGLE
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

print("🔧 Applying immediate fix...")

# ============================================================================
# FIXED DATASET CLASS
# ============================================================================

class MultimodalDataset(Dataset):
    """Fixed dataset class - no heavy_augment parameter"""
    def __init__(self, data_dir, split='train', num_frames=8, audio_duration=3.0, 
                 sample_rate=16000, augment=False):
        self.data_dir = data_dir
        self.split = split
        self.num_frames = num_frames
        self.audio_duration = audio_duration
        self.sample_rate = sample_rate
        self.audio_samples = int(audio_duration * sample_rate)
        
        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # Load samples
        self.samples = self._load_samples()
        print(f"📊 {split}: {len(self.samples)} samples")
    
    def _load_samples(self):
        """Load samples from any directory structure"""
        samples = []
        
        # Find all images
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Simple labeling heuristic
                    label = 1 if 'fake' in root.lower() else 0
                    samples.append((os.path.join(root, file), label))
        
        # Split 80/20
        split_idx = int(0.8 * len(samples))
        if self.split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load and transform image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Create video (repeat frame)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
            
            # Create dummy audio
            audio = torch.randn(self.audio_samples) * 0.1
            
            return video, audio, label
            
        except:
            # Return dummy data on error
            video = torch.zeros(self.num_frames, 3, 224, 224)
            audio = torch.zeros(self.audio_samples)
            return video, audio, label

# ============================================================================
# SIMPLE TEACHER MODEL
# ============================================================================

class SimpleTeacher(nn.Module):
    """Simplified teacher model"""
    def __init__(self, num_classes=2, visual_frames=8):
        super().__init__()
        
        # Visual backbone
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        backbone_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Simple temporal processing
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Audio processing
        self.audio_net = nn.Sequential(
            nn.Linear(48000, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(backbone_features + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, video_frames, audio_waveform, return_features=False):
        B, T, C, H, W = video_frames.shape
        
        # Process video
        frames = video_frames.view(B*T, C, H, W)
        visual_feat = self.backbone(frames)
        visual_feat = visual_feat.view(B, T, -1).mean(dim=1)  # Average over time
        
        # Process audio
        audio_feat = self.audio_net(audio_waveform)
        
        # Combine and classify
        combined = torch.cat([visual_feat, audio_feat], dim=1)
        output = self.classifier(combined)
        
        if return_features:
            return output, {'visual_feat': visual_feat, 'audio_feat': audio_feat}
        return output

def create_teacher_model(num_classes=2, visual_frames=8):
    return SimpleTeacher(num_classes, visual_frames)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_teacher_model():
    """Fixed training function"""
    print("🚀 Starting teacher model training...")
    
    # Config
    config = {
        'epochs': 10,
        'batch_size': 8,
        'lr': 1e-4,
        'num_frames': 8,
        'audio_duration': 3.0,
        'sample_rate': 16000
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Find data
    data_dirs = [d for d in os.listdir('/kaggle/input/') if os.path.isdir(f'/kaggle/input/{d}')]
    print(f"📁 Available datasets: {data_dirs}")
    
    # Use first available dataset
    if not data_dirs:
        print("❌ No datasets found!")
        return None, 0
    
    data_dir = f'/kaggle/input/{data_dirs[0]}'
    print(f"📂 Using: {data_dir}")
    
    # Create datasets (FIXED - no heavy_augment parameter)
    train_dataset = MultimodalDataset(
        data_dir=data_dir,
        split='train',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        sample_rate=config['sample_rate'],
        augment=True  # Only standard augment parameter
    )
    
    val_dataset = MultimodalDataset(
        data_dir=data_dir,
        split='val',
        num_frames=config['num_frames'],
        audio_duration=config['audio_duration'],
        sample_rate=config['sample_rate'],
        augment=False
    )
    
    if len(train_dataset) == 0:
        print("❌ No training samples!")
        return None, 0
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    model = create_teacher_model(num_classes=2, visual_frames=config['num_frames'])
    model = model.to(device)
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    best_acc = 0
    history = []
    
    for epoch in range(config['epochs']):
        print(f"\n📅 Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for frames, audio, labels in tqdm(train_loader, desc='Training'):
            frames, audio, labels = frames.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, audio, labels in tqdm(val_loader, desc='Validation'):
                frames, audio, labels = frames.to(device), audio.to(device), labels.to(device)
                outputs = model(frames, audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"📊 Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"📊 Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch,
                'config': config
            }, '/kaggle/working/teacher_model.pt')
            print(f"✅ Best model saved! Acc: {val_acc:.2f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader)
        })
    
    # Save history
    with open('/kaggle/working/teacher_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎯 Training completed! Best accuracy: {best_acc:.2f}%")
    return model, best_acc

# ============================================================================
# RUN TRAINING
# ============================================================================

print("✅ Fix applied! Starting training...")
model, accuracy = train_teacher_model()

if model:
    print(f"\n🏆 SUCCESS!")
    print(f"📊 Final accuracy: {accuracy:.2f}%")
    print(f"💾 Model saved to: /kaggle/working/teacher_model.pt")
else:
    print("❌ Training failed - check your dataset")