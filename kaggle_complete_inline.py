#!/usr/bin/env python3
"""
Complete Inline Kaggle Training Script
Copy this entire script into a Kaggle notebook cell
No external datasets required - everything is self-contained
"""

# ============================================================================
# COMPLETE KAGGLE TRAINING - COPY THIS ENTIRE CELL
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import subprocess
import sys

# Install packages
print("📦 Installing packages...")
packages = ['torch', 'torchvision', 'torchaudio', 'transformers', 'opencv-python', 'scikit-learn']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class SimpleTeacher(nn.Module):
    """Simplified teacher model for Kaggle training"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
        
        # Simple audio processing
        self.audio_net = nn.Sequential(
            nn.Linear(48000, 1024),  # 3 seconds at 16kHz
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.backbone.classifier.in_features + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
    
    def forward(self, video, audio=None, return_features=False):
        # Process video
        if len(video.shape) == 5:  # [B, T, C, H, W]
            B, T = video.shape[:2]
            video = video.view(B*T, *video.shape[2:])
            visual_feat = self.backbone(video)
            visual_feat = visual_feat.view(B, T, -1).mean(dim=1)  # Average over time
        else:
            visual_feat = self.backbone(video)
        
        # Process audio
        if audio is not None:
            audio_feat = self.audio_net(audio)
            combined = torch.cat([visual_feat, audio_feat], dim=1)
        else:
            # Pad with zeros if no audio
            audio_feat = torch.zeros(visual_feat.shape[0], 64, device=visual_feat.device)
            combined = torch.cat([visual_feat, audio_feat], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        if return_features:
            return output, {'visual_feat': visual_feat, 'audio_feat': audio_feat}
        return output

class SimpleStudent(nn.Module):
    """Lightweight student model"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        self.backbone.classifier = nn.Linear(self.backbone.classifier[0].in_features, num_classes)
    
    def forward(self, video, audio=None, return_features=False):
        if len(video.shape) == 5:  # [B, T, C, H, W]
            B, T = video.shape[:2]
            video = video.view(B*T, *video.shape[2:])
            features = self.backbone(video)
            features = features.view(B, T, -1).mean(dim=1)
        else:
            features = self.backbone(video)
        
        if return_features:
            return features, {'visual_feat': features}
        return features

# ============================================================================
# DATASET CLASS
# ============================================================================

class KaggleDeepfakeDataset(Dataset):
    """Dataset class that works with any Kaggle image dataset"""
    def __init__(self, data_dir, split='train', transform=None):
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Find all images
        self.samples = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    # Simple heuristic: fake if 'fake' in path, else real
                    label = 1 if 'fake' in root.lower() else 0
                    self.samples.append((os.path.join(root, file), label))
        
        # Split data 80/20
        split_idx = int(0.8 * len(self.samples))
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"📊 {split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Create video sequence (repeat frame 8 times)
            video = image.unsqueeze(0).repeat(8, 1, 1, 1)  # [T, C, H, W]
            
            # Create dummy audio (3 seconds at 16kHz)
            audio = torch.randn(48000) * 0.1  # Small random audio
            
            return video, audio, label
            
        except Exception as e:
            print(f"⚠️ Error loading {img_path}: {e}")
            # Return dummy data
            video = torch.zeros(8, 3, 224, 224)
            audio = torch.zeros(48000)
            return video, audio, label

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def find_data_directory():
    """Find the best data directory from Kaggle inputs"""
    possible_dirs = []
    
    for item in os.listdir('/kaggle/input/'):
        item_path = f'/kaggle/input/{item}'
        if os.path.isdir(item_path):
            # Count image files
            img_count = 0
            for root, dirs, files in os.walk(item_path):
                img_count += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
            
            if img_count > 10:  # At least 10 images
                possible_dirs.append((item_path, img_count))
    
    if possible_dirs:
        # Return directory with most images
        best_dir = max(possible_dirs, key=lambda x: x[1])
        print(f"📁 Using data directory: {best_dir[0]} ({best_dir[1]} images)")
        return best_dir[0]
    else:
        print("❌ No suitable data directory found!")
        return None

def train_model(model_type='student'):
    """Train teacher or student model"""
    print(f"🚀 Training {model_type} model...")
    
    # Configuration
    config = {
        'epochs': 10 if model_type == 'teacher' else 5,
        'batch_size': 16 if model_type == 'teacher' else 32,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"⚙️ Config: {config}")
    
    # Find data
    data_dir = find_data_directory()
    if not data_dir:
        print("❌ No data found! Please add a dataset with images.")
        return None, 0
    
    # Create datasets
    train_dataset = KaggleDeepfakeDataset(data_dir, 'train')
    val_dataset = KaggleDeepfakeDataset(data_dir, 'val')
    
    if len(train_dataset) == 0:
        print("❌ No training samples found!")
        return None, 0
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    # Create model
    if model_type == 'teacher':
        model = SimpleTeacher(num_classes=2)
    else:
        model = SimpleStudent(num_classes=2)
    
    model = model.to(config['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
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
        
        pbar = tqdm(train_loader, desc='Training')
        for video, audio, labels in pbar:
            video = video.to(config['device'])
            audio = audio.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for video, audio, labels in tqdm(val_loader, desc='Validation'):
                video = video.to(config['device'])
                audio = audio.to(config['device'])
                labels = labels.to(config['device'])
                
                outputs = model(video, audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"📊 Train: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")
        print(f"📊 Val:   Loss={val_loss/len(val_loader):.4f}, Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch,
                'config': config
            }, f'/kaggle/working/{model_type}_model.pt')
            print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader)
        })
    
    # Save history
    with open(f'/kaggle/working/{model_type}_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎯 {model_type.title()} training completed!")
    print(f"📊 Best accuracy: {best_acc:.2f}%")
    print(f"💾 Model saved: /kaggle/working/{model_type}_model.pt")
    
    return model, best_acc

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    print("🎯 E-Raksha Kaggle Training Pipeline")
    print("="*50)
    
    # Show available data
    print("📁 Available datasets:")
    for item in os.listdir('/kaggle/input/'):
        print(f"  - {item}")
    
    # Train student model (faster)
    print("\n🚀 Starting student model training...")
    student_model, student_acc = train_model('student')
    
    if student_model and student_acc > 60:  # If student is decent
        print(f"\n🎓 Student model successful! ({student_acc:.1f}%)")
        
        # Optionally train teacher if time permits
        response = input("Train teacher model too? (y/n): ")
        if response.lower() == 'y':
            print("\n🚀 Starting teacher model training...")
            teacher_model, teacher_acc = train_model('teacher')
            print(f"🎓 Teacher model completed! ({teacher_acc:.1f}%)")
    
    print("\n🏁 Training pipeline completed!")
    print("📁 Check /kaggle/working/ for saved models")

# Run the training
if __name__ == "__main__":
    main()