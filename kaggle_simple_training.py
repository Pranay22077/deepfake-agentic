#!/usr/bin/env python3
"""
Simplified Kaggle Training Script
Works without complex source code setup
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Simple dataset class
class SimpleDeepfakeDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load image paths and labels
        self.samples = []
        
        # Look for real/fake folders
        real_dir = os.path.join(data_dir, split, 'real')
        fake_dir = os.path.join(data_dir, split, 'fake')
        
        if os.path.exists(real_dir):
            for img_file in os.listdir(real_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(real_dir, img_file), 0))
        
        if os.path.exists(fake_dir):
            for img_file in os.listdir(fake_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(fake_dir, img_file), 1))
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except:
            # Return dummy data if image loading fails
            return torch.zeros(3, 224, 224), label

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        self.backbone.classifier = nn.Linear(self.backbone.classifier[0].in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

def train_simple_model():
    """Train a simple model for testing"""
    print("🚀 Starting simple training...")
    
    # Configuration
    config = {
        'epochs': 5,
        'batch_size': 32,
        'lr': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # Find data directory
    data_dirs = [
        '/kaggle/input/video-test-samples-5',
        '/kaggle/input/deepfake-faces-dataset',
        '/kaggle/input'
    ]
    
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break
    
    if not data_dir:
        print("❌ No data directory found!")
        return
    
    print(f"Using data from: {data_dir}")
    
    # Create datasets
    try:
        train_dataset = SimpleDeepfakeDataset(data_dir, 'train')
        val_dataset = SimpleDeepfakeDataset(data_dir, 'val')
    except:
        print("⚠️ Standard train/val split not found, using single directory")
        # Create a simple dataset from available images
        all_samples = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Simple heuristic: assume 'fake' in path means fake
                    label = 1 if 'fake' in root.lower() else 0
                    all_samples.append((os.path.join(root, file), label))
        
        # Split 80/20
        split_idx = int(0.8 * len(all_samples))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        class SimpleDataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                try:
                    image = Image.open(img_path).convert('RGB')
                    image = self.transform(image)
                    return image, label
                except:
                    return torch.zeros(3, 224, 224), label
        
        train_dataset = SimpleDataset(train_samples)
        val_dataset = SimpleDataset(val_samples)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Create model
    model = SimpleModel(num_classes=2)
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Training loop
    best_acc = 0
    history = []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc='Training'):
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
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
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'epoch': epoch
            }, '/kaggle/working/simple_model.pt')
            print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_loss': train_loss/len(train_loader),
            'val_loss': val_loss/len(val_loader)
        })
    
    # Save history
    with open('/kaggle/working/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n🎯 Training completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Model saved to: /kaggle/working/simple_model.pt")
    
    return model, best_acc

if __name__ == "__main__":
    model, accuracy = train_simple_model()