#!/usr/bin/env python3
"""
Kaggle Script: Fix the Broken Deepfake Detection Model
Train a proper ResNet18 model that actually works on real deepfake data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import pickle
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Kaggle paths
KAGGLE_INPUT = '/kaggle/input'
KAGGLE_WORKING = '/kaggle/working'

class StudentModel(nn.Module):
    """Fixed ResNet18-based deepfake detector"""
    def __init__(self, num_classes=2, pretrained=True):
        super(StudentModel, self).__init__()
        
        # Use pretrained ResNet18 for better feature extraction
        if pretrained:
            self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Replace final layer with proper architecture
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DFDCDataset(Dataset):
    """DFDC Dataset for deepfake detection"""
    def __init__(self, data_dir, metadata_file, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Create samples list
        self.samples = []
        for video_name, info in self.metadata.items():
            if info['label'] in ['REAL', 'FAKE']:
                self.samples.append({
                    'video': video_name,
                    'label': 1 if info['label'] == 'FAKE' else 0  # 0=real, 1=fake
                })
        
        # Limit samples if specified
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Dataset loaded: {len(self.samples)} samples")
        real_count = sum(1 for s in self.samples if s['label'] == 0)
        fake_count = sum(1 for s in self.samples if s['label'] == 1)
        print(f"Real: {real_count}, Fake: {fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = os.path.join(self.data_dir, sample['video'])
        label = sample['label']
        
        # Extract face from video
        face = self.extract_face_from_video(video_path)
        
        if face is None:
            # Fallback to random face if extraction fails
            face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Convert to PIL Image
        face = Image.fromarray(face)
        
        if self.transform:
            face = self.transform(face)
        
        return face, label
    
    def extract_face_from_video(self, video_path, target_size=(224, 224)):
        """Extract a face from video"""
        if not os.path.exists(video_path):
            return None
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Try to get a frame from middle of video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return None
        
        # Jump to middle frame
        middle_frame = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Simple face detection
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Add padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face = frame[y1:y2, x1:x2]
            else:
                # Fallback to center crop
                h, w = frame.shape[:2]
                size = min(h, w)
                y_start = (h - size) // 2
                x_start = (w - size) // 2
                face = frame[y_start:y_start+size, x_start:x_start+size]
            
            # Resize to target size
            face = cv2.resize(face, target_size)
            return face
            
        except Exception as e:
            print(f"Face extraction failed for {video_path}: {e}")
            return None

def find_dfdc_dataset():
    """Find DFDC dataset in Kaggle input"""
    possible_paths = [
        '/kaggle/input/deepfake-detection-challenge',
        '/kaggle/input/dfdc-dataset',
        '/kaggle/input/deepfake-faces-dataset'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            return path
    
    print("DFDC dataset not found. Available datasets:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            print(f"  - {item}")
    
    return None

def train_fixed_model():
    """Train a proper deepfake detection model"""
    print("🔧 Training Fixed Deepfake Detection Model")
    print("=" * 60)
    
    # Find dataset
    dataset_path = find_dfdc_dataset()
    if not dataset_path:
        print("❌ No DFDC dataset found!")
        return None, 0, []
    
    # Look for metadata file
    metadata_files = [
        os.path.join(dataset_path, 'train_sample_videos', 'metadata.json'),
        os.path.join(dataset_path, 'metadata.json'),
        os.path.join(dataset_path, 'train_metadata.json')
    ]
    
    metadata_file = None
    for mf in metadata_files:
        if os.path.exists(mf):
            metadata_file = mf
            break
    
    if not metadata_file:
        print("❌ No metadata file found!")
        return None, 0, []
    
    print(f"Using metadata: {metadata_file}")
    
    # Find video directory
    video_dirs = [
        os.path.join(dataset_path, 'train_sample_videos'),
        os.path.join(dataset_path, 'videos'),
        dataset_path
    ]
    
    video_dir = None
    for vd in video_dirs:
        if os.path.exists(vd) and any(f.endswith('.mp4') for f in os.listdir(vd) if os.path.isfile(os.path.join(vd, f))):
            video_dir = vd
            break
    
    if not video_dir:
        print("❌ No video directory found!")
        return None, 0, []
    
    print(f"Using videos from: {video_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = DFDCDataset(video_dir, metadata_file, transform=None, max_samples=1000)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_dataset)))
    
    # Create separate datasets with transforms
    train_dataset = DFDCDataset(video_dir, metadata_file, transform=train_transform, max_samples=train_size)
    val_dataset = DFDCDataset(video_dir, metadata_file, transform=val_transform, max_samples=None)
    val_dataset.samples = [val_dataset.samples[i] for i in range(train_size, min(len(val_dataset.samples), train_size + val_size))]
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = StudentModel(num_classes=2, pretrained=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Training loop
    best_acc = 0
    best_auc = 0
    train_history = []
    
    num_epochs = 15
    
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch+1}/{num_epochs}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        all_train_preds = []
        all_train_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # For AUC calculation
            probs = torch.softmax(output, dim=1)
            all_train_preds.extend(probs[:, 1].cpu().detach().numpy())
            all_train_labels.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.1f}%'
            })
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                # For metrics
                probs = torch.softmax(output, dim=1)
                all_val_preds.extend(pred.cpu().numpy())
                all_val_labels.extend(target.cpu().numpy())
                all_val_probs.extend(probs[:, 1].cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        try:
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            val_auc = roc_auc_score(all_val_labels, all_val_probs)
        except:
            train_auc = val_auc = 0.5
        
        scheduler.step()
        
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
        
        # Save history
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_auc = val_auc
            
            # Save model PROPERLY with all BatchNorm statistics
            model_save_path = os.path.join(KAGGLE_WORKING, 'fixed_deepfake_model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),  # This includes ALL parameters
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'best_auc': best_auc,
                'config': {
                    'architecture': 'ResNet18',
                    'num_classes': 2,
                    'pretrained': True,
                    'input_size': [3, 224, 224]
                }
            }, model_save_path)
            
            print(f"✅ New best model saved! Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
    
    print(f"\n🎯 Training completed!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best AUC: {best_auc:.4f}")
    
    # Test model behavior
    print("\n🧪 Testing model behavior:")
    test_model_behavior(model, device, val_transform)
    
    # Save training info
    training_info = {
        'best_accuracy': best_acc,
        'best_auc': best_auc,
        'architecture': 'ResNet18',
        'parameters': total_params,
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'training_history': train_history,
        'model_info': {
            'architecture': 'ResNet18',
            'input_size': [3, 224, 224],
            'num_classes': 2,
            'parameters': total_params
        }
    }
    
    with open(os.path.join(KAGGLE_WORKING, 'fixed_training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return model, best_acc, train_history

def test_model_behavior(model, device, transform):
    """Test model with different inputs to verify it's not broken"""
    model.eval()
    
    # Create test images
    test_cases = [
        ("Random Noise", np.random.randint(0, 255, (224, 224, 3))),
        ("All Black", np.zeros((224, 224, 3))),
        ("All White", np.ones((224, 224, 3)) * 255),
        ("Gradient", np.array([[[i, j, (i+j)//2] for j in range(224)] for i in range(224)])),
    ]
    
    with torch.no_grad():
        for name, image_array in test_cases:
            # Convert to PIL and apply transform
            pil_image = Image.fromarray(image_array.astype(np.uint8))
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            # Get prediction
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()
            
            class_name = "Real" if pred_class == 0 else "Fake"
            print(f"{name:15} | {class_name:4} | Conf: {confidence:.3f} | Raw: [{output[0][0].item():6.2f}, {output[0][1].item():6.2f}]")

if __name__ == "__main__":
    print("🚀 Starting Fixed Model Training on Kaggle")
    print("=" * 60)
    
    try:
        model, best_acc, history = train_fixed_model()
        
        if model is not None:
            print(f"\n✅ SUCCESS!")
            print(f"Fixed model trained with {best_acc:.2f}% accuracy")
            print(f"Model saved as: fixed_deepfake_model.pt")
            print(f"Training info saved as: fixed_training_info.json")
            
            # List output files
            print(f"\n📁 Output files:")
            for file in os.listdir(KAGGLE_WORKING):
                if file.endswith(('.pt', '.json')):
                    size = os.path.getsize(os.path.join(KAGGLE_WORKING, file)) / 1024 / 1024
                    print(f"  - {file} ({size:.1f} MB)")
        else:
            print("❌ Training failed!")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()