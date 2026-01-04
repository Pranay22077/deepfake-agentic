#!/usr/bin/env python3
"""
Simple Kaggle Training Script - Uses OpenCV to avoid PIL issues
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from facenet_pytorch import MTCNN
import warnings
warnings.filterwarnings('ignore')

# Model Definition
import torchvision.models as models

class StudentModel(nn.Module):
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def save_face_opencv(face_array, path):
    """Save face using OpenCV to avoid PIL issues"""
    try:
        # Convert RGB to BGR for OpenCV
        face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
        # Save with OpenCV
        success = cv2.imwrite(path, face_bgr)
        return success
    except Exception as e:
        print(f"Error saving {path}: {e}")
        return False

class VideoProcessor:
    def __init__(self, data_dir="/kaggle/input/video-data-sample/data"):
        self.data_dir = data_dir
        self.mtcnn = MTCNN(
            keep_all=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7]
        )
        
    def extract_faces_from_video(self, video_path, max_faces=5):
        """Extract faces from video"""
        faces = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return faces
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return faces
        
        # Sample frames
        step = max(1, total_frames // (max_faces * 2))
        frame_count = 0
        
        while len(faces) < max_faces:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                # Convert to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                try:
                    # Detect face
                    box, prob = self.mtcnn.detect(Image.fromarray(rgb_frame))
                    
                    if box is not None and len(box) > 0 and prob is not None and prob[0] > 0.9:
                        x1, y1, x2, y2 = map(int, box[0])
                        
                        # Add padding
                        h, w = rgb_frame.shape[:2]
                        padding = 20
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)
                        
                        # Extract face
                        face = rgb_frame[y1:y2, x1:x2]
                        if face.size > 0:
                            # Resize face
                            face_resized = cv2.resize(face, (224, 224))
                            faces.append(face_resized)
                            
                except Exception as e:
                    pass
                    
            frame_count += 1
            
        cap.release()
        return faces
    def process_dataset(self, output_dir="/kaggle/working/processed_data"):
        """Process entire dataset"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/real", exist_ok=True)
        os.makedirs(f"{output_dir}/fake", exist_ok=True)
        
        real_count = 0
        fake_count = 0
        
        # Process real videos
        real_dir = os.path.join(self.data_dir, "real")
        if os.path.exists(real_dir):
            print("Processing real videos...")
            video_files = [f for f in os.listdir(real_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in tqdm(video_files):
                video_path = os.path.join(real_dir, video_file)
                faces = self.extract_faces_from_video(video_path, max_faces=5)
                
                for i, face in enumerate(faces):
                    face_path = f"{output_dir}/real/{video_file}_{i:03d}.jpg"
                    if save_face_opencv(face, face_path):
                        real_count += 1
        
        # Process fake videos
        fake_dir = os.path.join(self.data_dir, "fake")
        if os.path.exists(fake_dir):
            print("Processing fake videos...")
            video_files = [f for f in os.listdir(fake_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in tqdm(video_files):
                video_path = os.path.join(fake_dir, video_file)
                faces = self.extract_faces_from_video(video_path, max_faces=5)
                
                for i, face in enumerate(faces):
                    face_path = f"{output_dir}/fake/{video_file}_{i:03d}.jpg"
                    if save_face_opencv(face, face_path):
                        fake_count += 1
        
        print(f"Extracted {real_count} real faces and {fake_count} fake faces")
        return real_count, fake_count

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Load samples
        for label, folder in [(0, 'real'), (1, 'fake')]:
            folder_path = os.path.join(data_dir, folder)
            if os.path.exists(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.endswith(('.jpg', '.png')):
                        self.samples.append((os.path.join(folder_path, img_file), label))
        
        print(f"Dataset: {len(self.samples)} samples")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # Return dummy if error
            return torch.zeros(3, 224, 224), label

def train_model(data_dir, epochs=8, batch_size=16):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Simple transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    dataset = DeepfakeDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("No data found!")
        return None, 0
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = StudentModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
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
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '/kaggle/working/baseline_student.pt')
            print(f'Best model saved! Acc: {val_acc:.2f}%')
    
    return model, best_acc
def main():
    """Main function"""
    print("E-Raksha Simple Training")
    print("="*40)
    
    # Process videos
    processor = VideoProcessor()
    real_count, fake_count = processor.process_dataset()
    
    if real_count == 0 and fake_count == 0:
        print("No faces found!")
        print("Check your dataset structure:")
        print("/kaggle/input/video-data-sample/data/real/ (videos)")
        print("/kaggle/input/video-data-sample/data/fake/ (videos)")
        return
    
    # Train
    print(f"\nTraining with {real_count} real + {fake_count} fake faces...")
    model, best_acc = train_model("/kaggle/working/processed_data")
    
    if model is None:
        print("Training failed!")
        return
    
    # Test saved model
    print("\nTesting saved model...")
    test_model = StudentModel()
    test_model.load_state_dict(torch.load('/kaggle/working/baseline_student.pt'))
    test_model.eval()
    
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        output = test_model(dummy)
        print(f"Model works! Output: {output.shape}")
    
    # Save info
    info = {
        'best_accuracy': best_acc,
        'real_faces': real_count,
        'fake_faces': fake_count,
        'total_faces': real_count + fake_count
    }
    
    with open('/kaggle/working/training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*40)
    print("TRAINING COMPLETE!")
    print("="*40)
    print(f"Dataset: {real_count} real + {fake_count} fake")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved: baseline_student.pt")
    print("="*40)
    
    print("\nDownload these files:")
    print("   - baseline_student.pt")
    print("   - training_info.json")

if __name__ == "__main__":
    main()