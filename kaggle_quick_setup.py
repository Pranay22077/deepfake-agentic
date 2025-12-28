#!/usr/bin/env python3
"""
Quick Kaggle Setup - Copy this directly into your Kaggle notebook
Run this as the first cell in your Kaggle notebook
"""

# ============================================================================
# KAGGLE QUICK SETUP - COPY THIS ENTIRE CELL TO YOUR KAGGLE NOTEBOOK
# ============================================================================

import os
import sys
import subprocess

def kaggle_quick_setup():
    """Quick setup for Kaggle notebook"""
    print("🚀 E-Raksha Kaggle Quick Setup")
    print("="*50)
    
    # 1. Install packages
    print("📦 Installing required packages...")
    packages = [
        'torch', 'torchvision', 'torchaudio', 
        'transformers', 'opencv-python', 'librosa',
        'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    for pkg in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
            print(f"✅ {pkg}")
        except:
            print(f"⚠️ {pkg} failed")
    
    # 2. Setup directories
    print("\n📁 Creating directories...")
    dirs = [
        '/kaggle/working/src/models',
        '/kaggle/working/src/train',
        '/kaggle/working/models',
        '/kaggle/working/logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # 3. Add to Python path
    sys.path.insert(0, '/kaggle/working')
    sys.path.insert(0, '/kaggle/working/src')
    
    # 4. Copy source files if dataset exists
    source_found = False
    possible_sources = [
        '/kaggle/input/eraksha-source-code',
        '/kaggle/input/e-raksha-deepfake-source-code',
        '/kaggle/input/deepfake-source'
    ]
    
    for source in possible_sources:
        if os.path.exists(source):
            print(f"📋 Found source at: {source}")
            
            # Copy files
            import shutil
            try:
                if os.path.exists(f"{source}/src"):
                    shutil.copytree(f"{source}/src", "/kaggle/working/src", dirs_exist_ok=True)
                    print("✅ src/ copied")
                
                # Copy individual scripts
                scripts = [
                    'kaggle_train_teacher.py',
                    'kaggle_simple_training.py'
                ]
                
                for script in scripts:
                    if os.path.exists(f"{source}/{script}"):
                        shutil.copy2(f"{source}/{script}", f"/kaggle/working/{script}")
                        print(f"✅ {script} copied")
                
                source_found = True
                break
            except Exception as e:
                print(f"⚠️ Copy failed: {e}")
    
    if not source_found:
        print("⚠️ Source dataset not found. Creating minimal files...")
        create_minimal_files()
    
    print("\n🎯 Setup complete!")
    print("Available inputs:", os.listdir('/kaggle/input/'))
    
    return source_found

def create_minimal_files():
    """Create minimal files for basic training"""
    
    # Minimal teacher model
    teacher_code = '''
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleTeacher(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)
    
    def forward(self, x, audio=None, return_features=False):
        if len(x.shape) == 5:  # Video input [B, T, C, H, W]
            B, T = x.shape[:2]
            x = x.view(B*T, *x.shape[2:])
            out = self.backbone(x)
            out = out.view(B, T, -1).mean(dim=1)  # Average over time
        else:
            out = self.backbone(x)
        
        if return_features:
            return out, {'visual_feat': out}
        return out

def create_teacher_model(num_classes=2, visual_frames=8):
    return SimpleTeacher(num_classes)
'''
    
    with open('/kaggle/working/src/models/teacher.py', 'w') as f:
        f.write(teacher_code)
    
    # Minimal dataset class
    dataset_code = '''
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class MultimodalDataset(Dataset):
    def __init__(self, data_dir, split='train', num_frames=8, audio_duration=3.0, 
                 sample_rate=16000, augment=False, heavy_augment=False):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Find samples
        self.samples = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    label = 1 if 'fake' in root.lower() else 0
                    self.samples.append((os.path.join(root, file), label))
        
        # Split data
        if split == 'train':
            self.samples = self.samples[:int(0.8 * len(self.samples))]
        else:
            self.samples = self.samples[int(0.8 * len(self.samples)):]
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            
            # Create dummy video (repeat frame)
            video = image.unsqueeze(0).repeat(8, 1, 1, 1)  # [T, C, H, W]
            
            # Create dummy audio
            audio = torch.zeros(48000)  # 3 seconds at 16kHz
            
            return video, audio, label
        except:
            # Return dummy data if loading fails
            video = torch.zeros(8, 3, 224, 224)
            audio = torch.zeros(48000)
            return video, audio, label
'''
    
    with open('/kaggle/working/src/train/train_teacher.py', 'w') as f:
        f.write(dataset_code)
    
    print("✅ Minimal files created")

# Run setup
kaggle_quick_setup()

print("""
🎯 NEXT STEPS:
1. If you uploaded the source dataset, you're ready to train!
2. If not, you can still run basic training with the minimal setup
3. Run your training script or use the simple training option

📝 TRAINING OPTIONS:
- Full pipeline: exec(open('/kaggle/working/kaggle_train_teacher.py').read())
- Simple training: exec(open('/kaggle/working/kaggle_simple_training.py').read())
""")