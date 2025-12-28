#!/usr/bin/env python3
"""
Kaggle Notebook Setup
Run this first in your Kaggle notebook to set up the environment
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def setup_kaggle_environment():
    """Complete Kaggle environment setup"""
    print("🔧 Setting up Kaggle environment for E-Raksha Step 2...")
    
    # 1. Install required packages
    print("📦 Installing packages...")
    packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'transformers',
        'opencv-python',
        'librosa',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed")
        except:
            print(f"⚠️ {package} installation failed")
    
    # 2. Create directory structure
    print("📁 Creating directory structure...")
    directories = [
        '/kaggle/working/src/models',
        '/kaggle/working/src/train', 
        '/kaggle/working/src/opt',
        '/kaggle/working/src/eval',
        '/kaggle/working/src/preprocess',
        '/kaggle/working/src/agent',
        '/kaggle/working/export',
        '/kaggle/working/config',
        '/kaggle/working/models',
        '/kaggle/working/logs',
        '/kaggle/working/teacher_predictions',
        '/kaggle/working/eval_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # 3. Copy source files from input dataset
    print("📋 Copying source files...")
    
    # Check if source dataset is available
    source_paths = [
        '/kaggle/input/eraksha-deepfake-source',
        '/kaggle/input/eraksha-source-code',
        '/kaggle/input/deepfake-source'
    ]
    
    source_path = None
    for path in source_paths:
        if os.path.exists(path):
            source_path = path
            break
    
    if source_path:
        print(f"Found source code at: {source_path}")
        
        # Copy all source files
        import shutil
        
        # Copy src directory
        if os.path.exists(f"{source_path}/src"):
            shutil.copytree(f"{source_path}/src", "/kaggle/working/src", dirs_exist_ok=True)
            print("✅ src/ copied")
        
        # Copy export directory  
        if os.path.exists(f"{source_path}/export"):
            shutil.copytree(f"{source_path}/export", "/kaggle/working/export", dirs_exist_ok=True)
            print("✅ export/ copied")
            
        # Copy config directory
        if os.path.exists(f"{source_path}/config"):
            shutil.copytree(f"{source_path}/config", "/kaggle/working/config", dirs_exist_ok=True)
            print("✅ config/ copied")
            
    else:
        print("❌ Source code dataset not found!")
        print("Available inputs:")
        for item in os.listdir('/kaggle/input/'):
            print(f"  - {item}")
        print("\n🔧 Manual setup required - see instructions below")
        return False
    
    # 4. Add working directory to Python path
    sys.path.insert(0, '/kaggle/working')
    sys.path.insert(0, '/kaggle/working/src')
    
    print("✅ Python path updated")
    
    # 5. Verify setup
    print("🔍 Verifying setup...")
    
    required_files = [
        '/kaggle/working/src/models/teacher.py',
        '/kaggle/working/src/models/student.py',
        '/kaggle/working/src/train/train_teacher.py',
        '/kaggle/working/src/preprocess/augmentation.py'
    ]
    
    all_good = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    if all_good:
        print("🎯 Setup complete! Ready for training.")
        return True
    else:
        print("⚠️ Some files missing. Check your source dataset.")
        return False

def manual_file_creation():
    """Create essential files manually if dataset upload failed"""
    print("📝 Creating essential files manually...")
    
    # Create a minimal teacher model file
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
        # Simple implementation for testing
        if len(x.shape) == 5:  # [B, T, C, H, W]
            B, T = x.shape[:2]
            x = x.view(B*T, *x.shape[2:])
            features = self.backbone(x)
            features = features.view(B, T, -1).mean(dim=1)
        else:
            features = self.backbone(x)
        
        if return_features:
            return features, {'visual_feat': features}
        return features

def create_teacher_model(num_classes=2, visual_frames=8):
    return SimpleTeacher(num_classes)
'''
    
    with open('/kaggle/working/src/models/teacher.py', 'w') as f:
        f.write(teacher_code)
    
    print("✅ Basic teacher model created")
    
    # Create other essential files...
    # (Add more files as needed)

# Run setup
if __name__ == "__main__":
    success = setup_kaggle_environment()
    
    if not success:
        print("\n" + "="*50)
        print("MANUAL SETUP INSTRUCTIONS")
        print("="*50)
        print("1. Upload your source code as a Kaggle dataset")
        print("2. Add it as input to this notebook") 
        print("3. Re-run this setup cell")
        print("4. Or use manual_file_creation() for basic files")
        
        # Offer manual creation
        response = input("Create basic files manually? (y/n): ")
        if response.lower() == 'y':
            manual_file_creation()
    
    print(f"\n🚀 Ready to start training!")