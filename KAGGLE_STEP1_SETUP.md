# Kaggle Step 1: Environment Setup & Dataset Preparation

## Overview
Step 1 prepares your Kaggle environment with the DFDC dataset and sets up everything needed for deepfake detection training.

## Prerequisites

### 1. Kaggle Account Requirements
- Kaggle account with phone verification
- GPU quota available (P100 or T4 recommended)
- Internet access enabled for package installation
- DFDC dataset downloaded from Kaggle competition

### 2. Create Kaggle Notebook
1. Go to Kaggle Notebooks
2. Create New Notebook
3. Settings:
   - **Accelerator**: GPU P100 or T4
   - **Internet**: ON
   - **Environment**: Python 3.7+

### 3. Add Required Datasets
Add these datasets as input to your notebook:
- **DFDC Dataset**: The competition dataset you downloaded
- **Your Source Code**: Upload your E-Raksha source code as a dataset

## Step 1 Setup Script

Run this in your first Kaggle notebook cell:

```python
# Step 1: Environment Setup and Dataset Preparation
import os
import sys
import subprocess
import torch
import cv2
import numpy as np
from pathlib import Path
import json
import shutil

print("E-Raksha Step 1: Kaggle Environment Setup")
print("=" * 50)

# 1. Install required packages
def install_packages():
    packages = [
        'facenet-pytorch',
        'librosa',
        'transformers',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
        except Exception as e:
            print(f"Warning: Could not install {package}: {e}")

install_packages()

# 2. Check GPU availability
def check_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {device_name}")
        print(f"   Memory: {memory_gb:.1f} GB")
        return True
    else:
        print("❌ No GPU available - training will be very slow")
        return False

gpu_ok = check_gpu()

# 3. Explore DFDC dataset structure
def explore_dfdc_dataset():
    print("\n📁 Exploring DFDC Dataset Structure...")
    
    # Find DFDC dataset path
    input_paths = [p for p in os.listdir('/kaggle/input') if 'dfdc' in p.lower()]
    
    if not input_paths:
        print("❌ DFDC dataset not found in /kaggle/input/")
        print("Available datasets:", os.listdir('/kaggle/input'))
        return False
    
    dfdc_path = f"/kaggle/input/{input_paths[0]}"
    print(f"✅ Found DFDC dataset at: {dfdc_path}")
    
    # Explore structure
    for root, dirs, files in os.walk(dfdc_path):
        level = root.replace(dfdc_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Show first few files
        subindent = ' ' * 2 * (level + 1)
        for i, file in enumerate(files[:5]):
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")
        
        # Don't go too deep
        if level > 2:
            break
    
    return True

dataset_ok = explore_dfdc_dataset()

# 4. Set up working directories
def setup_directories():
    print("\n📂 Setting up working directories...")
    
    dirs = [
        '/kaggle/working/data/processed/train/real',
        '/kaggle/working/data/processed/train/fake', 
        '/kaggle/working/data/processed/val/real',
        '/kaggle/working/data/processed/val/fake',
        '/kaggle/working/models',
        '/kaggle/working/logs',
        '/kaggle/working/export'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created: {dir_path}")

setup_directories()

# 5. Copy source code if available
def setup_source_code():
    print("\n📋 Setting up source code...")
    
    # Look for source code dataset
    source_paths = [p for p in os.listdir('/kaggle/input') if 'source' in p.lower() or 'eraksha' in p.lower()]
    
    if source_paths:
        source_path = f"/kaggle/input/{source_paths[0]}"
        print(f"✅ Found source code at: {source_path}")
        
        # Copy to working directory
        if os.path.exists(f"{source_path}/src"):
            shutil.copytree(f"{source_path}/src", "/kaggle/working/src", dirs_exist_ok=True)
            print("✅ Copied src/ directory")
        
        return True
    else:
        print("⚠️  No source code dataset found - you'll need to upload it")
        return False

source_ok = setup_source_code()

# 6. Create configuration
def create_config():
    print("\n⚙️  Creating training configuration...")
    
    config = {
        "step1_setup": {
            "gpu_available": gpu_ok,
            "dataset_found": dataset_ok,
            "source_code_ready": source_ok,
            "setup_complete": gpu_ok and dataset_ok
        },
        "paths": {
            "dfdc_input": "/kaggle/input",
            "working_dir": "/kaggle/working",
            "processed_data": "/kaggle/working/data/processed",
            "models_dir": "/kaggle/working/models",
            "logs_dir": "/kaggle/working/logs"
        },
        "training_config": {
            "batch_size": 8 if gpu_ok else 2,
            "num_workers": 2,
            "epochs": 25,
            "learning_rate": 1e-4,
            "num_frames": 8,
            "audio_duration": 3.0,
            "sample_rate": 16000
        }
    }
    
    with open('/kaggle/working/step1_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to step1_config.json")
    return config

config = create_config()

# 7. Summary
print("\n" + "=" * 50)
print("📊 STEP 1 SETUP SUMMARY")
print("=" * 50)

status_items = [
    ("GPU Available", gpu_ok),
    ("DFDC Dataset Found", dataset_ok), 
    ("Source Code Ready", source_ok),
    ("Directories Created", True),
    ("Configuration Saved", True)
]

for item, status in status_items:
    icon = "✅" if status else "❌"
    print(f"{icon} {item}")

setup_complete = all([gpu_ok, dataset_ok])

if setup_complete:
    print("\n🎉 STEP 1 COMPLETE!")
    print("Ready to proceed with data preprocessing and training.")
    print("\nNext steps:")
    print("1. Run data preprocessing script")
    print("2. Start teacher model training")
    print("3. Proceed with student distillation")
else:
    print("\n⚠️  STEP 1 INCOMPLETE")
    print("Please resolve the issues above before proceeding.")

print("=" * 50)
```

## 🔍 What This Script Does

### 1. **Environment Setup**
- Installs required Python packages
- Checks GPU availability and memory
- Verifies Kaggle environment is ready

### 2. **Dataset Discovery**
- Locates your DFDC dataset in Kaggle inputs
- Explores the dataset structure
- Validates data availability

### 3. **Directory Structure**
- Creates organized working directories
- Sets up paths for processed data
- Prepares model and log directories

### 4. **Source Code Setup**
- Copies your E-Raksha source code
- Makes training scripts available
- Prepares for model training

### 5. **Configuration Creation**
- Generates training configuration
- Sets optimal parameters for your GPU
- Saves setup status for next steps

## ✅ Success Criteria

After running Step 1, you should see:
- ✅ GPU Available (P100 or T4)
- ✅ DFDC Dataset Found
- ✅ Source Code Ready
- ✅ Directories Created
- ✅ Configuration Saved

## 🚨 Common Issues & Solutions

### Issue: No GPU Available
**Solution**: Check notebook settings, ensure GPU is selected

### Issue: DFDC Dataset Not Found
**Solution**: Add DFDC dataset as input to your notebook

### Issue: Source Code Missing
**Solution**: Upload your E-Raksha code as a Kaggle dataset

### Issue: Package Installation Fails
**Solution**: Enable internet access in notebook settings

## 🎯 Next Steps

Once Step 1 is complete:
1. **Data Preprocessing**: Extract faces and audio from DFDC videos
2. **Teacher Training**: Train the heavy multimodal teacher model
3. **Student Distillation**: Create lightweight mobile model
4. **Model Optimization**: Prepare for mobile deployment

## 📝 Notes

- This setup takes 5-10 minutes to complete
- GPU quota is required for efficient training
- Internet access needed for package installation
- DFDC dataset should be added as notebook input

---

**Estimated Time**: 5-10 minutes
**Requirements**: Kaggle GPU notebook with DFDC dataset
**Next**: Data preprocessing and teacher model training