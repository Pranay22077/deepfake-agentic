#!/usr/bin/env python3
"""
E-Raksha Step 1: Kaggle Environment Setup & Dataset Preparation
Run this first in your Kaggle notebook to set up everything for training
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)

def print_step(step_num, title):
    """Print step header"""
    print(f"\n{step_num}. {title}")
    print("-" * 40)

def install_packages():
    """Install required packages for deepfake detection"""
    print_step("1", "Installing Required Packages")
    
    packages = [
        'facenet-pytorch',
        'librosa', 
        'transformers',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'opencv-python'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package, '-q'
            ])
            print(f"[OK] {package} installed")
        except Exception as e:
            print(f"[WARNING] Could not install {package}: {e}")
    
    print("[OK] Package installation complete")

def check_environment():
    """Check GPU and environment setup"""
    print_step("2", "Checking Environment")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU Available: {device_name}")
            print(f"   Memory: {memory_gb:.1f} GB")
            gpu_ok = True
        else:
            print("[ERROR] No GPU available - training will be very slow")
            gpu_ok = False
    except ImportError:
        print("[ERROR] PyTorch not available")
        gpu_ok = False
    
    # Check Python version
    python_version = sys.version_info
    print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return gpu_ok

def explore_datasets():
    """Explore available datasets in Kaggle input"""
    print_step("3", "Exploring Available Datasets")
    
    input_dir = "/kaggle/input"
    if not os.path.exists(input_dir):
        print("[ERROR] Not running in Kaggle environment")
        return False, None
    
    datasets = os.listdir(input_dir)
    print(f"Available datasets: {len(datasets)}")
    
    # Look for DFDC dataset
    dfdc_datasets = [d for d in datasets if 'dfdc' in d.lower()]
    
    if dfdc_datasets:
        dfdc_path = os.path.join(input_dir, dfdc_datasets[0])
        print(f"[OK] Found DFDC dataset: {dfdc_datasets[0]}")
        
        # Explore structure
        print("Dataset structure:")
        for root, dirs, files in os.walk(dfdc_path):
            level = root.replace(dfdc_path, '').count(os.sep)
            if level > 2:  # Don't go too deep
                continue
                
            indent = "  " * level
            folder_name = os.path.basename(root) or dfdc_datasets[0]
            print(f"{indent}{folder_name}/")
            
            # Show sample files
            if files and level < 2:
                subindent = "  " * (level + 1)
                sample_files = files[:3]
                for file in sample_files:
                    print(f"{subindent}{file}")
                if len(files) > 3:
                    print(f"{subindent}... and {len(files) - 3} more files")
        
        return True, dfdc_path
    else:
        print("[ERROR] DFDC dataset not found")
        print("Available datasets:", datasets)
        return False, None

def setup_directories():
    """Create working directory structure"""
    print_step("4", "Setting Up Working Directories")
    
    directories = [
        "/kaggle/working/data/raw",
        "/kaggle/working/data/processed/train/real",
        "/kaggle/working/data/processed/train/fake",
        "/kaggle/working/data/processed/val/real", 
        "/kaggle/working/data/processed/val/fake",
        "/kaggle/working/models",
        "/kaggle/working/export",
        "/kaggle/working/logs",
        "/kaggle/working/src"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Created: {directory}")
    
    print("[OK] Directory structure ready")

def setup_source_code():
    """Copy source code if available"""
    print_step("5", "Setting Up Source Code")
    
    input_dir = "/kaggle/input"
    datasets = os.listdir(input_dir)
    
    # Look for source code dataset
    source_datasets = [d for d in datasets if any(keyword in d.lower() 
                      for keyword in ['source', 'eraksha', 'code'])]
    
    if source_datasets:
        source_path = os.path.join(input_dir, source_datasets[0])
        print(f"[OK] Found source code: {source_datasets[0]}")
        
        # Copy source files
        src_path = os.path.join(source_path, "src")
        if os.path.exists(src_path):
            shutil.copytree(src_path, "/kaggle/working/src", dirs_exist_ok=True)
            print("[OK] Copied src/ directory")
        
        # Copy other important files
        important_files = ["requirements.txt", "config"]
        for file_name in important_files:
            src_file = os.path.join(source_path, file_name)
            if os.path.exists(src_file):
                if os.path.isdir(src_file):
                    shutil.copytree(src_file, f"/kaggle/working/{file_name}", dirs_exist_ok=True)
                else:
                    shutil.copy2(src_file, "/kaggle/working/")
                print(f"[OK] Copied {file_name}")
        
        return True
    else:
        print("[WARNING] No source code dataset found")
        print("Please upload your E-Raksha source code as a Kaggle dataset")
        return False

def create_configuration(gpu_available, dataset_found, source_ready):
    """Create training configuration"""
    print_step("6", "Creating Training Configuration")
    
    config = {
        "step1_status": {
            "gpu_available": gpu_available,
            "dataset_found": dataset_found,
            "source_code_ready": source_ready,
            "setup_complete": gpu_available and dataset_found,
            "timestamp": str(Path().cwd())
        },
        "paths": {
            "input_dir": "/kaggle/input",
            "working_dir": "/kaggle/working", 
            "processed_data": "/kaggle/working/data/processed",
            "models_dir": "/kaggle/working/models",
            "export_dir": "/kaggle/working/export",
            "logs_dir": "/kaggle/working/logs"
        },
        "training": {
            "batch_size": 8 if gpu_available else 2,
            "num_workers": 2,
            "epochs": 25,
            "learning_rate": 1e-4,
            "num_frames": 8,
            "audio_duration": 3.0,
            "sample_rate": 16000,
            "image_size": 224
        },
        "model": {
            "teacher_backbone": "efficientnet-b4",
            "student_backbone": "mobilenet_v3_small",
            "audio_model": "wav2vec2-base",
            "num_classes": 2
        }
    }
    
    # Save configuration
    config_path = "/kaggle/working/step1_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"[OK] Configuration saved to {config_path}")
    return config

def print_summary(gpu_ok, dataset_ok, source_ok):
    """Print setup summary"""
    print_header("STEP 1 SETUP SUMMARY")
    
    status_items = [
        ("GPU Available", gpu_ok, "Required for efficient training"),
        ("DFDC Dataset Found", dataset_ok, "Main training dataset"),
        ("Source Code Ready", source_ok, "E-Raksha training scripts"),
        ("Directories Created", True, "Working directory structure"),
        ("Configuration Saved", True, "Training parameters")
    ]
    
    print("Status Check:")
    for item, status, description in status_items:
        icon = "[OK]" if status else "[ERROR]"
        print(f"{icon} {item:<20} - {description}")
    
    setup_complete = gpu_ok and dataset_ok
    
    print("\n" + "-" * 60)
    if setup_complete:
        print("STEP 1 SETUP COMPLETE!")
        print("\nYou're ready to proceed with:")
        print("• Data preprocessing (extract faces and audio)")
        print("• Teacher model training (heavy multimodal model)")
        print("• Student distillation (lightweight mobile model)")
        print("• Model optimization (pruning and quantization)")
        
        print("\nNext: Run the data preprocessing script")
    else:
        print("STEP 1 SETUP INCOMPLETE")
        print("\nPlease resolve these issues:")
        if not gpu_ok:
            print("• Enable GPU in notebook settings")
        if not dataset_ok:
            print("• Add DFDC dataset as notebook input")
        if not source_ok:
            print("• Upload E-Raksha source code as dataset")
    
    print("=" * 60)

def main():
    """Main setup function"""
    print_header("E-Raksha Kaggle Step 1 Setup")
    print("This script prepares your Kaggle environment for deepfake detection training")
    
    # Run setup steps
    install_packages()
    gpu_ok = check_environment()
    dataset_ok, dataset_path = explore_datasets()
    setup_directories()
    source_ok = setup_source_code()
    config = create_configuration(gpu_ok, dataset_ok, source_ok)
    
    # Print summary
    print_summary(gpu_ok, dataset_ok, source_ok)
    
    return config

if __name__ == "__main__":
    config = main()