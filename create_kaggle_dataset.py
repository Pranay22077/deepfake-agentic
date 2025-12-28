#!/usr/bin/env python3
"""
Create Clean Kaggle Dataset
Creates a zip file with only the necessary source code files for Kaggle
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_kaggle_source_zip():
    """Create a clean zip file for Kaggle upload"""
    print("🔧 Creating clean Kaggle source dataset...")
    
    # Create temporary directory with clean name
    temp_dir = Path("eraksha_source_clean")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir()
    
    # Files and directories to include
    include_items = [
        # Source code directories
        "src/",
        "export/", 
        "config/",
        
        # Individual files
        "requirements.txt",
        "README.md",
        
        # Kaggle scripts
        "kaggle_train_teacher.py",
        "kaggle_distill_student.py", 
        "kaggle_optimize_models.py",
        "kaggle_step2_complete.py",
        "kaggle_simple_training.py",
        "kaggle_notebook_setup.py"
    ]
    
    # Copy files to clean directory
    for item in include_items:
        source_path = Path(item)
        
        if source_path.exists():
            target_path = temp_dir / item
            
            if source_path.is_dir():
                # Copy directory
                shutil.copytree(source_path, target_path, ignore=ignore_patterns)
                print(f"✅ Copied directory: {item}")
            else:
                # Copy file
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                print(f"✅ Copied file: {item}")
        else:
            print(f"⚠️ Not found: {item}")
    
    # Create zip file
    zip_path = "eraksha-source-code.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = Path(root) / file
                # Create archive path without the temp directory prefix
                archive_path = file_path.relative_to(temp_dir)
                zipf.write(file_path, archive_path)
                
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    
    # Get zip file size
    zip_size = os.path.getsize(zip_path) / (1024 * 1024)  # MB
    
    print(f"\n🎯 Kaggle dataset created successfully!")
    print(f"📁 File: {zip_path}")
    print(f"📊 Size: {zip_size:.1f} MB")
    print(f"\n📋 Next steps:")
    print(f"1. Go to kaggle.com/datasets")
    print(f"2. Click 'New Dataset'")
    print(f"3. Upload {zip_path}")
    print(f"4. Title: 'E-Raksha Deepfake Source Code'")
    print(f"5. Make it public or private")
    print(f"6. Add it as input to your notebook")
    
    return zip_path

def ignore_patterns(dir, files):
    """Ignore patterns for copying directories"""
    ignore = set()
    
    for file in files:
        # Ignore common unwanted files
        if (file.startswith('.') or 
            file.endswith('.pyc') or
            file.endswith('.pyo') or
            file == '__pycache__' or
            file.endswith('.log') or
            file.endswith('.tmp')):
            ignore.add(file)
    
    return ignore

def verify_zip_contents(zip_path):
    """Verify the contents of the created zip file"""
    print(f"\n🔍 Verifying zip contents...")
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        files = zipf.namelist()
        
        print(f"📁 Total files: {len(files)}")
        
        # Check for essential files
        essential_files = [
            'src/models/teacher.py',
            'src/models/student.py',
            'src/train/train_teacher.py',
            'kaggle_train_teacher.py',
            'requirements.txt'
        ]
        
        print(f"\n✅ Essential files check:")
        for file in essential_files:
            if file in files:
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file} - MISSING!")
        
        # Show directory structure
        print(f"\n📂 Directory structure:")
        dirs = set()
        for file in files:
            parts = file.split('/')
            for i in range(len(parts)):
                dir_path = '/'.join(parts[:i+1])
                if '.' not in parts[i]:  # It's a directory
                    dirs.add(dir_path)
        
        for dir in sorted(dirs):
            level = dir.count('/')
            indent = "  " * level
            name = dir.split('/')[-1] if '/' in dir else dir
            print(f"{indent}📁 {name}/")
        
        # Check for forbidden characters
        forbidden_chars = ['&', '<', '>', ':', '"', '|', '?', '*']
        problematic_files = []
        
        for file in files:
            for char in forbidden_chars:
                if char in file:
                    problematic_files.append(file)
                    break
        
        if problematic_files:
            print(f"\n⚠️ Files with forbidden characters:")
            for file in problematic_files:
                print(f"  - {file}")
        else:
            print(f"\n✅ No forbidden characters found!")

if __name__ == "__main__":
    zip_path = create_kaggle_source_zip()
    verify_zip_contents(zip_path)