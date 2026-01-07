"""
COMPREHENSIVE DATASET INSPECTOR
Inspect all 4 datasets to understand their structure and find real/fake labels
"""

import os
import json
import csv
import pandas as pd
from pathlib import Path
from collections import defaultdict

def inspect_faceforensics(data_dir):
    """Inspect FaceForensics++ dataset structure"""
    
    print(f"🔍 INSPECTING FACEFORENSICS++ DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    print(f"📂 Base directory: {data_dir}")
    
    # List all subdirectories
    if data_dir.exists():
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"📁 Found {len(subdirs)} subdirectories:")
        for subdir in subdirs:
            print(f"   - {subdir.name}")
        
        # Look for CSV folder specifically
        csv_folder = data_dir / 'csv'
        if csv_folder.exists():
            print(f"\n📋 Found CSV folder: {csv_folder}")
            csv_files = list(csv_folder.glob('*.csv'))
            print(f"📄 CSV files found: {len(csv_files)}")
            
            for csv_file in csv_files:
                print(f"\n📊 Analyzing: {csv_file.name}")
                try:
                    df = pd.read_csv(csv_file)
                    print(f"   Columns: {list(df.columns)}")
                    print(f"   Rows: {len(df)}")
                    
                    # Show sample data
                    print(f"   Sample data:")
                    for i, row in df.head(3).iterrows():
                        print(f"     {dict(row)}")
                    
                    # Analyze labels if present
                    if 'label' in df.columns:
                        label_counts = df['label'].value_counts()
                        print(f"   Label distribution: {dict(label_counts)}")
                    
                except Exception as e:
                    print(f"   ❌ Error reading CSV: {e}")
        
        # Look for video directories
        print(f"\n🎬 Looking for video directories...")
        video_dirs = []
        for subdir in subdirs:
            video_files = []
            for ext in ['.mp4', '.avi', '.mov']:
                video_files.extend(list(subdir.rglob(f'*{ext}')))
            
            if video_files:
                video_dirs.append((subdir.name, len(video_files)))
                print(f"   📹 {subdir.name}: {len(video_files)} videos")
        
        return {
            'csv_files': csv_files if 'csv_files' in locals() else [],
            'video_dirs': video_dirs,
            'structure': 'csv_based' if csv_folder.exists() else 'folder_based'
        }
    
    else:
        print(f"❌ Directory not found: {data_dir}")
        return None

def inspect_celebdf(data_dir):
    """Inspect Celeb-DF v2 dataset structure"""
    
    print(f"\n🔍 INSPECTING CELEB-DF V2 DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    print(f"📂 Base directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # List all files and directories
    items = list(data_dir.iterdir())
    print(f"📁 Found {len(items)} items:")
    
    txt_files = []
    video_dirs = []
    
    for item in items:
        if item.is_file() and item.suffix == '.txt':
            txt_files.append(item)
            print(f"   📄 {item.name}")
        elif item.is_dir():
            video_count = 0
            for ext in ['.mp4', '.avi', '.mov']:
                video_count += len(list(item.glob(f'*{ext}')))
            
            if video_count > 0:
                video_dirs.append((item.name, video_count))
                print(f"   📹 {item.name}: {video_count} videos")
            else:
                print(f"   📁 {item.name}: (empty or no videos)")
    
    # Analyze TXT files
    for txt_file in txt_files:
        print(f"\n📊 Analyzing: {txt_file.name}")
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            print(f"   Total lines: {len(lines)}")
            
            # Analyze format
            sample_lines = [line.strip() for line in lines[:5] if line.strip()]
            print(f"   Sample lines:")
            for line in sample_lines:
                print(f"     {line}")
            
            # Count labels if format is "label filename"
            if sample_lines:
                label_counts = defaultdict(int)
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split(' ', 1)
                        if len(parts) >= 1:
                            label = parts[0]
                            label_counts[label] += 1
                
                print(f"   Label distribution: {dict(label_counts)}")
        
        except Exception as e:
            print(f"   ❌ Error reading TXT: {e}")
    
    return {
        'txt_files': txt_files,
        'video_dirs': video_dirs,
        'structure': 'txt_based'
    }

def inspect_wilddeepfake(data_dir):
    """Inspect Wild Deepfake dataset structure"""
    
    print(f"\n🔍 INSPECTING WILD DEEPFAKE DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    print(f"📂 Base directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Look for folder structure
    splits = ['test', 'train', 'valid']
    structure = {}
    
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            print(f"\n📁 Found {split} directory")
            
            # Look for real/fake subdirectories
            real_dir = split_dir / 'real'
            fake_dir = split_dir / 'fake'
            
            real_count = 0
            fake_count = 0
            
            if real_dir.exists():
                for ext in ['.mp4', '.avi', '.mov']:
                    real_count += len(list(real_dir.glob(f'*{ext}')))
                print(f"   📹 Real videos: {real_count}")
            
            if fake_dir.exists():
                for ext in ['.mp4', '.avi', '.mov']:
                    fake_count += len(list(fake_dir.glob(f'*{ext}')))
                print(f"   📹 Fake videos: {fake_count}")
            
            structure[split] = {'real': real_count, 'fake': fake_count}
    
    return {
        'structure': 'folder_based',
        'splits': structure
    }

def inspect_dfdc(data_dir):
    """Inspect DFDC dataset structure"""
    
    print(f"\n🔍 INSPECTING DFDC DATASET")
    print("="*60)
    
    data_dir = Path(data_dir)
    print(f"📂 Base directory: {data_dir}")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return None
    
    # Look for chunk directories
    chunks_found = []
    
    for i in range(50):  # Check up to 50 chunks
        chunk_name = f"dfdc_train_part_{i:02d}"
        chunk_dir = data_dir / chunk_name
        
        if chunk_dir.exists():
            print(f"\n📁 Found chunk: {chunk_name}")
            
            # Check for subdirectory structure
            subdir_name = f"dfdc_train_part_{i}"
            subdir_path = chunk_dir / subdir_name
            
            actual_path = subdir_path if subdir_path.exists() else chunk_dir
            
            # Look for metadata.json
            metadata_file = actual_path / "metadata.json"
            video_count = 0
            
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_count += len(list(actual_path.glob(f'*{ext}')))
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    real_count = sum(1 for entry in metadata.values() if entry.get('label') == 'REAL')
                    fake_count = sum(1 for entry in metadata.values() if entry.get('label') == 'FAKE')
                    
                    print(f"   📊 Metadata entries: {len(metadata)}")
                    print(f"   📹 Video files: {video_count}")
                    print(f"   📈 Real: {real_count} ({real_count/len(metadata)*100:.1f}%)")
                    print(f"   📈 Fake: {fake_count} ({fake_count/len(metadata)*100:.1f}%)")
                    
                    chunks_found.append({
                        'chunk': i,
                        'metadata_entries': len(metadata),
                        'video_files': video_count,
                        'real': real_count,
                        'fake': fake_count
                    })
                
                except Exception as e:
                    print(f"   ❌ Error reading metadata: {e}")
            else:
                print(f"   ❌ No metadata.json found")
    
    return {
        'structure': 'json_based',
        'chunks': chunks_found
    }

def main():
    """Main inspection function"""
    
    print("🔍 COMPREHENSIVE DATASET STRUCTURE INSPECTOR")
    print("="*80)
    print("🎯 Goal: Understand all 4 datasets before training")
    print("📊 Find real vs fake distribution for each dataset")
    print()
    
    # Dataset paths (update these to match your Kaggle setup)
    datasets = {
        'FaceForensics++': '/kaggle/input/ff-c23',
        'Celeb-DF v2': '/kaggle/input/celeb-df-v2',
        'Wild Deepfake': '/kaggle/input/wild-deepfake',
        'DFDC': '/kaggle/input/dfdc-10-deepfake-detection'
    }
    
    results = {}
    
    # Inspect each dataset
    results['faceforensics'] = inspect_faceforensics(datasets['FaceForensics++'])
    results['celebdf'] = inspect_celebdf(datasets['Celeb-DF v2'])
    results['wilddeepfake'] = inspect_wilddeepfake(datasets['Wild Deepfake'])
    results['dfdc'] = inspect_dfdc(datasets['DFDC'])
    
    # Summary
    print(f"\n📋 INSPECTION SUMMARY")
    print("="*80)
    
    for dataset_name, result in results.items():
        if result:
            print(f"\n✅ {dataset_name.upper()}:")
            print(f"   Structure: {result.get('structure', 'unknown')}")
            
            if dataset_name == 'dfdc' and 'chunks' in result:
                total_real = sum(chunk['real'] for chunk in result['chunks'])
                total_fake = sum(chunk['fake'] for chunk in result['chunks'])
                total_videos = total_real + total_fake
                
                print(f"   Chunks: {len(result['chunks'])}")
                print(f"   Total videos: {total_videos}")
                print(f"   Real: {total_real} ({total_real/total_videos*100:.1f}%)")
                print(f"   Fake: {total_fake} ({total_fake/total_videos*100:.1f}%)")
            
            elif dataset_name == 'wilddeepfake' and 'splits' in result:
                for split, counts in result['splits'].items():
                    total = counts['real'] + counts['fake']
                    if total > 0:
                        print(f"   {split}: {total} videos ({counts['real']} real, {counts['fake']} fake)")
        else:
            print(f"\n❌ {dataset_name.upper()}: Not found or error")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Update dataset loaders based on findings")
    print(f"   2. Fix label extraction for each format")
    print(f"   3. Ensure balanced training data")
    print(f"   4. Start progressive training")

if __name__ == "__main__":
    main()