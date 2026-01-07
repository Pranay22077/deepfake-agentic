"""
DFDC METADATA INSPECTOR
Inspect the JSON metadata structure to understand how labels are stored
"""
import os
import json
from pathlib import Path
import pandas as pd

def inspect_dfdc_metadata(data_dir, chunk_idx=0):
    """Inspect DFDC metadata structure for a specific chunk"""
    
    print(f"🔍 INSPECTING DFDC METADATA FOR CHUNK {chunk_idx}")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Find chunk directory
    chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
    chunk_dir = data_dir / chunk_name
    
    if not chunk_dir.exists():
        print(f"❌ Chunk directory not found: {chunk_dir}")
        return None
    
    # Look for subdirectory structure
    subdir_name = f"dfdc_train_part_{chunk_idx}"
    subdir_path = chunk_dir / subdir_name
    
    if subdir_path.exists():
        chunk_path = subdir_path
        print(f"📂 Using subdirectory: {chunk_path}")
    else:
        chunk_path = chunk_dir
        print(f"📂 Using main directory: {chunk_path}")
    
    # Find metadata.json
    metadata_file = chunk_path / "metadata.json"
    
    if not metadata_file.exists():
        print(f"❌ No metadata.json found in {chunk_path}")
        return None
    
    print(f"✅ Found metadata file: {metadata_file}")
    
    # Load and inspect metadata
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"\n📊 METADATA ANALYSIS:")
        print(f"   Total entries: {len(metadata)}")
        
        # Show first few entries to understand structure
        print(f"\n📋 SAMPLE ENTRIES (first 5):")
        sample_keys = list(metadata.keys())[:5]
        
        for i, key in enumerate(sample_keys):
            entry = metadata[key]
            print(f"\n   Entry {i+1}: {key}")
            print(f"   Data: {entry}")
            
            # Analyze entry structure
            if isinstance(entry, dict):
                for field, value in entry.items():
                    print(f"     - {field}: {value} (type: {type(value).__name__})")
        
        # Analyze label distribution
        print(f"\n📈 LABEL ANALYSIS:")
        
        # Try different possible label field names
        label_fields = ['label', 'Label', 'LABEL', 'fake', 'real', 'is_fake', 'target']
        
        for field in label_fields:
            labels = []
            for key, entry in metadata.items():
                if isinstance(entry, dict) and field in entry:
                    labels.append(entry[field])
                elif isinstance(entry, str) and field == 'label':
                    # Sometimes the entry itself might be the label
                    labels.append(entry)
            
            if labels:
                print(f"\n   Field '{field}' found:")
                unique_labels = list(set(labels))
                print(f"   Unique values: {unique_labels}")
                
                # Count distribution
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                print(f"   Distribution:")
                for label, count in label_counts.items():
                    percentage = count / len(labels) * 100
                    print(f"     - {label}: {count} ({percentage:.1f}%)")
                
                return metadata, field, labels
        
        # If no standard label field found, show all possible fields
        print(f"\n   No standard label field found. Available fields:")
        all_fields = set()
        for entry in metadata.values():
            if isinstance(entry, dict):
                all_fields.update(entry.keys())
        
        print(f"   Fields across all entries: {sorted(all_fields)}")
        
        return metadata, None, None
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None

def analyze_video_files(chunk_path):
    """Analyze video files in the chunk directory"""
    
    print(f"\n🎬 VIDEO FILE ANALYSIS:")
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(chunk_path.glob(f"*{ext}")))
    
    print(f"   Total video files: {len(video_files)}")
    
    if video_files:
        print(f"   Sample filenames:")
        for i, video_file in enumerate(video_files[:5]):
            print(f"     {i+1}. {video_file.name}")
    
    return video_files

def main():
    """Main inspection function"""
    
    # Configuration - UPDATE THESE PATHS
    DATA_DIR = "/kaggle/input/dfdc-10"  # Update this path
    CHUNK_IDX = 0  # Start with chunk 0
    
    print("🔍 DFDC METADATA STRUCTURE INSPECTOR")
    print("="*60)
    print(f"📂 Data directory: {DATA_DIR}")
    print(f"📊 Inspecting chunk: {CHUNK_IDX}")
    print()
    
    # Inspect metadata
    result = inspect_dfdc_metadata(DATA_DIR, CHUNK_IDX)
    
    if result and result[0]:  # If metadata was loaded successfully
        metadata, label_field, labels = result
        
        # Also analyze video files
        data_dir = Path(DATA_DIR)
        chunk_name = f"dfdc_train_part_{CHUNK_IDX:02d}"
        chunk_dir = data_dir / chunk_name
        
        subdir_name = f"dfdc_train_part_{CHUNK_IDX}"
        subdir_path = chunk_dir / subdir_name
        
        chunk_path = subdir_path if subdir_path.exists() else chunk_dir
        video_files = analyze_video_files(chunk_path)
        
        # Cross-reference metadata with video files
        print(f"\n🔗 METADATA-VIDEO CROSS-REFERENCE:")
        
        video_names = {video.stem for video in video_files}
        metadata_names = set(metadata.keys())
        
        print(f"   Videos with metadata: {len(metadata_names & video_names)}")
        print(f"   Videos without metadata: {len(video_names - metadata_names)}")
        print(f"   Metadata without videos: {len(metadata_names - video_names)}")
        
        if len(video_names - metadata_names) > 0:
            print(f"   Sample videos without metadata:")
            for i, name in enumerate(list(video_names - metadata_names)[:3]):
                print(f"     - {name}")
        
        print(f"\n✅ INSPECTION COMPLETE!")
        print(f"📋 Summary:")
        print(f"   - Metadata entries: {len(metadata)}")
        print(f"   - Video files: {len(video_files)}")
        print(f"   - Label field: {label_field if label_field else 'NOT FOUND'}")
        
        if label_field and labels:
            real_count = labels.count('REAL') if 'REAL' in labels else labels.count(0) if 0 in labels else 0
            fake_count = labels.count('FAKE') if 'FAKE' in labels else labels.count(1) if 1 in labels else 0
            print(f"   - Real videos: {real_count}")
            print(f"   - Fake videos: {fake_count}")
    
    else:
        print(f"❌ Could not inspect metadata for chunk {CHUNK_IDX}")

if __name__ == "__main__":
    main()