"""
DEBUG DFDC METADATA - Find the filename mismatch issue
"""
import os
import json
from pathlib import Path

def debug_filename_matching(data_dir, chunk_idx=0):
    """Debug why video files and metadata don't match"""
    
    print(f"🔍 DEBUGGING FILENAME MATCHING FOR CHUNK {chunk_idx}")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Find chunk directory
    chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
    chunk_dir = data_dir / chunk_name
    
    subdir_name = f"dfdc_train_part_{chunk_idx}"
    subdir_path = chunk_dir / subdir_name
    
    chunk_path = subdir_path if subdir_path.exists() else chunk_dir
    print(f"📂 Chunk path: {chunk_path}")
    
    # Load metadata
    metadata_file = chunk_path / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Get video files
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(list(chunk_path.glob(f"*{ext}")))
    
    print(f"📊 Metadata entries: {len(metadata)}")
    print(f"🎬 Video files: {len(video_files)}")
    
    # Show sample metadata keys
    print(f"\n📋 SAMPLE METADATA KEYS:")
    metadata_keys = list(metadata.keys())[:10]
    for i, key in enumerate(metadata_keys):
        print(f"   {i+1}. '{key}'")
    
    # Show sample video filenames
    print(f"\n🎬 SAMPLE VIDEO FILENAMES:")
    for i, video_file in enumerate(video_files[:10]):
        print(f"   {i+1}. '{video_file.name}' (stem: '{video_file.stem}')")
    
    # Try different matching strategies
    print(f"\n🔍 TESTING MATCHING STRATEGIES:")
    
    # Strategy 1: Direct filename match
    direct_matches = 0
    for video_file in video_files:
        if video_file.name in metadata:
            direct_matches += 1
    print(f"   Strategy 1 - Direct filename: {direct_matches} matches")
    
    # Strategy 2: Stem match (without extension)
    stem_matches = 0
    for video_file in video_files:
        if video_file.stem in metadata:
            stem_matches += 1
    print(f"   Strategy 2 - Stem only: {stem_matches} matches")
    
    # Strategy 3: Add .mp4 to stems
    stem_mp4_matches = 0
    for video_file in video_files:
        key_with_ext = video_file.stem + ".mp4"
        if key_with_ext in metadata:
            stem_mp4_matches += 1
    print(f"   Strategy 3 - Stem + .mp4: {stem_mp4_matches} matches")
    
    # Find the best strategy
    strategies = [
        ("Direct filename", direct_matches),
        ("Stem only", stem_matches), 
        ("Stem + .mp4", stem_mp4_matches)
    ]
    
    best_strategy = max(strategies, key=lambda x: x[1])
    print(f"\n✅ BEST STRATEGY: {best_strategy[0]} with {best_strategy[1]} matches")
    
    # If we found a good strategy, analyze the distribution
    if best_strategy[1] > 0:
        print(f"\n📊 ANALYZING DISTRIBUTION WITH BEST STRATEGY:")
        
        real_count = 0
        fake_count = 0
        matched_videos = []
        
        for video_file in video_files:
            key = None
            
            if best_strategy[0] == "Direct filename":
                key = video_file.name
            elif best_strategy[0] == "Stem only":
                key = video_file.stem
            elif best_strategy[0] == "Stem + .mp4":
                key = video_file.stem + ".mp4"
            
            if key and key in metadata:
                label = metadata[key].get('label', 'UNKNOWN')
                matched_videos.append((video_file.name, label))
                
                if label == 'REAL':
                    real_count += 1
                elif label == 'FAKE':
                    fake_count += 1
        
        total_matched = len(matched_videos)
        print(f"   Total matched: {total_matched}")
        print(f"   Real videos: {real_count} ({real_count/total_matched*100:.1f}%)")
        print(f"   Fake videos: {fake_count} ({fake_count/total_matched*100:.1f}%)")
        
        # Show some examples
        print(f"\n📋 SAMPLE MATCHED VIDEOS:")
        for i, (filename, label) in enumerate(matched_videos[:10]):
            print(f"   {i+1}. {filename} -> {label}")
    
    return best_strategy

def check_multiple_chunks(data_dir, max_chunks=3):
    """Check multiple chunks to see if distribution varies"""
    
    print(f"\n🔍 CHECKING MULTIPLE CHUNKS FOR DISTRIBUTION")
    print("="*60)
    
    total_real = 0
    total_fake = 0
    
    for chunk_idx in range(max_chunks):
        print(f"\n📊 CHUNK {chunk_idx}:")
        
        try:
            strategy = debug_filename_matching(data_dir, chunk_idx)
            
            # Quick count for this chunk
            data_dir_path = Path(data_dir)
            chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
            chunk_dir = data_dir_path / chunk_name
            
            subdir_name = f"dfdc_train_part_{chunk_idx}"
            subdir_path = chunk_dir / subdir_name
            
            chunk_path = subdir_path if subdir_path.exists() else chunk_dir
            
            if not (chunk_path / "metadata.json").exists():
                print(f"   ❌ No metadata found")
                continue
            
            with open(chunk_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            
            chunk_real = sum(1 for entry in metadata.values() if entry.get('label') == 'REAL')
            chunk_fake = sum(1 for entry in metadata.values() if entry.get('label') == 'FAKE')
            
            total_real += chunk_real
            total_fake += chunk_fake
            
            print(f"   Real: {chunk_real}, Fake: {chunk_fake}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📈 OVERALL DISTRIBUTION ACROSS {max_chunks} CHUNKS:")
    total_videos = total_real + total_fake
    if total_videos > 0:
        print(f"   Total videos: {total_videos}")
        print(f"   Real: {total_real} ({total_real/total_videos*100:.1f}%)")
        print(f"   Fake: {total_fake} ({total_fake/total_videos*100:.1f}%)")
        
        if abs(total_real/total_videos - 0.5) < 0.1:  # Within 10% of 50%
            print(f"   ✅ Distribution is roughly balanced!")
        else:
            print(f"   ⚠️ Distribution is imbalanced")

def main():
    """Main debug function"""
    
    DATA_DIR = "/kaggle/input/dfdc-10"
    
    print("🔍 DFDC METADATA DEBUG TOOL")
    print("="*60)
    print("🎯 Goal: Find why video files don't match metadata keys")
    print("📊 Expected: 50:50 real/fake distribution")
    print()
    
    # Debug chunk 0 in detail
    debug_filename_matching(DATA_DIR, 0)
    
    # Check multiple chunks
    check_multiple_chunks(DATA_DIR, 3)
    
    print(f"\n✅ DEBUG COMPLETE!")

if __name__ == "__main__":
    main()