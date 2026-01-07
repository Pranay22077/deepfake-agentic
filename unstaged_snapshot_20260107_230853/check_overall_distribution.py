"""
CHECK OVERALL DFDC DISTRIBUTION
Check if the 50:50 distribution is across all chunks combined
"""
import json
from pathlib import Path

def check_all_chunks_distribution(data_dir):
    """Check distribution across all available chunks"""
    
    print(f"📊 CHECKING OVERALL DFDC DISTRIBUTION")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    total_real = 0
    total_fake = 0
    total_chunks_found = 0
    chunk_details = []
    
    # Check chunks 0-49 (DFDC can have up to 50 chunks)
    for chunk_idx in range(50):
        chunk_name = f"dfdc_train_part_{chunk_idx:02d}"
        chunk_dir = data_dir / chunk_name
        
        if not chunk_dir.exists():
            continue
        
        # Check for subdirectory structure
        subdir_name = f"dfdc_train_part_{chunk_idx}"
        subdir_path = chunk_dir / subdir_name
        
        chunk_path = subdir_path if subdir_path.exists() else chunk_dir
        metadata_file = chunk_path / "metadata.json"
        
        if not metadata_file.exists():
            continue
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            chunk_real = sum(1 for entry in metadata.values() if entry.get('label') == 'REAL')
            chunk_fake = sum(1 for entry in metadata.values() if entry.get('label') == 'FAKE')
            chunk_total = len(metadata)
            
            total_real += chunk_real
            total_fake += chunk_fake
            total_chunks_found += 1
            
            chunk_details.append({
                'chunk': chunk_idx,
                'real': chunk_real,
                'fake': chunk_fake,
                'total': chunk_total,
                'real_pct': chunk_real / chunk_total * 100 if chunk_total > 0 else 0
            })
            
            print(f"Chunk {chunk_idx:2d}: {chunk_real:4d} real, {chunk_fake:4d} fake, {chunk_total:4d} total ({chunk_real/chunk_total*100:5.1f}% real)")
            
        except Exception as e:
            print(f"Chunk {chunk_idx:2d}: Error - {e}")
    
    print(f"\n📈 OVERALL SUMMARY:")
    print(f"   Chunks found: {total_chunks_found}")
    print(f"   Total videos: {total_real + total_fake}")
    print(f"   Real videos: {total_real}")
    print(f"   Fake videos: {total_fake}")
    
    if total_real + total_fake > 0:
        overall_real_pct = total_real / (total_real + total_fake) * 100
        overall_fake_pct = total_fake / (total_real + total_fake) * 100
        
        print(f"   Real percentage: {overall_real_pct:.2f}%")
        print(f"   Fake percentage: {overall_fake_pct:.2f}%")
        
        # Check if it's balanced
        if abs(overall_real_pct - 50.0) < 5.0:  # Within 5% of 50%
            print(f"   ✅ BALANCED: Distribution is roughly 50:50!")
        else:
            print(f"   ⚠️ IMBALANCED: Distribution is not 50:50")
        
        # Show most balanced chunks
        print(f"\n🎯 MOST BALANCED CHUNKS:")
        balanced_chunks = sorted(chunk_details, key=lambda x: abs(x['real_pct'] - 50.0))
        
        for chunk in balanced_chunks[:5]:
            print(f"   Chunk {chunk['chunk']:2d}: {chunk['real_pct']:5.1f}% real ({chunk['real']:3d}/{chunk['total']:3d})")
        
        # Show most imbalanced chunks
        print(f"\n⚠️ MOST IMBALANCED CHUNKS:")
        imbalanced_chunks = sorted(chunk_details, key=lambda x: abs(x['real_pct'] - 50.0), reverse=True)
        
        for chunk in imbalanced_chunks[:5]:
            print(f"   Chunk {chunk['chunk']:2d}: {chunk['real_pct']:5.1f}% real ({chunk['real']:3d}/{chunk['total']:3d})")

def main():
    DATA_DIR = "/kaggle/input/dfdc-10"
    
    print("📊 DFDC OVERALL DISTRIBUTION CHECKER")
    print("="*60)
    print("🎯 Checking if 50:50 distribution exists across all chunks")
    print()
    
    check_all_chunks_distribution(DATA_DIR)

if __name__ == "__main__":
    main()