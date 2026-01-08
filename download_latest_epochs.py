"""
E-Raksha Checkpoint Download Utility

Automated checkpoint collection and packaging system.
Finds all model checkpoints and creates organized download packages.

Author: E-Raksha Team
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import glob
import zipfile

def find_and_zip_all_checkpoints():
    """
    Find all checkpoint files and create organized download package.
    
    Returns:
        str: Path to created zip file
    """
    print("🔍 COLLECTING ALL CHECKPOINT FILES")
    print("="*60)
    
    # Checkpoint search directories
    search_dirs = [
        "/kaggle/working",
        "/kaggle/working/checkpoints", 
        "/kaggle/working/downloads"
    ]
    
    # Collect all checkpoint files
    all_checkpoints = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"📂 Searching: {search_dir}")
            
            # Find model files (.pt and .zip)
            pt_files = glob.glob(os.path.join(search_dir, "*.pt"))
            zip_files = glob.glob(os.path.join(search_dir, "*.zip"))
            
            all_files = pt_files + zip_files
            
            for file_path_str in all_files:
                file_path = Path(file_path_str)
                
                # Skip output zip files
                if file_path.name.startswith("ALL_CHECKPOINTS_"):
                    continue
                
                try:
                    file_stat = file_path.stat()
                    
                    all_checkpoints.append({
                        'path': file_path,
                        'name': file_path.name,
                        'size_mb': file_stat.st_size / (1024 * 1024),
                        'modified_time': file_stat.st_mtime,
                        'modified_datetime': datetime.fromtimestamp(file_stat.st_mtime),
                        'type': file_path.suffix
                    })
                except Exception as e:
                    print(f"   ⚠️ Error reading {file_path.name}: {e}")
    
    if not all_checkpoints:
        print("❌ No checkpoint files (.pt or .zip) found!")
        print("\n📋 Let's check what files exist:")
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                print(f"\n📂 {search_dir}:")
                try:
                    files = os.listdir(search_dir)
                    if files:
                        for f in files[:20]:  # Show first 20 files
                            print(f"   • {f}")
                    else:
                        print("   (empty)")
                except Exception as e:
                    print(f"   Error: {e}")
        return
    
    print(f"\n📊 Found {len(all_checkpoints)} checkpoint files")
    
    # Sort by modification time (newest first)
    all_checkpoints.sort(key=lambda x: x['modified_time'], reverse=True)
    
    # Show all found checkpoints
    print(f"\n📋 ALL CHECKPOINTS (sorted by time):")
    total_size = 0
    for i, ckpt in enumerate(all_checkpoints, 1):
        print(f"   {i}. {ckpt['name']} ({ckpt['type']})")
        print(f"      Size: {ckpt['size_mb']:.1f} MB")
        print(f"      Modified: {ckpt['modified_datetime']}")
        total_size += ckpt['size_mb']
    
    print(f"\n📊 Total size: {total_size:.1f} MB")
    
    # Get latest 2
    latest_2 = all_checkpoints[:2]
    
    print(f"\n🎯 LATEST 2 CHECKPOINTS:")
    for i, ckpt in enumerate(latest_2, 1):
        print(f"   {i}. {ckpt['name']} ({ckpt['size_mb']:.1f} MB)")
        print(f"      Modified: {ckpt['modified_datetime']}")
    
    # Create zip file with all checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"ALL_CHECKPOINTS_{timestamp}.zip"
    zip_path = Path("/kaggle/working") / zip_filename
    
    print(f"\n📦 CREATING ZIP FILE...")
    print(f"   Output: {zip_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, ckpt in enumerate(all_checkpoints, 1):
                src_path = ckpt['path']
                
                print(f"\n   {i}/{len(all_checkpoints)} Adding: {ckpt['name']} ({ckpt['size_mb']:.1f} MB)")
                
                try:
                    # Add file to zip with just the filename (no directory structure)
                    zipf.write(src_path, arcname=ckpt['name'])
                    print(f"      ✅ Added successfully")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
        
        # Verify zip was created
        if zip_path.exists():
            zip_size = zip_path.stat().st_size / (1024 * 1024)
            print(f"\n{'='*60}")
            print(f"✅ ZIP FILE CREATED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"\n📦 File: {zip_filename}")
            print(f"📊 Size: {zip_size:.1f} MB")
            print(f"📁 Location: /kaggle/working/{zip_filename}")
            print(f"📋 Contains: {len(all_checkpoints)} checkpoint files")
            
            print(f"\n💡 DOWNLOAD INSTRUCTIONS:")
            print(f"   1. In Kaggle, click on 'Output' tab")
            print(f"   2. Look for: {zip_filename}")
            print(f"   3. Click download button")
            print(f"   4. Extract the zip to get all checkpoint files")
            
            print(f"\n📋 FILES IN ZIP:")
            for ckpt in all_checkpoints:
                print(f"   • {ckpt['name']} ({ckpt['size_mb']:.1f} MB)")
            
            return zip_path
        else:
            print(f"\n❌ Failed to create zip file!")
            return None
            
    except Exception as e:
        print(f"\n❌ Error creating zip: {e}")
        return None

if __name__ == "__main__":
    find_and_zip_all_checkpoints()
