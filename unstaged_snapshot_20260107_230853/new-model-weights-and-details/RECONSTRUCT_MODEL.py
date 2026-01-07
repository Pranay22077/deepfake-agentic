"""
Reconstruct the model from chunks
"""
import os
from pathlib import Path

def reconstruct_model():
    chunks = []
    chunk_num = 0
    
    while True:
        chunk_path = Path(f"model_chunk_{chunk_num:03d}.bin")
        if not chunk_path.exists():
            break
        
        with open(chunk_path, 'rb') as f:
            chunks.append(f.read())
        chunk_num += 1
    
    # Reconstruct original file
    with open("stage2_celebdf_complete_20260105_211721.zip", 'wb') as f:
        for chunk in chunks:
            f.write(chunk)
    
    print(f"✅ Reconstructed {len(chunks)} chunks into stage2_celebdf_complete_20260105_211721.zip")

if __name__ == "__main__":
    reconstruct_model()
