#!/usr/bin/env python3
"""
Step 1 Verification Script
Check if all Step 1 outputs are ready for Step 2
"""

import os
import json
import torch
import zipfile
from pathlib import Path

def verify_step1_outputs(extract_path="./kaggle_outputs"):
    """Verify Step 1 completion"""
    print("Step 1 Verification")
    print("=" * 50)
    
    # Check for required files (model can be .pt or .pkl)
    required_files = {
        'training_info.json': 'Training metrics and config',
        'README.txt': 'Training summary',
        'download_summary.json': 'Download metadata'
    }
    
    # Check for model file (either .pt or .pkl)
    model_files = {
        'baseline_student.pt': 'Trained model weights (PyTorch format)',
        'baseline_student.pkl': 'Trained model weights (Pickle format)'
    }
    
    verification_results = {}
    
    # Check if extraction path exists
    if not os.path.exists(extract_path):
        print(f"Error: {extract_path} not found")
        print("Please extract your kaggle_outputs_20251228_043850.zip first")
        return False
    
    print(f"Checking files in: {extract_path}")
    print("-" * 30)
    
    # Verify each required file
    for filename, description in required_files.items():
        filepath = os.path.join(extract_path, filename)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"[OK] {filename} - {description} ({file_size:.1f}MB)")
            verification_results[filename] = True
        else:
            print(f"[MISSING] {filename} - {description}")
            verification_results[filename] = False
    
    # Check for model file (either format)
    model_found = None
    for filename, description in model_files.items():
        filepath = os.path.join(extract_path, filename)
        
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"[OK] {filename} - {description} ({file_size:.1f}MB)")
            verification_results['model_file'] = True
            model_found = filepath
            break
    
    if not model_found:
        print(f"[MISSING] Model file - No baseline_student.pt or baseline_student.pkl found")
        verification_results['model_file'] = False
    
    # Check model file specifically
    if model_found:
        try:
            if model_found.endswith('.pt'):
                # Try to load PyTorch model
                checkpoint = torch.load(model_found, map_location='cpu')
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        print(f"[OK] Model format: Full PyTorch checkpoint with metadata")
                        if 'config' in checkpoint:
                            config = checkpoint['config']
                            print(f"     Architecture: {config.get('model_type', 'Unknown')}")
                            print(f"     Input size: {config.get('input_size', 'Unknown')}")
                    else:
                        print(f"[OK] Model format: PyTorch state dict only")
                else:
                    print(f"[WARNING] Model format: Unknown PyTorch structure")
                    
            elif model_found.endswith('.pkl'):
                # Try to load pickle model
                import pickle
                with open(model_found, 'rb') as f:
                    model_data = pickle.load(f)
                
                if isinstance(model_data, dict):
                    param_count = len(model_data)
                    print(f"[OK] Model format: Pickle file with {param_count} parameters")
                    print(f"     Architecture: ResNet18 (from training info)")
                    print(f"     Format: Parameter dictionary")
                else:
                    print(f"[WARNING] Model format: Unknown pickle structure")
                
        except Exception as e:
            print(f"[ERROR] Cannot load model: {e}")
            verification_results['model_loadable'] = False
        else:
            verification_results['model_loadable'] = True
    
    # Check training info
    info_path = os.path.join(extract_path, 'training_info.json')
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            print(f"[OK] Training Info:")
            print(f"     Best Accuracy: {info.get('best_accuracy', 'Unknown'):.2f}%")
            print(f"     Dataset: {info.get('real_faces', 0)} real + {info.get('fake_faces', 0)} fake")
            print(f"     Architecture: {info.get('model_info', {}).get('architecture', 'Unknown')}")
            print(f"     Parameters: {info.get('model_info', {}).get('parameters', 'Unknown'):,}")
            
        except Exception as e:
            print(f"[ERROR] Cannot read training info: {e}")
    
    # Check for processed data
    processed_data_path = os.path.join(extract_path, 'processed_data')
    if os.path.exists(processed_data_path):
        real_faces = len([f for f in os.listdir(os.path.join(processed_data_path, 'real')) 
                         if f.endswith('.jpg')]) if os.path.exists(os.path.join(processed_data_path, 'real')) else 0
        fake_faces = len([f for f in os.listdir(os.path.join(processed_data_path, 'fake')) 
                         if f.endswith('.jpg')]) if os.path.exists(os.path.join(processed_data_path, 'fake')) else 0
        
        print(f"[OK] Processed Data: {real_faces} real + {fake_faces} fake face images")
        verification_results['processed_data'] = True
    else:
        print(f"[INFO] No processed data folder (normal if using different preprocessing)")
        verification_results['processed_data'] = False
    
    print("-" * 50)
    
    # Overall assessment
    critical_files = ['training_info.json']
    critical_passed = all(verification_results.get(f, False) for f in critical_files)
    model_exists = verification_results.get('model_file', False)
    model_loadable = verification_results.get('model_loadable', False)
    
    if critical_passed and model_exists and model_loadable:
        print("STEP 1 VERIFICATION: PASSED")
        print("Ready to proceed with Step 2!")
        print(f"Model file: {os.path.basename(model_found)}")
        return True
    else:
        print("STEP 1 VERIFICATION: FAILED")
        if not model_exists:
            print("- Missing model file (.pt or .pkl)")
        if not model_loadable:
            print("- Model cannot be loaded")
        if not critical_passed:
            print("- Missing critical metadata files")
        return False

def extract_kaggle_zip(zip_path):
    """Extract Kaggle outputs ZIP file"""
    if not os.path.exists(zip_path):
        print(f"ZIP file not found: {zip_path}")
        return False
    
    extract_path = "./kaggle_outputs"
    
    print(f"Extracting {zip_path} to {extract_path}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extraction complete: {extract_path}")
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def main():
    """Main verification function"""
    print("E-Raksha Step 1 Verification")
    print("=" * 50)
    
    # Look for ZIP file
    zip_candidates = [
        "kaggle_outputs_20251228_043850.zip",
        "kaggle_outputs.zip"
    ]
    
    zip_found = None
    for zip_name in zip_candidates:
        if os.path.exists(zip_name):
            zip_found = zip_name
            break
    
    if zip_found:
        print(f"Found ZIP file: {zip_found}")
        if extract_kaggle_zip(zip_found):
            verify_step1_outputs()
        else:
            print("Failed to extract ZIP file")
    else:
        print("ZIP file not found, checking for extracted files...")
        verify_step1_outputs()

if __name__ == "__main__":
    main()