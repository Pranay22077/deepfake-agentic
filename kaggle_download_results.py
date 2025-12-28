#!/usr/bin/env python3
"""
Kaggle Results Download Script
Downloads trained models and results from Kaggle after training completion
"""

import os
import sys
import json
import requests
import zipfile
from pathlib import Path
import argparse
import shutil

def download_kaggle_results(dataset_name, output_dir="models", api_key=None):
    """
    Download results from Kaggle dataset
    
    Args:
        dataset_name: Kaggle dataset name (e.g., "username/dataset-name")
        output_dir: Local directory to save results
        api_key: Kaggle API key (optional, uses kaggle.json if not provided)
    """
    
    print(f"Downloading results from Kaggle dataset: {dataset_name}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use Kaggle API to download
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        print("Downloading dataset files...")
        api.dataset_download_files(
            dataset_name, 
            path=output_dir, 
            unzip=True
        )
        
        print(f"Results downloaded to: {output_dir}")
        
        # List downloaded files
        downloaded_files = list(Path(output_dir).glob("*"))
        print(f"Downloaded {len(downloaded_files)} files:")
        for file in downloaded_files:
            print(f"  - {file.name}")
            
        return True
        
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        return False

def download_from_url(url, output_path):
    """Download file from direct URL"""
    print(f"Downloading from URL: {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return False

def extract_models(source_dir, target_dir="models"):
    """Extract model files from downloaded results"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Look for model files
    model_extensions = ['.pt', '.pth', '.onnx', '.torchscript']
    model_files = []
    
    for ext in model_extensions:
        model_files.extend(source_path.glob(f"**/*{ext}"))
    
    print(f"Found {len(model_files)} model files:")
    
    for model_file in model_files:
        target_file = target_path / model_file.name
        shutil.copy2(model_file, target_file)
        print(f"  Copied: {model_file.name} -> {target_file}")
    
    return model_files

def main():
    parser = argparse.ArgumentParser(description='Download Kaggle Training Results')
    parser.add_argument('--dataset', help='Kaggle dataset name (username/dataset-name)')
    parser.add_argument('--url', help='Direct download URL')
    parser.add_argument('--output', default='models', help='Output directory')
    parser.add_argument('--extract', action='store_true', help='Extract model files only')
    
    args = parser.parse_args()
    
    if args.dataset:
        # Download from Kaggle dataset
        success = download_kaggle_results(args.dataset, args.output)
        
    elif args.url:
        # Download from direct URL
        filename = args.url.split('/')[-1]
        if not filename.endswith(('.zip', '.tar.gz', '.pt', '.pth')):
            filename = 'download.zip'
        
        output_path = os.path.join(args.output, filename)
        success = download_from_url(args.url, output_path)
        
        # Extract if it's an archive
        if success and filename.endswith('.zip'):
            print("Extracting archive...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(args.output)
            print("Extraction complete")
            
    else:
        print("Please provide either --dataset or --url")
        return
    
    if success and args.extract:
        print("\nExtracting model files...")
        extract_models(args.output)
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()