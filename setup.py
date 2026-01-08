#!/usr/bin/env python3
"""
E-Raksha Setup Script

Simple setup script for initial development environment configuration.
Handles basic directory creation and environment validation.

Author: E-Raksha Team
Created: Initial development phase
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'models/checkpoints',
        'logs',
        'outputs',
        'test-videos'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_dependencies():
    """Check if core dependencies are available"""
    try:
        import torch
        import cv2
        import numpy as np
        print("✓ Core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False

def main():
    """Main setup function"""
    print("E-Raksha Development Environment Setup")
    print("=" * 40)
    
    create_directories()
    
    if check_dependencies():
        print("\n✓ Setup completed successfully!")
    else:
        print("\n✗ Setup incomplete - install requirements first")
        print("Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()