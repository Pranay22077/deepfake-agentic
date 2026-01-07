"""
Compare LL Models: Old ResNet18 vs New EfficientNet-B4
Evaluate both models on test data to see improvement
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ===== OLD MODEL ARCHITECTURE (ResNet18-based) =====
class OldLowLightModel(nn.Module):
    """
    Old LL-Model: ResNet18-based Low-light Specialist
    Matches the actual trained model architecture
    """
    
    def __init__(self, num_classes=2):
        super(OldLowLightModel, self).__init__()
        
        # ResNet18 backbone (matches trained models)
        self.backbone = models.resnet18(weights=None)  # Don't load pretrained to avoid conflicts
        
        # Replace final layer with th