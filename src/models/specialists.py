#!/usr/bin/env python3
"""
Specialist Model Architectures
Person 2: CM-Model (Compression) & RR-Model (Re-recording)
Person 3: LL-Model (Low-light) & TM-Model (Temporal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpecialistModel(nn.Module):
    """
    Base specialist model architecture
    Used by all specialist models with consistent interface
    """
    
    def __init__(self, num_classes=2, model_type="base"):
        super(SpecialistModel, self).__init__()
        
        self.model_type = model_type
        
        # ResNet18 backbone (consistent with team architecture)
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Replace final layer with specialist head
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class CompressionModel(SpecialistModel):
    """
    CM-Model: Compression Specialist
    Person 2's model for handling compressed videos (WhatsApp, Instagram, etc.)
    """
    
    def __init__(self, num_classes=2):
        super(CompressionModel, self).__init__(num_classes, "compression")
        
        # Add compression-specific preprocessing layers
        self.compression_adapter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize to [0,1]
        )
    
    def forward(self, x):
        # Apply compression adaptation
        x_adapted = self.compression_adapter(x)
        
        # Residual connection to preserve original information
        x_enhanced = x + 0.1 * x_adapted
        
        return self.backbone(x_enhanced)

class RerecordingModel(SpecialistModel):
    """
    RR-Model: Re-recording Specialist  
    Person 2's model for handling re-recorded/screen-captured videos
    """
    
    def __init__(self, num_classes=2):
        super(RerecordingModel, self).__init__(num_classes, "rerecording")
        
        # Add moiré pattern detection layers
        self.moire_detector = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Detect and compensate for moiré patterns
        moire_compensation = self.moire_detector(x)
        x_compensated = x - 0.05 * moire_compensation
        
        return self.backbone(x_compensated)

class LowLightModel(SpecialistModel):
    """
    LL-Model: Low-light Specialist
    Person 3's model for handling low-light/dark videos
    """
    
    def __init__(self, num_classes=2):
        super(LowLightModel, self).__init__(num_classes, "lowlight")
        
        # Add brightness enhancement layers
        self.brightness_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Noise reduction layers
        self.noise_reducer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Enhance brightness for low-light conditions
        brightness_enhanced = self.brightness_enhancer(x)
        
        # Apply enhancement with adaptive weighting
        mean_brightness = torch.mean(x, dim=[2, 3], keepdim=True)
        enhancement_weight = torch.clamp(1.0 - mean_brightness / 0.5, 0.0, 1.0)
        x_enhanced = x + enhancement_weight * brightness_enhanced
        
        # Reduce noise
        noise_reduction = self.noise_reducer(x_enhanced)
        x_final = x_enhanced - 0.1 * noise_reduction
        
        return self.backbone(x_final)

class TemporalModel(SpecialistModel):
    """
    TM-Model: Temporal Specialist
    Person 3's model for handling temporal inconsistencies
    """
    
    def __init__(self, num_classes=2, sequence_length=8):
        super(TemporalModel, self).__init__(num_classes, "temporal")
        
        self.sequence_length = sequence_length
        
        # Modify backbone to handle sequences
        # Remove the final FC layer to get features
        self.backbone.fc = nn.Identity()
        
        # Add temporal processing layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64*2 from bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] for temporal model or [B, C, H, W] for single frame
        """
        if len(x.shape) == 4:
            # Single frame input, treat as sequence of length 1
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        
        B, T, C, H, W = x.shape
        
        # Process each frame through backbone
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)  # [B*T, 512]
        features = features.view(B, T, 512)  # [B, T, 512]
        
        # Temporal convolution
        features_conv = features.transpose(1, 2)  # [B, 512, T]
        temporal_features = self.temporal_conv(features_conv)  # [B, 128, T]
        temporal_features = temporal_features.transpose(1, 2)  # [B, T, 128]
        
        # LSTM processing
        lstm_out, _ = self.lstm(temporal_features)  # [B, T, 128]
        
        # Aggregate temporal information (mean pooling)
        aggregated = torch.mean(lstm_out, dim=1)  # [B, 128]
        
        # Final classification
        output = self.classifier(aggregated)
        
        return output

# Factory functions for creating specialist models

def create_compression_model(num_classes=2):
    """Create CM-Model (Compression Specialist)"""
    return CompressionModel(num_classes)

def create_rerecording_model(num_classes=2):
    """Create RR-Model (Re-recording Specialist)"""
    return RerecordingModel(num_classes)

def create_lowlight_model(num_classes=2):
    """Create LL-Model (Low-light Specialist)"""
    return LowLightModel(num_classes)

def create_temporal_model(num_classes=2, sequence_length=8):
    """Create TM-Model (Temporal Specialist)"""
    return TemporalModel(num_classes, sequence_length)

def load_specialist_model(model_path, model_type, device='cpu'):
    """
    Load a specialist model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of specialist model ('cm', 'rr', 'll', 'tm')
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Create model based on type
    if model_type.lower() == 'cm':
        model = create_compression_model()
    elif model_type.lower() == 'rr':
        model = create_rerecording_model()
    elif model_type.lower() == 'll':
        model = create_lowlight_model()
    elif model_type.lower() == 'tm':
        model = create_temporal_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('best_acc', 'Unknown')
        print(f"[OK] Loaded {model_type.upper()}-Model: {accuracy}% accuracy")
    else:
        model.load_state_dict(checkpoint)
        print(f"[OK] Loaded {model_type.upper()}-Model (basic format)")
    
    model.to(device)
    model.eval()
    
    return model

if __name__ == "__main__":
    # Test all specialist models
    print("[TEST] Testing Specialist Models...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test single frame input
    test_input = torch.randn(2, 3, 224, 224).to(device)
    
    # Test sequence input for temporal model
    test_sequence = torch.randn(2, 8, 3, 224, 224).to(device)
    
    models_to_test = [
        ("CM-Model", create_compression_model()),
        ("RR-Model", create_rerecording_model()),
        ("LL-Model", create_lowlight_model()),
        ("TM-Model", create_temporal_model())
    ]
    
    for name, model in models_to_test:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            if name == "TM-Model":
                # Test both single frame and sequence
                output_single = model(test_input)
                output_sequence = model(test_sequence)
                print(f"[OK] {name}: Single frame {output_single.shape}, Sequence {output_sequence.shape}")
            else:
                output = model(test_input)
                print(f"[OK] {name}: {output.shape}")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")
    
    print("[OK] All specialist models working correctly!")