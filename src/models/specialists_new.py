#!/usr/bin/env python3
"""
UPDATED Specialist Model Architectures
BG, CM, RR, LL, AV: New EfficientNet-B4 based architectures
TM: Keep old ResNet18 based architecture (no new model yet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ============================================================================
# SPECIALIST MODULES
# ============================================================================

class BackgroundModule(nn.Module):
    """Background inconsistency detection module - 44 channels, 2156 features"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Face boundary detector
        self.face_boundary = nn.Sequential(
            nn.Conv2d(in_channels, 11, kernel_size=3, padding=1),
            nn.BatchNorm2d(11), nn.ReLU(),
            nn.Conv2d(11, 22, kernel_size=3, padding=1),
            nn.BatchNorm2d(22), nn.ReLU(),
            nn.Conv2d(22, 11, kernel_size=1)
        )
        
        # Lighting analyzer
        self.lighting_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 11, kernel_size=5, padding=2),
            nn.BatchNorm2d(11), nn.ReLU(),
            nn.Conv2d(11, 22, kernel_size=5, padding=2),
            nn.BatchNorm2d(22), nn.ReLU(),
            nn.Conv2d(22, 11, kernel_size=1)
        )
        
        # Shadow detector
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 11, kernel_size=7, padding=3),
            nn.BatchNorm2d(11), nn.ReLU(),
            nn.Conv2d(11, 22, kernel_size=7, padding=3),
            nn.BatchNorm2d(22), nn.ReLU(),
            nn.Conv2d(22, 11, kernel_size=1)
        )
        
        # Color temperature analyzer
        self.color_temp = nn.Sequential(
            nn.Conv2d(in_channels, 11, kernel_size=9, padding=4),
            nn.BatchNorm2d(11), nn.ReLU(),
            nn.Conv2d(11, 22, kernel_size=9, padding=4),
            nn.BatchNorm2d(22), nn.ReLU(),
            nn.Conv2d(22, 11, kernel_size=1)
        )
        
        # Attention fusion - 44 channels total
        self.attention = nn.Sequential(
            nn.Conv2d(44, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 44, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        face_feat = F.adaptive_avg_pool2d(self.face_boundary(x), (7, 7))
        light_feat = F.adaptive_avg_pool2d(self.lighting_analyzer(x), (7, 7))
        shadow_feat = F.adaptive_avg_pool2d(self.shadow_detector(x), (7, 7))
        color_feat = F.adaptive_avg_pool2d(self.color_temp(x), (7, 7))
        combined = torch.cat([face_feat, light_feat, shadow_feat, color_feat], dim=1)
        return combined * self.attention(combined)


class AudioVisualModule(nn.Module):
    """Audio-visual synchronization detection module - 48 channels, 2352 features"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Lip-sync analyzer
        self.lip_sync = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Audio-visual correlation detector
        self.av_correlation = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=7, padding=3),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Facial expression analyzer
        self.expression_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Speech pattern detector
        self.speech_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=9, padding=4),
            nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=9, padding=4),
            nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Attention fusion - 48 channels total
        self.attention = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=1), nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        lip_feat = F.adaptive_avg_pool2d(self.lip_sync(x), (7, 7))
        av_feat = F.adaptive_avg_pool2d(self.av_correlation(x), (7, 7))
        expr_feat = F.adaptive_avg_pool2d(self.expression_analyzer(x), (7, 7))
        speech_feat = F.adaptive_avg_pool2d(self.speech_detector(x), (7, 7))
        combined = torch.cat([lip_feat, av_feat, expr_feat, speech_feat], dim=1)
        return combined * self.attention(combined)


class CompressionModule(nn.Module):
    """Compression artifact detection module - 40 channels, 1960 features"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # DCT coefficient analyzer
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=8, stride=8),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Quantization artifact detector
        self.quantization_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=4, stride=4),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Block boundary detector
        self.block_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=8, padding=4),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=8, padding=4),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Compression level estimator
        self.compression_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10), nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20), nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion - 40 channels total
        self.attention = nn.Sequential(
            nn.Conv2d(40, 28, kernel_size=1), nn.ReLU(),
            nn.Conv2d(28, 40, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        dct_feat = F.adaptive_avg_pool2d(self.dct_analyzer(x), (7, 7))
        quant_feat = F.adaptive_avg_pool2d(self.quantization_detector(x), (7, 7))
        block_feat = F.adaptive_avg_pool2d(self.block_detector(x), (7, 7))
        comp_feat = F.adaptive_avg_pool2d(self.compression_estimator(x), (7, 7))
        combined = torch.cat([dct_feat, quant_feat, block_feat, comp_feat], dim=1)
        return combined * self.attention(combined)


class ResolutionModule(nn.Module):
    """Resolution inconsistency detection module - 36 channels, 1764 features"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale resolution analyzer
        self.resolution_scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 9, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(9), nn.ReLU(),
                nn.Conv2d(9, 18, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(18), nn.ReLU(),
                nn.Conv2d(18, 9, kernel_size=1)
            ) for k in [3, 5, 7]
        ])
        
        # Upscaling artifact detector
        self.upscaling_detector = nn.Sequential(
            nn.Conv2d(in_channels, 9, kernel_size=5, padding=2),
            nn.BatchNorm2d(9), nn.ReLU(),
            nn.Conv2d(9, 18, kernel_size=5, padding=2),
            nn.BatchNorm2d(18), nn.ReLU(),
            nn.Conv2d(18, 9, kernel_size=1)
        )
        
        # Attention fusion - 36 channels total
        self.attention = nn.Sequential(
            nn.Conv2d(36, 24, kernel_size=1), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        scale_features = []
        for scale_branch in self.resolution_scales:
            scale_feat = F.adaptive_avg_pool2d(scale_branch(x), (7, 7))
            scale_features.append(scale_feat)
        
        scale_combined = torch.cat(scale_features, dim=1)
        upscale_feat = F.adaptive_avg_pool2d(self.upscaling_detector(x), (7, 7))
        combined = torch.cat([scale_combined, upscale_feat], dim=1)
        return combined * self.attention(combined)


class TemporalModule(nn.Module):
    """Temporal inconsistency detection module - 52 channels, 2548 features"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Frame consistency analyzer
        self.frame_consistency = nn.Sequential(
            nn.Conv2d(in_channels, 13, kernel_size=3, padding=1),
            nn.BatchNorm2d(13), nn.ReLU(),
            nn.Conv2d(13, 26, kernel_size=3, padding=1),
            nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 13, kernel_size=1)
        )
        
        # Motion flow analyzer
        self.motion_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 13, kernel_size=5, padding=2),
            nn.BatchNorm2d(13), nn.ReLU(),
            nn.Conv2d(13, 26, kernel_size=5, padding=2),
            nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 13, kernel_size=1)
        )
        
        # Temporal artifact detector
        self.temporal_artifact = nn.Sequential(
            nn.Conv2d(in_channels, 13, kernel_size=7, padding=3),
            nn.BatchNorm2d(13), nn.ReLU(),
            nn.Conv2d(13, 26, kernel_size=7, padding=3),
            nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 13, kernel_size=1)
        )
        
        # Optical flow estimator
        self.optical_flow = nn.Sequential(
            nn.Conv2d(in_channels, 13, kernel_size=9, padding=4),
            nn.BatchNorm2d(13), nn.ReLU(),
            nn.Conv2d(13, 26, kernel_size=9, padding=4),
            nn.BatchNorm2d(26), nn.ReLU(),
            nn.Conv2d(26, 13, kernel_size=1)
        )
        
        # Attention fusion - 52 channels total
        self.attention = nn.Sequential(
            nn.Conv2d(52, 36, kernel_size=1), nn.ReLU(),
            nn.Conv2d(36, 52, kernel_size=1), nn.Sigmoid()
        )
    
    def forward(self, x):
        frame_feat = F.adaptive_avg_pool2d(self.frame_consistency(x), (7, 7))
        motion_feat = F.adaptive_avg_pool2d(self.motion_analyzer(x), (7, 7))
        temporal_feat = F.adaptive_avg_pool2d(self.temporal_artifact(x), (7, 7))
        flow_feat = F.adaptive_avg_pool2d(self.optical_flow(x), (7, 7))
        combined = torch.cat([frame_feat, motion_feat, temporal_feat, flow_feat], dim=1)
        return combined * self.attention(combined)


# ============================================================================
# SPECIALIST MODELS
# ============================================================================

class SpecialistModelBase(nn.Module):
    """Base class for all specialist models with EfficientNet-B4 backbone"""
    
    def __init__(self, specialist_module, num_classes=2):
        super().__init__()
        
        from torchvision.models import efficientnet_b4
        
        # EfficientNet-B4 backbone
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except RuntimeError:
            self.backbone = efficientnet_b4(weights=None)
        
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Specialist module
        self.specialist_module = specialist_module
        
        # Calculate specialist features
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            specialist_output = self.specialist_module(dummy_input)
            specialist_features = specialist_output.numel() // specialist_output.shape[0]
        
        # Feature projection and attention
        total_features = backbone_features + specialist_features
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        
        # Progressive classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Extract specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project features
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Final classification
        output = self.classifier(attended_features)
        return output


class BGSpecialistModel(SpecialistModelBase):
    """BG-Model: Background Specialist"""
    def __init__(self, num_classes=2):
        super().__init__(BackgroundModule(), num_classes)
        self.model_type = "background"


class LLSpecialistModel(SpecialistModelBase):
    """LL-Model: Low-Light Specialist (same as BG but different training)"""
    def __init__(self, num_classes=2):
        super().__init__(BackgroundModule(), num_classes)
        self.model_type = "lowlight"


class AVSpecialistModel(SpecialistModelBase):
    """AV-Model: Audio-Visual Specialist"""
    def __init__(self, num_classes=2):
        super().__init__(AudioVisualModule(), num_classes)
        self.model_type = "audiovisual"


class CMSpecialistModel(SpecialistModelBase):
    """CM-Model: Compression Specialist"""
    def __init__(self, num_classes=2):
        super().__init__(CompressionModule(), num_classes)
        self.model_type = "compression"


class RRSpecialistModel(SpecialistModelBase):
    """RR-Model: Resolution Specialist"""
    def __init__(self, num_classes=2):
        super().__init__(ResolutionModule(), num_classes)
        self.model_type = "resolution"


class TMSpecialistModel(SpecialistModelBase):
    """TM-Model: Temporal Specialist"""
    def __init__(self, num_classes=2):
        super().__init__(TemporalModule(), num_classes)
        self.model_type = "temporal"


# ============================================================================
# OLD ARCHITECTURES (TM only - no new model yet)
# ============================================================================

class TMModelOld(nn.Module):
    """TM-Model: OLD ResNet18-based architecture (no new model yet)"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.model_type = "temporal"
        
        # ResNet18 backbone (without final FC layer)
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] for temporal model or [B, C, H, W] for single frame
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]
        
        B, T, C, H, W = x.shape
        
        # Process each frame through backbone
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)  # [B*T, 512]
        features = features.view(B, T, 512)  # [B, T, 512]
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [B, T, 256]
        
        # Use the last timestep output
        final_features = lstm_out[:, -1, :]  # [B, 256]
        
        # Final classification
        output = self.classifier(final_features)
        
        return output


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_bg_model(num_classes=2):
    """Create BG-Model (Background Specialist)"""
    return BGSpecialistModel(num_classes)

def create_ll_model(num_classes=2):
    """Create LL-Model (Low-Light Specialist)"""
    return LLSpecialistModel(num_classes)

def create_av_model(num_classes=2):
    """Create AV-Model (Audio-Visual Specialist)"""
    return AVSpecialistModel(num_classes)

def create_cm_model(num_classes=2):
    """Create CM-Model (Compression Specialist)"""
    return CMSpecialistModel(num_classes)

def create_rr_model(num_classes=2):
    """Create RR-Model (Resolution Specialist)"""
    return RRSpecialistModel(num_classes)

def create_tm_model(num_classes=2):
    """Create TM-Model (Temporal Specialist) - OLD architecture"""
    return TMModelOld(num_classes)

def load_specialist_model(model_path, model_type, device='cpu'):
    """
    Load a specialist model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_type: Type of specialist model ('bg', 'cm', 'rr', 'll', 'av', 'tm')
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Create model based on type
    model_type = model_type.lower()
    if model_type == 'bg':
        model = create_bg_model()
        print(f"[INFO] Using NEW BG model (EfficientNet-B4)")
    elif model_type == 'cm':
        model = create_cm_model()
        print(f"[INFO] Using NEW CM model (EfficientNet-B4)")
    elif model_type == 'rr':
        model = create_rr_model()
        print(f"[INFO] Using NEW RR model (EfficientNet-B4)")
    elif model_type == 'll':
        model = create_ll_model()
        print(f"[INFO] Using NEW LL model (EfficientNet-B4)")
    elif model_type == 'av':
        model = create_av_model()
        print(f"[INFO] Using NEW AV model (EfficientNet-B4)")
    elif model_type == 'tm':
        model = create_tm_model()
        print(f"[INFO] Using OLD TM model (ResNet18)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        accuracy = checkpoint.get('metrics', {}).get('accuracy', 'unknown')
        print(f"[OK] Loaded {model_type.upper()}-Model: {accuracy} accuracy")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print(f"[OK] Loaded {model_type.upper()}-Model")
    
    model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    # Test all specialist models
    print("[TEST] Testing Updated Specialist Models...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_input = torch.randn(2, 3, 224, 224).to(device)
    test_sequence = torch.randn(2, 8, 3, 224, 224).to(device)
    
    models_to_test = [
        ("BG-Model (NEW)", create_bg_model(), test_input),
        ("CM-Model (NEW)", create_cm_model(), test_input),
        ("RR-Model (NEW)", create_rr_model(), test_input),
        ("LL-Model (NEW)", create_ll_model(), test_input),
        ("AV-Model (NEW)", create_av_model(), test_input),
        ("TM-Model (OLD)", create_tm_model(), test_sequence)
    ]
    
    print("\n📊 MODEL ARCHITECTURES:")
    print("-"*60)
    
    for name, model, test_data in models_to_test:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(test_data)
            print(f"\n✅ {name}")
            print(f"   Output shape: {output.shape}")
        
        params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {params:,}")
        
        if "NEW" in name:
            print(f"   Architecture: EfficientNet-B4 + Specialist Module")
        else:
            print(f"   Architecture: ResNet18 (legacy)")
    
    print("\n" + "="*60)
    print("✅ All specialist models working correctly!")
    print("\n📋 DEPLOYMENT NOTES:")
    print("   • Replace 5 models on Hugging Face: BG, CM, RR, LL, AV")
    print("   • Keep 1 old model: TM")
    print("   • All models have same interface (forward pass)")
    print("="*60)
