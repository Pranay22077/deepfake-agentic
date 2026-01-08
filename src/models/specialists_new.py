"""
E-Raksha Specialist Models with EfficientNet-B4 Backbone

This module contains all specialist deepfake detection models, each designed
to detect specific types of manipulation artifacts. All models use EfficientNet-B4
as the backbone with custom specialist modules for domain-specific analysis.

Models included:
- BG: Background/Lighting inconsistency detection
- AV: Audio-Visual synchronization analysis  
- CM: Compression artifact detection
- RR: Resolution/Re-recording pattern detection
- LL: Low-light condition analysis
- TM: Temporal consistency analysis

Author: E-Raksha Team
Created: Initial development phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4


# ============================================================================
# BACKGROUND/LIGHTING (BG) MODULE - 44 CHANNELS
# ============================================================================
class BackgroundLightingModule(nn.Module):
    """
    Background and lighting inconsistency detection module.
    
    Detects manipulation artifacts in background textures, lighting direction,
    shadow consistency, and color temperature. Outputs 44 channels total.
    """
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Background texture analyzer (16 channels)
        self.bg_texture = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Lighting direction detector (12 channels)
        self.lighting_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow consistency checker (8 channels)
        self.shadow_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=9, padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=9, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Color temperature analyzer (8 channels)
        self.color_temp = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Attention fusion (44 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(16 + 12 + 8 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 44, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        bg_feat = self.bg_texture(x)
        bg_feat = F.adaptive_avg_pool2d(bg_feat, (7, 7))
        
        light_feat = self.lighting_detector(x)
        light_feat = F.adaptive_avg_pool2d(light_feat, (7, 7))
        
        shadow_feat = self.shadow_checker(x)
        shadow_feat = F.adaptive_avg_pool2d(shadow_feat, (7, 7))
        
        color_feat = self.color_temp(x)
        color_feat = F.adaptive_avg_pool2d(color_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([bg_feat, light_feat, shadow_feat, color_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# AUDIO-VISUAL (AV) MODULE - 48 CHANNELS
# ============================================================================
class AudioVisualModule(nn.Module):
    """Audio-visual inconsistency detection module - 48 channels"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Lip region analyzer (16 channels)
        self.lip_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1)
        )
        
        # Facial motion detector (12 channels)
        self.motion_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=7, padding=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Temporal consistency checker (10 channels)
        self.temporal_checker = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Audio-visual correlation analyzer (10 channels) - kernel 3x3
        self.av_correlation = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion (48 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(16 + 12 + 10 + 10, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        lip_feat = self.lip_analyzer(x)
        lip_feat = F.adaptive_avg_pool2d(lip_feat, (7, 7))
        
        motion_feat = self.motion_detector(x)
        motion_feat = F.adaptive_avg_pool2d(motion_feat, (7, 7))
        
        temporal_feat = self.temporal_checker(x)
        temporal_feat = F.adaptive_avg_pool2d(temporal_feat, (7, 7))
        
        av_feat = self.av_correlation(x)
        av_feat = F.adaptive_avg_pool2d(av_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([lip_feat, motion_feat, temporal_feat, av_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# RESOLUTION (RR) MODULE - 36 CHANNELS
# ============================================================================
class ResolutionModule(nn.Module):
    """Resolution inconsistency detection module - 36 channels"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale resolution analyzer (10 channels)
        self.resolution_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Upscaling artifact detector (8 channels)
        self.upscaling_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Edge sharpness checker (8 channels)
        self.edge_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Pixel interpolation detector (10 channels)
        self.interpolation_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=7, padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=7, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion (36 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(10 + 8 + 8 + 10, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 36, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        res_feat = self.resolution_analyzer(x)
        res_feat = F.adaptive_avg_pool2d(res_feat, (7, 7))
        
        ups_feat = self.upscaling_detector(x)
        ups_feat = F.adaptive_avg_pool2d(ups_feat, (7, 7))
        
        edge_feat = self.edge_checker(x)
        edge_feat = F.adaptive_avg_pool2d(edge_feat, (7, 7))
        
        interp_feat = self.interpolation_detector(x)
        interp_feat = F.adaptive_avg_pool2d(interp_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([res_feat, ups_feat, edge_feat, interp_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# COMPRESSION (CM) MODULE - 40 CHANNELS
# ============================================================================
class CompressionModule(nn.Module):
    """Compression artifact detection module - 40 channels"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # DCT coefficient analyzer (12 channels)
        self.dct_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=8, stride=8),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Quantization artifact detector (10 channels)
        self.quant_detector = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Block boundary checker (8 channels)
        self.block_checker = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Compression level estimator (10 channels)
        self.compression_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion (40 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(12 + 10 + 8 + 10, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 40, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        dct_feat = self.dct_analyzer(x)
        dct_feat = F.adaptive_avg_pool2d(dct_feat, (7, 7))
        
        quant_feat = self.quant_detector(x)
        quant_feat = F.adaptive_avg_pool2d(quant_feat, (7, 7))
        
        block_feat = self.block_checker(x)
        block_feat = F.adaptive_avg_pool2d(block_feat, (7, 7))
        
        comp_feat = self.compression_estimator(x)
        comp_feat = F.adaptive_avg_pool2d(comp_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([dct_feat, quant_feat, block_feat, comp_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# LOW-LIGHT (LL) MODULE - 68 CHANNELS
# ============================================================================
class EnhancedLowLightModule(nn.Module):
    """Enhanced low-light analysis with multi-scale features - 68 channels"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Multi-scale luminance analysis (3 branches × 16 channels = 48 channels)
        self.luminance_branch = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ) for k in [3, 5, 7]
        ])
        
        # Noise pattern detector (12 channels)
        self.noise_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=5, padding=2),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Shadow/highlight inconsistency detector (8 channels)
        self.shadow_detector = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=7, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Attention mechanism for feature fusion (68 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(48 + 12 + 8, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 68, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Multi-scale luminance analysis
        lum_features = []
        for branch in self.luminance_branch:
            lum_feat = branch(x)
            lum_feat = F.adaptive_avg_pool2d(lum_feat, (7, 7))
            lum_features.append(lum_feat)
        
        lum_combined = torch.cat(lum_features, dim=1)  # 48 channels
        
        # Noise analysis
        noise_features = self.noise_detector(x)
        noise_features = F.adaptive_avg_pool2d(noise_features, (7, 7))  # 12 channels
        
        # Shadow analysis
        shadow_features = self.shadow_detector(x)
        shadow_features = F.adaptive_avg_pool2d(shadow_features, (7, 7))  # 8 channels
        
        # Combine all features
        combined = torch.cat([lum_combined, noise_features, shadow_features], dim=1)  # 68 channels
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# SPECIALIST MODELS
# ============================================================================

class SpecialistModelBase(nn.Module):
    """Base class for all specialist models with EfficientNet-B4 backbone"""
    
    def __init__(self, specialist_module, specialist_channels, num_classes=2):
        super().__init__()
        
        # EfficientNet-B4 backbone
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except:
            self.backbone = efficientnet_b4(weights=None)
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Specialist module
        self.specialist_module = specialist_module
        specialist_features = specialist_channels * 7 * 7
        
        # Feature projection and attention (matching training script)
        total_features = backbone_features + specialist_features
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier (matching training script: 1024 -> 512 -> 256 -> 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Specialist features
        specialist_features = self.specialist_module(x)
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Classification
        output = self.classifier(attended_features)
        return output


class BGSpecialistModel(SpecialistModelBase):
    """BG-Model: Background Specialist - 44 channels"""
    def __init__(self, num_classes=2):
        super().__init__(BackgroundLightingModule(), 44, num_classes)


class AVSpecialistModel(SpecialistModelBase):
    """AV-Model: Audio-Visual Specialist - 48 channels"""
    def __init__(self, num_classes=2):
        super().__init__(AudioVisualModule(), 48, num_classes)


class CMSpecialistModel(nn.Module):
    """CM-Model: Compression Specialist - 40 channels
    Note: This model has a different architecture where specialist components
    are directly on the model, not inside a specialist_module wrapper
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # EfficientNet-B4 backbone
        try:
            self.backbone = efficientnet_b4(weights='IMAGENET1K_V1')
        except:
            self.backbone = efficientnet_b4(weights=None)
        self.backbone.classifier = nn.Identity()
        backbone_features = 1792
        
        # Specialist components (directly on model, not in specialist_module)
        # DCT coefficient analyzer (12 channels)
        self.specialist_dct_analyzer = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=8, stride=8),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Quantization artifact detector (10 channels)
        self.specialist_quant_detector = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Block boundary checker (8 channels)
        self.specialist_block_checker = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1)
        )
        
        # Compression level estimator (10 channels)
        self.specialist_compression_estimator = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20, 10, kernel_size=1)
        )
        
        # Attention fusion (40 channels total)
        self.specialist_attention = nn.Sequential(
            nn.Conv2d(12 + 10 + 8 + 10, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 40, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature projection and attention
        specialist_features = 40 * 7 * 7
        total_features = backbone_features + specialist_features
        num_heads = 8
        adjusted_features = ((total_features + num_heads - 1) // num_heads) * num_heads
        
        self.feature_projection = nn.Linear(total_features, adjusted_features)
        
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=adjusted_features,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(adjusted_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Backbone features
        backbone_features = self.backbone.features(x)
        backbone_features = self.backbone.avgpool(backbone_features)
        backbone_features = torch.flatten(backbone_features, 1)
        
        # Specialist features
        dct_feat = self.specialist_dct_analyzer(x)
        dct_feat = F.adaptive_avg_pool2d(dct_feat, (7, 7))
        
        quant_feat = self.specialist_quant_detector(x)
        quant_feat = F.adaptive_avg_pool2d(quant_feat, (7, 7))
        
        block_feat = self.specialist_block_checker(x)
        block_feat = F.adaptive_avg_pool2d(block_feat, (7, 7))
        
        comp_feat = self.specialist_compression_estimator(x)
        comp_feat = F.adaptive_avg_pool2d(comp_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([dct_feat, quant_feat, block_feat, comp_feat], dim=1)
        attention_weights = self.specialist_attention(combined)
        specialist_features = combined * attention_weights
        specialist_features = torch.flatten(specialist_features, 1)
        
        # Combine and project
        combined_features = torch.cat([backbone_features, specialist_features], dim=1)
        projected_features = self.feature_projection(combined_features)
        
        # Apply attention
        projected_reshaped = projected_features.unsqueeze(1)
        attended_features, _ = self.feature_attention(
            projected_reshaped, projected_reshaped, projected_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Classification
        output = self.classifier(attended_features)
        return output


class RRSpecialistModel(SpecialistModelBase):
    """RR-Model: Resolution Specialist - 36 channels"""
    def __init__(self, num_classes=2):
        super().__init__(ResolutionModule(), 36, num_classes)


class LLSpecialistModel(SpecialistModelBase):
    """LL-Model: Low-Light Specialist - 68 channels"""
    def __init__(self, num_classes=2):
        super().__init__(EnhancedLowLightModule(), 68, num_classes)


class TMSpecialistModel(SpecialistModelBase):
    """TM-Model: Temporal Specialist - 52 channels (NEW EfficientNet-B4 based)"""
    def __init__(self, num_classes=2):
        super().__init__(TemporalModule(), 52, num_classes)


# ============================================================================
# TEMPORAL (TM) MODULE - 52 CHANNELS
# ============================================================================
class TemporalModule(nn.Module):
    """Temporal inconsistency detection module - 52 channels"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Frame-to-frame consistency checker (14 channels)
        self.frame_consistency = nn.Sequential(
            nn.Conv2d(in_channels, 14, kernel_size=5, padding=2),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Conv2d(14, 28, kernel_size=5, padding=2),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 14, kernel_size=1)
        )
        
        # Motion flow analyzer (12 channels)
        self.motion_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=7, padding=3),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=7, padding=3),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Temporal artifact detector (12 channels)
        self.temporal_detector = nn.Sequential(
            nn.Conv2d(in_channels, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=1)
        )
        
        # Optical flow validator (14 channels)
        self.optical_flow = nn.Sequential(
            nn.Conv2d(in_channels, 14, kernel_size=9, padding=4),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Conv2d(14, 28, kernel_size=9, padding=4),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.Conv2d(28, 14, kernel_size=1)
        )
        
        # Attention fusion (52 channels total)
        self.attention = nn.Sequential(
            nn.Conv2d(14 + 12 + 12 + 14, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 52, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        frame_feat = self.frame_consistency(x)
        frame_feat = F.adaptive_avg_pool2d(frame_feat, (7, 7))
        
        motion_feat = self.motion_analyzer(x)
        motion_feat = F.adaptive_avg_pool2d(motion_feat, (7, 7))
        
        temporal_feat = self.temporal_detector(x)
        temporal_feat = F.adaptive_avg_pool2d(temporal_feat, (7, 7))
        
        optical_feat = self.optical_flow(x)
        optical_feat = F.adaptive_avg_pool2d(optical_feat, (7, 7))
        
        # Combine and apply attention
        combined = torch.cat([frame_feat, motion_feat, temporal_feat, optical_feat], dim=1)
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return attended_features


# ============================================================================
# OLD TM MODEL (ResNet18-based)
# ============================================================================

import torchvision.models as models

class TMModelOld(nn.Module):
    """TM-Model: OLD ResNet18-based architecture"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # ResNet18 backbone
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
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        features = self.backbone(x_flat)
        features = features.view(B, T, 512)
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]
        output = self.classifier(final_features)
        
        return output


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_bg_model(num_classes=2):
    """Create BG-Model"""
    return BGSpecialistModel(num_classes)

def create_av_model(num_classes=2):
    """Create AV-Model"""
    return AVSpecialistModel(num_classes)

def create_cm_model(num_classes=2):
    """Create CM-Model"""
    return CMSpecialistModel(num_classes)

def create_rr_model(num_classes=2):
    """Create RR-Model"""
    return RRSpecialistModel(num_classes)

def create_ll_model(num_classes=2):
    """Create LL-Model"""
    return LLSpecialistModel(num_classes)

def create_tm_model_new(num_classes=2):
    """Create TM-Model (NEW EfficientNet-B4 based)"""
    return TMSpecialistModel(num_classes)

def create_tm_model(num_classes=2):
    """Create TM-Model (OLD ResNet18 based)"""
    return TMModelOld(num_classes)


def load_specialist_model(model_path, model_type, device='cpu'):
    """Load a specialist model from checkpoint"""
    
    model_type = model_type.lower()
    if model_type == 'bg':
        model = create_bg_model()
    elif model_type == 'av':
        model = create_av_model()
    elif model_type == 'cm':
        model = create_cm_model()
    elif model_type == 'rr':
        model = create_rr_model()
    elif model_type == 'll':
        model = create_ll_model()
    elif model_type == 'tm_new':
        model = create_tm_model_new()
    elif model_type == 'tm':
        model = create_tm_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    return model
