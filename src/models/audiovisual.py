#!/usr/bin/env python3
"""
Person 4: AV-Model (Audio-Visual Specialist)
Specialized model for detecting lip-sync mismatches and audio-visual inconsistencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio.transforms as T
import cv2
import numpy as np
from typing import Tuple, Optional

class LipSyncDetector(nn.Module):
    """Specialized module for detecting lip-sync mismatches"""
    
    def __init__(self, visual_dim=512, audio_dim=256, hidden_dim=256):
        super().__init__()
        
        # Visual lip region encoder
        self.lip_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, visual_dim)
        )
        
        # Audio speech encoder
        self.speech_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            
            nn.Flatten(),
            nn.Linear(256 * 16, audio_dim)
        )
        
        # Sync correlation network
        self.sync_network = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, lip_frames, audio_segment):
        """
        Args:
            lip_frames: [B, T, C, H, W] - cropped lip regions
            audio_segment: [B, audio_length] - corresponding audio
        Returns:
            sync_score: [B, 1] - lip-sync correlation score (0-1)
        """
        B, T = lip_frames.shape[:2]
        
        # Process lip frames
        lip_frames = lip_frames.view(B * T, *lip_frames.shape[2:])
        lip_features = self.lip_encoder(lip_frames)  # [B*T, visual_dim]
        lip_features = lip_features.view(B, T, -1).mean(dim=1)  # [B, visual_dim]
        
        # Process audio
        audio_features = self.speech_encoder(audio_segment.unsqueeze(1))  # [B, audio_dim]
        
        # Compute sync correlation
        combined = torch.cat([lip_features, audio_features], dim=1)
        sync_score = self.sync_network(combined)
        
        return sync_score

class AVModel(nn.Module):
    """Audio-Visual Specialist Model for deepfake detection"""
    
    def __init__(self, num_classes=2, visual_frames=8):
        super().__init__()
        self.num_frames = visual_frames
        
        # Visual branch - ResNet18 backbone
        self.visual_backbone = models.resnet18(weights='DEFAULT')
        visual_features = self.visual_backbone.fc.in_features
        self.visual_backbone.fc = nn.Identity()
        
        # Temporal processing for video
        self.temporal_conv = nn.Conv1d(visual_features, 256, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Audio branch - Mel spectrogram processing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256)
        )
        
        # Lip-sync detector
        self.lip_sync_detector = LipSyncDetector(visual_dim=256, audio_dim=256)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 1, 256),  # visual + audio + lip_sync
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Auxiliary heads
        self.lip_sync_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def extract_lip_region(self, frame, face_landmarks=None):
        """Extract lip region from face frame"""
        # Simplified lip extraction - in practice would use face landmarks
        h, w = frame.shape[-2:]
        lip_region = frame[..., h//2:h*3//4, w//4:w*3//4]  # Approximate lip region
        return F.interpolate(lip_region, size=(64, 64), mode='bilinear')
    
    def forward(self, video_frames, audio_waveform=None, return_features=False):
        """
        Args:
            video_frames: [B, T, C, H, W] or [B, C, H, W]
            audio_waveform: [B, audio_length] (optional)
            return_features: bool
        Returns:
            logits: [B, num_classes]
            features: dict (if return_features=True)
        """
        # Handle both single frame and multi-frame input
        if len(video_frames.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = video_frames.shape
            
            # Process each frame through visual backbone
            frames = video_frames.view(B*T, C, H, W)
            frame_features = self.visual_backbone(frames)  # [B*T, visual_features]
            frame_features = frame_features.view(B, T, -1)  # [B, T, visual_features]
            
            # Temporal processing
            temp_features = frame_features.transpose(1, 2)  # [B, visual_features, T]
            temp_features = self.temporal_conv(temp_features)  # [B, 256, T]
            visual_feat = self.temporal_pool(temp_features).squeeze(-1)  # [B, 256]
            
        else:  # [B, C, H, W] - single frame
            visual_feat = self.visual_backbone(video_frames)  # [B, visual_features]
            # Project to expected dimension
            if visual_feat.shape[1] != 256:
                if not hasattr(self, 'visual_proj'):
                    self.visual_proj = nn.Linear(visual_feat.shape[1], 256).to(video_frames.device)
                visual_feat = self.visual_proj(visual_feat)
        
        # Audio processing
        if audio_waveform is not None:
            # Convert to mel spectrogram
            mel_spec = self.mel_transform(audio_waveform)  # [B, n_mels, time]
            mel_spec = mel_spec.unsqueeze(1)  # [B, 1, n_mels, time]
            audio_feat = self.audio_backbone(mel_spec)  # [B, 256]
            
            # Lip-sync analysis (if multi-frame)
            if len(video_frames.shape) == 5:
                # Extract lip regions
                lip_frames = []
                for t in range(T):
                    frame = video_frames[:, t]  # [B, C, H, W]
                    lip_region = self.extract_lip_region(frame)
                    lip_frames.append(lip_region)
                lip_frames = torch.stack(lip_frames, dim=1)  # [B, T, C, H, W]
                
                # Compute lip-sync score
                lip_sync_score = self.lip_sync_detector(lip_frames, audio_waveform)
            else:
                # Single frame - no temporal lip-sync analysis
                lip_sync_score = torch.ones(video_frames.shape[0], 1, device=video_frames.device) * 0.5
        else:
            # No audio - use zero features
            audio_feat = torch.zeros(visual_feat.shape[0], 256, device=visual_feat.device)
            lip_sync_score = torch.ones(visual_feat.shape[0], 1, device=visual_feat.device) * 0.5
        
        # Fusion
        combined_feat = torch.cat([visual_feat, audio_feat, lip_sync_score], dim=1)
        fused_feat = self.fusion(combined_feat)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        # Auxiliary outputs
        lip_sync_pred = self.lip_sync_head(fused_feat)
        
        if return_features:
            features = {
                'visual_feat': visual_feat,
                'audio_feat': audio_feat if audio_waveform is not None else None,
                'fused_feat': fused_feat,
                'lip_sync_score': lip_sync_score,
                'lip_sync_pred': lip_sync_pred
            }
            return logits, features
        
        return logits

def create_av_model(num_classes=2, visual_frames=8):
    """Factory function to create AV model"""
    return AVModel(num_classes=num_classes, visual_frames=visual_frames)

if __name__ == "__main__":
    # Test AV model
    model = create_av_model()
    
    # Test inputs
    video_multi = torch.randn(2, 8, 3, 224, 224)    # Multi frame
    video_single = torch.randn(2, 3, 224, 224)      # Single frame
    audio = torch.randn(2, 16000 * 3)               # 3 seconds at 16kHz
    
    print("Testing AV Model...")
    
    # Test multi-frame with audio
    logits1, features1 = model(video_multi, audio, return_features=True)
    print(f"Multi-frame + audio: {logits1.shape}")
    print(f"Lip-sync score: {features1['lip_sync_score'].shape}")
    print(f"Features keys: {list(features1.keys())}")
    
    # Test single frame with audio
    logits2 = model(video_single, audio)
    print(f"Single frame + audio: {logits2.shape}")
    
    # Test without audio
    logits3 = model(video_multi)
    print(f"Multi-frame (visual only): {logits3.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("AV Model test completed successfully!")