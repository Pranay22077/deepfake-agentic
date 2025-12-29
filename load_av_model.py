#!/usr/bin/env python3
"""
AV-Model Loader
Helper script to load the trained AV-Model for LangGraph integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio.transforms as T
import os
import json

class LipSyncDetector(nn.Module):
    """Lip-sync mismatch detection module"""
    
    def __init__(self, visual_dim=256, audio_dim=256, hidden_dim=128):
        super().__init__()
        
        # Visual lip encoder
        self.lip_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 16, visual_dim)
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            
            nn.Flatten(),
            nn.Linear(128 * 16, audio_dim)
        )
        
        # Sync correlation
        self.sync_net = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, lip_frames, audio):
        B, T = lip_frames.shape[:2]
        
        # Process lip frames
        lips = lip_frames.view(B * T, *lip_frames.shape[2:])
        lip_feat = self.lip_encoder(lips)
        lip_feat = lip_feat.view(B, T, -1).mean(dim=1)
        
        # Process audio
        audio_feat = self.audio_encoder(audio.unsqueeze(1))
        
        # Compute sync score
        combined = torch.cat([lip_feat, audio_feat], dim=1)
        sync_score = self.sync_net(combined)
        
        return sync_score

class AVModel(nn.Module):
    """Audio-Visual Deepfake Detection Model"""
    
    def __init__(self, num_classes=2, visual_frames=8):
        super().__init__()
        
        # Visual backbone
        self.visual_backbone = models.resnet18(weights='DEFAULT')
        visual_dim = self.visual_backbone.fc.in_features
        self.visual_backbone.fc = nn.Identity()
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(visual_dim, 256, 3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Audio processing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        self.audio_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 16, 256)
        )
        
        # Lip-sync detector
        self.lip_sync_detector = LipSyncDetector()
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 1, 256),  # visual + audio + lip_sync
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, video_frames, lip_regions, audio, return_features=False):
        B, T, C, H, W = video_frames.shape
        
        # Visual processing
        frames = video_frames.view(B*T, C, H, W)
        frame_features = self.visual_backbone(frames)
        frame_features = frame_features.view(B, T, -1)
        
        # Temporal aggregation
        temp_features = frame_features.transpose(1, 2)
        temp_features = self.temporal_conv(temp_features)
        visual_feat = self.temporal_pool(temp_features).squeeze(-1)
        
        # Audio processing
        mel_spec = self.mel_transform(audio)
        mel_spec = mel_spec.unsqueeze(1)
        audio_feat = self.audio_backbone(mel_spec)
        
        # Lip-sync analysis
        lip_sync_score = self.lip_sync_detector(lip_regions, audio)
        
        # Fusion
        combined = torch.cat([visual_feat, audio_feat, lip_sync_score], dim=1)
        fused_feat = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        if return_features:
            return logits, {
                'visual_feat': visual_feat,
                'audio_feat': audio_feat,
                'lip_sync_score': lip_sync_score,
                'fused_feat': fused_feat
            }
        
        return logits

def load_av_model(model_path='models/av_model_student.pt', device='cpu'):
    """Load the trained AV-Model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"AV-Model not found: {model_path}")
    
    # Create model
    model = AVModel(num_classes=2, visual_frames=8)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load config if available
    config_path = 'config/av_model_summary.json'
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    return model, checkpoint, config

if __name__ == "__main__":
    # Test loading
    try:
        model, checkpoint, config = load_av_model()
        print(f"✅ AV-Model loaded successfully!")
        print(f"📊 Accuracy: {checkpoint.get('best_acc', 0):.2f}%")
        print(f"🎯 Specialization: Audio-Visual Inconsistency Detection")
        
        # Test inference
        batch_size = 1
        video_frames = torch.randn(batch_size, 8, 3, 224, 224)
        lip_regions = torch.randn(batch_size, 8, 3, 64, 64)
        audio = torch.randn(batch_size, 48000)
        
        with torch.no_grad():
            logits, features = model(video_frames, lip_regions, audio, return_features=True)
            probs = torch.softmax(logits, dim=1)
        
        print(f"🧪 Test inference successful!")
        print(f"   Output shape: {logits.shape}")
        print(f"   Lip-sync score: {features['lip_sync_score'].item():.3f}")
        
    except Exception as e:
        print(f"❌ Error loading AV-Model: {e}")