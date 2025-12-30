import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    T = None

class AudioBranch(nn.Module):
    """Lightweight audio processing branch for student model"""
    def __init__(self, feature_dim=128, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Mel spectrogram transform (only if torchaudio is available)
        if TORCHAUDIO_AVAILABLE:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=64,
                n_fft=512,
                hop_length=256
            )
        else:
            self.mel_transform = None
        
        # Small CNN for mel spectrogram processing
        self.conv_layers = nn.Sequential(
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
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: [B, audio_length]
        Returns:
            features: [B, feature_dim]
        """
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(audio_waveform)  # [B, n_mels, time]
        mel_spec = mel_spec.unsqueeze(1)  # [B, 1, n_mels, time]
        
        # Process through CNN
        conv_out = self.conv_layers(mel_spec)  # [B, 128, 4, 4]
        conv_out = conv_out.view(conv_out.size(0), -1)  # [B, 128*4*4]
        
        # Project to feature dimension
        features = self.feature_proj(conv_out)
        
        return features

class StudentModel(nn.Module):
    """Compact multimodal student model for mobile deployment"""
    def __init__(self, num_classes=2, visual_frames=8, dropout=0.2):
        super(StudentModel, self).__init__()
        self.num_frames = visual_frames
        
        # Visual branch - MobileNetV3 Small
        self.visual_backbone = models.mobilenet_v3_small(weights='DEFAULT')
        # Remove classifier to get features
        visual_features = self.visual_backbone.classifier[0].in_features
        self.visual_backbone.classifier = nn.Identity()
        
        # Temporal processing for video
        self.temporal_conv = nn.Conv1d(visual_features, 256, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Audio branch
        self.audio_branch = AudioBranch(feature_dim=128)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),  # visual + audio features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, video_frames, audio_waveform=None, return_features=False):
        """
        Args:
            video_frames: [B, T, C, H, W] or [B, C, H, W] for single frame
            audio_waveform: [B, audio_length] (optional)
            return_features: bool
        Returns:
            logits: [B, num_classes]
            features: dict (if return_features=True)
        """
        # Handle both single frame and multi-frame input
        if len(video_frames.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = video_frames.shape
            # Process each frame
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
        
        # Audio processing (if available)
        if audio_waveform is not None:
            audio_feat = self.audio_branch(audio_waveform)
            # Concatenate visual and audio features
            combined_feat = torch.cat([visual_feat, audio_feat], dim=1)
        else:
            # Use only visual features, pad audio dimension with zeros
            audio_feat = torch.zeros(visual_feat.shape[0], 128, device=visual_feat.device)
            combined_feat = torch.cat([visual_feat, audio_feat], dim=1)
        
        # Fusion
        fused_feat = self.fusion(combined_feat)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        if return_features:
            features = {
                'visual_feat': visual_feat,
                'audio_feat': audio_feat if audio_waveform is not None else None,
                'fused_feat': fused_feat
            }
            return logits, features
        
        return logits

# Legacy compatibility
class MultiModalStudent(StudentModel):
    """Alias for backward compatibility"""
    pass

def create_student_model(num_classes=2, visual_frames=8):
    """Factory function to create student model"""
    return StudentModel(num_classes=num_classes, visual_frames=visual_frames)

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_student_model()
    
    # Test inputs
    video_single = torch.randn(2, 3, 224, 224)      # Single frame
    video_multi = torch.randn(2, 8, 3, 224, 224)    # Multi frame
    audio = torch.randn(2, 16000 * 3)               # 3 seconds at 16kHz
    
    print("Testing student model...")
    
    # Test single frame (visual only)
    logits1 = model(video_single)
    print(f"Single frame (visual only): {logits1.shape}")
    
    # Test single frame with audio
    logits2 = model(video_single, audio)
    print(f"Single frame + audio: {logits2.shape}")
    
    # Test multi-frame with audio
    logits3, features = model(video_multi, audio, return_features=True)
    print(f"Multi-frame + audio: {logits3.shape}")
    print(f"Features keys: {list(features.keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"Estimated model size: {model_size_mb:.2f} MB")