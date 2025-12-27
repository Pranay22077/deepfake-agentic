import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torchaudio.transforms as T

class VisualBranch(nn.Module):
    """EfficientNet-B4 backbone with temporal processing"""
    def __init__(self, num_frames=8, feature_dim=512):
        super().__init__()
        # Use EfficientNet-B4 as backbone
        self.backbone = models.efficientnet_b4(weights='DEFAULT')
        # Remove classifier to get features
        self.backbone.classifier = nn.Identity()
        backbone_dim = 1792  # EfficientNet-B4 feature dimension
        
        # Temporal processing
        self.temporal_conv = nn.Conv1d(backbone_dim, feature_dim, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_norm = nn.LayerNorm(feature_dim)
        
        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(feature_dim, feature_dim//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, video_frames):
        """
        Args:
            video_frames: [B, T, C, H, W] where T is number of frames
        Returns:
            features: [B, feature_dim]
        """
        B, T, C, H, W = video_frames.shape
        
        # Process each frame through backbone
        frames = video_frames.view(B*T, C, H, W)
        frame_features = self.backbone(frames)  # [B*T, backbone_dim]
        frame_features = frame_features.view(B, T, -1)  # [B, T, backbone_dim]
        
        # Temporal convolution
        temp_features = frame_features.transpose(1, 2)  # [B, backbone_dim, T]
        temp_features = self.temporal_conv(temp_features)  # [B, feature_dim, T]
        temp_features = temp_features.transpose(1, 2)  # [B, T, feature_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(temp_features)  # [B, T, feature_dim]
        
        # Global temporal pooling
        pooled = torch.mean(lstm_out, dim=1)  # [B, feature_dim]
        pooled = self.temporal_norm(pooled)
        pooled = self.dropout(pooled)
        
        return pooled

class AudioBranch(nn.Module):
    """Wav2Vec2-based audio processing"""
    def __init__(self, feature_dim=256, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Use smaller Wav2Vec2 config for efficiency
        config = Wav2Vec2Config(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            vocab_size=32
        )
        
        # Initialize Wav2Vec2 model
        self.wav2vec = Wav2Vec2Model(config)
        
        # Audio preprocessing
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=80,
            n_fft=1024,
            hop_length=256
        )
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(feature_dim)
        )
        
    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: [B, audio_length] raw audio
        Returns:
            features: [B, feature_dim]
        """
        # Process through Wav2Vec2
        with torch.no_grad():
            # Use feature extractor only for efficiency
            features = self.wav2vec.feature_extractor(audio_waveform)
            features = features.transpose(1, 2)  # [B, seq_len, hidden_size]
        
        # Global average pooling
        pooled = torch.mean(features, dim=1)  # [B, hidden_size]
        
        # Project to desired dimension
        output = self.feature_proj(pooled)
        
        return output

class FusionModule(nn.Module):
    """Transformer-based fusion of visual and audio features"""
    def __init__(self, visual_dim=512, audio_dim=256, fusion_dim=512, num_heads=8):
        super().__init__()
        
        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.audio_proj = nn.Linear(audio_dim, fusion_dim)
        
        # Positional embeddings
        self.visual_pos = nn.Parameter(torch.randn(1, 1, fusion_dim))
        self.audio_pos = nn.Parameter(torch.randn(1, 1, fusion_dim))
        
        # Transformer fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim*2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, visual_feat, audio_feat):
        """
        Args:
            visual_feat: [B, visual_dim]
            audio_feat: [B, audio_dim]
        Returns:
            fused_feat: [B, fusion_dim]
        """
        B = visual_feat.shape[0]
        
        # Project to common dimension
        v_proj = self.visual_proj(visual_feat).unsqueeze(1)  # [B, 1, fusion_dim]
        a_proj = self.audio_proj(audio_feat).unsqueeze(1)   # [B, 1, fusion_dim]
        
        # Add positional embeddings
        v_proj = v_proj + self.visual_pos
        a_proj = a_proj + self.audio_pos
        
        # Concatenate for transformer
        combined = torch.cat([v_proj, a_proj], dim=1)  # [B, 2, fusion_dim]
        
        # Transformer fusion
        fused = self.transformer(combined)  # [B, 2, fusion_dim]
        
        # Flatten and project
        fused_flat = fused.view(B, -1)  # [B, 2*fusion_dim]
        output = self.output_proj(fused_flat)
        
        return output

class TeacherModel(nn.Module):
    """Complete multimodal teacher model"""
    def __init__(self, num_classes=2, visual_frames=8):
        super().__init__()
        
        # Modality branches
        self.visual_branch = VisualBranch(num_frames=visual_frames, feature_dim=512)
        self.audio_branch = AudioBranch(feature_dim=256)
        
        # Fusion module
        self.fusion_module = FusionModule(
            visual_dim=512, 
            audio_dim=256, 
            fusion_dim=512
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Optional auxiliary heads
        self.lip_sync_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, video_frames, audio_waveform, return_features=False):
        """
        Args:
            video_frames: [B, T, C, H, W]
            audio_waveform: [B, audio_length]
            return_features: bool, whether to return intermediate features
        Returns:
            logits: [B, num_classes]
            features: dict with intermediate features (if return_features=True)
        """
        # Extract modality features
        visual_feat = self.visual_branch(video_frames)
        audio_feat = self.audio_branch(audio_waveform)
        
        # Fuse modalities
        fused_feat = self.fusion_module(visual_feat, audio_feat)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        # Auxiliary outputs
        lip_sync_score = self.lip_sync_head(fused_feat)
        
        if return_features:
            features = {
                'visual_feat': visual_feat,
                'audio_feat': audio_feat,
                'fused_feat': fused_feat,
                'lip_sync_score': lip_sync_score
            }
            return logits, features
        
        return logits

def create_teacher_model(num_classes=2, visual_frames=8):
    """Factory function to create teacher model"""
    return TeacherModel(num_classes=num_classes, visual_frames=visual_frames)

if __name__ == "__main__":
    # Test model creation and forward pass
    model = create_teacher_model()
    
    # Test inputs
    video = torch.randn(2, 8, 3, 224, 224)  # [B, T, C, H, W]
    audio = torch.randn(2, 16000 * 3)       # [B, 3 seconds at 16kHz]
    
    # Forward pass
    logits, features = model(video, audio, return_features=True)
    
    print(f"Model created successfully!")
    print(f"Video input: {video.shape}")
    print(f"Audio input: {audio.shape}")
    print(f"Logits output: {logits.shape}")
    print(f"Features keys: {list(features.keys())}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")