import torch
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
import cv2
import numpy as np
import random
from typing import Tuple, Optional
import io
import subprocess
import tempfile
import os

class VideoAugmentation:
    """Heavy augmentations for teacher training robustness"""
    
    def __init__(self, 
                 compression_prob=0.3,
                 brightness_prob=0.4,
                 contrast_prob=0.4,
                 blur_prob=0.2,
                 noise_prob=0.3):
        self.compression_prob = compression_prob
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        
        # Color augmentations
        self.color_jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        )
        
    def apply_compression(self, frames: torch.Tensor, quality: int = None) -> torch.Tensor:
        """Simulate video compression artifacts"""
        if quality is None:
            quality = random.randint(15, 50)  # Low to medium quality
            
        B, T, C, H, W = frames.shape
        compressed_frames = []
        
        for b in range(B):
            batch_frames = []
            for t in range(T):
                frame = frames[b, t].permute(1, 2, 0).numpy()  # [H, W, C]
                frame = (frame * 255).astype(np.uint8)
                
                # JPEG compression simulation
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
                _, encoded = cv2.imencode('.jpg', frame, encode_param)
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                
                # Convert back to tensor
                frame_tensor = torch.from_numpy(decoded).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
                batch_frames.append(frame_tensor)
            
            compressed_frames.append(torch.stack(batch_frames))
        
        return torch.stack(compressed_frames)
    
    def apply_blur(self, frames: torch.Tensor, kernel_size: int = None) -> torch.Tensor:
        """Apply motion blur or gaussian blur"""
        if kernel_size is None:
            kernel_size = random.choice([3, 5, 7])
            
        B, T, C, H, W = frames.shape
        blurred = frames.clone()
        
        for b in range(B):
            for t in range(T):
                for c in range(C):
                    frame = frames[b, t, c].numpy()
                    if random.random() < 0.5:
                        # Gaussian blur
                        blurred[b, t, c] = torch.from_numpy(
                            cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
                        )
                    else:
                        # Motion blur
                        kernel = np.zeros((kernel_size, kernel_size))
                        kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                        kernel = kernel / kernel_size
                        blurred[b, t, c] = torch.from_numpy(
                            cv2.filter2D(frame, -1, kernel)
                        )
        
        return blurred
    
    def add_noise(self, frames: torch.Tensor, noise_std: float = None) -> torch.Tensor:
        """Add gaussian noise"""
        if noise_std is None:
            noise_std = random.uniform(0.01, 0.05)
            
        noise = torch.randn_like(frames) * noise_std
        return torch.clamp(frames + noise, 0, 1)
    
    def simulate_screen_recording(self, frames: torch.Tensor) -> torch.Tensor:
        """Simulate screen recording artifacts"""
        # Add slight brightness variation
        brightness_factor = random.uniform(0.8, 1.2)
        frames = frames * brightness_factor
        
        # Add moire patterns (simplified)
        B, T, C, H, W = frames.shape
        x = torch.arange(W).float()
        y = torch.arange(H).float()
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Create subtle moire pattern
        freq = random.uniform(0.1, 0.3)
        moire = 0.02 * torch.sin(freq * xx) * torch.sin(freq * yy)
        moire = moire.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
        
        frames = frames + moire
        return torch.clamp(frames, 0, 1)
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations"""
        # Compression
        if random.random() < self.compression_prob:
            frames = self.apply_compression(frames)
        
        # Blur
        if random.random() < self.blur_prob:
            frames = self.apply_blur(frames)
        
        # Noise
        if random.random() < self.noise_prob:
            frames = self.add_noise(frames)
        
        # Color augmentations
        if random.random() < self.brightness_prob:
            B, T, C, H, W = frames.shape
            for b in range(B):
                for t in range(T):
                    frames[b, t] = self.color_jitter(frames[b, t])
        
        # Screen recording simulation
        if random.random() < 0.2:
            frames = self.simulate_screen_recording(frames)
        
        return frames

class AudioAugmentation:
    """Audio augmentations for robustness"""
    
    def __init__(self,
                 pitch_prob=0.3,
                 speed_prob=0.3,
                 noise_prob=0.4,
                 compression_prob=0.2):
        self.pitch_prob = pitch_prob
        self.speed_prob = speed_prob
        self.noise_prob = noise_prob
        self.compression_prob = compression_prob
    
    def pitch_shift(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Pitch shifting"""
        shift_factor = random.uniform(0.8, 1.2)  # ±20% pitch change
        
        # Use torchaudio pitch shift
        pitch_shift_transform = torchaudio.transforms.PitchShift(
            sample_rate=sample_rate,
            n_steps=int(12 * np.log2(shift_factor))  # Convert to semitones
        )
        
        return pitch_shift_transform(waveform)
    
    def time_stretch(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Time stretching without pitch change"""
        stretch_factor = random.uniform(0.9, 1.1)  # ±10% speed change
        
        # Simple resampling approach
        original_length = waveform.shape[-1]
        new_length = int(original_length / stretch_factor)
        
        # Resample
        resampled = F.interpolate(
            waveform.unsqueeze(0), 
            size=new_length, 
            mode='linear', 
            align_corners=False
        ).squeeze(0)
        
        # Pad or trim to original length
        if new_length > original_length:
            resampled = resampled[..., :original_length]
        else:
            padding = original_length - new_length
            resampled = F.pad(resampled, (0, padding))
        
        return resampled
    
    def add_noise(self, waveform: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """Add background noise"""
        if snr_db is None:
            snr_db = random.uniform(10, 30)  # 10-30 dB SNR
        
        # Generate noise
        noise = torch.randn_like(waveform)
        
        # Calculate noise power for desired SNR
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * torch.sqrt(noise_power / torch.mean(noise ** 2))
        
        return waveform + noise
    
    def simulate_compression(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Simulate audio compression (MP3-like)"""
        # Simple low-pass filtering to simulate compression
        cutoff_freq = random.uniform(4000, 8000)  # Hz
        
        # Create low-pass filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Simple butterworth-like filtering using conv1d
        # This is a simplified version - in practice you'd use proper filter design
        kernel_size = 15
        kernel = torch.ones(1, 1, kernel_size) / kernel_size
        
        # Apply filtering
        padded = F.pad(waveform.unsqueeze(0), (kernel_size//2, kernel_size//2), mode='reflect')
        filtered = F.conv1d(padded, kernel, padding=0)
        
        return filtered.squeeze(0)
    
    def __call__(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """Apply random audio augmentations"""
        # Pitch shift
        if random.random() < self.pitch_prob:
            waveform = self.pitch_shift(waveform, sample_rate)
        
        # Time stretch
        if random.random() < self.speed_prob:
            waveform = self.time_stretch(waveform, sample_rate)
        
        # Add noise
        if random.random() < self.noise_prob:
            waveform = self.add_noise(waveform)
        
        # Compression simulation
        if random.random() < self.compression_prob:
            waveform = self.simulate_compression(waveform, sample_rate)
        
        return waveform

class MultimodalAugmentation:
    """Combined video and audio augmentations"""
    
    def __init__(self, 
                 video_aug_prob=0.7,
                 audio_aug_prob=0.6,
                 sync_corruption_prob=0.1):
        self.video_aug = VideoAugmentation()
        self.audio_aug = AudioAugmentation()
        self.video_aug_prob = video_aug_prob
        self.audio_aug_prob = audio_aug_prob
        self.sync_corruption_prob = sync_corruption_prob
    
    def corrupt_sync(self, video_frames: torch.Tensor, audio_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Introduce audio-video synchronization issues"""
        # Shift audio relative to video
        shift_samples = random.randint(-8000, 8000)  # ±0.5 seconds at 16kHz
        
        if shift_samples > 0:
            # Delay audio
            audio_waveform = F.pad(audio_waveform, (shift_samples, 0))
        elif shift_samples < 0:
            # Advance audio
            audio_waveform = audio_waveform[..., abs(shift_samples):]
        
        return video_frames, audio_waveform
    
    def __call__(self, video_frames: torch.Tensor, audio_waveform: torch.Tensor, sample_rate: int = 16000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply multimodal augmentations"""
        # Video augmentations
        if random.random() < self.video_aug_prob:
            video_frames = self.video_aug(video_frames)
        
        # Audio augmentations
        if random.random() < self.audio_aug_prob:
            audio_waveform = self.audio_aug(audio_waveform, sample_rate)
        
        # Sync corruption (for negative examples)
        if random.random() < self.sync_corruption_prob:
            video_frames, audio_waveform = self.corrupt_sync(video_frames, audio_waveform)
        
        return video_frames, audio_waveform

# Factory functions
def create_teacher_augmentation():
    """Create augmentation pipeline for teacher training"""
    return MultimodalAugmentation(
        video_aug_prob=0.8,
        audio_aug_prob=0.7,
        sync_corruption_prob=0.15
    )

def create_student_augmentation():
    """Create lighter augmentation pipeline for student training"""
    return MultimodalAugmentation(
        video_aug_prob=0.6,
        audio_aug_prob=0.5,
        sync_corruption_prob=0.1
    )

if __name__ == "__main__":
    # Test augmentations
    print("Testing augmentation pipeline...")
    
    # Create test data
    video = torch.randn(2, 8, 3, 224, 224)  # [B, T, C, H, W]
    audio = torch.randn(2, 16000 * 3)       # [B, 3 seconds]
    
    # Test augmentations
    aug = create_teacher_augmentation()
    aug_video, aug_audio = aug(video, audio)
    
    print(f"Original video: {video.shape}")
    print(f"Augmented video: {aug_video.shape}")
    print(f"Original audio: {audio.shape}")
    print(f"Augmented audio: {aug_audio.shape}")
    print("Augmentation pipeline working correctly!")