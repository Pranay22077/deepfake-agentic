#!/usr/bin/env python3
"""
Agentic Cascade Core
Implements multi-stage deepfake detection with adaptive decision making
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import torchaudio

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model
from models.teacher import create_teacher_model
from preprocess.extract_faces import FaceExtractor
from preprocess.extract_audio import AudioExtractor

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class StageType(Enum):
    PRECHECK = "precheck"
    STUDENT = "student"
    VERIFIER = "verifier"
    TEACHER = "teacher"

@dataclass
class AgentDecision:
    """Agent decision output"""
    label: str  # 'real' or 'fake'
    confidence: float
    stage_taken: StageType
    confidence_level: ConfidenceLevel
    processing_time: float
    explanations: Dict[str, Any]
    heatmap_files: List[str]
    metadata: Dict[str, Any]

class ConfidenceSmoothing:
    """Temporal confidence smoothing using EMA"""
    
    def __init__(self, alpha=0.6, window_size=8):
        self.alpha = alpha
        self.window_size = window_size
        self.history = []
    
    def update(self, confidence: float) -> float:
        """Update with new confidence and return smoothed value"""
        self.history.append(confidence)
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Apply exponential moving average
        if len(self.history) == 1:
            return confidence
        
        smoothed = self.history[0]
        for i in range(1, len(self.history)):
            smoothed = self.alpha * self.history[i] + (1 - self.alpha) * smoothed
        
        return smoothed
    
    def reset(self):
        """Reset history"""
        self.history = []

class HeatmapGenerator:
    """Generate explanation heatmaps using GradCAM"""
    
    def __init__(self, model, target_layer_name='visual_backbone.features'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
    
    def generate_heatmap(self, input_tensor, class_idx=1):
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].sum()
        class_score.backward()
        
        if self.gradients is None or self.activations is None:
            return None
        
        # Generate heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        return heatmap

class AgentCore:
    """Main agentic cascade system"""
    
    def __init__(self, 
                 student_model_path: str,
                 verifier_model_path: Optional[str] = None,
                 teacher_model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 device: str = 'cpu'):
        
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize models
        self.student_model = self._load_student_model(student_model_path)
        self.verifier_model = self._load_verifier_model(verifier_model_path)
        self.teacher_model = self._load_teacher_model(teacher_model_path)
        
        # Initialize components
        self.face_extractor = FaceExtractor()
        self.audio_extractor = AudioExtractor()
        self.confidence_smoother = ConfidenceSmoothing(
            alpha=self.config['smoothing']['alpha'],
            window_size=self.config['smoothing']['window_size']
        )
        
        # Initialize heatmap generator
        if self.student_model:
            self.heatmap_generator = HeatmapGenerator(self.student_model)
        
        # Decision thresholds
        self.high_thresh = self.config['thresholds']['high_confidence']
        self.low_thresh = self.config['thresholds']['low_confidence']
        
        # Logging
        self.log_file = Path(self.config['logging']['log_file'])
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load agent configuration"""
        default_config = {
            'thresholds': {
                'high_confidence': 0.85,
                'low_confidence': 0.15
            },
            'smoothing': {
                'alpha': 0.6,
                'window_size': 8
            },
            'preprocessing': {
                'num_frames': 8,
                'audio_duration': 3.0,
                'sample_rate': 16000
            },
            'logging': {
                'log_file': 'logs/agent_logs.jsonl',
                'save_heatmaps': True,
                'heatmap_dir': 'logs/heatmaps'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configs
            default_config.update(user_config)
        
        return default_config
    
    def _load_student_model(self, model_path: str):
        """Load student model"""
        if not os.path.exists(model_path):
            self.logger.warning(f"Student model not found: {model_path}")
            return None
        
        model = create_student_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded student model from {model_path}")
        return model
    
    def _load_verifier_model(self, model_path: Optional[str]):
        """Load verifier model (can be same as student or different)"""
        if not model_path or not os.path.exists(model_path):
            self.logger.info("No verifier model specified, using student model")
            return self.student_model
        
        # For now, use same architecture as student
        model = create_student_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded verifier model from {model_path}")
        return model
    
    def _load_teacher_model(self, model_path: Optional[str]):
        """Load teacher model (optional, for final stage)"""
        if not model_path or not os.path.exists(model_path):
            self.logger.info("No teacher model specified")
            return None
        
        model = create_teacher_model()
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Loaded teacher model from {model_path}")
        return model
    
    def _precheck(self, video_path: str) -> Tuple[bool, Dict]:
        """Stage 0: Basic heuristic checks"""
        checks = {
            'file_exists': os.path.exists(video_path),
            'file_size': 0,
            'duration': 0,
            'has_faces': False,
            'has_audio': False
        }
        
        if not checks['file_exists']:
            return False, checks
        
        # File size check
        checks['file_size'] = os.path.getsize(video_path)
        if checks['file_size'] == 0:
            return False, checks
        
        # Quick face detection check
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            checks['duration'] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            
            # Check first few frames for faces
            for _ in range(min(5, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
                ret, frame = cap.read()
                if ret:
                    faces = self.face_extractor.detect_faces(frame)
                    if len(faces) > 0:
                        checks['has_faces'] = True
                        break
        cap.release()
        
        # Quick audio check
        try:
            waveform, sr = torchaudio.load(video_path)
            checks['has_audio'] = waveform.numel() > 0
        except:
            checks['has_audio'] = False
        
        # Pass if has faces (minimum requirement)
        return checks['has_faces'], checks
    
    def _extract_features(self, video_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract video frames and audio features"""
        # Extract video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            # Sample frames uniformly
            num_frames = self.config['preprocessing']['num_frames']
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
        cap.release()
        
        # Convert frames to tensor
        if len(frames) == num_frames:
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        else:
            frames = torch.zeros(num_frames, 3, 224, 224)
        
        # Extract audio
        try:
            waveform, sr = torchaudio.load(video_path)
            
            # Resample if needed
            sample_rate = self.config['preprocessing']['sample_rate']
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Trim or pad to desired length
            audio_duration = self.config['preprocessing']['audio_duration']
            audio_samples = int(audio_duration * sample_rate)
            
            if waveform.shape[1] > audio_samples:
                start = (waveform.shape[1] - audio_samples) // 2
                waveform = waveform[:, start:start + audio_samples]
            else:
                padding = audio_samples - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            
            audio = waveform.squeeze(0)
        except:
            # Return silence if audio extraction fails
            audio_samples = int(self.config['preprocessing']['audio_duration'] * 
                              self.config['preprocessing']['sample_rate'])
            audio = torch.zeros(audio_samples)
        
        return frames, audio
    
    def _run_student_stage(self, frames: torch.Tensor, audio: torch.Tensor) -> Tuple[float, Dict]:
        """Stage 1: Fast student model inference"""
        if self.student_model is None:
            return 0.5, {}
        
        frames_batch = frames.unsqueeze(0).to(self.device)
        audio_batch = audio.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, features = self.student_model(frames_batch, audio_batch, return_features=True)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, 1].item()  # Fake probability
        
        return confidence, features
    
    def _run_verifier_stage(self, frames: torch.Tensor, audio: torch.Tensor) -> Tuple[float, Dict]:
        """Stage 2: Verifier model (more thorough)"""
        if self.verifier_model is None:
            return self._run_student_stage(frames, audio)
        
        frames_batch = frames.unsqueeze(0).to(self.device)
        audio_batch = audio.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, features = self.verifier_model(frames_batch, audio_batch, return_features=True)
            probs = torch.softmax(logits, dim=1)
            confidence = probs[0, 1].item()
        
        return confidence, features
    
    def _generate_explanations(self, frames: torch.Tensor, confidence: float, 
                             stage: StageType) -> Tuple[Dict, List[str]]:
        """Generate explanations and heatmaps"""
        explanations = {
            'confidence_raw': confidence,
            'stage_used': stage.value,
            'num_frames': frames.shape[0]
        }
        
        heatmap_files = []
        
        if self.config['logging']['save_heatmaps'] and self.heatmap_generator:
            try:
                # Generate heatmap for middle frame
                mid_frame_idx = frames.shape[0] // 2
                frame_tensor = frames[mid_frame_idx:mid_frame_idx+1].unsqueeze(0).to(self.device)
                
                heatmap = self.heatmap_generator.generate_heatmap(frame_tensor)
                
                if heatmap is not None:
                    # Save heatmap
                    heatmap_dir = Path(self.config['logging']['heatmap_dir'])
                    heatmap_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = int(time.time() * 1000)
                    heatmap_file = heatmap_dir / f"heatmap_{timestamp}.png"
                    
                    # Convert to image and save
                    heatmap_img = (heatmap * 255).astype(np.uint8)
                    heatmap_img = cv2.resize(heatmap_img, (224, 224))
                    cv2.imwrite(str(heatmap_file), heatmap_img)
                    
                    heatmap_files.append(str(heatmap_file))
            except Exception as e:
                self.logger.warning(f"Failed to generate heatmap: {e}")
        
        return explanations, heatmap_files
    
    def _log_decision(self, video_path: str, decision: AgentDecision):
        """Log decision to file"""
        log_entry = {
            'timestamp': time.time(),
            'video_path': video_path,
            'label': decision.label,
            'confidence': decision.confidence,
            'confidence_level': decision.confidence_level.value,
            'stage_taken': decision.stage_taken.value,
            'processing_time': decision.processing_time,
            'heatmap_files': decision.heatmap_files,
            'metadata': decision.metadata
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def predict(self, video_path: str) -> AgentDecision:
        """Main prediction function with agentic cascade"""
        start_time = time.time()
        
        # Stage 0: Precheck
        precheck_pass, precheck_info = self._precheck(video_path)
        if not precheck_pass:
            return AgentDecision(
                label='unknown',
                confidence=0.0,
                stage_taken=StageType.PRECHECK,
                confidence_level=ConfidenceLevel.LOW,
                processing_time=time.time() - start_time,
                explanations={'precheck_failed': precheck_info},
                heatmap_files=[],
                metadata=precheck_info
            )
        
        # Extract features
        frames, audio = self._extract_features(video_path)
        
        # Stage 1: Student model
        student_confidence, student_features = self._run_student_stage(frames, audio)
        
        # Apply confidence smoothing
        smoothed_confidence = self.confidence_smoother.update(student_confidence)
        
        # Decision policy
        stage_taken = StageType.STUDENT
        final_confidence = smoothed_confidence
        
        if smoothed_confidence >= self.high_thresh:
            # High confidence fake - immediate decision
            confidence_level = ConfidenceLevel.HIGH
            label = 'fake'
        elif smoothed_confidence <= self.low_thresh:
            # High confidence real - immediate decision
            confidence_level = ConfidenceLevel.HIGH
            label = 'real'
        else:
            # Medium confidence - escalate to verifier
            stage_taken = StageType.VERIFIER
            verifier_confidence, verifier_features = self._run_verifier_stage(frames, audio)
            final_confidence = verifier_confidence
            
            if verifier_confidence >= 0.5:
                label = 'fake'
                confidence_level = ConfidenceLevel.MEDIUM if verifier_confidence < 0.8 else ConfidenceLevel.HIGH
            else:
                label = 'real'
                confidence_level = ConfidenceLevel.MEDIUM if verifier_confidence > 0.2 else ConfidenceLevel.HIGH
        
        # Generate explanations
        explanations, heatmap_files = self._generate_explanations(frames, final_confidence, stage_taken)
        
        # Create decision
        decision = AgentDecision(
            label=label,
            confidence=final_confidence,
            stage_taken=stage_taken,
            confidence_level=confidence_level,
            processing_time=time.time() - start_time,
            explanations=explanations,
            heatmap_files=heatmap_files,
            metadata={
                'precheck_info': precheck_info,
                'student_confidence': student_confidence,
                'smoothed_confidence': smoothed_confidence
            }
        )
        
        # Log decision
        self._log_decision(video_path, decision)
        
        return decision
    
    def get_agent_recommendation(self, decision: AgentDecision) -> str:
        """Generate human-readable recommendation"""
        if decision.confidence_level == ConfidenceLevel.HIGH:
            if decision.label == 'fake':
                return "ALERT: High confidence deepfake detected. Immediate verification recommended."
            else:
                return "CLEAR: High confidence authentic content."
        elif decision.confidence_level == ConfidenceLevel.MEDIUM:
            return f"CAUTION: Medium confidence {decision.label}. Consider additional verification."
        else:
            return "UNCERTAIN: Low confidence prediction. Manual review strongly recommended."

def create_agent(student_model_path: str, 
                config_path: Optional[str] = None,
                device: str = 'cpu') -> AgentCore:
    """Factory function to create agent"""
    return AgentCore(
        student_model_path=student_model_path,
        config_path=config_path,
        device=device
    )

if __name__ == "__main__":
    # Test agent creation
    print("Testing agent core...")
    
    # This would normally use real model paths
    try:
        agent = create_agent("models/student_distilled.pt")
        print("Agent created successfully!")
        print(f"Configuration: {agent.config}")
    except Exception as e:
        print(f"Agent creation failed (expected without models): {e}")
        print("Agent architecture is ready for deployment!")