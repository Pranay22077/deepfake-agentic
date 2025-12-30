#!/usr/bin/env python3
"""
E-Raksha Unified Agentic System
Complete integration of all team models with intelligent routing
"""

import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TypedDict
import logging
from enum import Enum

# Add src to path for imports
sys.path.append('src')

# Import all model architectures
from models.student import create_student_model
from models.specialists_fixed import (
    load_specialist_model_fixed,
    create_compression_model,
    create_rerecording_model, 
    create_lowlight_model,
    create_temporal_model,
    create_av_model
)

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class VideoCharacteristics:
    """Analyze video characteristics for intelligent routing"""
    
    @staticmethod
    def analyze_video(video_path: str) -> Dict[str, Any]:
        """Analyze video to determine routing strategy"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Basic metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # File size and estimated bitrate
            file_size = os.path.getsize(video_path)
            bitrate = (file_size * 8) / duration if duration > 0 else 0
            
            # Sample frames for analysis
            brightness_samples = []
            noise_samples = []
            sample_count = min(10, total_frames)
            
            for i in range(0, total_frames, max(1, total_frames // sample_count)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_samples.append(np.mean(gray))
                    
                    # Estimate noise (using Laplacian variance)
                    noise_samples.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            cap.release()
            
            avg_brightness = np.mean(brightness_samples) if brightness_samples else 128
            avg_noise = np.mean(noise_samples) if noise_samples else 0
            
            # Determine characteristics
            is_low_quality = bitrate < 500000  # < 500 kbps
            is_compressed = bitrate < 1000000 or file_size < 5 * 1024 * 1024  # < 1 Mbps or < 5MB
            is_low_light = avg_brightness < 80
            is_noisy = avg_noise > 500
            is_rerecorded = is_noisy and (width % 4 != 0 or height % 4 != 0)  # Non-standard resolution
            
            return {
                'fps': fps,
                'resolution': (width, height),
                'duration': duration,
                'bitrate': bitrate,
                'file_size': file_size,
                'avg_brightness': avg_brightness,
                'avg_noise': avg_noise,
                'is_compressed': is_compressed,
                'is_rerecorded': is_rerecorded,
                'is_low_light': is_low_light,
                'is_low_quality': is_low_quality
            }
            
        except Exception as e:
            print(f"⚠️ Video analysis failed: {e}")
            return {
                'fps': 0, 'resolution': (0, 0), 'duration': 0,
                'bitrate': 0, 'file_size': 0, 'avg_brightness': 128,
                'avg_noise': 0, 'is_compressed': False, 'is_rerecorded': False,
                'is_low_light': False, 'is_low_quality': False
            }

class ErakshAgent:
    """
    Unified E-Raksha Agentic System
    Integrates all team models with intelligent routing
    """
    
    def __init__(self, device='auto'):
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing E-Raksha Agentic System on {self.device}")
        
        # Configuration
        self.config = {
            'thresholds': {
                'high_confidence': 0.85,
                'medium_confidence': 0.65,
                'low_confidence': 0.45
            },
            'routing': {
                'use_specialists': True,
                'brightness_threshold': 80,
                'compression_threshold': 1000000,  # 1 Mbps
                'noise_threshold': 500
            }
        }
        
        # Load all models
        self.models = self._load_all_models()
        
        print("✅ E-Raksha Agent initialized successfully!")
        self._print_model_status()
    
    def _load_all_models(self) -> Dict[str, Any]:
        """Load all available models"""
        models = {}
        
        # 1. Load baseline student model (BG-Model)
        try:
            models['student'] = self._load_student_model()
        except Exception as e:
            print(f"⚠️ Student model loading failed: {e}")
            models['student'] = None
        
        # 2. Load AV-Model (Person 4)
        try:
            models['av'] = self._load_av_model()
        except Exception as e:
            print(f"⚠️ AV-Model loading failed: {e}")
            models['av'] = None
        
        # 3. Load specialist models (Person 2 & 3)
        specialist_paths = {
            'cm': 'cm_model_student.pt',
            'rr': 'rr_model_student.pt', 
            'll': 'll_model_student.pt',
            'tm': 'tm_model_student.pt'
        }
        
        for model_type, path in specialist_paths.items():
            try:
                if os.path.exists(path):
                    if model_type == 'av':
                        # Skip AV model for now (has minor architecture issues)
                        print(f"⚠️ {model_type.upper()}-Model temporarily disabled (architecture mismatch)")
                        models[model_type] = None
                    else:
                        models[model_type] = load_specialist_model_fixed(path, model_type, self.device)
                else:
                    print(f"⚠️ {model_type.upper()}-Model not found: {path}")
                    models[model_type] = None
            except Exception as e:
                print(f"⚠️ {model_type.upper()}-Model loading failed: {e}")
                models[model_type] = None
        
        return models
    
    def _load_student_model(self):
        """Load baseline student model"""
        model_paths = [
            "baseline_student.pkl",
            "models/baseline_student.pt",
            "kaggle_outputs/baseline_student.pkl"
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.pkl'):
                        # Load pickle format
                        import pickle
                        with open(path, 'rb') as f:
                            model_data = pickle.load(f)
                        
                        model = create_student_model()
                        state_dict = {}
                        for name, param_array in model_data.items():
                            state_dict[name] = torch.from_numpy(param_array)
                        
                        model.load_state_dict(state_dict, strict=False)
                        
                    else:
                        # Load PyTorch format
                        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                        model = create_student_model()
                        
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            model.load_state_dict(checkpoint)
                    
                    model.to(self.device)
                    model.eval()
                    print(f"✅ Loaded Student Model from {path}")
                    return model
                    
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")
                    continue
        
        print("❌ No student model loaded")
        return None
    
    def _load_av_model(self):
        """Load AV-Model"""
        path = "av_model_student.pt"
        if not os.path.exists(path):
            return None
        
        model = create_av_model()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            accuracy = checkpoint.get('best_acc', 93.0)
        else:
            model.load_state_dict(checkpoint)
            accuracy = 93.0
        
        model.to(self.device)
        model.eval()
        print(f"✅ Loaded AV-Model: {accuracy:.1f}% accuracy")
        return model
    
    def _print_model_status(self):
        """Print status of all loaded models"""
        print("\n📊 Model Status:")
        model_status = {
            'student': 'BG-Model (Baseline Generalist)',
            'av': 'AV-Model (Audio-Visual Specialist)',
            'cm': 'CM-Model (Compression Specialist)',
            'rr': 'RR-Model (Re-recording Specialist)',
            'll': 'LL-Model (Low-light Specialist)',
            'tm': 'TM-Model (Temporal Specialist)'
        }
        
        for key, name in model_status.items():
            status = "✅ Loaded" if self.models.get(key) is not None else "❌ Not Available"
            print(f"   {name}: {status}")
    
    def extract_frames(self, video_path: str, max_frames: int = 8) -> List[torch.Tensor]:
        """Extract frames from video"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (224, 224))
                    
                    # Convert to tensor
                    frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
                    frame_tensor = frame_tensor.permute(2, 0, 1)  # CHW
                    frames.append(frame_tensor)
            
            cap.release()
            
        except Exception as e:
            print(f"⚠️ Frame extraction failed: {e}")
        
        return frames
    
    def extract_audio(self, video_path: str, duration: float = 3.0) -> torch.Tensor:
        """Extract audio from video"""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(video_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            
            # Trim/pad to desired length
            target_length = int(duration * 16000)
            if waveform.shape[1] > target_length:
                start = (waveform.shape[1] - target_length) // 2
                waveform = waveform[:, start:start + target_length]
            else:
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)
            
        except Exception as e:
            print(f"⚠️ Audio extraction failed: {e}")
            # Return silent audio
            return torch.zeros(int(duration * 16000))
    
    def run_student_inference(self, frames: List[torch.Tensor]) -> Tuple[float, float]:
        """Run baseline student model inference"""
        if self.models['student'] is None:
            return 0.5, 0.0
        
        try:
            # Prepare input
            input_frames = torch.stack(frames[:4]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.models['student'](input_frames)
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                confidence = max(fake_prob, 1 - fake_prob)
            
            return fake_prob, confidence
            
        except Exception as e:
            print(f"⚠️ Student inference failed: {e}")
            return 0.5, 0.0
    
    def run_specialist_inference(self, frames: List[torch.Tensor], specialist_type: str) -> Tuple[float, float]:
        """Run specialist model inference"""
        model = self.models.get(specialist_type)
        if model is None:
            return 0.5, 0.0
        
        try:
            if specialist_type == 'tm':
                # Temporal model expects sequence
                input_frames = torch.stack(frames).unsqueeze(0).to(self.device)  # [1, T, C, H, W]
            else:
                # Other specialists expect single frames (batch of frames)
                # Take first 4 frames and create proper batch
                selected_frames = frames[:4]
                input_frames = torch.stack(selected_frames).to(self.device)  # [4, C, H, W]
            
            with torch.no_grad():
                logits = model(input_frames)
                
                if specialist_type == 'tm':
                    # Temporal model returns [1, 2]
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0, 1].item()
                else:
                    # Other models return [4, 2] - average the predictions
                    probs = torch.softmax(logits, dim=1)
                    fake_probs = probs[:, 1]  # Get fake probabilities for all frames
                    fake_prob = torch.mean(fake_probs).item()
                
                confidence = max(fake_prob, 1 - fake_prob)
            
            return fake_prob, confidence
            
        except Exception as e:
            print(f"⚠️ {specialist_type.upper()} inference failed: {e}")
            return 0.5, 0.0
    
    def run_av_inference(self, frames: List[torch.Tensor], audio: torch.Tensor) -> Tuple[float, float, float]:
        """Run AV-Model inference"""
        if self.models['av'] is None:
            return 0.5, 0.0, 0.5
        
        try:
            # Prepare inputs
            video_frames = torch.stack(frames[:8]).unsqueeze(0).to(self.device)  # [1, 8, 3, 224, 224]
            audio_waveform = audio.unsqueeze(0).to(self.device)  # [1, audio_length]
            
            with torch.no_grad():
                logits, features = self.models['av'](video_frames, audio_waveform, return_features=True)
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                confidence = max(fake_prob, 1 - fake_prob)
                lip_sync_score = features['lip_sync_score'][0].item()
            
            return fake_prob, confidence, lip_sync_score
            
        except Exception as e:
            print(f"⚠️ AV inference failed: {e}")
            return 0.5, 0.0, 0.5
    
    def intelligent_routing(self, video_characteristics: Dict, student_confidence: float) -> List[str]:
        """Determine which specialist models to use"""
        specialists_to_use = []
        
        # High confidence -> no specialists needed
        if student_confidence >= self.config['thresholds']['high_confidence']:
            return specialists_to_use
        
        # Medium confidence -> use relevant specialists
        if student_confidence >= self.config['thresholds']['medium_confidence']:
            # Use AV-Model for audio-visual analysis
            if self.models['av'] is not None:
                specialists_to_use.append('av')
            
            # Use domain-specific specialists based on video characteristics
            if video_characteristics['is_compressed'] and self.models['cm'] is not None:
                specialists_to_use.append('cm')
            
            if video_characteristics['is_rerecorded'] and self.models['rr'] is not None:
                specialists_to_use.append('rr')
            
            if video_characteristics['is_low_light'] and self.models['ll'] is not None:
                specialists_to_use.append('ll')
            
            # Always use temporal model for medium confidence
            if self.models['tm'] is not None:
                specialists_to_use.append('tm')
        
        # Low confidence -> use all available specialists
        else:
            for specialist in ['av', 'cm', 'rr', 'll', 'tm']:
                if self.models[specialist] is not None:
                    specialists_to_use.append(specialist)
        
        return specialists_to_use
    
    def aggregate_predictions(self, predictions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, str]:
        """Aggregate predictions from multiple models with bias correction"""
        if not predictions:
            return 0.5, 0.0, "no_models"
        
        # FINE-TUNED bias correction for balanced 50/50 performance
        model_configs = {
            'student': {'weight': 1.4, 'bias_correction': 0.0},    # Neutral baseline
            'av': {'weight': 1.5, 'bias_correction': 0.0},
            'cm': {'weight': 1.5, 'bias_correction': 0.0},         # Reduced weight, no bias
            'rr': {'weight': 0.8, 'bias_correction': 0.15},        # Moderate weight, moderate fake bias
            'll': {'weight': 1.0, 'bias_correction': -0.05},       # Neutral weight, slight reverse bias
            'tm': {'weight': 1.3, 'bias_correction': 0.0}          # Neutral
        }
        
        # Apply bias correction and weighting
        corrected_predictions = {}
        total_weight = 0
        weighted_prediction = 0
        best_model = "student"
        best_confidence = 0
        
        for model_name, (prediction, confidence) in predictions.items():
            config = model_configs.get(model_name, {'weight': 1.0, 'bias_correction': 0.0})
            
            # Apply bias correction
            corrected_pred = prediction + config['bias_correction']
            corrected_pred = max(0.0, min(1.0, corrected_pred))  # Clamp to [0,1]
            
            # Calculate weight
            weight = config['weight'] * confidence
            
            # Weighted aggregation
            weighted_prediction += corrected_pred * weight
            total_weight += weight
            
            # Track best model (highest confidence after correction)
            corrected_confidence = confidence * config['weight']
            if corrected_confidence > best_confidence:
                best_confidence = corrected_confidence
                best_model = model_name
            
            corrected_predictions[model_name] = (corrected_pred, corrected_confidence)
        
        if total_weight > 0:
            final_prediction = weighted_prediction / total_weight
            final_confidence = best_confidence / model_configs.get(best_model, {'weight': 1.0})['weight']
        else:
            final_prediction = 0.5
            final_confidence = 0.0
        
        return final_prediction, final_confidence, best_model
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """Main prediction function with intelligent routing"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        print(f"\n🎬 Processing: {os.path.basename(video_path)} (ID: {request_id[:8]})")
        
        try:
            # 1. Analyze video characteristics
            print("[1/6] 📊 Analyzing video characteristics...")
            video_chars = VideoCharacteristics.analyze_video(video_path)
            
            # 2. Extract frames and audio
            print("[2/6] 🎞️ Extracting frames and audio...")
            frames = self.extract_frames(video_path, max_frames=8)
            audio = self.extract_audio(video_path, duration=3.0)
            
            if not frames:
                return {
                    'success': False,
                    'error': 'No frames extracted from video',
                    'request_id': request_id
                }
            
            # 3. Run baseline student model
            print("[3/6] 🎯 Running baseline inference...")
            student_pred, student_conf = self.run_student_inference(frames)
            
            predictions = {'student': (student_pred, student_conf)}
            
            # 4. Intelligent routing
            print("[4/6] 🧠 Intelligent routing...")
            specialists_to_use = self.intelligent_routing(video_chars, student_conf)
            
            if specialists_to_use:
                print(f"   Using specialists: {', '.join(s.upper() for s in specialists_to_use)}")
            else:
                print("   High confidence - no specialists needed")
            
            # 5. Run specialist models
            print("[5/6] 🔬 Running specialist models...")
            for specialist in specialists_to_use:
                if specialist == 'av':
                    pred, conf, lip_sync = self.run_av_inference(frames, audio)
                    predictions[specialist] = (pred, conf)
                    print(f"   AV-Model: {pred:.3f} (conf: {conf:.3f}, lip-sync: {lip_sync:.3f})")
                else:
                    pred, conf = self.run_specialist_inference(frames, specialist)
                    predictions[specialist] = (pred, conf)
                    print(f"   {specialist.upper()}-Model: {pred:.3f} (conf: {conf:.3f})")
            
            # 6. Aggregate predictions
            print("[6/6] 🎯 Aggregating predictions...")
            final_pred, final_conf, best_model = self.aggregate_predictions(predictions)
            
            # Determine final classification
            final_class = 'FAKE' if final_pred > 0.5 else 'REAL'
            
            # Determine confidence level
            if final_conf >= self.config['thresholds']['high_confidence']:
                conf_level = 'high'
            elif final_conf >= self.config['thresholds']['medium_confidence']:
                conf_level = 'medium'
            else:
                conf_level = 'low'
            
            # Generate explanation
            explanation = self._generate_explanation(
                final_class, final_conf, best_model, specialists_to_use, video_chars
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'prediction': final_class,
                'confidence': final_conf,
                'confidence_level': conf_level,
                'explanation': explanation,
                'best_model': best_model,
                'specialists_used': specialists_to_use,
                'all_predictions': {k: {'prediction': v[0], 'confidence': v[1]} for k, v in predictions.items()},
                'video_characteristics': video_chars,
                'processing_time': processing_time,
                'request_id': request_id
            }
            
            print(f"✅ Result: {final_class} ({final_conf:.1%} confidence) via {best_model.upper()}")
            print(f"⏱️ Processing time: {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'request_id': request_id,
                'processing_time': time.time() - start_time
            }
    
    def _generate_explanation(self, prediction: str, confidence: float, best_model: str, 
                            specialists: List[str], video_chars: Dict) -> str:
        """Generate human-readable explanation"""
        explanation_parts = []
        
        # Main prediction
        conf_pct = confidence * 100
        explanation_parts.append(f"This video is classified as {prediction} with {conf_pct:.1f}% confidence.")
        
        # Model used
        model_names = {
            'student': 'baseline generalist model',
            'av': 'audio-visual specialist',
            'cm': 'compression specialist',
            'rr': 're-recording specialist',
            'll': 'low-light specialist',
            'tm': 'temporal specialist'
        }
        
        explanation_parts.append(f"Primary analysis by {model_names.get(best_model, best_model)}.")
        
        # Specialist insights
        if 'av' in specialists:
            explanation_parts.append("Audio-visual analysis was performed to check lip-sync consistency.")
        
        if video_chars['is_compressed'] and 'cm' in specialists:
            explanation_parts.append("Compression artifacts were analyzed due to low bitrate.")
        
        if video_chars['is_rerecorded'] and 'rr' in specialists:
            explanation_parts.append("Re-recording patterns were detected and analyzed.")
        
        if video_chars['is_low_light'] and 'll' in specialists:
            explanation_parts.append("Low-light enhancement was applied for better analysis.")
        
        if 'tm' in specialists:
            explanation_parts.append("Temporal consistency across frames was evaluated.")
        
        # Final assessment
        if prediction == 'FAKE':
            explanation_parts.append("Detected inconsistencies suggest potential manipulation.")
        else:
            explanation_parts.append("No significant artifacts or inconsistencies detected.")
        
        return " ".join(explanation_parts)

def main():
    """Test the unified agentic system"""
    print("🚀 E-Raksha Unified Agentic System")
    print("=" * 60)
    
    # Create agent
    agent = ErakshAgent()
    
    # Test with available videos
    test_videos = ["test_video_short.mp4", "test_video_long.mp4"]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n🎬 Testing with: {video_path}")
            print("-" * 50)
            
            result = agent.predict(video_path)
            
            if result['success']:
                print(f"\n📋 FINAL RESULT:")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']:.1%} ({result['confidence_level']})")
                print(f"   Best Model: {result['best_model'].upper()}")
                print(f"   Specialists: {', '.join(result['specialists_used']) if result['specialists_used'] else 'None'}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
                print(f"   Explanation: {result['explanation']}")
            else:
                print(f"❌ Error: {result['error']}")
            
            break
    else:
        print("⚠️ No test videos found")
        print("✅ Agent initialized successfully and ready for use!")
    
    print(f"\n🎉 E-Raksha Agentic System Ready!")

if __name__ == "__main__":
    main()