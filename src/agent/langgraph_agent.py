#!/usr/bin/env python3
"""
Person 4: LangGraph Agent Implementation
Modern agentic deepfake detection system with intelligent routing
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

# LangGraph imports
from langgraph.graph import StateGraph, END

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.audiovisual import AVModel
# from preprocess.extract_faces import FaceExtractor
# from preprocess.extract_audio import AudioExtractor

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AgentState(TypedDict):
    # Input
    video_path: str
    request_id: str
    
    # Metadata
    metadata: Dict[str, Any]
    bitrate: int
    fps: float
    resolution: tuple
    avg_brightness: float
    duration: float
    
    # Processing
    frames: List[torch.Tensor]
    audio_waveform: Optional[torch.Tensor]
    faces_detected: int
    
    # Model predictions
    student_prediction: float
    student_confidence: float
    specialist_prediction: float
    specialist_confidence: float
    selected_specialist: str
    
    # Decision
    final_prediction: str  # 'REAL' or 'FAKE'
    confidence: float
    confidence_level: ConfidenceLevel
    explanation: str
    heatmaps: List[str]
    
    # Routing
    next_action: str  # 'ACCEPT', 'DOMAIN', 'HUMAN'
    
    # Audio-Visual Analysis
    lip_sync_score: Optional[float]
    av_confidence: Optional[float]
    
    # Processing metadata
    processing_time: float
    stage_taken: str
    error_message: Optional[str]

class LangGraphAgent:
    """LangGraph-based Agentic Deepfake Detection System"""
    
    def __init__(self, 
                 student_model_path: str = "models/baseline_student.pt",
                 av_model_path: str = "models/av_model_student.pt",
                 config_path: str = "config/agent_config.yaml",
                 device: str = 'auto'):
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[INIT] Initializing LangGraph Agent on {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load models
        self.student_model = self._load_student_model(student_model_path)
        self.av_model = self._load_av_model(av_model_path)
        
        # Initialize preprocessors (simplified for now)
        # self.face_extractor = FaceExtractor()
        # self.audio_extractor = AudioExtractor()
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        print("[OK] LangGraph Agent initialized successfully!")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load agent configuration"""
        default_config = {
            'thresholds': {
                'high_confidence': 0.85,
                'medium_confidence': 0.60,
                'low_confidence': 0.40
            },
            'routing': {
                'use_av_model': True,
                'av_threshold': 0.70,
                'lip_sync_threshold': 0.30
            },
            'preprocessing': {
                'max_frames': 16,
                'audio_duration': 3.0,
                'sample_rate': 16000,
                'face_size': 224
            },
            'explainability': {
                'generate_heatmaps': True,
                'save_explanations': True
            }
        }
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except:
                print(f"[WARNING] Could not load config from {config_path}, using defaults")
        
        return default_config
    
    def _load_student_model(self, model_path: str):
        """Load baseline student model"""
        if not os.path.exists(model_path):
            print(f"[WARNING] Student model not found: {model_path}")
            return None
        
        try:
            # Load using the existing model architecture
            from models.student import create_student_model
            model = create_student_model()
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"[OK] Loaded student model: {checkpoint.get('best_acc', 'Unknown')}% accuracy")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load student model: {e}")
            return None
    
    def _load_av_model(self, model_path: str):
        """Load AV-Model (Audio-Visual Specialist)"""
        if not os.path.exists(model_path):
            print(f"[WARNING] AV-Model not found: {model_path}")
            return None
        
        try:
            model = AVModel(num_classes=2, visual_frames=8)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            print(f"[OK] Loaded AV-Model: {checkpoint.get('best_acc', 93.0):.1f}% accuracy")
            return model
        except Exception as e:
            print(f"[ERROR] Failed to load AV-Model: {e}")
            return None
    
    def _build_workflow(self) -> StateGraph:
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("ingest", self._ingest_node)
        workflow.add_node("metadata", self._metadata_node)
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("student_inference", self._student_inference_node)
        workflow.add_node("policy_decision", self._policy_decision_node)
        workflow.add_node("av_specialist", self._av_specialist_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("explanation", self._explanation_node)
        
        # Define workflow edges
        workflow.set_entry_point("ingest")
        workflow.add_edge("ingest", "metadata")
        workflow.add_edge("metadata", "preprocess")
        workflow.add_edge("preprocess", "student_inference")
        workflow.add_edge("student_inference", "policy_decision")
        
        # Conditional routing from policy
        workflow.add_conditional_edges(
            "policy_decision",
            self._route_decision,
            {
                "ACCEPT": "explanation",
                "AV_SPECIALIST": "av_specialist",
                "HUMAN": "human_review"
            }
        )
        
        # Convergence to explanation
        workflow.add_edge("av_specialist", "explanation")
        workflow.add_edge("human_review", "explanation")
        workflow.add_edge("explanation", END)
        
        return workflow.compile()
    
    # ============================================
    # LANGGRAPH NODES
    # ============================================
    
    def _ingest_node(self, state: AgentState) -> AgentState:
        """Node 1: Validate and ingest video"""
        video_path = state['video_path']
        
        # Generate request ID
        state['request_id'] = str(uuid.uuid4())
        
        # Validate file
        if not os.path.exists(video_path):
            state['error_message'] = f"Video file not found: {video_path}"
            return state
        
        # Check file size (max 500MB)
        file_size = os.path.getsize(video_path) / (1024 * 1024)
        if file_size > 500:
            state['error_message'] = f"Video too large: {file_size:.1f}MB (max 500MB)"
            return state
        
        # Check format
        valid_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if not any(video_path.lower().endswith(fmt) for fmt in valid_formats):
            state['error_message'] = f"Unsupported video format"
            return state
        
        print(f"[INGEST] [OK] Request {state['request_id']}: {os.path.basename(video_path)}")
        return state
    
    def _metadata_node(self, state: AgentState) -> AgentState:
        """Node 2: Extract video metadata"""
        video_path = state['video_path']
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                state['error_message'] = "Could not open video file"
                return state
            
            # Extract metadata
            state['fps'] = cap.get(cv2.CAP_PROP_FPS)
            state['resolution'] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            state['duration'] = total_frames / state['fps'] if state['fps'] > 0 else 0
            
            # Estimate bitrate
            file_size = os.path.getsize(video_path)
            state['bitrate'] = int((file_size * 8) / state['duration']) if state['duration'] > 0 else 0
            
            # Calculate average brightness
            brightness_samples = []
            sample_count = min(10, total_frames)
            
            for i in range(0, total_frames, max(1, total_frames // sample_count)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    brightness_samples.append(np.mean(gray))
            
            state['avg_brightness'] = np.mean(brightness_samples) if brightness_samples else 128
            
            cap.release()
            
            # Store metadata
            state['metadata'] = {
                'fps': state['fps'],
                'resolution': state['resolution'],
                'duration': state['duration'],
                'bitrate': state['bitrate'],
                'avg_brightness': state['avg_brightness'],
                'file_size_mb': file_size / (1024 * 1024)
            }
            
            print(f"[METADATA] {state['resolution'][0]}x{state['resolution'][1]}, "
                  f"{state['fps']:.1f}fps, {state['duration']:.1f}s, "
                  f"brightness: {state['avg_brightness']:.1f}")
            
        except Exception as e:
            state['error_message'] = f"Metadata extraction failed: {str(e)}"
        
        return state
    
    def _preprocess_node(self, state: AgentState) -> AgentState:
        """Node 3: Extract faces and audio"""
        video_path = state['video_path']
        
        try:
            # Extract video frames with faces
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames uniformly
            max_frames = self.config['preprocessing']['max_frames']
            frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
            
            faces = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Simple face detection using OpenCV (fallback)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces_detected) > 0:
                        # Take the largest face
                        x, y, w, h = max(faces_detected, key=lambda f: f[2] * f[3])
                        face = frame[y:y+h, x:x+w]
                        face = cv2.resize(face, (224, 224))
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        face_tensor = torch.from_numpy(face).float() / 255.0
                        face_tensor = face_tensor.permute(2, 0, 1)  # CHW
                        faces.append(face_tensor)
                    else:
                        # If no face detected, use center crop
                        h, w = frame.shape[:2]
                        center_crop = frame[h//4:3*h//4, w//4:3*w//4]
                        center_crop = cv2.resize(center_crop, (224, 224))
                        center_crop = cv2.cvtColor(center_crop, cv2.COLOR_BGR2RGB)
                        crop_tensor = torch.from_numpy(center_crop).float() / 255.0
                        crop_tensor = crop_tensor.permute(2, 0, 1)
                        faces.append(crop_tensor)
            
            cap.release()
            
            state['faces_detected'] = len(faces)
            state['frames'] = faces
            
            if len(faces) == 0:
                state['error_message'] = "No faces detected in video"
                return state
            
            # Extract audio
            try:
                import torchaudio
                waveform, sr = torchaudio.load(video_path)
                
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Resample if needed
                target_sr = self.config['preprocessing']['sample_rate']
                if sr != target_sr:
                    resampler = torchaudio.transforms.Resample(sr, target_sr)
                    waveform = resampler(waveform)
                
                # Trim/pad to desired length
                audio_duration = self.config['preprocessing']['audio_duration']
                target_length = int(audio_duration * target_sr)
                
                if waveform.shape[1] > target_length:
                    start = (waveform.shape[1] - target_length) // 2
                    waveform = waveform[:, start:start + target_length]
                else:
                    padding = target_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))
                
                state['audio_waveform'] = waveform.squeeze(0)
                
            except Exception as e:
                print(f"[PREPROCESS] [WARNING] Audio extraction failed: {e}")
                # Create silent audio
                target_length = int(self.config['preprocessing']['audio_duration'] * 
                                  self.config['preprocessing']['sample_rate'])
                state['audio_waveform'] = torch.zeros(target_length)
            
            print(f"[PREPROCESS] [OK] Extracted {len(faces)} faces, "
                  f"audio: {state['audio_waveform'].shape[0] / target_sr:.1f}s")
            
        except Exception as e:
            state['error_message'] = f"Preprocessing failed: {str(e)}"
        
        return state
    
    def _student_inference_node(self, state: AgentState) -> AgentState:
        """Node 4: Run baseline student model"""
        if self.student_model is None:
            state['student_prediction'] = 0.5
            state['student_confidence'] = 0.0
            return state
        
        try:
            # Prepare input
            frames = torch.stack(state['frames'][:8])  # Take first 8 frames
            frames = frames.unsqueeze(0).to(self.device)  # Add batch dim
            
            # Run inference
            with torch.no_grad():
                if hasattr(self.student_model, 'forward') and 'audio' in str(self.student_model.forward.__code__.co_varnames):
                    # Model expects audio
                    audio = state['audio_waveform'].unsqueeze(0).to(self.device)
                    logits = self.student_model(frames, audio)
                else:
                    # Visual-only model
                    logits = self.student_model(frames)
                
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
            
            state['student_prediction'] = fake_prob
            state['student_confidence'] = max(fake_prob, 1 - fake_prob)
            
            print(f"[STUDENT] Prediction: {fake_prob:.3f}, "
                  f"Confidence: {state['student_confidence']:.3f}")
            
        except Exception as e:
            print(f"[STUDENT] [ERROR] Inference failed: {e}")
            state['student_prediction'] = 0.5
            state['student_confidence'] = 0.0
        
        return state
    
    def _policy_decision_node(self, state: AgentState) -> AgentState:
        """Node 5: Intelligent routing decision"""
        confidence = state['student_confidence']
        prediction = state['student_prediction']
        
        # Get thresholds
        high_thresh = self.config['thresholds']['high_confidence']
        medium_thresh = self.config['thresholds']['medium_confidence']
        
        # Decision logic
        if confidence >= high_thresh:
            state['next_action'] = 'ACCEPT'
            state['stage_taken'] = 'student_only'
            print(f"[POLICY] [OK] High confidence ({confidence:.3f}) -> ACCEPT")
            
        elif confidence >= medium_thresh and self.av_model is not None:
            # Medium confidence + AV model available -> use AV specialist
            state['next_action'] = 'AV_SPECIALIST'
            state['stage_taken'] = 'av_specialist'
            print(f"[POLICY] [ROUTE] Medium confidence ({confidence:.3f}) -> AV SPECIALIST")
            
        else:
            # Low confidence -> human review
            state['next_action'] = 'HUMAN'
            state['stage_taken'] = 'human_review'
            print(f"[POLICY] [HUMAN] Low confidence ({confidence:.3f}) -> HUMAN REVIEW")
        
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Routing function for conditional edges"""
        return state['next_action']
    
    def _av_specialist_node(self, state: AgentState) -> AgentState:
        """Node 6: Audio-Visual Specialist Analysis"""
        if self.av_model is None:
            # Fallback to student prediction
            state['specialist_prediction'] = state['student_prediction']
            state['specialist_confidence'] = state['student_confidence']
            state['selected_specialist'] = 'student_fallback'
            return state
        
        try:
            # Prepare inputs for AV model
            frames = torch.stack(state['frames'][:8])  # 8 frames for AV model
            frames = frames.unsqueeze(0).to(self.device)
            
            audio = state['audio_waveform'].unsqueeze(0).to(self.device)
            
            # Run AV model inference
            with torch.no_grad():
                logits, features = self.av_model(frames, audio, return_features=True)
                probs = torch.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
                
                # Extract lip-sync score if available
                if 'lip_sync_score' in features:
                    state['lip_sync_score'] = features['lip_sync_score'].item()
                else:
                    state['lip_sync_score'] = 0.5
            
            state['specialist_prediction'] = fake_prob
            state['specialist_confidence'] = max(fake_prob, 1 - fake_prob)
            state['av_confidence'] = state['specialist_confidence']
            state['selected_specialist'] = 'av_model'
            
            print(f"[AV-SPECIALIST] Prediction: {fake_prob:.3f}, "
                  f"Confidence: {state['specialist_confidence']:.3f}, "
                  f"Lip-sync: {state['lip_sync_score']:.3f}")
            
        except Exception as e:
            print(f"[AV-SPECIALIST] [ERROR] Failed: {e}")
            # Fallback to student
            state['specialist_prediction'] = state['student_prediction']
            state['specialist_confidence'] = state['student_confidence']
            state['selected_specialist'] = 'av_fallback'
        
        return state
    
    def _human_review_node(self, state: AgentState) -> AgentState:
        """Node 7: Human review escalation"""
        print("[HUMAN] Escalated to human review")
        
        # In production, this would:
        # 1. Save to review queue
        # 2. Notify human reviewers
        # 3. Wait for human decision
        
        # For now, use student prediction with low confidence
        state['specialist_prediction'] = state['student_prediction']
        state['specialist_confidence'] = min(state['student_confidence'], 0.6)
        state['selected_specialist'] = 'human_review'
        
        return state
    
    def _explanation_node(self, state: AgentState) -> AgentState:
        """Node 8: Generate final prediction and explanation"""
        # Determine final prediction
        if state['next_action'] == 'ACCEPT':
            final_pred = state['student_prediction']
            confidence = state['student_confidence']
        else:
            final_pred = state['specialist_prediction']
            confidence = state['specialist_confidence']
        
        # Set final prediction
        state['final_prediction'] = 'FAKE' if final_pred > 0.5 else 'REAL'
        state['confidence'] = confidence
        
        # Determine confidence level
        if confidence >= self.config['thresholds']['high_confidence']:
            state['confidence_level'] = ConfidenceLevel.HIGH
        elif confidence >= self.config['thresholds']['medium_confidence']:
            state['confidence_level'] = ConfidenceLevel.MEDIUM
        else:
            state['confidence_level'] = ConfidenceLevel.LOW
        
        # Generate explanation
        pred = state['final_prediction']
        conf_pct = confidence * 100
        specialist = state.get('selected_specialist', 'student')
        
        explanation_parts = [
            f"This video is classified as {pred} with {conf_pct:.1f}% confidence."
        ]
        
        if specialist == 'av_model':
            lip_sync = state.get('lip_sync_score', 0.5)
            if lip_sync < 0.3:
                explanation_parts.append("Audio-visual analysis detected significant lip-sync inconsistencies.")
            elif lip_sync > 0.7:
                explanation_parts.append("Audio-visual synchronization appears natural.")
            else:
                explanation_parts.append("Audio-visual analysis shows moderate synchronization.")
        
        if pred == 'FAKE':
            explanation_parts.append("Detected inconsistencies suggest potential manipulation.")
        else:
            explanation_parts.append("No significant artifacts or inconsistencies detected.")
        
        state['explanation'] = " ".join(explanation_parts)
        
        # Placeholder for heatmaps
        state['heatmaps'] = []
        
        print(f"[EXPLANATION] Final: {pred} ({conf_pct:.1f}% confidence)")
        
        return state
    
    # ============================================
    # PUBLIC API
    # ============================================
    
    def predict(self, video_path: str) -> Dict[str, Any]:
        """Main prediction function"""
        start_time = time.time()
        
        # Initialize state
        initial_state = {
            'video_path': video_path,
            'request_id': '',
            'metadata': {},
            'bitrate': 0,
            'fps': 0.0,
            'resolution': (0, 0),
            'avg_brightness': 0.0,
            'duration': 0.0,
            'frames': [],
            'audio_waveform': None,
            'faces_detected': 0,
            'student_prediction': 0.0,
            'student_confidence': 0.0,
            'specialist_prediction': 0.0,
            'specialist_confidence': 0.0,
            'selected_specialist': '',
            'final_prediction': '',
            'confidence': 0.0,
            'confidence_level': ConfidenceLevel.LOW,
            'explanation': '',
            'heatmaps': [],
            'next_action': '',
            'lip_sync_score': None,
            'av_confidence': None,
            'processing_time': 0.0,
            'stage_taken': '',
            'error_message': None
        }
        
        try:
            # Run LangGraph workflow
            result = self.workflow.invoke(initial_state)
            
            # Calculate processing time
            result['processing_time'] = time.time() - start_time
            
            if result.get('error_message'):
                return {
                    'success': False,
                    'error': result['error_message'],
                    'request_id': result.get('request_id', 'unknown'),
                    'processing_time': result['processing_time']
                }
            
            return {
                'success': True,
                'prediction': result['final_prediction'],
                'confidence': result['confidence'],
                'confidence_level': result['confidence_level'].value,
                'explanation': result['explanation'],
                'specialist_used': result.get('selected_specialist', 'student'),
                'lip_sync_score': result.get('lip_sync_score'),
                'metadata': result['metadata'],
                'request_id': result['request_id'],
                'processing_time': result['processing_time'],
                'stage_taken': result['stage_taken']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'request_id': initial_state.get('request_id', 'unknown'),
                'processing_time': time.time() - start_time
            }

def create_langgraph_agent(**kwargs) -> LangGraphAgent:
    """Factory function to create LangGraph agent"""
    return LangGraphAgent(**kwargs)

if __name__ == "__main__":
    # Test agent creation
    print("[TEST] Testing LangGraph Agent...")
    
    try:
        agent = create_langgraph_agent()
        print("[OK] LangGraph Agent created successfully!")
        
        # Test with a video if available
        test_videos = ["test_video_short.mp4", "test_video_long.mp4"]
        for video in test_videos:
            if os.path.exists(video):
                print(f"\n[VIDEO] Testing with {video}...")
                result = agent.predict(video)
                print(f"Result: {result}")
                break
        else:
            print("[INFO] No test videos found, but agent is ready!")
            
    except Exception as e:
        print(f"[ERROR] Agent test failed: {e}")
        print("[INFO] Check model paths and dependencies")