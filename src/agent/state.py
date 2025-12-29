#!/usr/bin/env python3
"""
Person 4: LangGraph Agent State Definition
Implements the state management for the agentic deepfake detection system
"""

from typing import TypedDict, List, Optional, Dict, Any
import torch
from enum import Enum

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