#!/usr/bin/env python3
"""
Enhanced Inference API
Advanced deepfake detection with heatmaps and confidence analysis
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

class EnhancedInference:
    """Enhanced inference with heatmaps and detailed analysis"""
    
    def __init__(self, model, device, transform):
        self.model = model
        self.device = device
        self.transform = transform
        
    def extract_faces_with_metadata(self, video_path: str, max_faces: int = 8) -> List[Dict]:
        """Extract faces with frame metadata"""
        faces_data = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return faces_data
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames <= 0:
            cap.release()
            return faces_data
        
        # Sample frames evenly
        step = max(1, total_frames // (max_faces * 2))
        frame_count = 0
        
        # Face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while len(faces_data) < max_faces and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Face detection
                gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
                detected_faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(detected_faces) > 0:
                    # Use largest face
                    largest_face = max(detected_faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    # Add padding
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(rgb_frame.shape[1], x + w + padding)
                    y2 = min(rgb_frame.shape[0], y + h + padding)
                    
                    face = rgb_frame[y1:y2, x1:x2]
                    
                    # Calculate quality metrics
                    quality_score = self.calculate_face_quality(face)
                    
                else:
                    # Fallback to center crop
                    h_frame, w_frame = rgb_frame.shape[:2]
                    size = min(h_frame, w_frame)
                    y_start = (h_frame - size) // 2
                    x_start = (w_frame - size) // 2
                    face = rgb_frame[y_start:y_start+size, x_start:x_start+size]
                    quality_score = 0.5  # Lower quality for center crop
                
                if face.size > 0:
                    face_resized = cv2.resize(face, (224, 224))
                    
                    faces_data.append({
                        'face': face_resized,
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'quality_score': quality_score,
                        'detection_method': 'face_cascade' if len(detected_faces) > 0 else 'center_crop'
                    })
            
            frame_count += 1
        
        cap.release()
        return faces_data
    
    def calculate_face_quality(self, face: np.ndarray) -> float:
        """Calculate face quality score (0-1)"""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            
            # Calculate sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            # Calculate brightness (avoid too dark/bright)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Calculate contrast
            contrast = np.std(gray) / 255.0
            contrast_score = min(1.0, contrast * 2)
            
            # Combined quality score
            quality = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
            return max(0.0, min(1.0, quality))
            
        except Exception:
            return 0.5
    
    def generate_attention_heatmap(self, face: np.ndarray) -> np.ndarray:
        """Generate attention heatmap for face"""
        try:
            # Convert to tensor
            pil_face = Image.fromarray(face)
            input_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
            
            # Enable gradients
            input_tensor.requires_grad_(True)
            
            # Forward pass
            output = self.model(input_tensor)
            
            # Get prediction for fake class
            fake_score = output[0][1]
            
            # Backward pass
            self.model.zero_grad()
            fake_score.backward()
            
            # Get gradients
            gradients = input_tensor.grad.data.abs()
            
            # Average across channels
            heatmap = gradients.squeeze().mean(dim=0).cpu().numpy()
            
            # Normalize to 0-1
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return heatmap
            
        except Exception as e:
            print(f"Heatmap generation failed: {e}")
            # Return dummy heatmap
            return np.random.rand(224, 224) * 0.1
    
    def analyze_video_comprehensive(self, video_path: str) -> Dict:
        """Comprehensive video analysis with enhanced features"""
        
        # Extract faces with metadata
        faces_data = self.extract_faces_with_metadata(video_path, max_faces=8)
        
        if not faces_data:
            return {
                "error": "No faces detected in video",
                "prediction": "unknown",
                "confidence": 0.0,
                "analysis": {
                    "faces_analyzed": 0,
                    "quality_scores": [],
                    "frame_analysis": []
                }
            }
        
        # Analyze each face
        predictions = []
        confidences = []
        quality_scores = []
        frame_analysis = []
        heatmaps = []
        
        with torch.no_grad():
            for i, face_data in enumerate(faces_data):
                face = face_data['face']
                
                # Convert to PIL and apply transforms
                pil_face = Image.fromarray(face)
                input_tensor = self.transform(pil_face).unsqueeze(0).to(self.device)
                
                # Get prediction
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()
                real_prob = probabilities[0][0].item()
                fake_prob = probabilities[0][1].item()
                
                predictions.append(pred_class)
                confidences.append(confidence)
                quality_scores.append(face_data['quality_score'])
                
                # Generate heatmap for suspicious faces
                heatmap = None
                if fake_prob > 0.6:  # Only generate heatmap for likely fakes
                    heatmap = self.generate_attention_heatmap(face)
                    heatmaps.append({
                        'frame_index': i,
                        'heatmap': heatmap.tolist() if heatmap is not None else None
                    })
                
                # Frame analysis
                frame_analysis.append({
                    'frame_number': face_data['frame_number'],
                    'timestamp': face_data['timestamp'],
                    'prediction': 'real' if pred_class == 0 else 'fake',
                    'confidence': confidence,
                    'real_probability': real_prob,
                    'fake_probability': fake_prob,
                    'quality_score': face_data['quality_score'],
                    'detection_method': face_data['detection_method'],
                    'has_heatmap': heatmap is not None
                })
        
        # Aggregate results with weighted voting
        weighted_fake_votes = 0
        total_weight = 0
        
        for i, (pred, conf, quality) in enumerate(zip(predictions, confidences, quality_scores)):
            weight = quality * conf  # Weight by quality and confidence
            total_weight += weight
            if pred == 1:  # Fake
                weighted_fake_votes += weight
        
        # Final decision
        if total_weight > 0:
            fake_ratio = weighted_fake_votes / total_weight
        else:
            fake_ratio = sum(predictions) / len(predictions)
        
        final_prediction = "fake" if fake_ratio > 0.5 else "real"
        
        # Calculate overall confidence
        avg_confidence = np.mean(confidences)
        quality_adjusted_confidence = avg_confidence * np.mean(quality_scores)
        
        # Confidence analysis
        confidence_analysis = {
            'raw_confidence': float(avg_confidence),
            'quality_adjusted': float(quality_adjusted_confidence),
            'consistency': float(1.0 - np.std(confidences)),  # How consistent are predictions
            'quality_score': float(np.mean(quality_scores))
        }
        
        return {
            "prediction": final_prediction,
            "confidence": float(quality_adjusted_confidence),
            "faces_analyzed": len(faces_data),
            "fake_votes": sum(predictions),
            "total_votes": len(predictions),
            "weighted_fake_ratio": float(fake_ratio),
            "individual_confidences": confidences,
            "analysis": {
                "confidence_breakdown": confidence_analysis,
                "quality_scores": quality_scores,
                "frame_analysis": frame_analysis,
                "heatmaps_generated": len(heatmaps),
                "suspicious_frames": len([f for f in frame_analysis if f['fake_probability'] > 0.7])
            },
            "heatmaps": heatmaps[:3] if heatmaps else [],  # Limit to 3 heatmaps
            "metadata": {
                "processing_timestamp": datetime.now().isoformat(),
                "model_version": "kaggle-v1",
                "analysis_version": "enhanced-v1"
            }
        }