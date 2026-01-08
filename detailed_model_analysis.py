#!/usr/bin/env python3
"""
E-Raksha Comprehensive Model Analysis Framework

Advanced testing and analysis system for specialist models with automated
training recommendations and performance optimization insights.

Features:
- Individual model performance analysis
- Bias detection and correction strategies  
- Specialization effectiveness measurement
- Automated training recommendations
- Performance optimization insights

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.specialists_new import (
    BGSpecialistModel, AVSpecialistModel, CMSpecialistModel,
    RRSpecialistModel, LLSpecialistModel, TMModelOld
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")

class ModelAnalyzer:
    """
    Comprehensive model analysis system with intelligent training recommendations.
    
    Provides detailed performance metrics, bias analysis, and automated
    suggestions for model improvement and training optimization.
    """
    
    def __init__(self):
        self.models = {}
        self.model_configs = {
            'bg': {
                'class': BGSpecialistModel,
                'path': 'models/baseline_student.pt',
                'name': 'Background/Lighting Specialist',
                'specialization': 'Background artifacts, lighting inconsistencies, shadow detection',
                'expected_strengths': ['Lighting manipulation', 'Background replacement', 'Shadow artifacts'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC']
            },
            'av': {
                'class': AVSpecialistModel,
                'path': 'models/av_model_student.pt',
                'name': 'Audio-Visual Specialist',
                'specialization': 'Lip-sync detection, audio-visual correlation',
                'expected_strengths': ['Lip-sync mismatch', 'Audio-visual desync', 'Facial motion'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC']
            },
            'cm': {
                'class': CMSpecialistModel,
                'path': 'models/cm_model_student.pt',
                'name': 'Compression Specialist',
                'specialization': 'Compression artifacts, DCT analysis, quantization patterns',
                'expected_strengths': ['JPEG artifacts', 'Video compression', 'Quantization noise'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC']
            },
            'rr': {
                'class': RRSpecialistModel,
                'path': 'models/rr_model_student.pt',
                'name': 'Resolution/Re-recording Specialist',
                'specialization': 'Resolution artifacts, upscaling detection, re-recording patterns',
                'expected_strengths': ['Resolution mismatch', 'Upscaling artifacts', 'Re-recording detection'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC']
            },
            'll': {
                'class': LLSpecialistModel,
                'path': 'models/ll_model_student.pt',
                'name': 'Low-Light Specialist',
                'specialization': 'Low-light enhancement artifacts, noise patterns',
                'expected_strengths': ['Low-light enhancement', 'Noise artifacts', 'Luminance manipulation'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC']
            },
            'tm': {
                'class': TMModelOld,
                'path': 'models/tm_model_student.pt',
                'name': 'Temporal Specialist',
                'specialization': 'Temporal consistency, frame-to-frame analysis',
                'expected_strengths': ['Temporal artifacts', 'Frame consistency', 'Motion blur'],
                'datasets': ['FaceForensics++', 'CelebDF', 'DFDC'],
                'note': 'BROKEN - Predicts all REAL'
            }
        }
    
    def load_models(self):
        """Load all models for analysis"""
        print("=" * 80)
        print("LOADING MODELS FOR COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        for model_key, config in self.model_configs.items():
            try:
                print(f"\nLoading {config['name']}...")
                model = config['class']()
                
                if os.path.exists(config['path']):
                    checkpoint = torch.load(config['path'], map_location=DEVICE, weights_only=False)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        metrics = checkpoint.get('metrics', {})
                    else:
                        state_dict = checkpoint
                        metrics = {}
                    
                    model.load_state_dict(state_dict, strict=True)
                    model.to(DEVICE)
                    model.eval()
                    
                    self.models[model_key] = {
                        'model': model,
                        'config': config,
                        'metrics': metrics
                    }
                    
                    print(f"  ✅ Loaded successfully")
                    if metrics:
                        print(f"  📊 Training metrics: {metrics}")
                else:
                    print(f"  ❌ Model file not found: {config['path']}")
                    
            except Exception as e:
                print(f"  ❌ Failed to load: {e}")
        
        print(f"\n✅ Loaded {len(self.models)} models for analysis")
    
    def extract_frames_with_metadata(self, video_path, num_frames=8):
        """Extract frames with detailed metadata for analysis"""
        frames = []
        metadata = {}
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        metadata.update({
            'fps': fps,
            'resolution': (width, height),
            'total_frames': total_frames,
            'duration': duration,
            'file_size': os.path.getsize(video_path),
            'bitrate': (os.path.getsize(video_path) * 8) / duration if duration > 0 else 0
        })
        
        if total_frames == 0:
            cap.release()
            return None, metadata
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Extract frames with quality analysis
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        brightness_values = []
        contrast_values = []
        blur_scores = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Quality analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                contrast = np.std(gray)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                brightness_values.append(brightness)
                contrast_values.append(contrast)
                blur_scores.append(blur_score)
                
                # Prepare for model input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frame_normalized = (frame_normalized - mean) / std
                frame_tensor = torch.from_numpy(frame_normalized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frames.append(frame_tensor)
        
        cap.release()
        
        # Add quality metrics to metadata
        metadata.update({
            'avg_brightness': np.mean(brightness_values),
            'avg_contrast': np.mean(contrast_values),
            'avg_blur_score': np.mean(blur_scores),
            'brightness_std': np.std(brightness_values),
            'contrast_std': np.std(contrast_values),
            'blur_std': np.std(blur_scores),
            'is_low_light': np.mean(brightness_values) < 80,
            'is_low_contrast': np.mean(contrast_values) < 30,
            'is_blurry': np.mean(blur_scores) < 100,
            'is_compressed': metadata['bitrate'] < 1000000,  # < 1 Mbps
            'is_low_resolution': width * height < 640 * 480
        })
        
        return torch.stack(frames) if len(frames) >= 4 else None, metadata
    
    def analyze_model_predictions(self, model_key, frames, metadata):
        """Detailed analysis of model predictions with confidence breakdown"""
        if model_key not in self.models:
            return None
        
        model_info = self.models[model_key]
        model = model_info['model']
        
        try:
            with torch.no_grad():
                if model_key == 'tm':
                    input_tensor = frames.unsqueeze(0).to(DEVICE)
                else:
                    input_tensor = frames[:4].to(DEVICE)
                
                logits = model(input_tensor)
                
                if model_key == 'tm':
                    probs = F.softmax(logits, dim=1)
                    fake_prob = probs[0, 1].item()
                    confidence_scores = [fake_prob]
                else:
                    probs = F.softmax(logits, dim=1)
                    fake_probs = probs[:, 1].cpu().numpy()
                    fake_prob = np.mean(fake_probs)
                    confidence_scores = fake_probs.tolist()
                
                # Detailed analysis
                analysis = {
                    'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL',
                    'fake_probability': float(fake_prob),
                    'confidence': float(max(fake_prob, 1 - fake_prob)),
                    'frame_predictions': confidence_scores,
                    'prediction_variance': float(np.var(confidence_scores)),
                    'prediction_consistency': 1.0 - float(np.var(confidence_scores)),
                    'logits_raw': logits.cpu().numpy().tolist() if logits.numel() < 20 else "too_large"
                }
                
                # Specialization-specific analysis
                analysis.update(self._get_specialization_analysis(model_key, fake_prob, metadata))
                
                return analysis
                
        except Exception as e:
            return {'error': str(e)}
    
    def _get_specialization_analysis(self, model_key, fake_prob, metadata):
        """Get specialization-specific analysis and recommendations"""
        config = self.model_configs[model_key]
        
        analysis = {
            'specialization': config['specialization'],
            'expected_strengths': config['expected_strengths']
        }
        
        # Model-specific analysis
        if model_key == 'bg':
            analysis.update({
                'lighting_assessment': 'suspicious' if metadata['is_low_light'] and fake_prob > 0.6 else 'normal',
                'brightness_factor': metadata['avg_brightness'],
                'contrast_factor': metadata['avg_contrast'],
                'shadow_likelihood': fake_prob if metadata['avg_contrast'] > 50 else 0.5
            })
        
        elif model_key == 'av':
            analysis.update({
                'lip_sync_assessment': 'mismatch_detected' if fake_prob > 0.7 else 'synchronized',
                'motion_consistency': 1.0 - metadata.get('blur_std', 0) / 100,
                'temporal_stability': 1.0 - fake_prob if fake_prob > 0.5 else fake_prob
            })
        
        elif model_key == 'cm':
            analysis.update({
                'compression_assessment': 'artifacts_detected' if fake_prob > 0.6 and metadata['is_compressed'] else 'clean',
                'bitrate_factor': metadata['bitrate'],
                'quality_degradation': fake_prob if metadata['is_compressed'] else 0.3
            })
        
        elif model_key == 'rr':
            analysis.update({
                'resolution_assessment': 'upscaling_detected' if fake_prob > 0.6 and metadata['is_low_resolution'] else 'native',
                'rerecording_likelihood': fake_prob if metadata['is_blurry'] else 0.4,
                'quality_consistency': metadata['avg_blur_score']
            })
        
        elif model_key == 'll':
            analysis.update({
                'lowlight_assessment': 'enhancement_detected' if fake_prob > 0.6 and metadata['is_low_light'] else 'natural',
                'noise_level': metadata['avg_blur_score'],
                'luminance_manipulation': fake_prob if metadata['is_low_light'] else 0.3
            })
        
        elif model_key == 'tm':
            analysis.update({
                'temporal_assessment': 'inconsistent' if fake_prob > 0.5 else 'consistent',
                'frame_stability': 1.0 - fake_prob,
                'motion_artifacts': fake_prob if metadata.get('blur_std', 0) > 20 else 0.2,
                'note': 'Model appears broken - predicts all REAL'
            })
        
        return analysis
    
    def generate_training_recommendations(self, model_key, results):
        """Generate detailed training recommendations based on analysis"""
        config = self.model_configs[model_key]
        
        # Analyze performance patterns
        real_predictions = [r for r in results if r['true_label'] == 'REAL']
        fake_predictions = [r for r in results if r['true_label'] == 'FAKE']
        
        real_accuracy = np.mean([1 if r['analysis']['prediction'] == 'REAL' else 0 for r in real_predictions])
        fake_accuracy = np.mean([1 if r['analysis']['prediction'] == 'FAKE' else 0 for r in fake_predictions])
        
        overall_accuracy = (real_accuracy * len(real_predictions) + fake_accuracy * len(fake_predictions)) / len(results)
        bias = real_accuracy - fake_accuracy
        
        # Confidence analysis
        avg_confidence = np.mean([r['analysis']['confidence'] for r in results])
        confidence_variance = np.var([r['analysis']['confidence'] for r in results])
        
        recommendations = {
            'model_name': config['name'],
            'current_performance': {
                'overall_accuracy': float(overall_accuracy),
                'real_accuracy': float(real_accuracy),
                'fake_accuracy': float(fake_accuracy),
                'bias': float(bias),
                'avg_confidence': float(avg_confidence),
                'confidence_variance': float(confidence_variance)
            },
            'performance_assessment': self._assess_performance(overall_accuracy, bias, avg_confidence),
            'training_recommendations': self._get_training_recommendations(model_key, overall_accuracy, bias, avg_confidence, results),
            'dataset_recommendations': self._get_dataset_recommendations(model_key, results),
            'architecture_recommendations': self._get_architecture_recommendations(model_key, results)
        }
        
        return recommendations
    
    def _assess_performance(self, accuracy, bias, confidence):
        """Assess model performance and identify issues"""
        assessment = []
        
        if accuracy < 0.6:
            assessment.append("POOR_ACCURACY: Model needs significant improvement")
        elif accuracy < 0.7:
            assessment.append("MODERATE_ACCURACY: Model needs refinement")
        else:
            assessment.append("GOOD_ACCURACY: Model performing well")
        
        if abs(bias) > 0.2:
            if bias > 0:
                assessment.append("REAL_BIAS: Model over-predicts REAL videos")
            else:
                assessment.append("FAKE_BIAS: Model over-predicts FAKE videos")
        else:
            assessment.append("BALANCED: Good real/fake balance")
        
        if confidence < 0.6:
            assessment.append("LOW_CONFIDENCE: Model is uncertain about predictions")
        elif confidence > 0.9:
            assessment.append("OVERCONFIDENT: Model may be overfitting")
        else:
            assessment.append("GOOD_CONFIDENCE: Appropriate confidence levels")
        
        return assessment
    
    def _get_training_recommendations(self, model_key, accuracy, bias, confidence, results):
        """Get specific training recommendations"""
        recommendations = []
        
        # General recommendations based on performance
        if accuracy < 0.6:
            recommendations.extend([
                "INCREASE_TRAINING_DATA: Add more diverse training samples",
                "ADJUST_LEARNING_RATE: Try lower learning rate for better convergence",
                "AUGMENTATION: Increase data augmentation diversity",
                "ARCHITECTURE_REVIEW: Consider model architecture improvements"
            ])
        
        if abs(bias) > 0.2:
            if bias > 0:  # Real bias
                recommendations.extend([
                    "BALANCE_DATASET: Add more challenging fake samples",
                    "HARD_NEGATIVE_MINING: Focus on hard-to-detect fake samples",
                    "FAKE_AUGMENTATION: Increase fake sample augmentation"
                ])
            else:  # Fake bias
                recommendations.extend([
                    "REAL_SAMPLE_DIVERSITY: Add more diverse real samples",
                    "REDUCE_OVERFITTING: Add regularization or dropout",
                    "REAL_AUGMENTATION: Increase real sample augmentation"
                ])
        
        # Model-specific recommendations
        if model_key == 'bg':
            recommendations.extend([
                "LIGHTING_DIVERSITY: Train on varied lighting conditions",
                "BACKGROUND_COMPLEXITY: Include complex background scenarios",
                "SHADOW_DETECTION: Focus on shadow manipulation detection"
            ])
        
        elif model_key == 'av':
            recommendations.extend([
                "LIPSYNC_SAMPLES: Increase lip-sync mismatch samples",
                "AUDIO_QUALITY: Train on varied audio quality levels",
                "TEMPORAL_CONSISTENCY: Focus on frame-to-frame consistency"
            ])
        
        elif model_key == 'cm':
            recommendations.extend([
                "COMPRESSION_LEVELS: Train on multiple compression levels",
                "ARTIFACT_DETECTION: Focus on JPEG/video compression artifacts",
                "QUALITY_DEGRADATION: Include quality-degraded samples"
            ])
        
        elif model_key == 'rr':
            recommendations.extend([
                "RESOLUTION_VARIETY: Train on multiple resolution levels",
                "UPSCALING_DETECTION: Focus on upscaling artifact detection",
                "RERECORDING_PATTERNS: Include re-recorded video samples"
            ])
        
        elif model_key == 'll':
            recommendations.extend([
                "LOWLIGHT_CONDITIONS: Train on low-light scenarios",
                "NOISE_PATTERNS: Include various noise types",
                "ENHANCEMENT_ARTIFACTS: Focus on enhancement artifact detection"
            ])
        
        elif model_key == 'tm':
            recommendations.extend([
                "TEMPORAL_CONSISTENCY: Fix temporal analysis architecture",
                "FRAME_SEQUENCE: Improve frame sequence processing",
                "MOTION_ANALYSIS: Better motion pattern detection",
                "ARCHITECTURE_FIX: Model appears broken - needs complete review"
            ])
        
        return recommendations
    
    def _get_dataset_recommendations(self, model_key, results):
        """Get dataset-specific recommendations"""
        recommendations = {
            'primary_datasets': self.model_configs[model_key]['datasets'],
            'additional_datasets': [],
            'data_balance': {},
            'specific_needs': []
        }
        
        # Analyze failure patterns to recommend datasets
        failed_cases = [r for r in results if r['analysis']['prediction'] != r['true_label']]
        
        if len(failed_cases) > len(results) * 0.4:  # > 40% failure rate
            recommendations['additional_datasets'].extend([
                'WildDeepfake', 'DeeperForensics', 'FaceSwapper'
            ])
        
        # Model-specific dataset needs
        if model_key == 'bg':
            recommendations['specific_needs'].extend([
                'Outdoor lighting variations',
                'Indoor/studio lighting',
                'Mixed lighting scenarios',
                'Shadow manipulation samples'
            ])
        
        elif model_key == 'av':
            recommendations['specific_needs'].extend([
                'Multi-language lip-sync samples',
                'Various audio qualities',
                'Silent video samples',
                'Audio-visual desync samples'
            ])
        
        elif model_key == 'cm':
            recommendations['specific_needs'].extend([
                'Multiple compression standards (H.264, H.265, VP9)',
                'Various bitrate levels',
                'Compression artifact samples',
                'Uncompressed reference videos'
            ])
        
        elif model_key == 'rr':
            recommendations['specific_needs'].extend([
                'Multiple resolution levels',
                'Upscaled video samples',
                'Re-recorded video samples',
                'Screen recording artifacts'
            ])
        
        elif model_key == 'll':
            recommendations['specific_needs'].extend([
                'Low-light video samples',
                'Night vision footage',
                'Enhanced low-light videos',
                'Various noise patterns'
            ])
        
        # Data balance recommendations
        real_acc = np.mean([1 if r['analysis']['prediction'] == 'REAL' else 0 
                           for r in results if r['true_label'] == 'REAL'])
        fake_acc = np.mean([1 if r['analysis']['prediction'] == 'FAKE' else 0 
                           for r in results if r['true_label'] == 'FAKE'])
        
        if real_acc < fake_acc - 0.1:
            recommendations['data_balance']['increase_real_samples'] = True
        elif fake_acc < real_acc - 0.1:
            recommendations['data_balance']['increase_fake_samples'] = True
        else:
            recommendations['data_balance']['current_balance'] = 'adequate'
        
        return recommendations
    
    def _get_architecture_recommendations(self, model_key, results):
        """Get architecture improvement recommendations"""
        recommendations = []
        
        # Analyze prediction patterns
        confidences = [r['analysis']['confidence'] for r in results]
        variances = [r['analysis'].get('prediction_variance', 0) for r in results]
        
        avg_confidence = np.mean(confidences)
        avg_variance = np.mean(variances)
        
        # General architecture recommendations
        if avg_confidence < 0.6:
            recommendations.extend([
                "INCREASE_MODEL_CAPACITY: Add more layers or parameters",
                "ATTENTION_MECHANISM: Add attention layers for better feature focus",
                "ENSEMBLE_COMPONENTS: Consider ensemble within the model"
            ])
        
        if avg_variance > 0.3:
            recommendations.extend([
                "STABILIZE_PREDICTIONS: Add batch normalization or layer normalization",
                "REGULARIZATION: Add dropout or weight decay",
                "CONSISTENT_ARCHITECTURE: Review architecture for stability"
            ])
        
        # Model-specific architecture recommendations
        if model_key == 'bg':
            recommendations.extend([
                "SPATIAL_ATTENTION: Add spatial attention for background focus",
                "MULTI_SCALE_FEATURES: Use multiple scales for lighting analysis",
                "COLOR_SPACE_ANALYSIS: Include multiple color space processing"
            ])
        
        elif model_key == 'av':
            recommendations.extend([
                "TEMPORAL_FUSION: Better temporal feature fusion",
                "CROSS_MODAL_ATTENTION: Audio-visual cross-attention",
                "SEQUENCE_MODELING: LSTM/Transformer for temporal modeling"
            ])
        
        elif model_key == 'cm':
            recommendations.extend([
                "FREQUENCY_ANALYSIS: Add DCT/FFT analysis layers",
                "MULTI_RESOLUTION: Process multiple resolution levels",
                "ARTIFACT_DETECTION: Specialized artifact detection modules"
            ])
        
        elif model_key == 'rr':
            recommendations.extend([
                "RESOLUTION_PYRAMID: Multi-resolution processing",
                "EDGE_DETECTION: Enhanced edge analysis modules",
                "QUALITY_ASSESSMENT: Built-in quality assessment"
            ])
        
        elif model_key == 'll':
            recommendations.extend([
                "NOISE_MODELING: Explicit noise modeling layers",
                "LUMINANCE_ANALYSIS: Specialized luminance processing",
                "ENHANCEMENT_DETECTION: Enhancement artifact detection"
            ])
        
        elif model_key == 'tm':
            recommendations.extend([
                "COMPLETE_REDESIGN: Model needs fundamental architecture review",
                "TEMPORAL_MODELING: Proper temporal sequence processing",
                "FRAME_CONSISTENCY: Frame-to-frame consistency analysis",
                "MOTION_ANALYSIS: Optical flow or motion analysis integration"
            ])
        
        return recommendations
    
    def run_comprehensive_analysis(self, max_videos_per_class=25):
        """Run comprehensive analysis on all models"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL ANALYSIS")
        print("=" * 80)
        
        # Load models
        self.load_models()
        
        if not self.models:
            print("❌ No models loaded. Cannot proceed with analysis.")
            return
        
        # Get test videos
        real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:max_videos_per_class]
        fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:max_videos_per_class]
        
        print(f"\n📹 Testing on {len(real_videos)} real + {len(fake_videos)} fake videos")
        
        # Analyze each model
        all_results = {}
        
        for model_key in self.models.keys():
            print(f"\n{'='*60}")
            print(f"ANALYZING {self.models[model_key]['config']['name'].upper()}")
            print(f"{'='*60}")
            
            model_results = []
            
            # Test on real videos
            print(f"\nTesting on REAL videos...")
            for video_path in tqdm(real_videos, desc="Real videos"):
                frames, metadata = self.extract_frames_with_metadata(video_path)
                if frames is not None:
                    analysis = self.analyze_model_predictions(model_key, frames, metadata)
                    if analysis:
                        model_results.append({
                            'video': video_path.name,
                            'true_label': 'REAL',
                            'metadata': metadata,
                            'analysis': analysis
                        })
            
            # Test on fake videos
            print(f"Testing on FAKE videos...")
            for video_path in tqdm(fake_videos, desc="Fake videos"):
                frames, metadata = self.extract_frames_with_metadata(video_path)
                if frames is not None:
                    analysis = self.analyze_model_predictions(model_key, frames, metadata)
                    if analysis:
                        model_results.append({
                            'video': video_path.name,
                            'true_label': 'FAKE',
                            'metadata': metadata,
                            'analysis': analysis
                        })
            
            # Generate recommendations
            recommendations = self.generate_training_recommendations(model_key, model_results)
            
            all_results[model_key] = {
                'results': model_results,
                'recommendations': recommendations,
                'summary': self._generate_model_summary(model_key, model_results)
            }
            
            # Print summary
            self._print_model_summary(model_key, all_results[model_key])
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        # Generate overall recommendations
        self._generate_overall_recommendations(all_results)
        
        return all_results
    
    def _generate_model_summary(self, model_key, results):
        """Generate summary statistics for a model"""
        if not results:
            return {}
        
        real_results = [r for r in results if r['true_label'] == 'REAL']
        fake_results = [r for r in results if r['true_label'] == 'FAKE']
        
        real_correct = sum(1 for r in real_results if r['analysis']['prediction'] == 'REAL')
        fake_correct = sum(1 for r in fake_results if r['analysis']['prediction'] == 'FAKE')
        
        return {
            'total_videos': len(results),
            'real_videos': len(real_results),
            'fake_videos': len(fake_results),
            'overall_accuracy': (real_correct + fake_correct) / len(results),
            'real_accuracy': real_correct / len(real_results) if real_results else 0,
            'fake_accuracy': fake_correct / len(fake_results) if fake_results else 0,
            'bias': (real_correct / len(real_results) - fake_correct / len(fake_results)) if real_results and fake_results else 0,
            'avg_confidence': np.mean([r['analysis']['confidence'] for r in results]),
            'confidence_std': np.std([r['analysis']['confidence'] for r in results])
        }
    
    def _print_model_summary(self, model_key, model_data):
        """Print detailed summary for a model"""
        config = self.models[model_key]['config']
        summary = model_data['summary']
        recommendations = model_data['recommendations']
        
        print(f"\n📊 {config['name']} SUMMARY:")
        print(f"   Specialization: {config['specialization']}")
        print(f"   Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"   Real Accuracy: {summary['real_accuracy']:.1%}")
        print(f"   Fake Accuracy: {summary['fake_accuracy']:.1%}")
        print(f"   Bias: {summary['bias']:+.1%}")
        print(f"   Avg Confidence: {summary['avg_confidence']:.1%}")
        
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        for assessment in recommendations['performance_assessment']:
            print(f"   • {assessment}")
        
        print(f"\n🔧 TOP TRAINING RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations['training_recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📚 DATASET NEEDS:")
        for need in recommendations['dataset_recommendations']['specific_needs'][:3]:
            print(f"   • {need}")
    
    def _save_comprehensive_results(self, all_results):
        """Save comprehensive analysis results"""
        # Prepare serializable results
        serializable_results = {}
        
        for model_key, data in all_results.items():
            serializable_results[model_key] = {
                'model_name': self.models[model_key]['config']['name'],
                'specialization': self.models[model_key]['config']['specialization'],
                'summary': data['summary'],
                'recommendations': data['recommendations'],
                'sample_results': data['results'][:5]  # Save first 5 results as samples
            }
        
        # Save to JSON
        with open('comprehensive_model_analysis.json', 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n💾 Comprehensive analysis saved to 'comprehensive_model_analysis.json'")
    
    def _generate_overall_recommendations(self, all_results):
        """Generate overall system recommendations"""
        print(f"\n" + "=" * 80)
        print("OVERALL SYSTEM RECOMMENDATIONS")
        print("=" * 80)
        
        # Analyze overall performance
        model_performances = {}
        for model_key, data in all_results.items():
            model_performances[model_key] = data['summary']['overall_accuracy']
        
        # Rank models by performance
        ranked_models = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 MODEL PERFORMANCE RANKING:")
        for i, (model_key, accuracy) in enumerate(ranked_models, 1):
            model_name = self.models[model_key]['config']['name']
            print(f"   {i}. {model_name}: {accuracy:.1%}")
        
        # System-wide recommendations
        print(f"\n🔧 SYSTEM-WIDE RECOMMENDATIONS:")
        
        best_model = ranked_models[0][0]
        worst_model = ranked_models[-1][0]
        
        print(f"   1. PRIORITIZE {self.models[best_model]['config']['name']} - Best performer")
        print(f"   2. URGENT FIX needed for {self.models[worst_model]['config']['name']} - Worst performer")
        
        # Check for broken models
        broken_models = [k for k, v in all_results.items() 
                        if v['summary']['overall_accuracy'] < 0.55 or abs(v['summary']['bias']) > 0.4]
        
        if broken_models:
            print(f"   3. BROKEN MODELS detected: {', '.join([self.models[k]['config']['name'] for k in broken_models])}")
        
        # Ensemble recommendations
        good_models = [k for k, v in all_results.items() if v['summary']['overall_accuracy'] > 0.65]
        if len(good_models) >= 3:
            print(f"   4. ENSEMBLE READY: {len(good_models)} models suitable for ensemble")
        else:
            print(f"   4. ENSEMBLE NOT READY: Only {len(good_models)} models performing well enough")
        
        print(f"\n📈 NEXT STEPS:")
        print(f"   1. Focus training efforts on worst-performing models")
        print(f"   2. Implement model-specific recommendations")
        print(f"   3. Increase dataset diversity for biased models")
        print(f"   4. Consider architecture improvements for low-confidence models")
        print(f"   5. Re-evaluate ensemble weights based on current performance")

def main():
    """Run comprehensive model analysis"""
    analyzer = ModelAnalyzer()
    results = analyzer.run_comprehensive_analysis(max_videos_per_class=25)
    
    print(f"\n✅ Comprehensive analysis complete!")
    print(f"📄 Detailed results saved to 'comprehensive_model_analysis.json'")
    print(f"🔍 Review the recommendations for each model to improve performance")

if __name__ == "__main__":
    main()