#!/usr/bin/env python3
"""
DETAILED BG MODEL ANALYSIS
Background/Lighting Specialist Model - Comprehensive Testing and Training Recommendations
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

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.specialists_new import BGSpecialistModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")

class BGModelAnalyzer:
    """Detailed analysis of Background/Lighting Specialist Model"""
    
    def __init__(self):
        self.model = None
        self.model_info = {
            'name': 'Background/Lighting Specialist',
            'specialization': 'Background artifacts, lighting inconsistencies, shadow detection',
            'architecture': 'EfficientNet-B4 + BG Specialist Module (44 channels)',
            'specialist_components': [
                'bg_texture (texture analysis)',
                'lighting_detector (lighting consistency)',
                'shadow_checker (shadow artifacts)',
                'color_temp (color temperature analysis)'
            ],
            'expected_strengths': [
                'Lighting manipulation detection',
                'Background replacement artifacts',
                'Shadow inconsistencies',
                'Color temperature mismatches',
                'Texture pattern analysis'
            ]
        }
    
    def load_model(self):
        """Load BG specialist model"""
        print("=" * 70)
        print("LOADING BG SPECIALIST MODEL")
        print("=" * 70)
        
        model_path = 'models/baseline_student.pt'
        
        try:
            self.model = BGSpecialistModel()
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metrics = checkpoint.get('metrics', {})
            else:
                state_dict = checkpoint
                metrics = {}
            
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(DEVICE)
            self.model.eval()
            
            print(f"✅ BG Model loaded successfully")
            print(f"📊 Architecture: {self.model_info['architecture']}")
            print(f"🎯 Specialization: {self.model_info['specialization']}")
            
            if metrics:
                print(f"📈 Training metrics: {metrics}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load BG model: {e}")
            return False
    
    def analyze_lighting_conditions(self, frame):
        """Analyze lighting conditions in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Brightness analysis
        brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Contrast analysis
        contrast = np.std(gray)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_peaks = len([i for i in range(1, 255) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
        
        # Shadow detection (simple)
        shadow_threshold = brightness * 0.3
        shadow_pixels = np.sum(gray < shadow_threshold)
        shadow_ratio = shadow_pixels / gray.size
        
        # Highlight detection
        highlight_threshold = brightness * 1.7
        highlight_pixels = np.sum(gray > highlight_threshold)
        highlight_ratio = highlight_pixels / gray.size
        
        return {
            'brightness': float(brightness),
            'brightness_std': float(brightness_std),
            'contrast': float(contrast),
            'histogram_peaks': int(hist_peaks),
            'shadow_ratio': float(shadow_ratio),
            'highlight_ratio': float(highlight_ratio),
            'lighting_quality': 'good' if 50 < brightness < 200 and contrast > 30 else 'poor',
            'is_low_light': brightness < 80,
            'is_overexposed': brightness > 200,
            'has_shadows': shadow_ratio > 0.1,
            'has_highlights': highlight_ratio > 0.05
        }
    
    def analyze_background_complexity(self, frame):
        """Analyze background complexity and texture"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Texture analysis using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        
        # Pattern detection (simple)
        pattern_score = texture_variance / (avg_gradient + 1)
        
        return {
            'texture_variance': float(texture_variance),
            'edge_density': float(edge_density),
            'avg_gradient': float(avg_gradient),
            'pattern_score': float(pattern_score),
            'complexity': 'high' if texture_variance > 500 else 'medium' if texture_variance > 100 else 'low',
            'is_textured': texture_variance > 200,
            'has_patterns': pattern_score > 50
        }
    
    def extract_frames_with_bg_analysis(self, video_path, num_frames=8):
        """Extract frames with BG-specific analysis"""
        frames = []
        bg_metadata = []
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None, None
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # BG-specific analysis
                lighting_analysis = self.analyze_lighting_conditions(frame_rgb)
                bg_analysis = self.analyze_background_complexity(frame_rgb)
                
                bg_metadata.append({
                    'frame_idx': idx,
                    'lighting': lighting_analysis,
                    'background': bg_analysis
                })
                
                # Prepare for model input
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frame_normalized = (frame_normalized - mean) / std
                frame_tensor = torch.from_numpy(frame_normalized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frames.append(frame_tensor)
        
        cap.release()
        return torch.stack(frames) if frames else None, bg_metadata
    
    def predict_with_analysis(self, frames, bg_metadata):
        """Run BG model prediction with detailed analysis"""
        if self.model is None:
            return None
        
        try:
            with torch.no_grad():
                input_tensor = frames[:4].to(DEVICE)
                logits = self.model(input_tensor)
                probs = F.softmax(logits, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()
                
                # Detailed analysis
                analysis = {
                    'prediction': 'FAKE' if np.mean(fake_probs) > 0.5 else 'REAL',
                    'fake_probability': float(np.mean(fake_probs)),
                    'confidence': float(max(np.mean(fake_probs), 1 - np.mean(fake_probs))),
                    'frame_predictions': fake_probs.tolist(),
                    'prediction_variance': float(np.var(fake_probs)),
                    'frame_consistency': 1.0 - float(np.var(fake_probs)),
                    
                    # BG-specific analysis
                    'lighting_assessment': self._assess_lighting_artifacts(fake_probs, bg_metadata),
                    'background_assessment': self._assess_background_artifacts(fake_probs, bg_metadata),
                    'shadow_assessment': self._assess_shadow_artifacts(fake_probs, bg_metadata),
                    'texture_assessment': self._assess_texture_artifacts(fake_probs, bg_metadata)
                }
                
                return analysis
                
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_lighting_artifacts(self, fake_probs, bg_metadata):
        """Assess lighting-related artifacts"""
        avg_fake_prob = np.mean(fake_probs)
        
        # Analyze lighting conditions across frames
        lighting_conditions = [frame['lighting'] for frame in bg_metadata]
        
        brightness_values = [lc['brightness'] for lc in lighting_conditions]
        contrast_values = [lc['contrast'] for lc in lighting_conditions]
        
        brightness_consistency = 1.0 - (np.std(brightness_values) / (np.mean(brightness_values) + 1))
        contrast_consistency = 1.0 - (np.std(contrast_values) / (np.mean(contrast_values) + 1))
        
        # Lighting artifact indicators
        has_low_light = any(lc['is_low_light'] for lc in lighting_conditions)
        has_overexposure = any(lc['is_overexposed'] for lc in lighting_conditions)
        inconsistent_lighting = brightness_consistency < 0.8
        
        assessment = {
            'lighting_consistency': float(brightness_consistency),
            'contrast_consistency': float(contrast_consistency),
            'avg_brightness': float(np.mean(brightness_values)),
            'brightness_variance': float(np.var(brightness_values)),
            'has_low_light': has_low_light,
            'has_overexposure': has_overexposure,
            'inconsistent_lighting': inconsistent_lighting,
            'lighting_manipulation_likelihood': float(avg_fake_prob) if inconsistent_lighting else 0.3,
            'recommendation': self._get_lighting_recommendation(avg_fake_prob, lighting_conditions)
        }
        
        return assessment
    
    def _assess_background_artifacts(self, fake_probs, bg_metadata):
        """Assess background-related artifacts"""
        avg_fake_prob = np.mean(fake_probs)
        
        # Analyze background complexity
        bg_conditions = [frame['background'] for frame in bg_metadata]
        
        texture_values = [bc['texture_variance'] for bc in bg_conditions]
        edge_densities = [bc['edge_density'] for bc in bg_conditions]
        
        texture_consistency = 1.0 - (np.std(texture_values) / (np.mean(texture_values) + 1))
        edge_consistency = 1.0 - (np.std(edge_densities) / (np.mean(edge_densities) + 1))
        
        assessment = {
            'texture_consistency': float(texture_consistency),
            'edge_consistency': float(edge_consistency),
            'avg_texture_variance': float(np.mean(texture_values)),
            'avg_edge_density': float(np.mean(edge_densities)),
            'background_complexity': bg_conditions[0]['complexity'] if bg_conditions else 'unknown',
            'has_patterns': any(bc['has_patterns'] for bc in bg_conditions),
            'background_replacement_likelihood': float(avg_fake_prob) if texture_consistency < 0.7 else 0.2,
            'recommendation': self._get_background_recommendation(avg_fake_prob, bg_conditions)
        }
        
        return assessment
    
    def _assess_shadow_artifacts(self, fake_probs, bg_metadata):
        """Assess shadow-related artifacts"""
        avg_fake_prob = np.mean(fake_probs)
        
        lighting_conditions = [frame['lighting'] for frame in bg_metadata]
        
        shadow_ratios = [lc['shadow_ratio'] for lc in lighting_conditions]
        highlight_ratios = [lc['highlight_ratio'] for lc in lighting_conditions]
        
        shadow_consistency = 1.0 - np.std(shadow_ratios)
        highlight_consistency = 1.0 - np.std(highlight_ratios)
        
        assessment = {
            'shadow_consistency': float(shadow_consistency),
            'highlight_consistency': float(highlight_consistency),
            'avg_shadow_ratio': float(np.mean(shadow_ratios)),
            'avg_highlight_ratio': float(np.mean(highlight_ratios)),
            'has_shadows': any(lc['has_shadows'] for lc in lighting_conditions),
            'has_highlights': any(lc['has_highlights'] for lc in lighting_conditions),
            'shadow_manipulation_likelihood': float(avg_fake_prob) if shadow_consistency < 0.8 else 0.25,
            'recommendation': self._get_shadow_recommendation(avg_fake_prob, lighting_conditions)
        }
        
        return assessment
    
    def _assess_texture_artifacts(self, fake_probs, bg_metadata):
        """Assess texture-related artifacts"""
        avg_fake_prob = np.mean(fake_probs)
        
        bg_conditions = [frame['background'] for frame in bg_metadata]
        pattern_scores = [bc['pattern_score'] for bc in bg_conditions]
        
        pattern_consistency = 1.0 - (np.std(pattern_scores) / (np.mean(pattern_scores) + 1))
        
        assessment = {
            'pattern_consistency': float(pattern_consistency),
            'avg_pattern_score': float(np.mean(pattern_scores)),
            'is_textured': any(bc['is_textured'] for bc in bg_conditions),
            'texture_manipulation_likelihood': float(avg_fake_prob) if pattern_consistency < 0.7 else 0.2,
            'recommendation': self._get_texture_recommendation(avg_fake_prob, bg_conditions)
        }
        
        return assessment
    
    def _get_lighting_recommendation(self, fake_prob, lighting_conditions):
        """Get lighting-specific training recommendations"""
        recommendations = []
        
        if fake_prob > 0.7:
            recommendations.append("Strong lighting manipulation detected")
        elif fake_prob > 0.5:
            recommendations.append("Possible lighting inconsistencies")
        
        if any(lc['is_low_light'] for lc in lighting_conditions):
            recommendations.append("Train on more low-light scenarios")
        
        if any(lc['is_overexposed'] for lc in lighting_conditions):
            recommendations.append("Train on overexposed lighting conditions")
        
        brightness_variance = np.var([lc['brightness'] for lc in lighting_conditions])
        if brightness_variance > 1000:
            recommendations.append("Train on consistent lighting scenarios")
        
        return recommendations
    
    def _get_background_recommendation(self, fake_prob, bg_conditions):
        """Get background-specific training recommendations"""
        recommendations = []
        
        if fake_prob > 0.7:
            recommendations.append("Strong background manipulation detected")
        
        complexities = [bc['complexity'] for bc in bg_conditions]
        if 'low' in complexities:
            recommendations.append("Train on simple background scenarios")
        if 'high' in complexities:
            recommendations.append("Train on complex background scenarios")
        
        if any(bc['has_patterns'] for bc in bg_conditions):
            recommendations.append("Focus on pattern-based background detection")
        
        return recommendations
    
    def _get_shadow_recommendation(self, fake_prob, lighting_conditions):
        """Get shadow-specific training recommendations"""
        recommendations = []
        
        shadow_ratios = [lc['shadow_ratio'] for lc in lighting_conditions]
        avg_shadow_ratio = np.mean(shadow_ratios)
        
        if fake_prob > 0.6 and avg_shadow_ratio > 0.15:
            recommendations.append("Shadow manipulation likely detected")
        
        if avg_shadow_ratio < 0.05:
            recommendations.append("Train on videos with more shadow content")
        elif avg_shadow_ratio > 0.3:
            recommendations.append("Train on high-shadow scenarios")
        
        return recommendations
    
    def _get_texture_recommendation(self, fake_prob, bg_conditions):
        """Get texture-specific training recommendations"""
        recommendations = []
        
        if fake_prob > 0.6:
            recommendations.append("Texture manipulation detected")
        
        if not any(bc['is_textured'] for bc in bg_conditions):
            recommendations.append("Train on more textured backgrounds")
        
        return recommendations
    
    def generate_training_recommendations(self, results):
        """Generate comprehensive training recommendations for BG model"""
        print("\n" + "=" * 70)
        print("BG MODEL TRAINING RECOMMENDATIONS")
        print("=" * 70)
        
        # Performance analysis
        real_results = [r for r in results if r['true_label'] == 'REAL']
        fake_results = [r for r in results if r['true_label'] == 'FAKE']
        
        real_accuracy = np.mean([1 if r['analysis']['prediction'] == 'REAL' else 0 for r in real_results])
        fake_accuracy = np.mean([1 if r['analysis']['prediction'] == 'FAKE' else 0 for r in fake_results])
        overall_accuracy = (real_accuracy * len(real_results) + fake_accuracy * len(fake_results)) / len(results)
        bias = real_accuracy - fake_accuracy
        
        print(f"\n📊 CURRENT PERFORMANCE:")
        print(f"   Overall Accuracy: {overall_accuracy:.1%}")
        print(f"   Real Accuracy: {real_accuracy:.1%}")
        print(f"   Fake Accuracy: {fake_accuracy:.1%}")
        print(f"   Bias: {bias:+.1%}")
        
        # Analyze failure patterns
        failed_real = [r for r in real_results if r['analysis']['prediction'] != 'REAL']
        failed_fake = [r for r in fake_results if r['analysis']['prediction'] != 'FAKE']
        
        print(f"\n🔍 FAILURE ANALYSIS:")
        print(f"   Failed Real Videos: {len(failed_real)}/{len(real_results)}")
        print(f"   Failed Fake Videos: {len(failed_fake)}/{len(fake_results)}")
        
        # Specific recommendations
        recommendations = {
            'priority': 'HIGH' if overall_accuracy < 0.6 else 'MEDIUM' if overall_accuracy < 0.7 else 'LOW',
            'training_focus': [],
            'dataset_needs': [],
            'architecture_improvements': [],
            'data_augmentation': []
        }
        
        # Performance-based recommendations
        if overall_accuracy < 0.6:
            recommendations['training_focus'].extend([
                'URGENT: Fundamental model improvement needed',
                'Increase training data significantly',
                'Review model architecture',
                'Implement better data augmentation'
            ])
        
        if abs(bias) > 0.2:
            if bias > 0:  # Real bias
                recommendations['training_focus'].append('Add more challenging fake samples')
                recommendations['dataset_needs'].append('Hard-to-detect fake videos with background manipulation')
            else:  # Fake bias
                recommendations['training_focus'].append('Add more diverse real samples')
                recommendations['dataset_needs'].append('More natural real videos with varied backgrounds')
        
        # Lighting-specific recommendations
        lighting_issues = []
        for result in results:
            if 'lighting_assessment' in result['analysis']:
                la = result['analysis']['lighting_assessment']
                if la['inconsistent_lighting']:
                    lighting_issues.append('inconsistent_lighting')
                if la['has_low_light']:
                    lighting_issues.append('low_light')
                if la['has_overexposure']:
                    lighting_issues.append('overexposure')
        
        if lighting_issues:
            recommendations['dataset_needs'].extend([
                'More videos with consistent lighting',
                'Low-light video samples',
                'Overexposed video samples',
                'Mixed lighting condition videos'
            ])
        
        # Background-specific recommendations
        recommendations['dataset_needs'].extend([
            'Simple background videos (solid colors, minimal texture)',
            'Complex background videos (detailed scenes, patterns)',
            'Background replacement samples',
            'Green screen artifacts',
            'Outdoor vs indoor lighting scenarios'
        ])
        
        # Architecture improvements
        recommendations['architecture_improvements'].extend([
            'Enhance lighting detection module',
            'Improve shadow analysis components',
            'Add color temperature analysis',
            'Implement multi-scale texture analysis',
            'Consider attention mechanisms for background focus'
        ])
        
        # Data augmentation
        recommendations['data_augmentation'].extend([
            'Lighting condition variations',
            'Shadow manipulation augmentation',
            'Background replacement simulation',
            'Color temperature shifts',
            'Texture pattern modifications'
        ])
        
        # Print recommendations
        print(f"\n🎯 TRAINING RECOMMENDATIONS (Priority: {recommendations['priority']}):")
        for i, rec in enumerate(recommendations['training_focus'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📚 DATASET REQUIREMENTS:")
        for i, need in enumerate(recommendations['dataset_needs'], 1):
            print(f"   {i}. {need}")
        
        print(f"\n🏗️ ARCHITECTURE IMPROVEMENTS:")
        for i, imp in enumerate(recommendations['architecture_improvements'], 1):
            print(f"   {i}. {imp}")
        
        print(f"\n🔄 DATA AUGMENTATION:")
        for i, aug in enumerate(recommendations['data_augmentation'], 1):
            print(f"   {i}. {aug}")
        
        return recommendations
    
    def run_detailed_analysis(self, max_videos_per_class=25):
        """Run detailed BG model analysis"""
        if not self.load_model():
            return None
        
        # Get test videos
        real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:max_videos_per_class]
        fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:max_videos_per_class]
        
        print(f"\n📹 Testing BG model on {len(real_videos)} real + {len(fake_videos)} fake videos")
        
        results = []
        
        # Test on real videos
        print(f"\nTesting on REAL videos...")
        for video_path in tqdm(real_videos, desc="Real videos"):
            frames, bg_metadata = self.extract_frames_with_bg_analysis(video_path)
            if frames is not None and bg_metadata:
                analysis = self.predict_with_analysis(frames, bg_metadata)
                if analysis and 'error' not in analysis:
                    results.append({
                        'video': video_path.name,
                        'true_label': 'REAL',
                        'bg_metadata': bg_metadata,
                        'analysis': analysis
                    })
        
        # Test on fake videos
        print(f"Testing on FAKE videos...")
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            frames, bg_metadata = self.extract_frames_with_bg_analysis(video_path)
            if frames is not None and bg_metadata:
                analysis = self.predict_with_analysis(frames, bg_metadata)
                if analysis and 'error' not in analysis:
                    results.append({
                        'video': video_path.name,
                        'true_label': 'FAKE',
                        'bg_metadata': bg_metadata,
                        'analysis': analysis
                    })
        
        # Generate recommendations
        recommendations = self.generate_training_recommendations(results)
        
        # Save detailed results
        detailed_results = {
            'model_info': self.model_info,
            'test_summary': {
                'total_videos': len(results),
                'real_videos': len([r for r in results if r['true_label'] == 'REAL']),
                'fake_videos': len([r for r in results if r['true_label'] == 'FAKE'])
            },
            'recommendations': recommendations,
            'sample_results': results[:5]  # Save first 5 for inspection
        }
        
        with open('bg_model_detailed_analysis.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed BG model analysis saved to 'bg_model_detailed_analysis.json'")
        
        return results

def main():
    """Run detailed BG model analysis"""
    analyzer = BGModelAnalyzer()
    results = analyzer.run_detailed_analysis(max_videos_per_class=25)
    
    if results:
        print(f"\n✅ BG model analysis complete!")
        print(f"📄 Check 'bg_model_detailed_analysis.json' for detailed results")
    else:
        print(f"\n❌ BG model analysis failed!")

if __name__ == "__main__":
    main()