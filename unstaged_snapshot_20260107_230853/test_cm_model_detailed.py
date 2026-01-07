#!/usr/bin/env python3
"""
DETAILED CM MODEL ANALYSIS
Compression Specialist Model - Comprehensive Testing and Training Recommendations
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

sys.path.insert(0, str(Path(__file__).parent / "src"))
from models.specialists_new import CMSpecialistModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")

class CMModelAnalyzer:
    """Detailed analysis of Compression Specialist Model"""
    
    def __init__(self):
        self.model = None
        self.model_info = {
            'name': 'Compression Specialist Model',
            'specialization': 'Compression artifacts, DCT analysis, quantization patterns',
            'architecture': 'EfficientNet-B4 + CM Specialist Module (40 channels)',
            'specialist_components': [
                'specialist_dct_analyzer (DCT coefficient analysis)',
                'specialist_quant_detector (quantization artifact detection)',
                'specialist_block_checker (block artifact detection)',
                'specialist_compression_estimator (compression level estimation)'
            ],
            'expected_strengths': [
                'JPEG compression artifacts',
                'Video compression detection',
                'Quantization noise patterns',
                'Block artifacts',
                'DCT coefficient anomalies',
                'Compression level estimation'
            ]
        }
    
    def load_model(self):
        """Load CM specialist model"""
        print("=" * 70)
        print("LOADING CM SPECIALIST MODEL")
        print("=" * 70)
        
        model_path = 'models/cm_model_student.pt'
        
        try:
            self.model = CMSpecialistModel()
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
            
            print(f"✅ CM Model loaded successfully")
            print(f"📊 Architecture: {self.model_info['architecture']}")
            print(f"🎯 Specialization: {self.model_info['specialization']}")
            
            if metrics:
                print(f"📈 Training metrics: {metrics}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load CM model: {e}")
            return False
    
    def analyze_compression_artifacts(self, frame):
        """Analyze compression artifacts in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Block artifact detection (8x8 DCT blocks)
        h, w = gray.shape
        block_artifacts = 0
        block_count = 0
        
        for y in range(0, h-8, 8):
            for x in range(0, w-8, 8):
                block = gray[y:y+8, x:x+8]
                
                # Detect block boundaries (sudden intensity changes)
                if y > 0:  # Check top boundary
                    top_diff = np.mean(np.abs(block[0, :] - gray[y-1, x:x+8]))
                    if top_diff > 10:  # Threshold for block artifact
                        block_artifacts += 1
                
                if x > 0:  # Check left boundary
                    left_diff = np.mean(np.abs(block[:, 0] - gray[y:y+8, x-1]))
                    if left_diff > 10:
                        block_artifacts += 1
                
                block_count += 1
        
        block_artifact_ratio = block_artifacts / (block_count * 2) if block_count > 0 else 0
        
        # Quantization noise detection
        # High frequency noise patterns typical of compression
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        high_freq_noise = np.std(laplacian)
        
        # DCT-like pattern detection
        dct_frame = cv2.dct(gray.astype(np.float32))
        high_freq_energy = np.sum(np.abs(dct_frame[4:, 4:]))  # High frequency components
        total_energy = np.sum(np.abs(dct_frame))
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Compression quality estimation
        # Based on JPEG quality metrics
        quality_score = 100 - (high_freq_noise / 10)  # Rough estimation
        quality_score = max(0, min(100, quality_score))
        
        return {
            'block_artifact_ratio': float(block_artifact_ratio),
            'high_freq_noise': float(high_freq_noise),
            'high_freq_ratio': float(high_freq_ratio),
            'estimated_quality': float(quality_score),
            'has_block_artifacts': block_artifact_ratio > 0.1,
            'has_quantization_noise': high_freq_noise > 50,
            'compression_level': 'high' if quality_score < 50 else 'medium' if quality_score < 80 else 'low',
            'artifact_severity': 'severe' if block_artifact_ratio > 0.3 else 'moderate' if block_artifact_ratio > 0.1 else 'mild'
        }
    
    def analyze_video_compression(self, video_path):
        """Analyze video-level compression characteristics"""
        cap = cv2.VideoCapture(str(video_path))
        
        # Video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # File size and bitrate
        file_size = os.path.getsize(video_path)
        bitrate = (file_size * 8) / duration if duration > 0 else 0
        
        # Estimated compression ratio
        uncompressed_size = width * height * 3 * total_frames  # RGB
        compression_ratio = uncompressed_size / file_size if file_size > 0 else 1
        
        cap.release()
        
        return {
            'file_size': file_size,
            'bitrate': bitrate,
            'compression_ratio': compression_ratio,
            'resolution': (width, height),
            'fps': fps,
            'duration': duration,
            'is_highly_compressed': bitrate < 500000,  # < 500 kbps
            'is_low_quality': compression_ratio > 100,
            'compression_assessment': 'high' if bitrate < 500000 else 'medium' if bitrate < 2000000 else 'low'
        }
    
    def extract_frames_with_compression_analysis(self, video_path, num_frames=8):
        """Extract frames with compression-specific analysis"""
        frames = []
        compression_metadata = []
        
        # Video-level analysis
        video_compression = self.analyze_video_compression(video_path)
        
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
                
                # Compression-specific analysis
                compression_analysis = self.analyze_compression_artifacts(frame_rgb)
                
                frame_metadata = {
                    'frame_idx': idx,
                    'compression': compression_analysis,
                    'video_compression': video_compression
                }
                compression_metadata.append(frame_metadata)
                
                # Prepare for model input
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frame_normalized = frame_resized.astype(np.float32) / 255.0
                frame_normalized = (frame_normalized - mean) / std
                frame_tensor = torch.from_numpy(frame_normalized).float()
                frame_tensor = frame_tensor.permute(2, 0, 1)
                frames.append(frame_tensor)
        
        cap.release()
        return torch.stack(frames) if frames else None, compression_metadata
    
    def predict_with_compression_analysis(self, frames, compression_metadata):
        """Run CM model prediction with detailed compression analysis"""
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
                    
                    # Compression-specific analysis
                    'compression_assessment': self._assess_compression_artifacts(fake_probs, compression_metadata),
                    'quality_assessment': self._assess_quality_degradation(fake_probs, compression_metadata),
                    'artifact_assessment': self._assess_artifact_patterns(fake_probs, compression_metadata),
                    'dct_assessment': self._assess_dct_anomalies(fake_probs, compression_metadata)
                }
                
                return analysis
                
        except Exception as e:
            return {'error': str(e)}
    
    def _assess_compression_artifacts(self, fake_probs, compression_metadata):
        """Assess compression-related artifacts"""
        avg_fake_prob = np.mean(fake_probs)
        
        # Analyze compression characteristics
        video_compression = compression_metadata[0]['video_compression']
        frame_compressions = [frame['compression'] for frame in compression_metadata]
        
        # Block artifact analysis
        block_artifact_ratios = [fc['block_artifact_ratio'] for fc in frame_compressions]
        avg_block_artifacts = np.mean(block_artifact_ratios)
        block_consistency = 1.0 - np.std(block_artifact_ratios)
        
        # Quality analysis
        quality_scores = [fc['estimated_quality'] for fc in frame_compressions]
        avg_quality = np.mean(quality_scores)
        quality_consistency = 1.0 - (np.std(quality_scores) / 100)
        
        assessment = {
            'video_bitrate': video_compression['bitrate'],
            'compression_ratio': video_compression['compression_ratio'],
            'avg_block_artifacts': float(avg_block_artifacts),
            'block_consistency': float(block_consistency),
            'avg_quality_score': float(avg_quality),
            'quality_consistency': float(quality_consistency),
            'is_highly_compressed': video_compression['is_highly_compressed'],
            'compression_manipulation_likelihood': float(avg_fake_prob) if avg_block_artifacts > 0.2 else 0.3,
            'recommendation': self._get_compression_recommendation(avg_fake_prob, video_compression, frame_compressions)
        }
        
        return assessment
    
    def _assess_quality_degradation(self, fake_probs, compression_metadata):
        """Assess quality degradation patterns"""
        avg_fake_prob = np.mean(fake_probs)
        
        frame_compressions = [frame['compression'] for frame in compression_metadata]
        
        # High frequency noise analysis
        noise_levels = [fc['high_freq_noise'] for fc in frame_compressions]
        avg_noise = np.mean(noise_levels)
        noise_consistency = 1.0 - (np.std(noise_levels) / (avg_noise + 1))
        
        # High frequency ratio analysis
        hf_ratios = [fc['high_freq_ratio'] for fc in frame_compressions]
        avg_hf_ratio = np.mean(hf_ratios)
        
        assessment = {
            'avg_noise_level': float(avg_noise),
            'noise_consistency': float(noise_consistency),
            'avg_high_freq_ratio': float(avg_hf_ratio),
            'has_quantization_noise': any(fc['has_quantization_noise'] for fc in frame_compressions),
            'quality_degradation_likelihood': float(avg_fake_prob) if avg_noise > 60 else 0.25,
            'recommendation': self._get_quality_recommendation(avg_fake_prob, frame_compressions)
        }
        
        return assessment
    
    def _assess_artifact_patterns(self, fake_probs, compression_metadata):
        """Assess artifact patterns"""
        avg_fake_prob = np.mean(fake_probs)
        
        frame_compressions = [frame['compression'] for frame in compression_metadata]
        
        # Artifact severity analysis
        artifact_severities = [fc['artifact_severity'] for fc in frame_compressions]
        severe_count = sum(1 for sev in artifact_severities if sev == 'severe')
        moderate_count = sum(1 for sev in artifact_severities if sev == 'moderate')
        
        assessment = {
            'severe_artifacts': severe_count,
            'moderate_artifacts': moderate_count,
            'artifact_distribution': {
                'severe': severe_count / len(artifact_severities),
                'moderate': moderate_count / len(artifact_severities),
                'mild': (len(artifact_severities) - severe_count - moderate_count) / len(artifact_severities)
            },
            'artifact_manipulation_likelihood': float(avg_fake_prob) if severe_count > 0 else 0.2,
            'recommendation': self._get_artifact_recommendation(avg_fake_prob, frame_compressions)
        }
        
        return assessment
    
    def _assess_dct_anomalies(self, fake_probs, compression_metadata):
        """Assess DCT-related anomalies"""
        avg_fake_prob = np.mean(fake_probs)
        
        frame_compressions = [frame['compression'] for frame in compression_metadata]
        
        # High frequency energy analysis
        hf_ratios = [fc['high_freq_ratio'] for fc in frame_compressions]
        avg_hf_ratio = np.mean(hf_ratios)
        hf_consistency = 1.0 - np.std(hf_ratios)
        
        assessment = {
            'avg_high_freq_energy': float(avg_hf_ratio),
            'high_freq_consistency': float(hf_consistency),
            'dct_anomaly_likelihood': float(avg_fake_prob) if hf_consistency < 0.8 else 0.2,
            'recommendation': self._get_dct_recommendation(avg_fake_prob, frame_compressions)
        }
        
        return assessment
    
    def _get_compression_recommendation(self, fake_prob, video_compression, frame_compressions):
        """Get compression-specific recommendations"""
        recommendations = []
        
        if fake_prob > 0.7:
            recommendations.append("Strong compression manipulation detected")
        
        if video_compression['is_highly_compressed']:
            recommendations.append("Train on highly compressed videos")
        
        if any(fc['has_block_artifacts'] for fc in frame_compressions):
            recommendations.append("Focus on block artifact detection")
        
        return recommendations
    
    def _get_quality_recommendation(self, fake_prob, frame_compressions):
        """Get quality-specific recommendations"""
        recommendations = []
        
        avg_quality = np.mean([fc['estimated_quality'] for fc in frame_compressions])
        
        if fake_prob > 0.6 and avg_quality < 50:
            recommendations.append("Low quality compression manipulation detected")
        
        if any(fc['has_quantization_noise'] for fc in frame_compressions):
            recommendations.append("Train on quantization noise patterns")
        
        return recommendations
    
    def _get_artifact_recommendation(self, fake_prob, frame_compressions):
        """Get artifact-specific recommendations"""
        recommendations = []
        
        severe_artifacts = sum(1 for fc in frame_compressions if fc['artifact_severity'] == 'severe')
        
        if fake_prob > 0.6 and severe_artifacts > 0:
            recommendations.append("Severe compression artifacts detected")
        
        return recommendations
    
    def _get_dct_recommendation(self, fake_prob, frame_compressions):
        """Get DCT-specific recommendations"""
        recommendations = []
        
        if fake_prob > 0.6:
            recommendations.append("DCT coefficient anomalies detected")
            recommendations.append("Enhance DCT analysis module")
        
        return recommendations
    
    def generate_training_recommendations(self, results):
        """Generate comprehensive training recommendations for CM model"""
        print("\n" + "=" * 70)
        print("CM MODEL TRAINING RECOMMENDATIONS")
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
        
        # Compression-specific analysis
        compression_issues = []
        for result in results:
            if 'compression_assessment' in result['analysis']:
                ca = result['analysis']['compression_assessment']
                if ca['is_highly_compressed']:
                    compression_issues.append('high_compression')
                if ca['avg_block_artifacts'] > 0.2:
                    compression_issues.append('block_artifacts')
        
        print(f"\n🔍 COMPRESSION ANALYSIS:")
        print(f"   High Compression Issues: {compression_issues.count('high_compression')}")
        print(f"   Block Artifact Issues: {compression_issues.count('block_artifacts')}")
        
        # Generate recommendations
        recommendations = {
            'priority': 'HIGH' if overall_accuracy < 0.6 else 'MEDIUM' if overall_accuracy < 0.7 else 'LOW',
            'training_focus': [],
            'dataset_needs': [],
            'architecture_improvements': [],
            'compression_specific': []
        }
        
        # Performance-based recommendations
        if overall_accuracy < 0.6:
            recommendations['training_focus'].extend([
                'URGENT: Improve compression artifact detection',
                'Enhance DCT analysis capabilities',
                'Better quantization noise detection'
            ])
        
        # Compression-specific recommendations
        recommendations['compression_specific'].extend([
            'Train on multiple compression standards (H.264, H.265, VP9, AV1)',
            'Include various bitrate levels (100kbps to 10Mbps)',
            'Add JPEG compression artifact samples',
            'Include uncompressed reference videos',
            'Train on re-compressed videos (multiple generations)',
            'Add streaming compression artifacts'
        ])
        
        recommendations['dataset_needs'].extend([
            'Low bitrate compressed videos',
            'High compression ratio samples',
            'Block artifact examples',
            'Quantization noise samples',
            'DCT coefficient manipulation examples',
            'Quality degradation sequences'
        ])
        
        recommendations['architecture_improvements'].extend([
            'Enhance DCT analysis module',
            'Improve block artifact detection',
            'Add frequency domain analysis',
            'Implement compression level estimation',
            'Add multi-scale compression analysis'
        ])
        
        # Print recommendations
        print(f"\n🎯 TRAINING RECOMMENDATIONS (Priority: {recommendations['priority']}):")
        for i, rec in enumerate(recommendations['training_focus'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📚 COMPRESSION-SPECIFIC NEEDS:")
        for i, need in enumerate(recommendations['compression_specific'], 1):
            print(f"   {i}. {need}")
        
        print(f"\n🏗️ ARCHITECTURE IMPROVEMENTS:")
        for i, imp in enumerate(recommendations['architecture_improvements'], 1):
            print(f"   {i}. {imp}")
        
        return recommendations
    
    def run_detailed_analysis(self, max_videos_per_class=25):
        """Run detailed CM model analysis"""
        if not self.load_model():
            return None
        
        # Get test videos
        real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:max_videos_per_class]
        fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:max_videos_per_class]
        
        print(f"\n📹 Testing CM model on {len(real_videos)} real + {len(fake_videos)} fake videos")
        
        results = []
        
        # Test on real videos
        print(f"\nTesting on REAL videos...")
        for video_path in tqdm(real_videos, desc="Real videos"):
            frames, compression_metadata = self.extract_frames_with_compression_analysis(video_path)
            if frames is not None and compression_metadata:
                analysis = self.predict_with_compression_analysis(frames, compression_metadata)
                if analysis and 'error' not in analysis:
                    results.append({
                        'video': video_path.name,
                        'true_label': 'REAL',
                        'compression_metadata': compression_metadata,
                        'analysis': analysis
                    })
        
        # Test on fake videos
        print(f"Testing on FAKE videos...")
        for video_path in tqdm(fake_videos, desc="Fake videos"):
            frames, compression_metadata = self.extract_frames_with_compression_analysis(video_path)
            if frames is not None and compression_metadata:
                analysis = self.predict_with_compression_analysis(frames, compression_metadata)
                if analysis and 'error' not in analysis:
                    results.append({
                        'video': video_path.name,
                        'true_label': 'FAKE',
                        'compression_metadata': compression_metadata,
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
            'sample_results': results[:5]
        }
        
        with open('cm_model_detailed_analysis.json', 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed CM model analysis saved to 'cm_model_detailed_analysis.json'")
        
        return results

def main():
    """Run detailed CM model analysis"""
    analyzer = CMModelAnalyzer()
    results = analyzer.run_detailed_analysis(max_videos_per_class=25)
    
    if results:
        print(f"\n✅ CM model analysis complete!")
        print(f"📄 Check 'cm_model_detailed_analysis.json' for detailed results")
    else:
        print(f"\n❌ CM model analysis failed!")

if __name__ == "__main__":
    main()