#!/usr/bin/env python3
"""
Robustness Testing Script
Tests model performance under various adversarial conditions
"""

import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model
from preprocess.augmentation import VideoAugmentation, AudioAugmentation

class RobustnessEvaluator:
    """Comprehensive robustness evaluation"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize augmentation modules
        self.video_aug = VideoAugmentation()
        self.audio_aug = AudioAugmentation()
        
        # Results storage
        self.results = {}
    
    def test_compression_robustness(self, video_frames, audio_waveform, quality_levels=[10, 20, 30, 40, 50]):
        """Test robustness to video compression"""
        print("Testing compression robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different compression levels
        for quality in quality_levels:
            compressed_frames = self.video_aug.apply_compression(video_frames, quality)
            
            with torch.no_grad():
                logits = self.model(compressed_frames, audio_waveform)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'quality_{quality}'] = prob
        
        return results
    
    def test_noise_robustness(self, video_frames, audio_waveform, noise_levels=[0.01, 0.02, 0.05, 0.1]):
        """Test robustness to noise"""
        print("Testing noise robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different noise levels
        for noise_std in noise_levels:
            # Video noise
            video_noise = torch.randn_like(video_frames) * noise_std
            noisy_video = torch.clamp(video_frames + video_noise, 0, 1)
            
            # Audio noise
            audio_noise = torch.randn_like(audio_waveform) * noise_std * torch.std(audio_waveform)
            noisy_audio = audio_waveform + audio_noise
            
            with torch.no_grad():
                logits = self.model(noisy_video, noisy_audio)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'noise_{noise_std}'] = prob
        
        return results
    
    def test_blur_robustness(self, video_frames, audio_waveform, kernel_sizes=[3, 5, 7, 9]):
        """Test robustness to blur"""
        print("Testing blur robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different blur levels
        for kernel_size in kernel_sizes:
            blurred_frames = self.video_aug.apply_blur(video_frames, kernel_size)
            
            with torch.no_grad():
                logits = self.model(blurred_frames, audio_waveform)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'blur_{kernel_size}'] = prob
        
        return results
    
    def test_brightness_robustness(self, video_frames, audio_waveform, brightness_factors=[0.5, 0.7, 1.3, 1.5]):
        """Test robustness to brightness changes"""
        print("Testing brightness robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different brightness levels
        for factor in brightness_factors:
            bright_frames = torch.clamp(video_frames * factor, 0, 1)
            
            with torch.no_grad():
                logits = self.model(bright_frames, audio_waveform)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'brightness_{factor}'] = prob
        
        return results
    
    def test_audio_pitch_robustness(self, video_frames, audio_waveform, pitch_factors=[0.8, 0.9, 1.1, 1.2]):
        """Test robustness to audio pitch changes"""
        print("Testing audio pitch robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different pitch levels
        for factor in pitch_factors:
            # Simple pitch shift approximation using resampling
            original_length = audio_waveform.shape[-1]
            new_length = int(original_length / factor)
            
            # Resample
            pitched_audio = F.interpolate(
                audio_waveform.unsqueeze(0), 
                size=new_length, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            
            # Pad or trim to original length
            if new_length > original_length:
                pitched_audio = pitched_audio[..., :original_length]
            else:
                padding = original_length - new_length
                pitched_audio = F.pad(pitched_audio, (0, padding))
            
            with torch.no_grad():
                logits = self.model(video_frames, pitched_audio)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'pitch_{factor}'] = prob
        
        return results
    
    def test_frame_dropout(self, video_frames, audio_waveform, dropout_rates=[0.1, 0.2, 0.3, 0.5]):
        """Test robustness to frame dropout"""
        print("Testing frame dropout robustness...")
        
        results = {}
        
        # Original prediction
        with torch.no_grad():
            original_logits = self.model(video_frames, audio_waveform)
            original_prob = torch.softmax(original_logits, dim=1)[0, 1].item()
        
        results['original'] = original_prob
        
        # Test different dropout rates
        for dropout_rate in dropout_rates:
            B, T, C, H, W = video_frames.shape
            
            # Create dropout mask
            keep_prob = 1 - dropout_rate
            mask = torch.rand(B, T, 1, 1, 1) < keep_prob
            
            # Apply dropout (replace dropped frames with zeros)
            dropped_frames = video_frames * mask.float()
            
            with torch.no_grad():
                logits = self.model(dropped_frames, audio_waveform)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            
            results[f'dropout_{dropout_rate}'] = prob
        
        return results
    
    def run_comprehensive_test(self, video_frames, audio_waveform):
        """Run all robustness tests"""
        print("Running comprehensive robustness evaluation...")
        
        # Move inputs to device
        video_frames = video_frames.to(self.device)
        audio_waveform = audio_waveform.to(self.device)
        
        # Run all tests
        self.results['compression'] = self.test_compression_robustness(video_frames, audio_waveform)
        self.results['noise'] = self.test_noise_robustness(video_frames, audio_waveform)
        self.results['blur'] = self.test_blur_robustness(video_frames, audio_waveform)
        self.results['brightness'] = self.test_brightness_robustness(video_frames, audio_waveform)
        self.results['audio_pitch'] = self.test_audio_pitch_robustness(video_frames, audio_waveform)
        self.results['frame_dropout'] = self.test_frame_dropout(video_frames, audio_waveform)
        
        return self.results
    
    def calculate_robustness_metrics(self):
        """Calculate overall robustness metrics"""
        metrics = {}
        
        for test_name, test_results in self.results.items():
            original_prob = test_results['original']
            
            # Calculate stability (how much predictions change)
            deviations = []
            for key, prob in test_results.items():
                if key != 'original':
                    deviation = abs(prob - original_prob)
                    deviations.append(deviation)
            
            if deviations:
                metrics[test_name] = {
                    'mean_deviation': np.mean(deviations),
                    'max_deviation': np.max(deviations),
                    'stability_score': 1.0 - np.mean(deviations)  # Higher is more stable
                }
        
        # Overall robustness score
        stability_scores = [m['stability_score'] for m in metrics.values()]
        metrics['overall'] = {
            'robustness_score': np.mean(stability_scores),
            'worst_case_stability': np.min(stability_scores)
        }
        
        return metrics
    
    def plot_robustness_results(self, save_path=None):
        """Plot robustness test results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        test_names = list(self.results.keys())
        
        for i, test_name in enumerate(test_names):
            if i >= len(axes):
                break
                
            test_results = self.results[test_name]
            
            # Extract data for plotting
            conditions = []
            probabilities = []
            
            for key, prob in test_results.items():
                conditions.append(key)
                probabilities.append(prob)
            
            # Plot
            axes[i].plot(range(len(conditions)), probabilities, 'o-')
            axes[i].set_title(f'{test_name.title()} Robustness')
            axes[i].set_xlabel('Condition')
            axes[i].set_ylabel('Fake Probability')
            axes[i].set_xticks(range(len(conditions)))
            axes[i].set_xticklabels(conditions, rotation=45, ha='right')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        # Remove empty subplots
        for i in range(len(test_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness plot saved to: {save_path}")
        
        plt.show()

def create_test_data(num_frames=8, audio_duration=3.0, sample_rate=16000):
    """Create synthetic test data"""
    # Create synthetic video frames
    video_frames = torch.randn(1, num_frames, 3, 224, 224)
    video_frames = torch.clamp(video_frames * 0.5 + 0.5, 0, 1)  # Normalize to [0,1]
    
    # Create synthetic audio
    audio_samples = int(audio_duration * sample_rate)
    audio_waveform = torch.randn(1, audio_samples) * 0.1
    
    return video_frames, audio_waveform

def main():
    parser = argparse.ArgumentParser(description='Test Model Robustness')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='eval_results', help='Output directory for results')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model}")
    
    # Load model
    device = torch.device(args.device)
    model = create_student_model()
    checkpoint = torch.load(args.model, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create test data
    print("Creating test data...")
    video_frames, audio_waveform = create_test_data(
        args.num_frames, args.audio_duration
    )
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator(model, device)
    
    # Run robustness tests
    results = evaluator.run_comprehensive_test(video_frames, audio_waveform)
    
    # Calculate metrics
    metrics = evaluator.calculate_robustness_metrics()
    
    # Print results
    print("\n" + "="*50)
    print("ROBUSTNESS TEST RESULTS")
    print("="*50)
    
    for test_name, test_results in results.items():
        print(f"\n{test_name.upper()} TEST:")
        for condition, prob in test_results.items():
            print(f"  {condition}: {prob:.4f}")
    
    print("\n" + "="*50)
    print("ROBUSTNESS METRICS")
    print("="*50)
    
    for test_name, test_metrics in metrics.items():
        if test_name == 'overall':
            print(f"\nOVERALL ROBUSTNESS:")
            print(f"  Robustness Score: {test_metrics['robustness_score']:.4f}")
            print(f"  Worst Case Stability: {test_metrics['worst_case_stability']:.4f}")
        else:
            print(f"\n{test_name.upper()}:")
            print(f"  Mean Deviation: {test_metrics['mean_deviation']:.4f}")
            print(f"  Max Deviation: {test_metrics['max_deviation']:.4f}")
            print(f"  Stability Score: {test_metrics['stability_score']:.4f}")
    
    # Save results
    results_file = output_dir / 'robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'metrics': metrics,
            'test_config': {
                'model_path': args.model,
                'num_frames': args.num_frames,
                'audio_duration': args.audio_duration,
                'device': args.device
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    if args.plot:
        plot_file = output_dir / 'robustness_plot.png'
        evaluator.plot_robustness_results(str(plot_file))
    
    print("\nRobustness evaluation completed!")

if __name__ == "__main__":
    main()