#!/usr/bin/env python3
"""
Model Quantization Script
Implements post-training quantization for mobile deployment
"""

import os
import sys
import torch
import torch.quantization as quantization
import torch.nn as nn
import argparse
import json
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.student import create_student_model

class QuantizationCalibrator:
    """Calibration dataset for static quantization"""
    
    def __init__(self, model, calibration_data, device='cpu'):
        self.model = model
        self.calibration_data = calibration_data
        self.device = device
    
    def calibrate(self, num_batches=10):
        """Run calibration on sample data"""
        self.model.eval()
        
        print(f"Running calibration with {num_batches} batches...")
        
        with torch.no_grad():
            for i, (video, audio) in enumerate(self.calibration_data):
                if i >= num_batches:
                    break
                
                video = video.to(self.device)
                audio = audio.to(self.device)
                
                # Forward pass for calibration
                _ = self.model(video, audio)
                
                if (i + 1) % 5 == 0:
                    print(f"Calibrated {i + 1}/{num_batches} batches")

def create_calibration_data(batch_size=4, num_batches=10):
    """Create dummy calibration data"""
    calibration_data = []
    
    for _ in range(num_batches):
        # Create dummy video and audio data
        video = torch.randn(batch_size, 8, 3, 224, 224)  # Multi-frame
        audio = torch.randn(batch_size, 16000 * 3)       # 3 seconds
        calibration_data.append((video, audio))
    
    return calibration_data

def dynamic_quantization(model):
    """Apply dynamic quantization (weights only)"""
    print("Applying dynamic quantization...")
    
    # Quantize linear layers
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},  # Layers to quantize
        dtype=torch.qint8
    )
    
    return quantized_model

def static_quantization(model, calibration_data, device='cpu'):
    """Apply static quantization (weights + activations)"""
    print("Applying static quantization...")
    
    # Set quantization config
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Fuse modules for better quantization
    model = fuse_model_modules(model)
    
    # Prepare model for quantization
    model_prepared = quantization.prepare(model)
    
    # Calibrate with sample data
    calibrator = QuantizationCalibrator(model_prepared, calibration_data, device)
    calibrator.calibrate()
    
    # Convert to quantized model
    quantized_model = quantization.convert(model_prepared)
    
    return quantized_model

def fuse_model_modules(model):
    """Fuse conv-bn-relu modules for better quantization"""
    print("Fusing modules...")
    
    # This is a simplified version - in practice, you'd need to identify
    # specific modules to fuse based on your model architecture
    try:
        # Example fusing for common patterns
        for name, module in model.named_modules():
            if hasattr(module, 'conv_layers'):
                # Fuse conv-bn-relu in audio branch
                if hasattr(module, 'conv_layers') and isinstance(module.conv_layers, nn.Sequential):
                    # This would need to be customized based on actual architecture
                    pass
    except Exception as e:
        print(f"Module fusion failed (continuing without fusion): {e}")
    
    return model

def quantization_aware_training_setup(model):
    """Setup model for quantization-aware training"""
    print("Setting up quantization-aware training...")
    
    # Set QAT config
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare model for QAT
    model_prepared = quantization.prepare_qat(model)
    
    return model_prepared

def measure_inference_time(model, test_data, device='cpu', num_runs=100):
    """Measure inference time"""
    model.eval()
    model = model.to(device)
    
    times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            video, audio = test_data
            _ = model(video.to(device), audio.to(device))
        
        # Measure
        for _ in range(num_runs):
            video, audio = test_data
            video, audio = video.to(device), audio.to(device)
            
            start_time = time.time()
            _ = model(video, audio)
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb

def main():
    parser = argparse.ArgumentParser(description='Quantize Student Model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--out', required=True, help='Output path for quantized model')
    parser.add_argument('--mode', choices=['dynamic', 'static', 'qat'], 
                       default='dynamic', help='Quantization mode')
    parser.add_argument('--calibration_batches', type=int, default=10,
                       help='Number of calibration batches for static quantization')
    parser.add_argument('--measure_performance', action='store_true',
                       help='Measure inference performance')
    
    args = parser.parse_args()
    
    device = torch.device('cpu')  # Quantization typically done on CPU
    
    print(f"Loading model from {args.model}")
    
    # Load model
    model = create_student_model()
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Measure original model
    original_size = calculate_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")
    
    # Create test data for performance measurement
    test_video = torch.randn(1, 8, 3, 224, 224)
    test_audio = torch.randn(1, 16000 * 3)
    test_data = (test_video, test_audio)
    
    if args.measure_performance:
        original_time, original_std = measure_inference_time(model, test_data, device)
        print(f"Original inference time: {original_time*1000:.2f} ± {original_std*1000:.2f} ms")
    
    # Apply quantization
    if args.mode == 'dynamic':
        quantized_model = dynamic_quantization(model)
        
    elif args.mode == 'static':
        # Create calibration data
        calibration_data = create_calibration_data(
            batch_size=1, 
            num_batches=args.calibration_batches
        )
        quantized_model = static_quantization(model, calibration_data, device)
        
    elif args.mode == 'qat':
        # This would require retraining - just setup here
        quantized_model = quantization_aware_training_setup(model)
        print("QAT setup complete. Model needs to be retrained with QAT.")
    
    # Measure quantized model
    quantized_size = calculate_model_size(quantized_model)
    size_reduction = (1 - quantized_size / original_size) * 100
    
    print(f"\nQuantization Results:")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {size_reduction:.1f}%")
    
    if args.measure_performance and args.mode != 'qat':
        quantized_time, quantized_std = measure_inference_time(
            quantized_model, test_data, device
        )
        speedup = original_time / quantized_time
        print(f"Quantized inference time: {quantized_time*1000:.2f} ± {quantized_std*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")
    
    # Test model accuracy (basic forward pass)
    print("\nTesting quantized model...")
    try:
        with torch.no_grad():
            original_output = model(test_video, test_audio)
            if args.mode != 'qat':
                quantized_output = quantized_model(test_video, test_audio)
                
                # Compare outputs
                output_diff = torch.mean(torch.abs(original_output - quantized_output))
                print(f"Output difference (MAE): {output_diff:.6f}")
            
            print("Forward pass successful!")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return
    
    # Save quantized model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    if args.mode == 'qat':
        # Save QAT-prepared model
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'quantization_info': {
                'mode': args.mode,
                'original_size_mb': original_size,
                'status': 'prepared_for_qat'
            }
        }, args.out)
    else:
        # Save quantized model
        torch.save({
            'model': quantized_model,
            'quantization_info': {
                'mode': args.mode,
                'original_size_mb': original_size,
                'quantized_size_mb': quantized_size,
                'size_reduction_percent': size_reduction,
                'calibration_batches': args.calibration_batches if args.mode == 'static' else None
            }
        }, args.out)
    
    # Save quantization report
    report_path = args.out.replace('.pt', '_quantization_report.json')
    report_data = {
        'mode': args.mode,
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'size_reduction_percent': size_reduction
    }
    
    if args.measure_performance and args.mode != 'qat':
        report_data.update({
            'original_inference_ms': original_time * 1000,
            'quantized_inference_ms': quantized_time * 1000,
            'speedup': speedup
        })
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nQuantized model saved to: {args.out}")
    print(f"Quantization report saved to: {report_path}")

if __name__ == "__main__":
    main()