#!/usr/bin/env python3
"""
TorchScript Export Script
Exports models to TorchScript format for mobile deployment
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.student import create_student_model
from models.teacher import create_teacher_model

def export_to_torchscript(model, example_inputs, output_path, method='trace'):
    """
    Export model to TorchScript
    
    Args:
        model: PyTorch model
        example_inputs: Tuple of example inputs
        output_path: Path to save TorchScript model
        method: 'trace' or 'script'
    """
    model.eval()
    
    print(f"Exporting model using {method} method...")
    
    with torch.no_grad():
        if method == 'trace':
            # Tracing method (recommended for most cases)
            traced_model = torch.jit.trace(model, example_inputs)
        elif method == 'script':
            # Scripting method (for models with control flow)
            traced_model = torch.jit.script(model)
        else:
            raise ValueError("Method must be 'trace' or 'script'")
    
    # Save the model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model.save(output_path)
    
    print(f"TorchScript model saved to: {output_path}")
    
    return traced_model

def test_torchscript_model(torchscript_path, example_inputs, original_model=None):
    """Test TorchScript model and compare with original"""
    print(f"Testing TorchScript model: {torchscript_path}")
    
    # Load TorchScript model
    loaded_model = torch.jit.load(torchscript_path)
    loaded_model.eval()
    
    # Test inference
    with torch.no_grad():
        start_time = time.time()
        ts_output = loaded_model(*example_inputs)
        inference_time = time.time() - start_time
    
    print(f"TorchScript inference time: {inference_time*1000:.2f} ms")
    print(f"TorchScript output shape: {ts_output.shape}")
    
    # Compare with original model if provided
    if original_model is not None:
        original_model.eval()
        with torch.no_grad():
            original_output = original_model(*example_inputs)
        
        # Calculate difference
        diff = torch.mean(torch.abs(ts_output - original_output))
        print(f"Output difference (MAE): {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ TorchScript model matches original model")
        else:
            print("⚠ TorchScript model differs from original model")
    
    return loaded_model

def get_model_size(model_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def optimize_torchscript(model, example_inputs):
    """Apply TorchScript optimizations"""
    print("Applying TorchScript optimizations...")
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_inputs)
    
    # Optimize for mobile
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    # Freeze the model (removes training-specific operations)
    frozen_model = torch.jit.freeze(optimized_model)
    
    return frozen_model

def main():
    parser = argparse.ArgumentParser(description='Export Models to TorchScript')
    parser.add_argument('--model', required=True, help='Path to PyTorch model checkpoint')
    parser.add_argument('--model_type', choices=['student', 'teacher'], default='student',
                       help='Type of model to export')
    parser.add_argument('--output', required=True, help='Output path for TorchScript model')
    parser.add_argument('--method', choices=['trace', 'script'], default='trace',
                       help='TorchScript export method')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply mobile optimizations')
    parser.add_argument('--test', action='store_true',
                       help='Test exported model')
    parser.add_argument('--num_frames', type=int, default=8, help='Number of video frames')
    parser.add_argument('--audio_duration', type=float, default=3.0, help='Audio duration')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for export')
    
    args = parser.parse_args()
    
    print(f"Exporting {args.model_type} model to TorchScript...")
    
    # Create model
    if args.model_type == 'student':
        model = create_student_model()
    else:
        model = create_teacher_model()
    
    # Load checkpoint
    checkpoint = torch.load(args.model, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model = checkpoint['model']
    else:
        # Assume checkpoint is the model state dict
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create example inputs
    sample_rate = 16000
    audio_samples = int(args.audio_duration * sample_rate)
    
    if args.model_type == 'student':
        # Student model can handle both single frame and multi-frame
        example_video = torch.randn(args.batch_size, args.num_frames, 3, 224, 224)
        example_audio = torch.randn(args.batch_size, audio_samples)
        example_inputs = (example_video, example_audio)
    else:
        # Teacher model expects multi-frame input
        example_video = torch.randn(args.batch_size, args.num_frames, 3, 224, 224)
        example_audio = torch.randn(args.batch_size, audio_samples)
        example_inputs = (example_video, example_audio)
    
    print(f"Example video input: {example_video.shape}")
    print(f"Example audio input: {example_audio.shape}")
    
    # Test original model first
    print("Testing original model...")
    with torch.no_grad():
        original_output = model(*example_inputs)
    print(f"Original model output shape: {original_output.shape}")
    
    # Apply optimizations if requested
    if args.optimize:
        model = optimize_torchscript(model, example_inputs)
        print("Applied mobile optimizations")
    
    # Export to TorchScript
    torchscript_model = export_to_torchscript(
        model, example_inputs, args.output, args.method
    )
    
    # Get file size
    model_size = get_model_size(args.output)
    print(f"TorchScript model size: {model_size:.2f} MB")
    
    # Test exported model
    if args.test:
        test_torchscript_model(args.output, example_inputs, model)
    
    # Save export info
    export_info = {
        'model_type': args.model_type,
        'export_method': args.method,
        'optimized': args.optimize,
        'input_shapes': {
            'video': list(example_video.shape),
            'audio': list(example_audio.shape)
        },
        'output_shape': list(original_output.shape),
        'model_size_mb': model_size,
        'num_frames': args.num_frames,
        'audio_duration': args.audio_duration,
        'sample_rate': sample_rate
    }
    
    info_path = args.output.replace('.pt', '_export_info.json')
    with open(info_path, 'w') as f:
        json.dump(export_info, f, indent=2)
    
    print(f"Export info saved to: {info_path}")
    print("TorchScript export completed successfully!")

if __name__ == "__main__":
    main()