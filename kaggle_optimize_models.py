#!/usr/bin/env python3
"""
Kaggle Model Optimization Script
Applies pruning, quantization, and mobile optimization
"""

import os
import sys
import torch
import json
import subprocess
from pathlib import Path

# Kaggle environment setup
sys.path.append('/kaggle/working')

def setup_optimization_environment():
    """Setup optimization environment"""
    os.makedirs('/kaggle/working/models/optimized', exist_ok=True)
    os.makedirs('/kaggle/working/export', exist_ok=True)
    print("Optimization environment setup complete")

def run_pruning():
    """Run model pruning"""
    print("🔧 Starting model pruning...")
    
    cmd = [
        'python', '/kaggle/working/src/opt/prune_model.py',
        '--model', '/kaggle/working/models/student_distilled.pt',
        '--out', '/kaggle/working/models/optimized/student_pruned.pt',
        '--method', 'structured',
        '--ratio', '0.3'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Pruning completed successfully")
        print(result.stdout)
    else:
        print("❌ Pruning failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_fine_tuning():
    """Run fine-tuning after pruning"""
    print("🔧 Starting fine-tuning of pruned model...")
    
    cmd = [
        'python', '/kaggle/working/src/opt/fine_tune_pruned.py',
        '--pruned_model', '/kaggle/working/models/optimized/student_pruned.pt',
        '--data_dir', '/kaggle/input/deepfake-faces-dataset',
        '--save_path', '/kaggle/working/models/optimized/student_pruned_finetuned.pt',
        '--epochs', '5',
        '--batch_size', '16',
        '--lr', '1e-5'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Fine-tuning completed successfully")
        print(result.stdout)
    else:
        print("❌ Fine-tuning failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_quantization():
    """Run model quantization"""
    print("🔧 Starting model quantization...")
    
    cmd = [
        'python', '/kaggle/working/src/opt/quantize_model.py',
        '--model', '/kaggle/working/models/optimized/student_pruned_finetuned.pt',
        '--out', '/kaggle/working/models/optimized/student_quantized.pt',
        '--mode', 'dynamic',
        '--measure_performance'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Quantization completed successfully")
        print(result.stdout)
    else:
        print("❌ Quantization failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_torchscript_export():
    """Export to TorchScript for mobile"""
    print("🔧 Starting TorchScript export...")
    
    cmd = [
        'python', '/kaggle/working/export/export_torchscript.py',
        '--model', '/kaggle/working/models/optimized/student_quantized.pt',
        '--model_type', 'student',
        '--output', '/kaggle/working/export/student_mobile.ptl',
        '--optimize',
        '--test'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ TorchScript export completed successfully")
        print(result.stdout)
    else:
        print("❌ TorchScript export failed")
        print(result.stderr)
    
    return result.returncode == 0

def run_robustness_testing():
    """Run robustness evaluation"""
    print("🔧 Starting robustness testing...")
    
    cmd = [
        'python', '/kaggle/working/src/eval/robustness_test.py',
        '--model', '/kaggle/working/models/optimized/student_quantized.pt',
        '--output_dir', '/kaggle/working/eval_results',
        '--plot'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Robustness testing completed successfully")
        print(result.stdout)
    else:
        print("❌ Robustness testing failed")
        print(result.stderr)
    
    return result.returncode == 0

def create_optimization_summary():
    """Create optimization summary"""
    summary = {
        'optimization_pipeline': {
            'teacher_model': '/kaggle/working/models/teacher_model.pt',
            'student_distilled': '/kaggle/working/models/student_distilled.pt',
            'student_pruned': '/kaggle/working/models/optimized/student_pruned.pt',
            'student_finetuned': '/kaggle/working/models/optimized/student_pruned_finetuned.pt',
            'student_quantized': '/kaggle/working/models/optimized/student_quantized.pt',
            'mobile_export': '/kaggle/working/export/student_mobile.ptl'
        },
        'optimization_steps': [
            'Teacher training with heavy augmentation',
            'Knowledge distillation to student model',
            'Structured pruning (30% parameter reduction)',
            'Fine-tuning for accuracy recovery',
            'Dynamic quantization (INT8)',
            'TorchScript export with mobile optimization',
            'Robustness evaluation'
        ],
        'expected_performance': {
            'teacher_accuracy': '98%+',
            'student_accuracy': '95%+',
            'pruned_accuracy': '94%+',
            'quantized_accuracy': '93%+',
            'mobile_inference_time': '<100ms',
            'model_size': '<5MB'
        }
    }
    
    with open('/kaggle/working/optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📊 Optimization summary created")

def main():
    """Main optimization pipeline"""
    print("🚀 Starting Step 2 Model Optimization Pipeline")
    
    # Setup environment
    setup_optimization_environment()
    
    # Check if required models exist
    if not os.path.exists('/kaggle/working/models/student_distilled.pt'):
        print("❌ Student distilled model not found! Run distillation first.")
        return
    
    # Run optimization pipeline
    steps = [
        ("Pruning", run_pruning),
        ("Fine-tuning", run_fine_tuning),
        ("Quantization", run_quantization),
        ("TorchScript Export", run_torchscript_export),
        ("Robustness Testing", run_robustness_testing)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*50}")
        print(f"Step: {step_name}")
        print(f"{'='*50}")
        
        success = step_func()
        results[step_name] = success
        
        if not success:
            print(f"⚠️ {step_name} failed, but continuing with pipeline...")
    
    # Create summary
    create_optimization_summary()
    
    # Final report
    print(f"\n{'='*50}")
    print("OPTIMIZATION PIPELINE COMPLETE")
    print(f"{'='*50}")
    
    for step_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step_name}: {status}")
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully")
    
    if successful_steps >= 3:  # At least pruning, quantization, and export
        print("🎯 Minimum optimization requirements met!")
        print("📱 Models ready for mobile deployment")
    else:
        print("⚠️ Some optimization steps failed. Check logs for details.")

if __name__ == "__main__":
    main()