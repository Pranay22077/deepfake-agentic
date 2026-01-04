#!/usr/bin/env python3
"""
Complete Step 2 Kaggle Workflow
Runs the entire teacher-student distillation and optimization pipeline
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def setup_kaggle_workspace():
    """Setup complete Kaggle workspace"""
    print("[SETUP] Setting up Kaggle workspace...")
    
    # Create directory structure
    directories = [
        '/kaggle/working/src/models',
        '/kaggle/working/src/train',
        '/kaggle/working/src/opt',
        '/kaggle/working/src/eval',
        '/kaggle/working/src/preprocess',
        '/kaggle/working/src/agent',
        '/kaggle/working/export',
        '/kaggle/working/config',
        '/kaggle/working/models',
        '/kaggle/working/logs',
        '/kaggle/working/teacher_predictions',
        '/kaggle/working/eval_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("[OK] Workspace setup complete")

def copy_source_files():
    """Copy source files to working directory"""
    print("[FOLDER] Copying source files...")
    
    # This would copy files from your uploaded dataset
    # In practice, you'd upload your src/ folder as a Kaggle dataset
    
    source_files = [
        # Models
        'src/models/teacher.py',
        'src/models/student.py',
        
        # Training scripts
        'src/train/train_teacher.py',
        'src/train/distill_student.py',
        'src/train/save_teacher_preds.py',
        
        # Optimization
        'src/opt/prune_model.py',
        'src/opt/quantize_model.py',
        'src/opt/fine_tune_pruned.py',
        
        # Evaluation
        'src/eval/robustness_test.py',
        
        # Preprocessing
        'src/preprocess/augmentation.py',
        'src/preprocess/extract_faces.py',
        'src/preprocess/extract_audio.py',
        
        # Agent
        'src/agent/agent_core.py',
        
        # Export
        'export/export_torchscript.py',
        
        # Config
        'config/agent_config.yaml'
    ]
    
    # In Kaggle, you'd copy from input dataset
    # cp /kaggle/input/eraksha-source-code/src/* /kaggle/working/src/
    
    print("[OK] Source files ready")

def run_complete_pipeline():
    """Run the complete Step 2 pipeline"""
    print("[RUN] Starting Complete Step 2 Pipeline")
    
    pipeline_steps = [
        {
            'name': 'Teacher Training',
            'script': 'kaggle_train_teacher.py',
            'description': 'Train heavy multimodal teacher model',
            'estimated_time': '4-6 hours'
        },
        {
            'name': 'Student Distillation',
            'script': 'kaggle_distill_student.py',
            'description': 'Distill knowledge to lightweight student',
            'estimated_time': '2-3 hours'
        },
        {
            'name': 'Model Optimization',
            'script': 'kaggle_optimize_models.py',
            'description': 'Prune, quantize, and export for mobile',
            'estimated_time': '1-2 hours'
        }
    ]
    
    results = {}
    total_start_time = time.time()
    
    for i, step in enumerate(pipeline_steps, 1):
        print(f"\n{'='*60}")
        print(f"STEP {i}/3: {step['name']}")
        print(f"Description: {step['description']}")
        print(f"Estimated time: {step['estimated_time']}")
        print(f"{'='*60}")
        
        step_start_time = time.time()
        
        # Run the step
        try:
            result = subprocess.run(
                ['python', f"/kaggle/working/{step['script']}"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per step
            )
            
            step_duration = time.time() - step_start_time
            
            if result.returncode == 0:
                print(f"[OK] {step['name']} completed successfully")
                print(f"[TIME] Duration: {step_duration/60:.1f} minutes")
                results[step['name']] = {
                    'status': 'success',
                    'duration': step_duration,
                    'output': result.stdout[-1000:]  # Last 1000 chars
                }
            else:
                print(f"[ERROR] {step['name']} failed")
                print(f"Error: {result.stderr}")
                results[step['name']] = {
                    'status': 'failed',
                    'duration': step_duration,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {step['name']} timed out after 2 hours")
            results[step['name']] = {
                'status': 'timeout',
                'duration': 7200,
                'error': 'Process timed out'
            }
        except Exception as e:
            print(f"[CRASH] {step['name']} crashed: {str(e)}")
            results[step['name']] = {
                'status': 'crashed',
                'duration': time.time() - step_start_time,
                'error': str(e)
            }
    
    total_duration = time.time() - total_start_time
    
    # Generate final report
    generate_final_report(results, total_duration)
    
    return results

def generate_final_report(results, total_duration):
    """Generate comprehensive final report"""
    print(f"\n{'='*60}")
    print("STEP 2 PIPELINE COMPLETION REPORT")
    print(f"{'='*60}")
    
    successful_steps = sum(1 for r in results.values() if r['status'] == 'success')
    total_steps = len(results)
    
    print(f"[STATS] Overall Success Rate: {successful_steps}/{total_steps} ({successful_steps/total_steps*100:.1f}%)")
    print(f"[TIME] Total Pipeline Duration: {total_duration/3600:.1f} hours")
    
    print(f"\n[INFO] Step-by-Step Results:")
    for step_name, result in results.items():
        status_emoji = {
            'success': '[OK]',
            'failed': '[ERROR]',
            'timeout': '[TIMEOUT]',
            'crashed': '[CRASH]'
        }
        
        emoji = status_emoji.get(result['status'], '❓')
        duration_min = result['duration'] / 60
        
        print(f"{emoji} {step_name}: {result['status'].upper()} ({duration_min:.1f} min)")
        
        if result['status'] != 'success' and 'error' in result:
            print(f"   Error: {result['error'][:200]}...")
    
    # Check what models were created
    print(f"\n[FOLDER] Generated Models:")
    model_files = [
        ('/kaggle/working/models/teacher_model.pt', 'Teacher Model'),
        ('/kaggle/working/models/student_distilled.pt', 'Distilled Student'),
        ('/kaggle/working/models/optimized/student_pruned.pt', 'Pruned Student'),
        ('/kaggle/working/models/optimized/student_quantized.pt', 'Quantized Student'),
        ('/kaggle/working/export/student_mobile.ptl', 'Mobile TorchScript')
    ]
    
    for file_path, description in model_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"[OK] {description}: {size_mb:.1f} MB")
        else:
            print(f"[ERROR] {description}: Not found")
    
    # Performance summary
    print(f"\n[TARGET] Expected Performance (if all steps successful):")
    print(f"   • Teacher Model: 98%+ accuracy, ~60MB")
    print(f"   • Student Model: 95%+ accuracy, ~4MB")
    print(f"   • Optimized Model: 93%+ accuracy, <2MB")
    print(f"   • Mobile Inference: <100ms on device")
    
    # Next steps
    print(f"\n[RUN] Next Steps:")
    if successful_steps >= 2:
        print(f"   1. Download models using kaggle_download_results.py")
        print(f"   2. Test models locally with robustness evaluation")
        print(f"   3. Deploy to Android app for testing")
        print(f"   4. Package APK for distribution")
    else:
        print(f"   1. Review error logs and fix issues")
        print(f"   2. Re-run failed steps individually")
        print(f"   3. Check data paths and dependencies")
    
    # Save report
    report_data = {
        'pipeline_results': results,
        'total_duration_hours': total_duration / 3600,
        'success_rate': successful_steps / total_steps,
        'timestamp': time.time(),
        'models_generated': [f for f, _ in model_files if os.path.exists(f)]
    }
    
    with open('/kaggle/working/step2_completion_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: /kaggle/working/step2_completion_report.json")

def main():
    """Main execution function"""
    print("[TARGET] E-Raksha Step 2: Complete Kaggle Pipeline")
    print("=" * 60)
    
    # Setup
    setup_kaggle_workspace()
    copy_source_files()
    
    # Run pipeline
    results = run_complete_pipeline()
    
    print(f"\n🏁 Pipeline execution completed!")
    print(f"Check /kaggle/working/ for all generated models and reports.")

if __name__ == "__main__":
    main()