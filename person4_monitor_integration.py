"""
Person 4: Training Monitor & Integration System
Monitors all training processes, handles integration, and provides real-time status
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import threading
import subprocess
import requests
from typing import Dict, List, Any
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitors all training processes and provides real-time status"""
    
    def __init__(self, output_dir='/kaggle/working', data_dir='/kaggle/input/dfdc-10'):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model types and their trainers
        self.model_types = ['bg', 'av', 'cm', 'rr', 'll', 'tm']
        self.trainer_assignments = {
            'bg': 'person1',
            'av': 'person1', 
            'cm': 'person2',
            'rr': 'person2',
            'll': 'person3',
            'tm': 'person3'
        }
        
        # Monitoring state
        self.monitoring_active = True
        self.status_history = []
        self.performance_metrics = defaultdict(list)
        self.chunk_availability = {}
        
        # Integration state
        self.integration_ready = False
        self.final_models = {}
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Training Monitor initialized")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check chunk availability
                self._check_chunk_availability()
                
                # Monitor training progress
                self._monitor_training_progress()
                
                # Check system resources
                self._monitor_system_resources()
                
                # Generate status report
                self._generate_status_report()
                
                # Check for integration readiness
                self._check_integration_readiness()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def _check_chunk_availability(self):
        """Check which data chunks are available"""
        available_chunks = []
        
        for chunk_idx in range(10):  # Only first 10 chunks (00-09)
            chunk_file = f"{chunk_idx:02d}.zip"
            chunk_path = self.data_dir / chunk_file
            
            if chunk_path.exists():
                file_size = chunk_path.stat().st_size / (1024**3)  # GB
                available_chunks.append({
                    'chunk_idx': chunk_idx,
                    'filename': chunk_file,
                    'size_gb': round(file_size, 2),
                    'available_since': chunk_path.stat().st_mtime
                })
        
        self.chunk_availability = {
            'total_chunks': 10,  # Only first 10 chunks (00-09)
            'available_chunks': len(available_chunks),
            'chunks': available_chunks,
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"Chunks available: {len(available_chunks)}/10")
    
    def _monitor_training_progress(self):
        """Monitor progress of all training processes"""
        training_status = {}
        
        for model_type in self.model_types:
            status = self._get_model_training_status(model_type)
            training_status[model_type] = status
            
            # Update performance metrics
            if status['metrics']:
                self.performance_metrics[model_type].append({
                    'timestamp': datetime.now().isoformat(),
                    'chunk': status['current_chunk'],
                    'metrics': status['metrics']
                })
        
        # Save training status
        status_file = self.output_dir / 'training_status.json'
        with open(status_file, 'w') as f:
            json.dump(training_status, f, indent=2)
        
        return training_status
    
    def _get_model_training_status(self, model_type):
        """Get training status for a specific model"""
        # Check for checkpoints
        checkpoint_pattern = f"{model_type}_model_chunk_*.pt"
        checkpoints = list(self.output_dir.glob(checkpoint_pattern))
        
        if not checkpoints:
            return {
                'status': 'not_started',
                'current_chunk': 0,
                'total_chunks': 10,  # Only first 10 chunks (00-09)
                'progress': 0.0,
                'metrics': None,
                'last_checkpoint': None,
                'trainer': self.trainer_assignments[model_type]
            }
        
        # Get latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        try:
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
            current_chunk = checkpoint_data.get('current_chunk', 0)
            metrics = checkpoint_data.get('metrics', {})
            
            # Check if training is active (recent checkpoint)
            last_modified = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
            time_since_update = datetime.now() - last_modified
            
            if time_since_update.total_seconds() < 3600:  # Less than 1 hour
                status = 'training'
            elif current_chunk >= 9:  # Completed all 10 chunks (0-9)
                status = 'completed'
            else:
                status = 'paused'
            
            return {
                'status': status,
                'current_chunk': current_chunk + 1,
                'total_chunks': 10,  # Only first 10 chunks (00-09)
                'progress': (current_chunk + 1) / 10 * 100,
                'metrics': metrics,
                'last_checkpoint': latest_checkpoint.name,
                'last_updated': last_modified.isoformat(),
                'trainer': self.trainer_assignments[model_type]
            }
            
        except Exception as e:
            logger.error(f"Error reading checkpoint for {model_type}: {e}")
            return {
                'status': 'error',
                'current_chunk': 0,
                'total_chunks': 10,  # Only first 10 chunks (00-09)
                'progress': 0.0,
                'metrics': None,
                'last_checkpoint': latest_checkpoint.name,
                'error': str(e),
                'trainer': self.trainer_assignments[model_type]
            }
    
    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # GPU usage (if available)
            gpu_info = {}
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    gpu_info[f'gpu_{i}'] = {
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    }
            except:
                gpu_info = {'error': 'GPU monitoring not available'}
            
            resource_status = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory': {
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percent': (disk.used / disk.total) * 100
                },
                'gpu': gpu_info
            }
            
            # Save resource status
            resource_file = self.output_dir / 'resource_status.json'
            with open(resource_file, 'w') as f:
                json.dump(resource_status, f, indent=2)
            
            # Check for resource warnings
            if memory.percent > 90:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.free < 5 * (1024**3):  # Less than 5GB free
                logger.warning(f"Low disk space: {disk.free / (1024**3):.1f}GB free")
            
            return resource_status
            
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")
            return None
    
    def _generate_status_report(self):
        """Generate comprehensive status report"""
        try:
            # Get current status
            training_status = self._monitor_training_progress()
            resource_status = self._monitor_system_resources()
            
            # Calculate overall progress
            total_progress = sum(status['progress'] for status in training_status.values()) / len(training_status)
            
            # Count models by status
            status_counts = defaultdict(int)
            for status in training_status.values():
                status_counts[status['status']] += 1
            
            # Generate report
            report = {
                'timestamp': datetime.now().isoformat(),
                'overall_progress': round(total_progress, 2),
                'models': {
                    'total': len(self.model_types),
                    'training': status_counts['training'],
                    'completed': status_counts['completed'],
                    'paused': status_counts['paused'],
                    'not_started': status_counts['not_started'],
                    'error': status_counts['error']
                },
                'chunks': self.chunk_availability,
                'training_details': training_status,
                'resources': resource_status,
                'integration_ready': self.integration_ready
            }
            
            # Save report
            report_file = self.output_dir / 'status_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Generate visual dashboard
            self._generate_dashboard(report)
            
            logger.info(f"Status Report - Overall Progress: {total_progress:.1f}%, "
                       f"Training: {status_counts['training']}, "
                       f"Completed: {status_counts['completed']}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating status report: {e}")
            return None
    
    def _generate_dashboard(self, report):
        """Generate visual dashboard"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Interceptor Training Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Overall Progress
            ax1 = axes[0, 0]
            progress_data = [report['overall_progress'], 100 - report['overall_progress']]
            colors = ['#2E8B57', '#D3D3D3']
            ax1.pie(progress_data, labels=['Completed', 'Remaining'], colors=colors, autopct='%1.1f%%')
            ax1.set_title('Overall Training Progress')
            
            # 2. Model Status
            ax2 = axes[0, 1]
            status_data = report['models']
            statuses = ['training', 'completed', 'paused', 'not_started', 'error']
            counts = [status_data.get(s, 0) for s in statuses]
            colors = ['#FFD700', '#32CD32', '#FFA500', '#D3D3D3', '#FF6347']
            bars = ax2.bar(statuses, counts, color=colors)
            ax2.set_title('Models by Status')
            ax2.set_ylabel('Count')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                            str(count), ha='center', va='bottom')
            
            # 3. Individual Model Progress
            ax3 = axes[0, 2]
            model_names = list(report['training_details'].keys())
            model_progress = [report['training_details'][m]['progress'] for m in model_names]
            bars = ax3.barh(model_names, model_progress, color='skyblue')
            ax3.set_title('Individual Model Progress')
            ax3.set_xlabel('Progress (%)')
            ax3.set_xlim(0, 100)
            
            # Add progress labels
            for bar, progress in zip(bars, model_progress):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{progress:.1f}%', ha='left', va='center')
            
            # 4. Chunk Availability
            ax4 = axes[1, 0]
            chunk_data = report['chunks']
            available = chunk_data['available_chunks']
            total = chunk_data['total_chunks']
            remaining = total - available
            
            chunk_pie_data = [available, remaining]
            chunk_colors = ['#4169E1', '#D3D3D3']
            ax4.pie(chunk_pie_data, labels=['Available', 'Pending'], colors=chunk_colors, autopct='%1.0f')
            ax4.set_title(f'Data Chunks ({available}/{total})')
            
            # 5. Resource Usage
            ax5 = axes[1, 1]
            if report['resources']:
                resources = ['CPU', 'Memory', 'Disk']
                usage = [
                    report['resources'].get('cpu_percent', 0),
                    report['resources']['memory']['percent'],
                    report['resources']['disk']['percent']
                ]
                colors = ['red' if u > 80 else 'orange' if u > 60 else 'green' for u in usage]
                bars = ax5.bar(resources, usage, color=colors)
                ax5.set_title('System Resource Usage')
                ax5.set_ylabel('Usage (%)')
                ax5.set_ylim(0, 100)
                
                # Add usage labels
                for bar, usage_val in zip(bars, usage):
                    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{usage_val:.1f}%', ha='center', va='bottom')
            
            # 6. Training Timeline
            ax6 = axes[1, 2]
            # Show recent performance metrics if available
            if self.performance_metrics:
                model_type = list(self.performance_metrics.keys())[0]
                recent_metrics = self.performance_metrics[model_type][-10:]  # Last 10 points
                
                if recent_metrics:
                    timestamps = [datetime.fromisoformat(m['timestamp']) for m in recent_metrics]
                    accuracies = [m['metrics'].get('accuracy', 0) * 100 for m in recent_metrics]
                    
                    ax6.plot(timestamps, accuracies, marker='o', linewidth=2, markersize=4)
                    ax6.set_title(f'Recent Accuracy Trend ({model_type.upper()})')
                    ax6.set_ylabel('Accuracy (%)')
                    ax6.tick_params(axis='x', rotation=45)
                    ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / 'training_dashboard.png'
            plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dashboard saved: {dashboard_file}")
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
    
    def _check_integration_readiness(self):
        """Check if all models are ready for integration"""
        try:
            final_models = {}
            
            for model_type in self.model_types:
                final_model_path = self.output_dir / f"{model_type}_model_final.pt"
                
                if final_model_path.exists():
                    # Load and validate model
                    try:
                        model_data = torch.load(final_model_path, map_location='cpu')
                        final_models[model_type] = {
                            'path': str(final_model_path),
                            'size_mb': round(final_model_path.stat().st_size / (1024**2), 2),
                            'chunks_processed': model_data.get('total_chunks_processed', 0),
                            'final_metrics': model_data.get('final_metrics', {}),
                            'model_type': model_data.get('model_type', model_type)
                        }
                    except Exception as e:
                        logger.error(f"Error validating {model_type} model: {e}")
            
            self.final_models = final_models
            self.integration_ready = len(final_models) == len(self.model_types)
            
            if self.integration_ready:
                logger.info("🎉 All models completed! Ready for integration.")
                self._prepare_integration()
            else:
                missing = set(self.model_types) - set(final_models.keys())
                logger.info(f"Integration status: {len(final_models)}/{len(self.model_types)} models ready. Missing: {missing}")
            
            return self.integration_ready
            
        except Exception as e:
            logger.error(f"Error checking integration readiness: {e}")
            return False
    
    def _prepare_integration(self):
        """Prepare final integration package"""
        try:
            integration_dir = self.output_dir / 'integration_package'
            integration_dir.mkdir(exist_ok=True)
            
            # Copy all final models
            for model_type, model_info in self.final_models.items():
                src_path = Path(model_info['path'])
                dst_path = integration_dir / f"{model_type}_model_student.pt"
                
                # Copy model file
                import shutil
                shutil.copy2(src_path, dst_path)
                
                logger.info(f"Copied {model_type} model to integration package")
            
            # Create integration metadata
            integration_metadata = {
                'created_at': datetime.now().isoformat(),
                'total_models': len(self.final_models),
                'models': self.final_models,
                'training_summary': self._generate_training_summary(),
                'system_info': {
                    'python_version': sys.version,
                    'torch_version': torch.__version__,
                    'platform': os.name
                }
            }
            
            metadata_file = integration_dir / 'integration_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(integration_metadata, f, indent=2)
            
            # Create model loader script
            self._create_model_loader(integration_dir)
            
            # Create README
            self._create_integration_readme(integration_dir)
            
            logger.info(f"Integration package prepared: {integration_dir}")
            
            return integration_dir
            
        except Exception as e:
            logger.error(f"Error preparing integration: {e}")
            return None
    
    def _generate_training_summary(self):
        """Generate comprehensive training summary"""
        summary = {
            'total_training_time': 'N/A',
            'total_chunks_processed': 10,  # Only first 10 chunks (00-09)
            'models_trained': len(self.final_models),
            'average_metrics': {},
            'best_performing_model': None,
            'training_efficiency': {}
        }
        
        try:
            # Calculate average metrics across all models
            all_accuracies = []
            all_f1_scores = []
            best_accuracy = 0
            best_model = None
            
            for model_type, model_info in self.final_models.items():
                metrics = model_info.get('final_metrics', {})
                
                if 'accuracy' in metrics:
                    accuracy = metrics['accuracy']
                    all_accuracies.append(accuracy)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_type
                
                if 'f1' in metrics:
                    all_f1_scores.append(metrics['f1'])
            
            if all_accuracies:
                summary['average_metrics']['accuracy'] = np.mean(all_accuracies)
                summary['average_metrics']['accuracy_std'] = np.std(all_accuracies)
            
            if all_f1_scores:
                summary['average_metrics']['f1'] = np.mean(all_f1_scores)
                summary['average_metrics']['f1_std'] = np.std(all_f1_scores)
            
            if best_model:
                summary['best_performing_model'] = {
                    'model_type': best_model,
                    'accuracy': best_accuracy
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating training summary: {e}")
            return summary
    
    def _create_model_loader(self, integration_dir):
        """Create model loader script for integration"""
        loader_script = '''
"""
Interceptor Models Loader
Loads all 6 specialist models for the agentic system
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

class ModelLoader:
    """Loads and manages all Interceptor specialist models"""
    
    def __init__(self, models_dir='.'):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.model_types = ['bg', 'av', 'cm', 'rr', 'll', 'tm']
        
    def load_all_models(self, device='cpu'):
        """Load all specialist models"""
        for model_type in self.model_types:
            model_path = self.models_dir / f"{model_type}_model_student.pt"
            
            if model_path.exists():
                try:
                    model_data = torch.load(model_path, map_location=device)
                    self.models[model_type] = model_data
                    print(f"✅ Loaded {model_type.upper()} model")
                except Exception as e:
                    print(f"❌ Error loading {model_type.upper()} model: {e}")
            else:
                print(f"⚠️  {model_type.upper()} model not found: {model_path}")
        
        return self.models
    
    def get_model(self, model_type):
        """Get specific model"""
        return self.models.get(model_type)
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {}
        for model_type, model_data in self.models.items():
            info[model_type] = {
                'loaded': True,
                'final_metrics': model_data.get('final_metrics', {}),
                'chunks_processed': model_data.get('total_chunks_processed', 0)
            }
        return info

# Example usage
if __name__ == "__main__":
    loader = ModelLoader()
    models = loader.load_all_models()
    
    print(f"\\nLoaded {len(models)} models:")
    for model_type in models:
        print(f"- {model_type.upper()}")
    
    # Print model info
    info = loader.get_model_info()
    print("\\nModel Information:")
    print(json.dumps(info, indent=2))
'''
        
        loader_file = integration_dir / 'model_loader.py'
        with open(loader_file, 'w') as f:
            f.write(loader_script)
    
    def _create_integration_readme(self, integration_dir):
        """Create README for integration package"""
        readme_content = f'''# Interceptor Models Integration Package

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This package contains all 6 trained specialist models for the Interceptor deepfake detection system.

## Models Included
{chr(10).join([f"- **{model_type.upper()}-Model**: ~160MB" for model_type in self.final_models])}

## Model Descriptions
- **BG-Model**: Background/Baseline deepfake detection
- **AV-Model**: Audio-Visual synchronization analysis  
- **CM-Model**: Compression artifact detection
- **RR-Model**: Resolution/Reconstruction inconsistency detection
- **LL-Model**: Low-light condition analysis
- **TM-Model**: Temporal consistency analysis

## Usage

### Loading Models
```python
from model_loader import ModelLoader

# Initialize loader
loader = ModelLoader()

# Load all models
models = loader.load_all_models(device='cuda')  # or 'cpu'

# Get specific model
bg_model = loader.get_model('bg')
```

### Integration with Agentic System
These models are designed to work with the LangGraph-based agentic routing system. Each model provides specialized analysis that contributes to the final deepfake detection decision.

## Training Summary
- **Total Models**: {len(self.final_models)}
- **Data Processed**: ~100GB across 10 chunks (00-09)
- **Training Approach**: Incremental learning with periodic checkpointing
- **Architecture**: EfficientNet-B4 backbone with specialist modules

## Model Performance
{chr(10).join([f"- **{model_type.upper()}**: {self.final_models[model_type]['final_metrics'].get('accuracy', 0):.3f} accuracy" for model_type in self.final_models if 'final_metrics' in self.final_models[model_type]])}

## Files
- `*_model_student.pt`: Trained model weights
- `model_loader.py`: Python script to load models
- `integration_metadata.json`: Detailed metadata
- `README.md`: This file

## Next Steps
1. Copy these models to your deployment environment
2. Update your agentic system to use the new models
3. Test the integrated system with sample videos
4. Deploy to production

## Support
For questions about model integration or usage, refer to the main project documentation.
'''
        
        readme_file = integration_dir / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.monitoring_active = False
        logger.info("Training monitoring stopped")
    
    def get_status_summary(self):
        """Get current status summary"""
        try:
            status_file = self.output_dir / 'status_report.json'
            if status_file.exists():
                with open(status_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error getting status summary: {e}")
            return None

def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Interceptor training processes')
    parser.add_argument('--data_dir', default='/kaggle/input/dfdc-10', help='Data directory')
    parser.add_argument('--output_dir', default='/kaggle/working', help='Output directory')
    parser.add_argument('--mode', choices=['monitor', 'status', 'integrate'], default='monitor', 
                       help='Operation mode')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = TrainingMonitor(args.output_dir, args.data_dir)
    
    if args.mode == 'monitor':
        # Run continuous monitoring
        logger.info("Starting continuous monitoring...")
        try:
            while True:
                time.sleep(60)  # Keep main thread alive
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            monitor.stop_monitoring()
    
    elif args.mode == 'status':
        # Generate single status report
        report = monitor._generate_status_report()
        if report:
            print(json.dumps(report, indent=2))
    
    elif args.mode == 'integrate':
        # Check integration readiness and prepare package
        if monitor._check_integration_readiness():
            integration_dir = monitor._prepare_integration()
            print(f"Integration package ready: {integration_dir}")
        else:
            print("Not all models are ready for integration")

if __name__ == "__main__":
    main()
'''