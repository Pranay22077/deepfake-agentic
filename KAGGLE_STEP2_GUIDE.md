# 🚀 Kaggle Step 2 Training Guide

## Overview
This guide covers the complete Step 2 training pipeline on Kaggle, including teacher training, student distillation, and mobile optimization.

## 📋 Prerequisites

### 1. Kaggle Account Setup
- Kaggle account with phone verification
- GPU quota available (P100 or T4 recommended)
- Internet access enabled for package installation

### 2. Dataset Preparation
Upload your deepfake dataset to Kaggle with structure:
```
deepfake-faces-dataset/
├── train/
│   ├── real/
│   └── fake/
└── val/
    ├── real/
    └── fake/
```

### 3. Source Code Upload
Create a Kaggle dataset with your source code:
```
eraksha-source-code/
├── src/
│   ├── models/
│   ├── train/
│   ├── opt/
│   ├── eval/
│   └── preprocess/
├── export/
├── config/
└── requirements.txt
```

## 🎯 Training Pipeline

### Phase 1: Teacher Model Training (4-6 hours)

**Objective**: Train heavy multimodal teacher model for knowledge distillation

**Script**: `kaggle_train_teacher.py`

**Configuration**:
```python
config = {
    'epochs': 25,
    'batch_size': 8,      # Small batch for heavy model
    'lr': 1e-4,
    'num_frames': 8,
    'audio_duration': 3.0,
    'sample_rate': 16000,
    'save_predictions': True
}
```

**Expected Outputs**:
- `teacher_model.pt` (~60MB, 98%+ accuracy)
- `teacher_predictions/` (for distillation)
- `teacher_history.json` (training metrics)

**Key Features**:
- EfficientNet-B4 visual backbone
- Wav2Vec2 audio processing
- Transformer multimodal fusion
- Heavy augmentation pipeline
- Lip-sync auxiliary loss

### Phase 2: Student Distillation (2-3 hours)

**Objective**: Distill teacher knowledge to lightweight student model

**Script**: `kaggle_distill_student.py`

**Configuration**:
```python
config = {
    'epochs': 15,
    'batch_size': 32,     # Larger batch for light model
    'lr': 1e-4,
    'alpha': 0.5,         # Hard/soft loss balance
    'temperature': 3.0,   # Distillation temperature
}
```

**Expected Outputs**:
- `student_distilled.pt` (~4MB, 95%+ accuracy)
- `student_distillation_history.json`

**Key Features**:
- MobileNetV3-Small backbone
- Lightweight audio branch
- Knowledge distillation loss
- Soft label learning

### Phase 3: Model Optimization (1-2 hours)

**Objective**: Optimize student model for mobile deployment

**Script**: `kaggle_optimize_models.py`

**Optimization Steps**:
1. **Structured Pruning** (30% parameter reduction)
2. **Fine-tuning** (accuracy recovery)
3. **Dynamic Quantization** (INT8 conversion)
4. **TorchScript Export** (mobile format)
5. **Robustness Testing** (evaluation)

**Expected Outputs**:
- `student_pruned.pt` (~2.5MB, 94%+ accuracy)
- `student_quantized.pt` (~1.2MB, 93%+ accuracy)
- `student_mobile.ptl` (TorchScript for Android)
- `robustness_results.json`

## 🔧 Kaggle Notebook Setup

### 1. Create New Notebook
- GPU: P100 or T4
- Internet: ON
- Environment: Python 3.7+

### 2. Install Dependencies
```python
!pip install torch torchvision torchaudio
!pip install transformers
!pip install opencv-python
!pip install librosa
!pip install scikit-learn
!pip install matplotlib seaborn
```

### 3. Import Datasets
```python
# Add your datasets as input
# - deepfake-faces-dataset
# - eraksha-source-code
```

### 4. Run Complete Pipeline
```python
# Copy the complete workflow script
exec(open('/kaggle/input/eraksha-source-code/kaggle_step2_complete.py').read())
```

## 📊 Expected Performance Metrics

| Model | Size | Accuracy | Inference Time | Use Case |
|-------|------|----------|----------------|----------|
| Teacher | ~60MB | 98%+ | 500ms | Knowledge source |
| Student | ~4MB | 95%+ | 80ms | Base mobile model |
| Pruned | ~2.5MB | 94%+ | 60ms | Optimized mobile |
| Quantized | ~1.2MB | 93%+ | 40ms | Production mobile |

## 🚨 Common Issues & Solutions

### 1. GPU Memory Issues
```python
# Reduce batch size
config['batch_size'] = 4  # For teacher
config['batch_size'] = 16  # For student

# Enable gradient checkpointing
torch.utils.checkpoint.checkpoint_sequential()
```

### 2. Dataset Loading Errors
```python
# Check data paths
print(os.listdir('/kaggle/input/'))
print(os.listdir('/kaggle/input/deepfake-faces-dataset/'))

# Verify file structure
for root, dirs, files in os.walk('/kaggle/input/deepfake-faces-dataset/'):
    print(f"{root}: {len(files)} files")
```

### 3. Model Loading Issues
```python
# Check model compatibility
checkpoint = torch.load('model.pt', map_location='cpu')
print(checkpoint.keys())

# Handle different checkpoint formats
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
```

### 4. Audio Processing Errors
```python
# Install additional audio dependencies
!apt-get update
!apt-get install -y ffmpeg
!pip install ffmpeg-python

# Handle missing audio tracks
try:
    audio = extract_audio(video_path)
except:
    audio = torch.zeros(sample_rate * duration)  # Silence fallback
```

## 📥 Download Results

After training completion, download results using:

```python
# Download all models
!zip -r /kaggle/working/step2_models.zip /kaggle/working/models/
!zip -r /kaggle/working/step2_exports.zip /kaggle/working/export/

# Download specific files
files_to_download = [
    '/kaggle/working/models/teacher_model.pt',
    '/kaggle/working/models/student_distilled.pt',
    '/kaggle/working/models/optimized/student_quantized.pt',
    '/kaggle/working/export/student_mobile.ptl',
    '/kaggle/working/step2_completion_report.json'
]

for file in files_to_download:
    if os.path.exists(file):
        print(f"✅ {file} ready for download")
    else:
        print(f"❌ {file} not found")
```

## 🎯 Success Criteria

### Minimum Success (Ready for Step 3):
- ✅ Teacher model trained (95%+ accuracy)
- ✅ Student distilled (90%+ accuracy)
- ✅ At least one optimization step completed

### Full Success (Production Ready):
- ✅ All models trained successfully
- ✅ Complete optimization pipeline
- ✅ Mobile TorchScript export
- ✅ Robustness evaluation passed
- ✅ Model size <2MB
- ✅ Inference time <100ms

## 🚀 Next Steps After Kaggle Training

1. **Download Models**: Use `kaggle_download_results.py`
2. **Local Testing**: Test models with `src/eval/robustness_test.py`
3. **Android Integration**: Copy `student_mobile.ptl` to Android app
4. **APK Building**: Build and test Android application
5. **Performance Validation**: Verify on-device performance
6. **Production Deployment**: Package for distribution

## 📞 Troubleshooting

### If Training Fails:
1. Check GPU quota and availability
2. Verify dataset paths and structure
3. Review error logs in `/kaggle/working/logs/`
4. Reduce model complexity if memory issues
5. Run individual scripts instead of complete pipeline

### If Models Underperform:
1. Increase training epochs
2. Adjust learning rates
3. Modify augmentation strength
4. Check data quality and balance
5. Tune distillation parameters

### If Optimization Fails:
1. Skip problematic optimization steps
2. Use lighter pruning ratios
3. Try different quantization modes
4. Export without optimizations first
5. Test individual optimization scripts

---

**Estimated Total Time**: 7-11 hours on Kaggle GPU
**Estimated Cost**: $20-40 in Kaggle credits
**Success Rate**: 85%+ with proper setup