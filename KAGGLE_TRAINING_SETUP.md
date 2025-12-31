# Kaggle Training Setup Instructions

## Overview
Complete setup guide for training all 6 Interceptor models on Kaggle with ~100GB dataset (first 10 chunks only).

## Prerequisites
1. Kaggle account with phone verification
2. Dataset uploaded to: `www.kaggle.com/datasets/pranay22077/dfdc-10`
3. 4 separate Kaggle notebooks (one per person)

## Dataset Structure
```
dfdc-10/
├── 00.zip (~10 GB)
├── 01.zip (~10 GB)
├── 02.zip (~10 GB)
├── ...
└── 09.zip (~10 GB)
```
**Note: We only use first 10 chunks (00-09) for training, not all 25 chunks.**

## Person 1: BG-Model + AV-Model Training

### Kaggle Notebook Setup
1. Create new notebook: "Interceptor BG-AV Training"
2. Add dataset: `pranay22077/dfdc-10`
3. Enable GPU: T4 x2 or P100
4. Set language: Python

### Code Setup
```python
# Install dependencies
!pip install librosa scikit-learn psutil

# Upload training script
# Copy person1_bg_av_training.py content to cell

# Start BG model training
!python person1_bg_av_training.py --model bg --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

### Run AV Model (After BG Completes)
```python
# Start AV model training
!python person1_bg_av_training.py --model av --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

## Person 2: CM-Model + RR-Model Training

### Kaggle Notebook Setup
1. Create new notebook: "Interceptor CM-RR Training"
2. Add dataset: `pranay22077/dfdc-10`
3. Enable GPU: T4 x2 or P100
4. Set language: Python

### Code Setup
```python
# Install dependencies
!pip install scikit-learn psutil scipy scikit-image

# Upload training script
# Copy person2_cm_rr_training.py content to cell

# Start CM model training
!python person2_cm_rr_training.py --model cm --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

### Run RR Model (After CM Completes)
```python
# Start RR model training
!python person2_cm_rr_training.py --model rr --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

## Person 3: LL-Model + TM-Model Training

### Kaggle Notebook Setup
1. Create new notebook: "Interceptor LL-TM Training"
2. Add dataset: `pranay22077/dfdc-10`
3. Enable GPU: T4 x2 or P100
4. Set language: Python

### Code Setup
```python
# Install dependencies
!pip install scikit-learn psutil scipy scikit-image

# Upload training script
# Copy person3_ll_tm_training.py content to cell

# Start LL model training
!python person3_ll_tm_training.py --model ll --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

### Run TM Model (After LL Completes)
```python
# Start TM model training
!python person3_ll_tm_training.py --model tm --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

## Person 4: Monitoring & Integration

### Kaggle Notebook Setup
1. Create new notebook: "Interceptor Training Monitor"
2. Add dataset: `pranay22077/dfdc-10`
3. Enable CPU (sufficient for monitoring)
4. Set language: Python

### Code Setup
```python
# Install dependencies
!pip install matplotlib seaborn psutil

# Upload monitoring script
# Copy person4_monitor_integration.py content to cell

# Start monitoring
!python person4_monitor_integration.py --mode monitor --data_dir /kaggle/input/dfdc-10 --output_dir /kaggle/working
```

### Check Status
```python
# Get current status
!python person4_monitor_integration.py --mode status
```

### Integration (When All Models Complete)
```python
# Prepare integration package
!python person4_monitor_integration.py --mode integrate
```

## Training Schedule

### Phase 1: Start All Models (Day 1)
- Person 1: Start BG model
- Person 2: Start CM model  
- Person 3: Start LL model
- Person 4: Start monitoring

### Phase 2: Second Models (Day 3-4)
- Person 1: Start AV model (after BG completes)
- Person 2: Start RR model (after CM completes)
- Person 3: Start TM model (after LL completes)

### Phase 3: Integration (Day 6-7)
- Person 4: Prepare integration package
- All: Download final models

## Key Features

### Automatic Checkpointing
- Saves every 3 hours automatically
- Keeps only latest 2 checkpoints per model
- Resumes from latest checkpoint on restart

### Incremental Data Loading
- Processes chunks sequentially (00.zip → 01.zip → ... → 09.zip)
- Waits for chunks to be uploaded
- No bias towards fake data

### Resource Management
- Memory cleanup after each batch
- GPU cache clearing
- Storage optimization (max 20GB usage)

### Monitoring Dashboard
- Real-time progress tracking
- Resource usage monitoring
- Visual performance charts
- Integration readiness status

## Expected Timeline
- **BG Model**: ~12-15 hours (baseline, fastest)
- **AV Model**: ~15-18 hours (audio processing overhead)
- **CM Model**: ~14-16 hours (compression analysis)
- **RR Model**: ~13-15 hours (resolution analysis)
- **LL Model**: ~16-19 hours (low-light preprocessing)
- **TM Model**: ~18-21 hours (temporal analysis, most complex)

**Total Training Time**: ~2-3 days (with overlapping)

## Storage Management
- Each model checkpoint: ~160MB
- 2 checkpoints per model: ~320MB per model
- Total storage usage: ~2GB max
- Final models: ~1GB total

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in training scripts
# Change batch_size=16 to batch_size=8 or batch_size=4
```

### Slow Training
```python
# Check GPU utilization
!nvidia-smi

# Reduce num_workers if CPU bottleneck
# Change num_workers=4 to num_workers=2
```

### Chunk Not Found
```python
# Check dataset connection
import os
print(os.listdir('/kaggle/input/dfdc-10'))

# Wait for chunk upload (automatic in scripts)
```

### Session Timeout
- Kaggle sessions timeout after 12 hours
- Scripts automatically save checkpoints
- Restart notebook and re-run training command
- Training will resume from latest checkpoint

## Final Integration
When all models complete:
1. Person 4 runs integration mode
2. Download `integration_package.zip` from Kaggle
3. Upload models to HuggingFace: `Pran-ay-22077/interceptor-models`
4. Update production system with new models

## Quality Assurance
- Each model trained on full 471GB dataset
- No data bias (balanced real/fake sampling)
- Cross-validation on held-out chunks
- Performance monitoring throughout training
- Comprehensive metrics tracking

## Success Criteria
- All 6 models complete training on first 10 chunks
- Accuracy > 85% on validation data
- Models integrate successfully with agentic system
- Production deployment ready