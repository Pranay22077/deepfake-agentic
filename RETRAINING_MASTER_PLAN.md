# Interceptor Models Retraining Master Plan

## Overview
- **Total Data**: ~100GB split into 10 chunks (00.zip - 09.zip, ~10GB each)
- **Models**: 6 specialist models (BG, AV, CM, RR, LL, TM)
- **Team**: 4 persons with distributed workload
- **Strategy**: Incremental training with periodic checkpointing

## Data Distribution Strategy
- Process chunks sequentially: 00.zip → 01.zip → ... → 09.zip
- Each chunk contains balanced real/fake samples
- No bias towards fake data (use full available data)
- Automatic checkpoint saving every 3 hours
- Storage management: Keep only latest 2 checkpoints per model

## Team Assignment

### Person 1: BG-Model (Background/Baseline) + AV-Model (Audio-Visual)
- **BG-Model**: General deepfake detection baseline
- **AV-Model**: Audio-visual synchronization analysis
- **Files**: `person1_bg_av_training.py`

### Person 2: CM-Model (Compression) + RR-Model (Resolution/Reconstruction)  
- **CM-Model**: Compression artifact detection
- **RR-Model**: Resolution inconsistency detection
- **Files**: `person2_cm_rr_training.py`

### Person 3: LL-Model (Low-Light) + TM-Model (Temporal)
- **LL-Model**: Low-light condition analysis
- **TM-Model**: Temporal consistency analysis
- **Files**: `person3_ll_tm_training.py`

### Person 4: Integration & Monitoring
- **Role**: Monitor all training processes, handle integration
- **Files**: `person4_monitor_integration.py`

## Architecture Specifications
- **Base**: EfficientNet-B4 backbone for all models
- **Input**: 224x224 RGB frames
- **Output**: Binary classification (real/fake) with confidence
- **Training**: Progressive learning with curriculum scheduling
- **Optimization**: AdamW with cosine annealing

## Checkpoint Strategy
- Save every 3 hours (automatic)
- Keep only latest 2 checkpoints per model
- Model size: ~160MB per checkpoint
- Total storage usage: ~2GB max (6 models × 2 checkpoints × 160MB)

## Quality Assurance
- Balanced sampling from each chunk
- Cross-validation on held-out data
- Bias detection and correction
- Performance monitoring across all models