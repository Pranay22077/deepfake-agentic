# Specialist Models Training Guide

## Overview
This guide covers the training scripts for all 5 specialist models in the Interceptor deepfake detection system.

## Models and Their Specializations

### 1. LL (Low-Light) Model ✅ COMPLETE
- **Specialty**: Low-light and luminance inconsistencies
- **Scripts**: `multi_dataset_progressive_training.py` (all stages combined)
- **Status**: Training complete

### 2. BG (Background/Lighting) Model 🔄 IN PROGRESS
- **Specialty**: Background texture and lighting direction inconsistencies
- **Scripts**:
  - `train_bg_stage1_faceforensics.py` ✅
  - `train_bg_stage2_celebdf.py` ✅
  - `train_bg_stage4_dfdc.py` 🔄 (partial)

### 3. AV (Audio-Visual) Model ⏳ PENDING
- **Specialty**: Audio-visual synchronization and voice inconsistencies
- **Scripts** (to create):
  - `train_av_stage1_faceforensics.py`
  - `train_av_stage2_celebdf.py`
  - `train_av_stage4_dfdc.py`

### 4. CM (Compression) Model ⏳ PENDING
- **Specialty**: Compression artifacts and encoding inconsistencies
- **Scripts** (to create):
  - `train_cm_stage1_faceforensics.py`
  - `train_cm_stage2_celebdf.py`
  - `train_cm_stage4_dfdc.py`

### 5. RR (Resolution) Model ⏳ PENDING
- **Specialty**: Resolution mismatches and upscaling artifacts
- **Scripts** (to create):
  - `train_rr_stage1_faceforensics.py`
  - `train_rr_stage2_celebdf.py`
  - `train_rr_stage4_dfdc.py`

### 6. TM (Temporal) Model ⏳ PENDING
- **Specialty**: Temporal inconsistencies across frames
- **Scripts** (to create):
  - `train_tm_stage1_faceforensics.py`
  - `train_tm_stage2_celebdf.py`
  - `train_tm_stage4_dfdc.py`

## Training Pipeline

### Stage 1: FaceForensics++ (Foundation)
- **Dataset**: FaceForensics++ C23
- **Purpose**: Foundation training on clean, balanced data
- **Epochs**: 6
- **Learning Rate**: 1e-4
- **Loss**: CrossEntropyLoss

### Stage 2: Celeb-DF (Realism Adaptation)
- **Dataset**: Celeb-DF v2
- **Purpose**: Adapt to high-quality realistic deepfakes
- **Epochs**: 5
- **Learning Rate**: 5e-5
- **Loss**: Focal Loss
- **Input**: Stage 1 checkpoint

### Stage 4: DFDC (Large-Scale Training)
- **Dataset**: DFDC (10 chunks, ~100GB)
- **Purpose**: Large-scale diversity and robustness
- **Epochs**: 2 per chunk
- **Learning Rate**: 1e-5
- **Loss**: Weighted CrossEntropyLoss
- **Input**: Stage 2 checkpoint
- **Chunk Order**: [9, 8, 3, 5, 7, 2, 6, 4, 1, 0] (most to least balanced)

## Common Architecture Components

All specialist models share:
1. **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
2. **Specialist Module**: Custom CNN layers for specific artifact detection
3. **Feature Fusion**: Multi-head attention mechanism
4. **Classifier**: Deep MLP with dropout and batch normalization
5. **Checkpoint Management**: Automatic saving, compression, and cleanup
6. **Mixed Precision Training**: FP16 for faster training
7. **Balanced Sampling**: WeightedRandomSampler for class balance

## Specialist Module Architectures

### BG Module (Background/Lighting)
- Background texture analyzer (7x7 convolutions)
- Lighting direction detector (5x5 convolutions)
- Shadow consistency checker (9x9 convolutions)
- Color temperature analyzer (3x3 convolutions)
- Output: 44 channels × 7×7 = 2156 features

### AV Module (Audio-Visual)
- Lip-sync analyzer
- Voice frequency detector
- Audio-visual correlation checker
- Temporal audio consistency
- Output: 48 channels × 7×7 = 2352 features

### CM Module (Compression)
- DCT coefficient analyzer
- Quantization artifact detector
- Block boundary checker
- Compression level estimator
- Output: 40 channels × 7×7 = 1960 features

### RR Module (Resolution)
- Multi-scale resolution analyzer
- Upscaling artifact detector
- Edge sharpness checker
- Pixel interpolation detector
- Output: 36 channels × 7×7 = 1764 features

### TM Module (Temporal)
- Frame-to-frame consistency checker
- Motion flow analyzer
- Temporal artifact detector
- Optical flow validator
- Output: 52 channels × 7×7 = 2548 features

## Usage Instructions

### For Each Model:

1. **Stage 1 Training**:
   ```bash
   python train_<model>_stage1_faceforensics.py
   ```
   - Downloads checkpoint automatically
   - Saves to `/kaggle/working/checkpoints/`

2. **Stage 2 Training**:
   ```bash
   # Update STAGE1_CHECKPOINT path in script
   python train_<model>_stage2_celebdf.py
   ```

3. **Stage 4 Training**:
   ```bash
   # Update STAGE2_CHECKPOINT path in script
   python train_<model>_stage4_dfdc.py
   ```

## Integration with Agent Framework

After training all models, the checkpoints will be integrated into the agent framework:
- `ll_model_student.pt` (Low-Light)
- `bg_model_student.pt` (Background)
- `av_model_student.pt` (Audio-Visual)
- `cm_model_student.pt` (Compression)
- `rr_model_student.pt` (Resolution)
- `tm_model_student.pt` (Temporal)

The agent will load all 6 models and combine their predictions using weighted voting.

## Next Steps

1. Complete BG Stage 4 script
2. Create all AV model scripts (3 stages)
3. Create all CM model scripts (3 stages)
4. Create all RR model scripts (3 stages)
5. Create all TM model scripts (3 stages)
6. Test each script on Kaggle
7. Train all models sequentially
8. Integrate trained models into agent framework

## Notes

- All scripts follow the same architecture as `multi_dataset_progressive_training.py`
- Each script is ~1000-1500 lines (comprehensive but focused)
- Checkpoint management ensures efficient storage usage
- Mixed precision training speeds up training by ~2x
- Balanced sampling prevents bias toward majority class
