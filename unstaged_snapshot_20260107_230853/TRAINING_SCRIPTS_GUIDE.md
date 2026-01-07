# 🔧 INTERCEPTOR TRAINING SCRIPTS GUIDE

**Purpose**: Detailed guide for all training scripts and their usage  
**Target**: New account with no context  
**Last Updated**: January 6, 2026  

---

## 📋 SCRIPT OVERVIEW

### Current Training Status
**ACTIVE SCRIPT**: `resume_stage2_full_celebdf_training.py`  
**STATUS**: Running Epoch 1/10 on 6,011 videos  
**EXPECTED COMPLETION**: 6-8 hours  

---

## 🚀 MAIN TRAINING SCRIPTS

### 1. `multi_dataset_progressive_training.py` (ORIGINAL)
**Purpose**: Complete 4-dataset progressive training pipeline  
**Status**: Completed Stages 1-2, ready for Stages 3-4  

**Datasets Covered**:
- Stage 1: FaceForensics++ (7,000 videos) ✅ COMPLETED
- Stage 2: Celeb-DF v2 (518 videos) ✅ COMPLETED (but undertrained)
- Stage 3: Wild Deepfake (pending) 🔄 READY
- Stage 4: DFDC (10 chunks) 🔄 READY

**Key Features**:
- Automatic checkpoint management
- Progressive difficulty increase
- Stage-adaptive learning rates
- Mixed precision training
- Automatic downloads

**Usage**:
```python
# Update dataset paths in configuration section
DATASET_CONFIGS = {
    'faceforensics': {'path': '/kaggle/input/ff-c23'},
    'celebdf': {'path': '/kaggle/input/celeb-df-v2'},
    'wilddeepfake': {'path': '/kaggle/input/wild-deepfake'},
    'dfdc': {'path': '/kaggle/input/dfdc-10'}
}

# Run directly
python multi_dataset_progressive_training.py
```

**Outputs**:
- Stage checkpoints (compressed .zip files)
- Training metrics and logs
- Automatic download links

---

### 2. `resume_stage2_full_celebdf_training.py` (CURRENT)
**Purpose**: Fix Stage 2 undertrained issue by using FULL Celeb-DF dataset  
**Status**: CURRENTLY RUNNING  

**Problem Solved**: Original Stage 2 only used 518 test videos instead of full dataset  
**Solution**: Loads all 6,011 videos from all folders  

**Key Improvements**:
- 11.6x more training data (6,011 vs 518 videos)
- Proper folder structure loading
- Excludes test samples to avoid duplication
- Balanced sampling with class weights
- Focal loss for imbalanced data

**Dataset Loading**:
```
Celeb-real/     → 482 real videos
YouTube-real/   → 230 real videos  
Celeb-synthesis/ → 5,299 fake videos
Total: 6,011 training videos (after excluding 518 test samples)
```

**Usage**:
```bash
# 1. Upload your Stage 2 checkpoint to Kaggle dataset
# 2. Update paths in script:
CELEBDF_DATASET_PATH = '/kaggle/input/celeb-df-v2'
CHECKPOINT_PATH = '/kaggle/input/your-dataset/stage2_celebdf_complete_20260105_211721.pt'

# 3. Run script
python resume_stage2_full_celebdf_training.py
# Choose option 1 when prompted
```

**Expected Results**:
- Much better real detection (30-50% vs previous 5.84%)
- Overall accuracy improvement (60-70% vs 49.42%)
- Reduced bias (balanced real/fake detection)

---

## 🛠️ UTILITY SCRIPTS

### 3. `emergency_model_saver.py`
**Purpose**: Save model when Kaggle storage is full or downloads fail  
**Use Case**: Kaggle storage limitations, download issues  

**Features**:
- Creates lightweight model summaries
- Splits large models into downloadable chunks
- Generates reconstruction code
- Creates minimal working models

**Usage**:
```python
python emergency_model_saver.py
```

**Outputs**:
- `TRAINING_RESULTS.txt` - Performance summary
- `MODEL_ARCHITECTURE.txt` - Complete architecture
- `RECREATE_MODEL.py` - Model recreation code
- `model_chunk_*.bin` - Model in small pieces
- `RECONSTRUCT_MODEL.py` - Chunk reconstruction script

---

### 4. `organize_and_download_checkpoints.py`
**Purpose**: Organize and prepare all checkpoints for download  
**Use Case**: End of training, checkpoint management  

**Features**:
- Finds all checkpoint files
- Organizes by training stage
- Creates download summary
- Generates final model package

**Usage**:
```python
python organize_and_download_checkpoints.py
```

**Outputs**:
- `FINAL_DOWNLOADS/` folder with organized checkpoints
- `DOWNLOAD_SUMMARY.txt` with training results
- Clickable download links for each checkpoint

---

## 🔍 ANALYSIS SCRIPTS

### 5. `comprehensive_dataset_inspector.py`
**Purpose**: Analyze all 4 datasets before training  
**Use Case**: Understanding data distribution, debugging dataset issues  

**Features**:
- Analyzes FaceForensics++, Celeb-DF, Wild Deepfake, DFDC
- Shows real/fake distribution
- Identifies missing files or metadata issues
- Estimates training time and data size

**Usage**:
```python
# Update dataset paths
DATA_CONFIGS = {
    'faceforensics': '/kaggle/input/ff-c23',
    'celebdf': '/kaggle/input/celeb-df-v2',
    'wilddeepfake': '/kaggle/input/wild-deepfake',
    'dfdc': '/kaggle/input/dfdc-10'
}

python comprehensive_dataset_inspector.py
```

---

### 6. `debug_dfdc_metadata.py`
**Purpose**: Debug DFDC dataset metadata issues  
**Use Case**: DFDC chunk analysis, metadata validation  

**Features**:
- Analyzes all 10 DFDC chunks (00-09)
- Validates JSON metadata files
- Shows real/fake distribution per chunk
- Identifies problematic chunks

---

### 7. `check_overall_distribution.py`
**Purpose**: Quick overview of DFDC dataset distribution  
**Use Case**: Fast DFDC analysis  

---

## 📊 TRAINING CONFIGURATION GUIDE

### Learning Rate Schedule
```python
stage_configs = {
    1: {'lr': 1e-4, 'loss': 'ce'},        # FaceForensics: Standard
    2: {'lr': 5e-5, 'loss': 'focal'},     # Celeb-DF: Lower LR, focal loss
    3: {'lr': 3e-5, 'loss': 'focal'},     # Wild: Even lower
    4: {'lr': 1e-5, 'loss': 'weighted'}   # DFDC: Minimal LR, weighted loss
}
```

### Data Augmentation by Stage
```python
# Stage 1: Minimal augmentation
transforms = [RandomRotation(5), ColorJitter(0.1, 0.1)]

# Stage 2: Moderate augmentation  
transforms = [RandomRotation(10), ColorJitter(0.2, 0.2, 0.1), RandomAffine(translate=(0.05, 0.05))]

# Stage 3: Strong augmentation
transforms = [RandomRotation(15), ColorJitter(0.3, 0.3, 0.2, 0.1), RandomPerspective(0.1)]

# Stage 4: Maximum augmentation
transforms = [RandomRotation(20), ColorJitter(0.4, 0.4, 0.3, 0.1), GaussianBlur()]
```

### Checkpoint Management
```python
CHECKPOINT_MANAGEMENT = {
    'save_frequency_gb': 5.0,      # Save every 5GB processed
    'max_checkpoints': 3,          # Keep 3 latest checkpoints
    'auto_download': True,         # Auto-trigger downloads
    'compress_checkpoints': True   # Compress to save space
}
```

---

## 🚨 TROUBLESHOOTING GUIDE

### Common Issues & Solutions

#### 1. Checkpoint Loading Errors
**Error**: `weights_only=True` prevents loading  
**Solution**: Scripts now use `weights_only=False` for trusted checkpoints  

#### 2. Dataset Path Issues
**Error**: Dataset not found  
**Solution**: Update paths in configuration section of each script  

#### 3. Memory Issues
**Error**: CUDA out of memory  
**Solution**: Reduce batch size from 8 to 4 or 2  

#### 4. Storage Full
**Error**: Kaggle storage limit reached  
**Solution**: Run `emergency_model_saver.py` to create downloadable chunks  

#### 5. Download Failures
**Error**: Cannot download large checkpoints  
**Solution**: Use `organize_and_download_checkpoints.py` to create smaller files  

---

## 🎯 RECOMMENDED WORKFLOW

### For New Account Setup
1. **Upload Checkpoints**: Add model files to Kaggle dataset
2. **Add Datasets**: Ensure all 4 datasets are available
3. **Run Current Training**: Continue with `resume_stage2_full_celebdf_training.py`
4. **Monitor Progress**: Check training logs and metrics
5. **Save Checkpoints**: Regular saves every few epochs

### For Continuing Training
1. **Check Current Status**: See which stage is complete
2. **Load Latest Checkpoint**: Use best available model
3. **Run Next Stage**: Follow progressive training order
4. **Evaluate Results**: Compare metrics across stages
5. **Deploy Best Model**: Use highest-performing checkpoint

---

## 📈 EXPECTED TRAINING TIMELINE

### Stage 2 Enhanced (CURRENT)
- **Duration**: 6-8 hours
- **Data**: 6,011 videos
- **Expected**: Major improvement in real detection

### Stage 3: Wild Deepfake
- **Duration**: 3-4 hours  
- **Data**: ~2,000 videos (estimated)
- **Expected**: Real-world robustness

### Stage 4: DFDC
- **Duration**: 12-16 hours
- **Data**: ~20,000 videos across 10 chunks
- **Expected**: Large-scale generalization

### Total Remaining Time: 20-28 hours

---

## 💾 CHECKPOINT NAMING CONVENTION

```
stage{N}_{dataset}_{status}_{timestamp}.{ext}

Examples:
- stage1_faceforensics_complete_20260105_210858.zip
- stage2_celebdf_epoch1_20260105_211125.zip  
- stage2_celebdf_complete_20260105_211721.pt
- stage3_wilddeepfake_best_20260106_120000.pt
```

---

## 🔄 SCRIPT EXECUTION ORDER

### Current Recommended Sequence
1. ✅ **COMPLETED**: `multi_dataset_progressive_training.py` (Stages 1-2)
2. 🔄 **RUNNING**: `resume_stage2_full_celebdf_training.py` (Stage 2 Enhanced)
3. 🔜 **NEXT**: Continue with Stage 3 in `multi_dataset_progressive_training.py`
4. 🔜 **THEN**: Complete Stage 4 in `multi_dataset_progressive_training.py`
5. 🔜 **FINALLY**: `organize_and_download_checkpoints.py` for final model

---

**🎯 IMMEDIATE ACTION**: Let the current `resume_stage2_full_celebdf_training.py` complete its 10 epochs. This will provide a much better foundation for the remaining stages and significantly improve overall model performance.