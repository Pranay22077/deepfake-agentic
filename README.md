# E-Raksha Deepfake Detection Agent

Agentic AI system for real-time deepfake detection on edge devices.

## Project Structure
- `data/` - Raw downloaded datasets (not committed)
- `data_processed/` - Face crops and spectrograms
- `src/` - Source code modules
- `export/` - Exported models (TorchScript/ONNX)
- `website/` - Static site for APK hosting
- `docker/` - Docker configurations

## Status: Step 1 Complete
- [x] Project setup
- [x] Environment & dependencies (Python 3.11 + PyTorch)
- [x] Dataset preparation (structure ready)
- [x] Preprocessing pipeline (face extraction + audio spectrograms)
- [x] Baseline model (MobileNetV3 student model)
- [x] Local inference demo (with GradCAM explainability)
- [x] Model export (TorchScript + ONNX)
- [x] Website stub (responsive design with download links)

## What's Ready for Kaggle Training
- `src/models/student.py` - Lightweight MobileNetV3 model (1M params)
- `src/train/train_baseline.py` - Training script ready for datasets
- `src/preprocess/` - Face extraction and audio processing pipelines
- `export/export_models.py` - Model export for mobile deployment

## Next Steps (Step 2)
When you have trained model weights from Kaggle:
1. Place weights in `models/baseline_student.pt`
2. Run inference demo: `python src/inference/run_demo.py --video sample.mp4`
3. Export optimized models: `python export/export_models.py`
4. Build Android APK with embedded models

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
