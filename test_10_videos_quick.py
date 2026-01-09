"""
E-Raksha Quick 10-Video Test Suite

Rapid validation test using random video samples across all specialist models.
Provides fast performance overview and model comparison metrics.

Author: E-Raksha Team
"""

import sys
import os
import random
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.specialists_new import (
    BGSpecialistModel, AVSpecialistModel, CMSpecialistModel,
    RRSpecialistModel, LLSpecialistModel, TMModelOld
)
import torch.nn.functional as F
import cv2
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")

# Initialize model loading
print("\n📦 Loading specialist models...")

models = {}

# BG Model
print("Loading BG model...")
models['bg'] = BGSpecialistModel()
checkpoint = torch.load('models/baseline_student.pt', map_location=DEVICE, weights_only=False)
models['bg'].load_state_dict(checkpoint['model_state_dict'], strict=True)
models['bg'].to(DEVICE).eval()
print("  ✅ BG Model loaded")

# AV Model
print("Loading AV model...")
models['av'] = AVSpecialistModel()
checkpoint = torch.load('models/av_model_student.pt', map_location=DEVICE, weights_only=False)
models['av'].load_state_dict(checkpoint['model_state_dict'], strict=True)
models['av'].to(DEVICE).eval()
print("  ✅ AV Model loaded")

# CM Model
print("Loading CM model...")
models['cm'] = CMSpecialistModel()
checkpoint = torch.load('models/cm_model_student.pt', map_location=DEVICE, weights_only=False)
models['cm'].load_state_dict(checkpoint['model_state_dict'], strict=True)
models['cm'].to(DEVICE).eval()
print("  ✅ CM Model loaded")

# RR Model
print("Loading RR model...")
models['rr'] = RRSpecialistModel()
checkpoint = torch.load('models/rr_model_student.pt', map_location=DEVICE, weights_only=False)
models['rr'].load_state_dict(checkpoint['model_state_dict'], strict=True)
models['rr'].to(DEVICE).eval()
print("  ✅ RR Model loaded")

# LL Model
print("Loading LL model...")
models['ll'] = LLSpecialistModel()
checkpoint = torch.load('models/ll_model_student.pt', map_location=DEVICE, weights_only=False)
models['ll'].load_state_dict(checkpoint['model_state_dict'], strict=True)
models['ll'].to(DEVICE).eval()
print("  ✅ LL Model loaded")

# TM Model
print("Loading TM model...")
models['tm'] = TMModelOld()
checkpoint = torch.load('models/tm_model_student.pt', map_location=DEVICE, weights_only=False)
if 'model_state_dict' in checkpoint:
    models['tm'].load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    models['tm'].load_state_dict(checkpoint, strict=False)
models['tm'].to(DEVICE).eval()
print("  ✅ TM Model loaded")

print("\n✅ All 6 models loaded!")


def extract_frames(video_path, num_frames=8):
    """Extract frames from video"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < 4:
        return None
    
    return torch.stack(frames)


def predict_with_model(model, frames, model_name):
    """Run prediction with a single model"""
    try:
        with torch.no_grad():
            if model_name == 'tm':
                # TM expects [B, T, C, H, W]
                input_tensor = frames.unsqueeze(0).to(DEVICE)
            else:
                # Other models expect [B, C, H, W] - use first 4 frames
                input_tensor = frames[:4].to(DEVICE)
            
            logits = model(input_tensor)
            
            if model_name == 'tm':
                probs = F.softmax(logits, dim=1)
                fake_prob = probs[0, 1].item()
            else:
                probs = F.softmax(logits, dim=1)
                fake_prob = probs[:, 1].mean().item()
            
            return fake_prob
    except Exception as e:
        print(f"    Error with {model_name}: {e}")
        return 0.5


def ensemble_predict(all_predictions):
    """Ensemble prediction with equal weights"""
    # Equal weights for all models
    weights = {
        'bg': 1.0,
        'av': 1.0,
        'cm': 1.0,
        'rr': 1.0,
        'll': 1.0,
        'tm': 1.0
    }
    
    weighted_sum = 0
    total_weight = 0
    
    for model_name, pred in all_predictions.items():
        w = weights[model_name]
        weighted_sum += pred * w
        total_weight += w
    
    return weighted_sum / total_weight


# Get test videos
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
REAL_VIDEOS = list((TEST_DATA_ROOT / "real").glob("*.mp4"))
FAKE_VIDEOS = list((TEST_DATA_ROOT / "fake").glob("*.mp4"))

# Select 5 real and 5 fake randomly
random.seed(42)
test_real = random.sample(REAL_VIDEOS, min(5, len(REAL_VIDEOS)))
test_fake = random.sample(FAKE_VIDEOS, min(5, len(FAKE_VIDEOS)))

print(f"\n📹 Testing on {len(test_real)} real + {len(test_fake)} fake videos")
print("="*80)

results = []

for video_path in test_real + test_fake:
    is_real = "real" in str(video_path)
    true_label = "REAL" if is_real else "FAKE"
    
    print(f"\n🎬 {video_path.name} (TRUE: {true_label})")
    
    frames = extract_frames(video_path)
    if frames is None:
        print("  ❌ Could not extract frames")
        continue
    
    # Get predictions from all models
    predictions = {}
    for model_name, model in models.items():
        pred = predict_with_model(model, frames, model_name)
        predictions[model_name] = pred
        pred_label = "FAKE" if pred > 0.5 else "REAL"
        conf = max(pred, 1-pred)
        print(f"  {model_name.upper()}: {pred:.3f} ({pred_label}, {conf:.1%})")
    
    # Ensemble prediction
    ensemble_pred = ensemble_predict(predictions)
    ensemble_label = "FAKE" if ensemble_pred > 0.5 else "REAL"
    ensemble_conf = max(ensemble_pred, 1-ensemble_pred)
    
    correct = ensemble_label == true_label
    status = "✅" if correct else "❌"
    
    print(f"  📊 ENSEMBLE: {ensemble_pred:.3f} ({ensemble_label}, {ensemble_conf:.1%}) {status}")
    
    results.append({
        'video': video_path.name,
        'true_label': true_label,
        'predictions': predictions,
        'ensemble': ensemble_pred,
        'predicted': ensemble_label,
        'correct': correct
    })

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

correct_count = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = correct_count / total if total > 0 else 0

print(f"Overall Accuracy: {correct_count}/{total} ({accuracy:.1%})")

# Per-model accuracy
print("\n📈 Per-Model Performance:")
for model_name in models.keys():
    model_correct = 0
    for r in results:
        pred = r['predictions'][model_name]
        pred_label = "FAKE" if pred > 0.5 else "REAL"
        if pred_label == r['true_label']:
            model_correct += 1
    model_acc = model_correct / total if total > 0 else 0
    print(f"  {model_name.upper()}: {model_correct}/{total} ({model_acc:.1%})")

# Real vs Fake breakdown
real_results = [r for r in results if r['true_label'] == 'REAL']
fake_results = [r for r in results if r['true_label'] == 'FAKE']

real_correct = sum(1 for r in real_results if r['correct'])
fake_correct = sum(1 for r in fake_results if r['correct'])

print(f"\n📈 Real Detection: {real_correct}/{len(real_results)} ({real_correct/len(real_results)*100:.1f}%)")
print(f"📈 Fake Detection: {fake_correct}/{len(fake_results)} ({fake_correct/len(fake_results)*100:.1f}%)")
