"""
QUICK TEST: 10 videos with all HF models
"""

import sys
import os
from pathlib import Path
import random

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import cv2
import numpy as np

from models.specialists_new import (
    BGSpecialistModel, AVSpecialistModel, CMSpecialistModel,
    RRSpecialistModel, LLSpecialistModel, TMModelOld
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")


def load_models():
    """Load all models"""
    models = {}
    
    configs = {
        'bg': ('models/baseline_student.pt', BGSpecialistModel),
        'av': ('models/av_model_student.pt', AVSpecialistModel),
        'cm': ('models/cm_model_student.pt', CMSpecialistModel),
        'rr': ('models/rr_model_student.pt', RRSpecialistModel),
        'll': ('models/ll_model_student.pt', LLSpecialistModel),
        'tm': ('models/tm_model_student.pt', TMModelOld),
    }
    
    for name, (path, model_class) in configs.items():
        try:
            model = model_class()
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE).eval()
            models[name] = model
            print(f"✅ {name.upper()} loaded")
        except Exception as e:
            print(f"❌ {name.upper()} failed: {e}")
    
    return models


def extract_frames(video_path, num_frames=8):
    """Extract frames"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
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
    
    return torch.stack(frames) if len(frames) >= 4 else None


def predict(model, frames, model_name):
    """Get prediction"""
    with torch.no_grad():
        if model_name == 'tm':
            inp = frames.unsqueeze(0).to(DEVICE)
        else:
            inp = frames[:4].to(DEVICE)
        
        logits = model(inp)
        probs = F.softmax(logits, dim=1)
        
        if model_name == 'tm':
            return probs[0, 0].item(), probs[0, 1].item()  # real_prob, fake_prob
        else:
            return probs[:, 0].mean().item(), probs[:, 1].mean().item()


def main():
    print("\n" + "="*80)
    print("QUICK TEST: 10 VIDEOS")
    print("="*80)
    
    models = load_models()
    
    # Get 5 real + 5 fake
    real_videos = list((TEST_DATA_ROOT / "real").glob("*.mp4"))
    fake_videos = list((TEST_DATA_ROOT / "fake").glob("*.mp4"))
    
    random.seed(42)
    test_real = random.sample(real_videos, min(5, len(real_videos)))
    test_fake = random.sample(fake_videos, min(5, len(fake_videos)))
    
    print(f"\nTesting {len(test_real)} real + {len(test_fake)} fake videos\n")
    
    # Track results
    results = {name: {'correct': 0, 'total': 0, 'real_correct': 0, 'fake_correct': 0} 
               for name in models.keys()}
    
    for video_path in test_real + test_fake:
        is_real = "real" in str(video_path)
        true_label = "REAL" if is_real else "FAKE"
        
        print(f"\n{'='*60}")
        print(f"📹 {video_path.name} (TRUE: {true_label})")
        print(f"{'='*60}")
        
        frames = extract_frames(video_path)
        if frames is None:
            print("  ❌ Could not extract frames")
            continue
        
        for name, model in models.items():
            try:
                real_prob, fake_prob = predict(model, frames, name)
                pred_label = "FAKE" if fake_prob > 0.5 else "REAL"
                correct = pred_label == true_label
                
                results[name]['total'] += 1
                if correct:
                    results[name]['correct'] += 1
                    if is_real:
                        results[name]['real_correct'] += 1
                    else:
                        results[name]['fake_correct'] += 1
                
                status = "✅" if correct else "❌"
                print(f"  {name.upper():4}: P(real)={real_prob:.3f} P(fake)={fake_prob:.3f} -> {pred_label} {status}")
                
            except Exception as e:
                print(f"  {name.upper():4}: ERROR - {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<6} {'Accuracy':<12} {'Real Acc':<12} {'Fake Acc':<12}")
    print("-"*50)
    
    for name, r in results.items():
        if r['total'] > 0:
            acc = r['correct'] / r['total']
            real_acc = r['real_correct'] / 5 if 5 > 0 else 0
            fake_acc = (r['correct'] - r['real_correct']) / 5 if 5 > 0 else 0
            print(f"{name.upper():<6} {acc:.1%}         {real_acc:.1%}         {fake_acc:.1%}")


if __name__ == "__main__":
    main()
