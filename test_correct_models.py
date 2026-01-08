"""
E-Raksha Correct Model Architecture Testing

Comprehensive testing suite using verified specialist model architectures.
Validates all models with perfect architecture matches for accurate performance evaluation.

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import json

from models.specialists_new import (
    BGSpecialistModel, AVSpecialistModel, CMSpecialistModel,
    RRSpecialistModel, LLSpecialistModel, TMModelOld
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")


def load_model(name, model_class, ckpt_path):
    """Load model with strict=True to ensure perfect match"""
    print(f"\nLoading {name}...")
    
    model = model_class()
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        metrics = checkpoint.get('metrics', {})
    else:
        state_dict = checkpoint
        metrics = {}
    
    # Use strict=True to ensure perfect match
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    
    print(f"  ✅ Loaded with strict=True")
    if metrics:
        print(f"  Training: Acc={metrics.get('accuracy', 'N/A'):.3f}, "
              f"Real={metrics.get('real_accuracy', 'N/A'):.3f}, "
              f"Fake={metrics.get('fake_accuracy', 'N/A'):.3f}")
    
    return model, metrics


def extract_frames(video_path, num_frames=8):
    """Extract frames from video with ImageNet normalization"""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # ImageNet normalization constants
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            # Normalize: /255, then ImageNet normalization
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - mean) / std
            frame = torch.from_numpy(frame).float()
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
    
    cap.release()
    return torch.stack(frames) if len(frames) >= 4 else None


def predict(model, frames, model_name):
    """Run prediction"""
    with torch.no_grad():
        if model_name == 'tm':
            input_tensor = frames.unsqueeze(0).to(DEVICE)
        else:
            input_tensor = frames[:4].to(DEVICE)
        
        logits = model(input_tensor)
        
        if model_name == 'tm':
            probs = F.softmax(logits, dim=1)
            fake_prob = probs[0, 1].item()
        else:
            probs = F.softmax(logits, dim=1)
            fake_prob = probs[:, 1].mean().item()
        
        return fake_prob


def test_model(model, model_name, videos, labels):
    """Test a model on videos"""
    correct = 0
    real_correct = 0
    fake_correct = 0
    real_total = 0
    fake_total = 0
    
    for video_path, label in tqdm(zip(videos, labels), total=len(videos), desc=f"{model_name.upper()}"):
        frames = extract_frames(video_path)
        if frames is None:
            continue
        
        fake_prob = predict(model, frames, model_name)
        predicted = 1 if fake_prob > 0.5 else 0
        
        if label == 0:  # Real
            real_total += 1
            if predicted == 0:
                real_correct += 1
                correct += 1
        else:  # Fake
            fake_total += 1
            if predicted == 1:
                fake_correct += 1
                correct += 1
    
    total = real_total + fake_total
    return {
        'accuracy': correct / total if total > 0 else 0,
        'real_acc': real_correct / real_total if real_total > 0 else 0,
        'fake_acc': fake_correct / fake_total if fake_total > 0 else 0,
        'real_correct': real_correct,
        'real_total': real_total,
        'fake_correct': fake_correct,
        'fake_total': fake_total
    }


def main():
    print("=" * 70)
    print("TESTING WITH CORRECT ARCHITECTURES (specialists_new.py)")
    print("=" * 70)
    
    # Load all models
    models = {}
    model_configs = [
        ('bg', BGSpecialistModel, 'models/baseline_student.pt'),
        ('av', AVSpecialistModel, 'models/av_model_student.pt'),
        ('cm', CMSpecialistModel, 'models/cm_model_student.pt'),
        ('rr', RRSpecialistModel, 'models/rr_model_student.pt'),
        ('ll', LLSpecialistModel, 'models/ll_model_student.pt'),
        ('tm', TMModelOld, 'models/tm_model_student.pt'),
    ]
    
    for name, model_class, ckpt_path in model_configs:
        try:
            models[name], _ = load_model(name, model_class, ckpt_path)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n✅ Loaded {len(models)} models")
    
    # Get test videos
    real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:50]
    fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:50]
    
    all_videos = list(real_videos) + list(fake_videos)
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)
    
    print(f"\n📹 Testing on {len(real_videos)} real + {len(fake_videos)} fake = {len(all_videos)} videos")
    
    # Test each model
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        results[name] = test_model(model, name, all_videos, all_labels)
        r = results[name]
        print(f"\n{name.upper()} Results:")
        print(f"  Overall: {r['accuracy']:.1%}")
        print(f"  Real:    {r['real_acc']:.1%} ({r['real_correct']}/{r['real_total']})")
        print(f"  Fake:    {r['fake_acc']:.1%} ({r['fake_correct']}/{r['fake_total']})")
        print(f"  Bias:    {r['real_acc'] - r['fake_acc']:+.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<6} {'Overall':<10} {'Real':<10} {'Fake':<10} {'Bias':<10}")
    print("-" * 50)
    
    for name, r in results.items():
        bias = r['real_acc'] - r['fake_acc']
        print(f"{name.upper():<6} {r['accuracy']:.1%}      {r['real_acc']:.1%}      {r['fake_acc']:.1%}      {bias:+.1%}")
    
    # Save results
    with open('correct_models_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to correct_models_test_results.json")


if __name__ == "__main__":
    main()
