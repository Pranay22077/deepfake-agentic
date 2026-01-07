"""
TEST ALL HUGGING FACE MODELS
Tests all 6 models from models/ folder using correct architectures from specialists_new.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
import json

# Import correct model architectures
from models.specialists_new import (
    BGSpecialistModel, AVSpecialistModel, CMSpecialistModel,
    RRSpecialistModel, LLSpecialistModel, TMModelOld
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Test data paths
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")


def load_all_models():
    """Load all 6 models from Hugging Face (models/ folder)"""
    models = {}
    model_info = {}
    
    model_configs = {
        'bg': ('models/baseline_student.pt', BGSpecialistModel),
        'av': ('models/av_model_student.pt', AVSpecialistModel),
        'cm': ('models/cm_model_student.pt', CMSpecialistModel),
        'rr': ('models/rr_model_student.pt', RRSpecialistModel),
        'll': ('models/ll_model_student.pt', LLSpecialistModel),
        'tm': ('models/tm_model_student.pt', TMModelOld),
    }
    
    for name, (path, model_class) in model_configs.items():
        print(f"\nLoading {name.upper()} model from {path}...")
        
        if not os.path.exists(path):
            print(f"  ❌ File not found: {path}")
            continue
        
        try:
            # Create model
            model = model_class()
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
            
            # Get state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                metrics = checkpoint.get('metrics', {})
            else:
                state_dict = checkpoint
                metrics = {}
            
            # Load weights
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE)
            model.eval()
            
            models[name] = model
            model_info[name] = {
                'path': path,
                'metrics': metrics,
                'loaded': True
            }
            
            acc = metrics.get('accuracy', 'N/A')
            real_acc = metrics.get('real_accuracy', 'N/A')
            fake_acc = metrics.get('fake_accuracy', 'N/A')
            print(f"  ✅ Loaded successfully")
            print(f"     Training metrics: Acc={acc}, Real={real_acc}, Fake={fake_acc}")
            
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
            model_info[name] = {'path': path, 'error': str(e), 'loaded': False}
    
    return models, model_info


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
            
            return fake_prob, True
    except Exception as e:
        print(f"    Error with {model_name}: {e}")
        return 0.5, False


def test_single_model(model, model_name, real_videos, fake_videos):
    """Test a single model on all videos"""
    results = {
        'real_correct': 0, 'real_total': 0,
        'fake_correct': 0, 'fake_total': 0,
        'predictions': []
    }
    
    # Test real videos
    for video_path in tqdm(real_videos, desc=f"Testing {model_name.upper()} on REAL"):
        frames = extract_frames(video_path)
        if frames is None:
            continue
        
        fake_prob, success = predict_with_model(model, frames, model_name)
        if not success:
            continue
        
        predicted = "FAKE" if fake_prob > 0.5 else "REAL"
        correct = predicted == "REAL"
        
        results['real_total'] += 1
        if correct:
            results['real_correct'] += 1
        
        results['predictions'].append({
            'video': video_path.name,
            'true': 'REAL',
            'predicted': predicted,
            'fake_prob': fake_prob,
            'correct': correct
        })
    
    # Test fake videos
    for video_path in tqdm(fake_videos, desc=f"Testing {model_name.upper()} on FAKE"):
        frames = extract_frames(video_path)
        if frames is None:
            continue
        
        fake_prob, success = predict_with_model(model, frames, model_name)
        if not success:
            continue
        
        predicted = "FAKE" if fake_prob > 0.5 else "REAL"
        correct = predicted == "FAKE"
        
        results['fake_total'] += 1
        if correct:
            results['fake_correct'] += 1
        
        results['predictions'].append({
            'video': video_path.name,
            'true': 'FAKE',
            'predicted': predicted,
            'fake_prob': fake_prob,
            'correct': correct
        })
    
    return results


def main():
    print("=" * 80)
    print("TESTING ALL HUGGING FACE MODELS")
    print("=" * 80)
    
    # Load all models
    models, model_info = load_all_models()
    
    if not models:
        print("\n❌ No models loaded successfully!")
        return
    
    print(f"\n✅ Loaded {len(models)} models: {list(models.keys())}")
    
    # Get test videos
    real_videos = list((TEST_DATA_ROOT / "real").glob("*.mp4"))[:50]
    fake_videos = list((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:50]
    
    print(f"\n📹 Test set: {len(real_videos)} real + {len(fake_videos)} fake videos")
    
    # Test each model
    all_results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"TESTING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        results = test_single_model(model, model_name, real_videos, fake_videos)
        all_results[model_name] = results
        
        # Calculate metrics
        total = results['real_total'] + results['fake_total']
        correct = results['real_correct'] + results['fake_correct']
        accuracy = correct / total if total > 0 else 0
        
        real_acc = results['real_correct'] / results['real_total'] if results['real_total'] > 0 else 0
        fake_acc = results['fake_correct'] / results['fake_total'] if results['fake_total'] > 0 else 0
        
        bias = real_acc - fake_acc
        
        print(f"\n📊 {model_name.upper()} RESULTS:")
        print(f"   Overall Accuracy: {correct}/{total} ({accuracy:.1%})")
        print(f"   Real Detection:   {results['real_correct']}/{results['real_total']} ({real_acc:.1%})")
        print(f"   Fake Detection:   {results['fake_correct']}/{results['fake_total']} ({fake_acc:.1%})")
        print(f"   Bias (Real-Fake): {bias:+.1%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - ALL MODELS")
    print("=" * 80)
    
    print(f"\n{'Model':<8} {'Accuracy':<12} {'Real Acc':<12} {'Fake Acc':<12} {'Bias':<10}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        total = results['real_total'] + results['fake_total']
        correct = results['real_correct'] + results['fake_correct']
        accuracy = correct / total if total > 0 else 0
        real_acc = results['real_correct'] / results['real_total'] if results['real_total'] > 0 else 0
        fake_acc = results['fake_correct'] / results['fake_total'] if results['fake_total'] > 0 else 0
        bias = real_acc - fake_acc
        
        print(f"{model_name.upper():<8} {accuracy:.1%}        {real_acc:.1%}        {fake_acc:.1%}        {bias:+.1%}")
    
    # Save results
    with open('all_hf_models_test_results.json', 'w') as f:
        json.dump({
            'model_info': model_info,
            'results': {k: {
                'real_correct': v['real_correct'],
                'real_total': v['real_total'],
                'fake_correct': v['fake_correct'],
                'fake_total': v['fake_total'],
            } for k, v in all_results.items()}
        }, f, indent=2)
    
    print("\n✅ Results saved to all_hf_models_test_results.json")


if __name__ == "__main__":
    main()
