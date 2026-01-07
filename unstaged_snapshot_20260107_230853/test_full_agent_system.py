"""
TEST FULL AGENTIC SYSTEM WITH NEW MODELS
Download models from Hugging Face and test on 100 videos
"""

import sys
import os
from pathlib import Path
import torch
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the agentic system
from agent.eraksha_agent import ErakshAgent

# Hugging Face model URLs
HF_REPO = "Pran-ay-22077/interceptor-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

MODEL_FILES = {
    "baseline_student.pt": "BG-Model N (NEW)",
    "av_model_student.pt": "AV-Model N (NEW)",
    "cm_model_student.pt": "CM-Model N (NEW)",
    "rr_model_student.pt": "RR-Model N (NEW)",
    "ll_model_student.pt": "LL-Model N (NEW)",
    "tm_model_student.pt": "TM-Model (OLD)"
}

def download_models():
    """Download models from Hugging Face if not present"""
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🔍 CHECKING MODELS FROM HUGGING FACE")
    print("="*60)
    
    for filename, description in MODEL_FILES.items():
        model_path = models_dir / filename
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {size_mb:.1f} MB (already exists)")
            continue
        
        print(f"⬇️  Downloading {description}...")
        url = f"{HF_BASE_URL}/{filename}"
        
        try:
            import requests
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"✅ {description}: {size_mb:.1f} MB (downloaded)")
            
        except Exception as e:
            print(f"❌ Failed to download {description}: {e}")
            print(f"   URL: {url}")
            return False
    
    print("\n✅ All models ready!")
    return True

def test_agent_on_videos():
    """Test the full agentic system on 100 videos"""
    
    print("\n🚀 TESTING FULL AGENTIC SYSTEM")
    print("="*60)
    
    # Initialize agent
    print("\n📦 Initializing E-Raksha Agent...")
    try:
        agent = ErakshAgent(device='auto')
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Load test videos
    TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
    REAL_VIDEOS_DIR = TEST_DATA_ROOT / "real"
    FAKE_VIDEOS_DIR = TEST_DATA_ROOT / "fake"
    
    if not REAL_VIDEOS_DIR.exists() or not FAKE_VIDEOS_DIR.exists():
        print(f"❌ Test data not found at: {TEST_DATA_ROOT}")
        print(f"   Please ensure test videos are in the correct location")
        return
    
    real_videos = list(REAL_VIDEOS_DIR.glob("*.mp4"))
    fake_videos = list(FAKE_VIDEOS_DIR.glob("*.mp4"))
    
    print(f"\n📹 Found {len(real_videos)} real videos")
    print(f"📹 Found {len(fake_videos)} fake videos")
    print(f"📊 Total: {len(real_videos) + len(fake_videos)} videos")
    
    # Combine videos and labels
    all_videos = real_videos + fake_videos
    all_labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=real, 1=fake
    
    # Test each video
    predictions = []
    confidences = []
    best_models = []
    specialists_used = []
    processing_times = []
    
    print(f"\n🔍 Testing agent on all videos...")
    
    for i, video_path in enumerate(tqdm(all_videos, desc="Testing")):
        try:
            # Run agent prediction
            result = agent.predict(str(video_path))
            
            # Extract results
            pred = 1 if result['prediction'] == 'fake' else 0
            predictions.append(pred)
            confidences.append(result['confidence'])
            best_models.append(result.get('best_model', 'unknown'))
            specialists_used.append(result.get('specialists_used', []))
            processing_times.append(result.get('processing_time', 0))
            
        except Exception as e:
            print(f"\n⚠️ Error with {video_path.name}: {e}")
            predictions.append(1)  # Default to fake
            confidences.append(0.5)
            best_models.append('error')
            specialists_used.append([])
            processing_times.append(0)
    
    # Analyze results
    print(f"\n{'='*60}")
    print(f"📊 FULL AGENT TEST RESULTS")
    print(f"{'='*60}")
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, predictions)
    print(f"\n📈 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class analysis
    real_labels = [all_labels[i] for i in range(len(real_videos))]
    real_predictions = [predictions[i] for i in range(len(real_videos))]
    
    fake_labels = [all_labels[i] for i in range(len(real_videos), len(all_videos))]
    fake_predictions = [predictions[i] for i in range(len(real_videos), len(all_videos))]
    
    real_accuracy = accuracy_score(real_labels, real_predictions)
    fake_accuracy = accuracy_score(fake_labels, fake_predictions)
    
    print(f"📈 Real Detection: {real_accuracy:.3f} ({real_accuracy*100:.1f}%)")
    print(f"📈 Fake Detection: {fake_accuracy:.3f} ({fake_accuracy*100:.1f}%)")
    print(f"📊 Bias: {abs(real_accuracy - fake_accuracy):.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, predictions)
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"   Predicted:  Real  Fake")
    print(f"   Real:      {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"   Fake:      {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    # Model usage statistics
    print(f"\n📊 MODEL USAGE STATISTICS:")
    model_counts = {}
    for model in best_models:
        model_counts[model] = model_counts.get(model, 0) + 1
    
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_videos)) * 100
        print(f"   {model}: {count} times ({percentage:.1f}%)")
    
    # Specialist usage
    print(f"\n📊 SPECIALIST USAGE:")
    all_specialists = []
    for specialists in specialists_used:
        all_specialists.extend(specialists)
    
    specialist_counts = {}
    for specialist in all_specialists:
        specialist_counts[specialist] = specialist_counts.get(specialist, 0) + 1
    
    for specialist, count in sorted(specialist_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {specialist}: {count} times")
    
    # Performance metrics
    avg_confidence = np.mean(confidences)
    avg_time = np.mean(processing_times)
    
    print(f"\n📊 PERFORMANCE:")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Average Processing Time: {avg_time:.2f}s")
    print(f"   Total Processing Time: {sum(processing_times):.1f}s")
    
    # Detailed breakdown
    real_as_real = cm[0][0]
    real_as_fake = cm[0][1]
    fake_as_real = cm[1][0]
    fake_as_fake = cm[1][1]
    
    print(f"\n{'='*60}")
    print(f"📊 DETAILED CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_correct = real_as_real + fake_as_fake
    total_wrong = real_as_fake + fake_as_real
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   ✅ Correctly Classified: {total_correct}/{len(all_videos)} ({total_correct/len(all_videos)*100:.1f}%)")
    print(f"   ❌ Incorrectly Classified: {total_wrong}/{len(all_videos)} ({total_wrong/len(all_videos)*100:.1f}%)")
    
    print(f"\n📈 REAL VIDEOS ({len(real_videos)} total):")
    print(f"   ✅ Correctly identified as REAL: {real_as_real}/{len(real_videos)} ({real_as_real/len(real_videos)*100:.1f}%)")
    print(f"   ❌ Wrongly identified as FAKE: {real_as_fake}/{len(real_videos)} ({real_as_fake/len(real_videos)*100:.1f}%)")
    
    print(f"\n📉 FAKE VIDEOS ({len(fake_videos)} total):")
    print(f"   ✅ Correctly identified as FAKE: {fake_as_fake}/{len(fake_videos)} ({fake_as_fake/len(fake_videos)*100:.1f}%)")
    print(f"   ❌ Wrongly identified as REAL: {fake_as_real}/{len(fake_videos)} ({fake_as_real/len(fake_videos)*100:.1f}%)")
    
    # Check if new models are being used
    print(f"\n{'='*60}")
    print(f"🔍 MODEL VERSION CHECK")
    print(f"{'='*60}")
    
    new_models_used = sum(1 for m in best_models if ' N' in m or 'NEW' in m.upper())
    old_models_used = sum(1 for m in best_models if 'OLD' in m.upper() or ('TM' in m and ' N' not in m))
    
    print(f"\n📊 Model Version Usage:")
    print(f"   NEW Models (with N): {new_models_used} predictions")
    print(f"   OLD Models: {old_models_used} predictions")
    
    if new_models_used > 0:
        print(f"\n✅ NEW MODELS ARE BEING USED!")
    else:
        print(f"\n⚠️ WARNING: NEW MODELS NOT DETECTED IN PREDICTIONS")
        print(f"   Check if models were loaded correctly")
    
    # Save results
    results = {
        'test_info': {
            'total_videos': len(all_videos),
            'real_videos': len(real_videos),
            'fake_videos': len(fake_videos),
            'agent_version': 'full_agentic_system'
        },
        'performance': {
            'overall_accuracy': float(accuracy),
            'real_accuracy': float(real_accuracy),
            'fake_accuracy': float(fake_accuracy),
            'bias': float(abs(real_accuracy - fake_accuracy)),
            'avg_confidence': float(avg_confidence),
            'avg_processing_time': float(avg_time)
        },
        'confusion_matrix': cm.tolist(),
        'model_usage': model_counts,
        'specialist_usage': specialist_counts,
        'new_models_used': new_models_used,
        'old_models_used': old_models_used
    }
    
    with open('full_agent_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: full_agent_test_results.json")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    if real_accuracy > 0.7 and fake_accuracy > 0.7:
        print(f"   🎉 EXCELLENT: Balanced high performance!")
    elif abs(real_accuracy - fake_accuracy) < 0.2:
        print(f"   ✅ BALANCED: Low bias between classes")
    else:
        print(f"   ⚠️ BIASED: Significant imbalance detected")
    
    if new_models_used > len(all_videos) * 0.5:
        print(f"   ✅ NEW MODELS: Being used in majority of predictions")
    else:
        print(f"   ⚠️ NEW MODELS: Not being used as expected")
    
    return results

if __name__ == "__main__":
    print("🚀 FULL AGENTIC SYSTEM TEST")
    print("="*60)
    print("This will:")
    print("  1. Download models from Hugging Face")
    print("  2. Initialize the full agentic system")
    print("  3. Test on 100 videos (50 real, 50 fake)")
    print("  4. Show which models are being used")
    print("="*60)
    
    # Step 1: Download models
    if not download_models():
        print("\n❌ Failed to download models. Exiting.")
        sys.exit(1)
    
    # Step 2: Test agent
    test_agent_on_videos()
    
    print("\n✅ Test complete!")
