"""
E-Raksha Advanced Agent Validation - 100 Video Benchmark

Comprehensive testing of the advanced E-Raksha agent system with optimized models.
Tests enhanced ensemble logic, dynamic weights, and confidence calibration.

Models tested: BG (78%), AV (85%), CM (82%), RR (79%), LL (88%)
Target ensemble accuracy: 87%

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.eraksha_agent import ErakshAgent

# Test dataset configuration
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:50]
fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:50]

print("=" * 70)
print("E-RAKSHA ADVANCED AGENT VALIDATION - 100 VIDEOS")
print("Enhanced Models: BG (78%), AV (85%), CM (82%), RR (79%), LL (88%)")
print("Target Ensemble Accuracy: 87%")
print("=" * 70)

# Initialize advanced agent system
agent = ErakshAgent()

# Validation results
results = []
real_correct = 0
fake_correct = 0

print("\n" + "=" * 70)
print("TESTING 50 REAL VIDEOS")
print("=" * 70)

for video_path in tqdm(real_videos, desc="Real videos"):
    result = agent.predict(str(video_path))
    if result['success']:
        pred = result['prediction']
        conf = result['confidence']
        is_correct = pred == 'REAL'
        if is_correct:
            real_correct += 1
        results.append({
            'video': video_path.name,
            'true': 'REAL',
            'pred': pred,
            'confidence': conf,
            'correct': is_correct,
            'specialists_used': result.get('specialists_used', []),
            'all_predictions': result.get('all_predictions', {})
        })

print(f"\nReal videos: {real_correct}/50 correct ({real_correct/50*100:.1f}%)")

print("\n" + "=" * 70)
print("TESTING 50 FAKE VIDEOS")
print("=" * 70)

for video_path in tqdm(fake_videos, desc="Fake videos"):
    result = agent.predict(str(video_path))
    if result['success']:
        pred = result['prediction']
        conf = result['confidence']
        is_correct = pred == 'FAKE'
        if is_correct:
            fake_correct += 1
        results.append({
            'video': video_path.name,
            'true': 'FAKE',
            'pred': pred,
            'confidence': conf,
            'correct': is_correct,
            'specialists_used': result.get('specialists_used', []),
            'all_predictions': result.get('all_predictions', {})
        })

print(f"\nFake videos: {fake_correct}/50 correct ({fake_correct/50*100:.1f}%)")

# Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total = len(results)
total_correct = real_correct + fake_correct
accuracy = total_correct / total if total > 0 else 0

print(f"\n📊 OVERALL ACCURACY: {total_correct}/{total} ({accuracy*100:.1f}%)")
print(f"📈 Real Detection:   {real_correct}/50 ({real_correct/50*100:.1f}%)")
print(f"📉 Fake Detection:   {fake_correct}/50 ({fake_correct/50*100:.1f}%)")
print(f"📊 Bias (Real-Fake): {(real_correct/50 - fake_correct/50)*100:+.1f}%")

# Confusion matrix
print(f"\n📊 CONFUSION MATRIX:")
print(f"   Predicted:  REAL  FAKE")
print(f"   REAL:       {real_correct:4d}  {50-real_correct:4d}")
print(f"   FAKE:       {50-fake_correct:4d}  {fake_correct:4d}")

# Save results
output = {
    'summary': {
        'total': total,
        'correct': total_correct,
        'accuracy': accuracy,
        'real_correct': real_correct,
        'real_total': 50,
        'fake_correct': fake_correct,
        'fake_total': 50,
        'real_accuracy': real_correct / 50,
        'fake_accuracy': fake_correct / 50,
        'bias': (real_correct/50 - fake_correct/50)
    },
    'results': results
}

with open('agent_100_videos_test_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to agent_100_videos_test_results.json")

# Assessment
print(f"\n🎯 ASSESSMENT:")
if accuracy >= 0.7:
    print(f"   🎉 GOOD: {accuracy*100:.1f}% accuracy")
elif accuracy >= 0.6:
    print(f"   ✅ ACCEPTABLE: {accuracy*100:.1f}% accuracy")
else:
    print(f"   ⚠️ NEEDS IMPROVEMENT: {accuracy*100:.1f}% accuracy")

if abs(real_correct/50 - fake_correct/50) < 0.15:
    print(f"   ✅ BALANCED: Low bias between real/fake detection")
else:
    print(f"   ⚠️ BIASED: Significant imbalance in real/fake detection")
