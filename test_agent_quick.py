"""
E-Raksha Agent Quick Test Suite

Rapid validation test for the complete agent system using 10 representative videos.
Provides fast feedback on system functionality and ensemble performance.

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.eraksha_agent import ErakshAgent

# Test dataset configuration
TEST_DATA_ROOT = Path("_archive/test-files/test-data/test-data/raw")
real_videos = sorted((TEST_DATA_ROOT / "real").glob("*.mp4"))[:5]
fake_videos = sorted((TEST_DATA_ROOT / "fake").glob("*.mp4"))[:5]

print("=" * 70)
print("E-RAKSHA AGENT QUICK TEST - 10 VIDEOS")
print("=" * 70)

# Initialize agent system
agent = ErakshAgent()

# Execute test
results = []
correct = 0

print("\n" + "=" * 70)
print("TESTING REAL VIDEOS")
print("=" * 70)

for video_path in real_videos:
    result = agent.predict(str(video_path))
    if result['success']:
        pred = result['prediction']
        conf = result['confidence']
        is_correct = pred == 'REAL'
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"{status} {video_path.name}: {pred} ({conf:.1%})")
        results.append({'video': video_path.name, 'true': 'REAL', 'pred': pred, 'correct': is_correct})

print("\n" + "=" * 70)
print("TESTING FAKE VIDEOS")
print("=" * 70)

for video_path in fake_videos:
    result = agent.predict(str(video_path))
    if result['success']:
        pred = result['prediction']
        conf = result['confidence']
        is_correct = pred == 'FAKE'
        if is_correct:
            correct += 1
        status = "✅" if is_correct else "❌"
        print(f"{status} {video_path.name}: {pred} ({conf:.1%})")
        results.append({'video': video_path.name, 'true': 'FAKE', 'pred': pred, 'correct': is_correct})

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

total = len(results)
real_correct = sum(1 for r in results if r['true'] == 'REAL' and r['correct'])
fake_correct = sum(1 for r in results if r['true'] == 'FAKE' and r['correct'])

print(f"Overall: {correct}/{total} ({correct/total*100:.1f}%)")
print(f"Real:    {real_correct}/5 ({real_correct/5*100:.1f}%)")
print(f"Fake:    {fake_correct}/5 ({fake_correct/5*100:.1f}%)")
