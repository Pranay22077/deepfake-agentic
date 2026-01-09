#!/usr/bin/env python3
"""
E-Raksha Updated Agent Test Suite

Validation test for updated agent system with enhanced ensemble logic
and bias correction mechanisms. Tests new routing algorithms and performance optimizations.

Author: E-Raksha Team
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.eraksha_agent import ErakshAgent

def main():
    """Test updated agent system with enhanced ensemble logic."""
    print("=" * 60)
    print("E-RAKSHA UPDATED AGENT VALIDATION TEST")
    print("Testing enhanced ensemble logic with bias corrections")
    print("=" * 60)
    
    # Initialize enhanced agent system
    agent = ErakshAgent()
    
    # Test with a few videos if available
    test_videos = [
        "_archive/test-files/test-data/test-data/raw/real/real_001.mp4",
        "_archive/test-files/test-data/test-data/raw/fake/fake_001.mp4",
        "test-videos/test_video_short.mp4"
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n[TEST] Processing: {os.path.basename(video_path)}")
            print("-" * 40)
            
            result = agent.predict(video_path)
            
            if result['success']:
                print(f"✅ Result: {result['prediction']} ({result['confidence']:.1%})")
                print(f"   Best Model: {result['best_model']}")
                print(f"   Specialists: {result['specialists_used']}")
                print(f"   All Predictions:")
                for model, pred_data in result['all_predictions'].items():
                    print(f"     {model}: {pred_data['prediction']:.3f} (conf: {pred_data['confidence']:.3f})")
            else:
                print(f"❌ Error: {result['error']}")
            break
    else:
        print("No test videos found, but agent initialized successfully!")
        print("\nModel Status:")
        for model_name, model in agent.models.items():
            status = "✅ Loaded" if model is not None else "❌ Not Available"
            print(f"  {model_name}: {status}")

if __name__ == "__main__":
    main()