#!/usr/bin/env python3
"""Quick test to verify bias correction is working"""

from eraksha_agent import ErakshAgent
from pathlib import Path

def quick_test():
    agent = ErakshAgent()
    
    # Test with 2 real and 2 fake videos
    real_videos = list(Path('test-data/test-data/raw/real').glob('*.mp4'))[:2]
    fake_videos = list(Path('test-data/test-data/raw/fake').glob('*.mp4'))[:2]
    
    print('🧪 BIAS CORRECTION VERIFICATION')
    print('=' * 40)
    
    for video in real_videos + fake_videos:
        result = agent.predict(str(video))
        if result['success']:
            expected = 'REAL' if 'real' in str(video) else 'FAKE'
            correct = '✅' if result['prediction'] == expected else '❌'
            print(f'{correct} {video.name}: {result["prediction"]} ({result["confidence"]:.1%}) via {result["best_model"].upper()}')
            
            # Show all model predictions for analysis
            all_preds = result['all_predictions']
            print(f'   Models: ', end='')
            for model, pred_data in all_preds.items():
                print(f'{model.upper()}={pred_data["prediction"]:.2f} ', end='')
            print()
        else:
            print(f'❌ {video.name}: Error')
        print()

if __name__ == "__main__":
    quick_test()