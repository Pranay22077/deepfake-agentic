#!/usr/bin/env python3
"""
Debug Current Bias in Agentic System
Test with known real/fake videos to see actual behavior
"""

import os
from pathlib import Path
from eraksha_agent import ErakshAgent

def test_bias():
    """Test current bias with known videos"""
    
    print("🔍 DEBUGGING CURRENT BIAS")
    print("=" * 50)
    
    # Initialize agent
    agent = ErakshAgent()
    
    # Test with known real and fake videos
    real_dir = Path("test-data/test-data/raw/real")
    fake_dir = Path("test-data/test-data/raw/fake")
    
    # Test 5 real and 5 fake videos
    real_videos = list(real_dir.glob("*.mp4"))[:5]
    fake_videos = list(fake_dir.glob("*.mp4"))[:5]
    
    print(f"\n📊 Testing {len(real_videos)} real + {len(fake_videos)} fake videos")
    
    real_results = []
    fake_results = []
    
    # Test real videos
    print(f"\n🎬 REAL VIDEOS:")
    for i, video in enumerate(real_videos):
        result = agent.predict(str(video))
        if result['success']:
            prediction = result['prediction']
            confidence = result['confidence']
            best_model = result['best_model']
            
            # Show all model predictions
            all_preds = result['all_predictions']
            
            correct = "✅" if prediction == 'REAL' else "❌"
            print(f"{correct} {video.name}: {prediction} ({confidence:.1%}) via {best_model.upper()}")
            
            # Show individual model predictions
            print(f"   Models: ", end="")
            for model, pred_data in all_preds.items():
                pred_val = pred_data['prediction']
                conf_val = pred_data['confidence']
                print(f"{model.upper()}={pred_val:.2f}({conf_val:.2f}) ", end="")
            print()
            
            real_results.append({
                'correct': prediction == 'REAL',
                'prediction': prediction,
                'confidence': confidence,
                'best_model': best_model,
                'all_predictions': all_preds
            })
        print()
    
    # Test fake videos
    print(f"\n🎭 FAKE VIDEOS:")
    for i, video in enumerate(fake_videos):
        result = agent.predict(str(video))
        if result['success']:
            prediction = result['prediction']
            confidence = result['confidence']
            best_model = result['best_model']
            
            # Show all model predictions
            all_preds = result['all_predictions']
            
            correct = "✅" if prediction == 'FAKE' else "❌"
            print(f"{correct} {video.name}: {prediction} ({confidence:.1%}) via {best_model.upper()}")
            
            # Show individual model predictions
            print(f"   Models: ", end="")
            for model, pred_data in all_preds.items():
                pred_val = pred_data['prediction']
                conf_val = pred_data['confidence']
                print(f"{model.upper()}={pred_val:.2f}({conf_val:.2f}) ", end="")
            print()
            
            fake_results.append({
                'correct': prediction == 'FAKE',
                'prediction': prediction,
                'confidence': confidence,
                'best_model': best_model,
                'all_predictions': all_preds
            })
        print()
    
    # Analyze results
    print(f"\n📈 BIAS ANALYSIS:")
    print("=" * 30)
    
    real_correct = sum(1 for r in real_results if r['correct'])
    fake_correct = sum(1 for r in fake_results if r['correct'])
    
    real_accuracy = (real_correct / len(real_results)) * 100 if real_results else 0
    fake_accuracy = (fake_correct / len(fake_results)) * 100 if fake_results else 0
    
    print(f"Real Video Accuracy: {real_accuracy:.1f}% ({real_correct}/{len(real_results)})")
    print(f"Fake Video Accuracy: {fake_accuracy:.1f}% ({fake_correct}/{len(fake_results)})")
    print(f"Bias Gap: {real_accuracy - fake_accuracy:.1f}% (positive = REAL bias)")
    
    # Model usage analysis
    print(f"\n🤖 Model Usage:")
    model_usage = {}
    all_results = real_results + fake_results
    for result in all_results:
        model = result['best_model']
        model_usage[model] = model_usage.get(model, 0) + 1
    
    for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_results)) * 100
        print(f"   {model.upper()}: {count}/{len(all_results)} ({percentage:.1f}%)")
    
    # Individual model bias analysis
    print(f"\n🔬 Individual Model Bias:")
    model_stats = {}
    
    for result in real_results:
        for model, pred_data in result['all_predictions'].items():
            if model not in model_stats:
                model_stats[model] = {'real_sum': 0, 'real_count': 0, 'fake_sum': 0, 'fake_count': 0}
            model_stats[model]['real_sum'] += pred_data['prediction']
            model_stats[model]['real_count'] += 1
    
    for result in fake_results:
        for model, pred_data in result['all_predictions'].items():
            if model not in model_stats:
                model_stats[model] = {'real_sum': 0, 'real_count': 0, 'fake_sum': 0, 'fake_count': 0}
            model_stats[model]['fake_sum'] += pred_data['prediction']
            model_stats[model]['fake_count'] += 1
    
    for model, stats in model_stats.items():
        if stats['real_count'] > 0 and stats['fake_count'] > 0:
            avg_real_pred = stats['real_sum'] / stats['real_count']
            avg_fake_pred = stats['fake_sum'] / stats['fake_count']
            
            print(f"   {model.upper()}:")
            print(f"     Real videos avg prediction: {avg_real_pred:.3f}")
            print(f"     Fake videos avg prediction: {avg_fake_pred:.3f}")
            print(f"     Bias: {avg_real_pred - avg_fake_pred:.3f} (negative = REAL bias)")

if __name__ == "__main__":
    test_bias()