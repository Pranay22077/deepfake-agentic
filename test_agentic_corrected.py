#!/usr/bin/env python3
"""
Test Bias-Corrected Agentic System
Full evaluation with 50 real + 50 fake videos
"""

import os
import sys
import json
import time
from pathlib import Path

# Import the bias-corrected agent
from eraksha_agent import ErakshAgent

def test_agentic_system_corrected():
    """Test the bias-corrected agentic system with 100 videos"""
    
    print("🔧 TESTING BIAS-CORRECTED AGENTIC SYSTEM")
    print("=" * 60)
    
    # Initialize agent
    agent = ErakshAgent()
    
    # Test data paths
    real_dir = Path("test-data/test-data/raw/real")
    fake_dir = Path("test-data/test-data/raw/fake")
    
    # Get video files
    real_videos = list(real_dir.glob("*.mp4"))[:50]  # Real videos in real/ directory
    fake_videos = list(fake_dir.glob("*.mp4"))[:50]  # Fake videos in fake/ directory
    
    print(f"📊 Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")
    
    results = {
        'summary': {},
        'model_usage': {},
        'confidence_levels': {'high': 0, 'medium': 0, 'low': 0},
        'routing_decisions': {},
        'detailed_results': {'real': [], 'fake': []},
        'processing_times': []
    }
    
    total_correct = 0
    real_correct = 0
    fake_correct = 0
    
    # Test real videos
    print(f"\n🎬 Testing {len(real_videos)} REAL videos...")
    for i, video_path in enumerate(real_videos):
        print(f"[{i+1}/{len(real_videos)}] {video_path.name}")
        
        result = agent.predict(str(video_path))
        
        if result['success']:
            is_correct = result['prediction'] == 'REAL'
            if is_correct:
                real_correct += 1
                total_correct += 1
            
            # Track model usage
            best_model = result['best_model']
            results['model_usage'][best_model] = results['model_usage'].get(best_model, 0) + 1
            
            # Track confidence levels
            conf_level = result['confidence_level']
            results['confidence_levels'][conf_level] += 1
            
            # Track routing decisions
            num_specialists = len(result['specialists_used'])
            results['routing_decisions'][str(num_specialists)] = results['routing_decisions'].get(str(num_specialists), 0) + 1
            
            # Store detailed result
            results['detailed_results']['real'].append({
                'video': video_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'correct': is_correct,
                'best_model': best_model,
                'specialists_used': result['specialists_used'],
                'processing_time': result['processing_time']
            })
            
            results['processing_times'].append(result['processing_time'])
        
        else:
            print(f"❌ Error processing {video_path.name}: {result['error']}")
    
    # Test fake videos
    print(f"\n🎬 Testing {len(fake_videos)} FAKE videos...")
    for i, video_path in enumerate(fake_videos):
        print(f"[{i+1}/{len(fake_videos)}] {video_path.name}")
        
        result = agent.predict(str(video_path))
        
        if result['success']:
            is_correct = result['prediction'] == 'FAKE'
            if is_correct:
                fake_correct += 1
                total_correct += 1
            
            # Track model usage
            best_model = result['best_model']
            results['model_usage'][best_model] = results['model_usage'].get(best_model, 0) + 1
            
            # Track confidence levels
            conf_level = result['confidence_level']
            results['confidence_levels'][conf_level] += 1
            
            # Track routing decisions
            num_specialists = len(result['specialists_used'])
            results['routing_decisions'][str(num_specialists)] = results['routing_decisions'].get(str(num_specialists), 0) + 1
            
            # Store detailed result
            results['detailed_results']['fake'].append({
                'video': video_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'correct': is_correct,
                'best_model': best_model,
                'specialists_used': result['specialists_used'],
                'processing_time': result['processing_time']
            })
            
            results['processing_times'].append(result['processing_time'])
        
        else:
            print(f"❌ Error processing {video_path.name}: {result['error']}")
    
    # Calculate final metrics
    total_videos = len(real_videos) + len(fake_videos)
    overall_accuracy = (total_correct / total_videos) * 100
    real_accuracy = (real_correct / len(real_videos)) * 100
    fake_accuracy = (fake_correct / len(fake_videos)) * 100
    
    # Calculate precision, recall, F1
    true_positives = fake_correct
    false_positives = len(real_videos) - real_correct
    false_negatives = len(fake_videos) - fake_correct
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    results['summary'] = {
        'total_videos': total_videos,
        'overall_accuracy': overall_accuracy,
        'real_accuracy': real_accuracy,
        'fake_accuracy': fake_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_confidence': sum([r['confidence'] for r in results['detailed_results']['real'] + results['detailed_results']['fake']]) / total_videos,
        'avg_processing_time': sum(results['processing_times']) / len(results['processing_times'])
    }
    
    # Save results
    timestamp = int(time.time())
    results_file = f"agentic_evaluation_corrected_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n📊 BIAS-CORRECTED EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_videos})")
    print(f"Real Video Accuracy: {real_accuracy:.1f}% ({real_correct}/{len(real_videos)})")
    print(f"Fake Video Accuracy: {fake_accuracy:.1f}% ({fake_correct}/{len(fake_videos)})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    print(f"Average Confidence: {results['summary']['avg_confidence']:.1%}")
    print(f"Average Processing Time: {results['summary']['avg_processing_time']:.2f}s")
    
    print(f"\n🤖 Model Usage:")
    for model, count in sorted(results['model_usage'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_videos) * 100
        print(f"   {model.upper()}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n🎯 Confidence Distribution:")
    for level, count in results['confidence_levels'].items():
        percentage = (count / total_videos) * 100
        print(f"   {level.title()}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n📁 Results saved to: {results_file}")
    
    # Compare with original results
    try:
        with open('agentic_evaluation_results_1767041294.json', 'r') as f:
            original_results = json.load(f)
        
        print(f"\n📈 IMPROVEMENT COMPARISON")
        print("=" * 40)
        print(f"Overall Accuracy: {original_results['summary']['overall_accuracy']:.1f}% → {overall_accuracy:.1f}% ({overall_accuracy - original_results['summary']['overall_accuracy']:+.1f}%)")
        print(f"Real Accuracy: {original_results['summary']['real_accuracy']:.1f}% → {real_accuracy:.1f}% ({real_accuracy - original_results['summary']['real_accuracy']:+.1f}%)")
        print(f"Fake Accuracy: {original_results['summary']['fake_accuracy']:.1f}% → {fake_accuracy:.1f}% ({fake_accuracy - original_results['summary']['fake_accuracy']:+.1f}%)")
        print(f"F1-Score: {original_results['summary']['f1_score']:.3f} → {f1_score:.3f} ({f1_score - original_results['summary']['f1_score']:+.3f})")
        
        # Model usage comparison
        print(f"\n🔄 Model Usage Changes:")
        original_usage = original_results['model_usage']
        for model in set(list(original_usage.keys()) + list(results['model_usage'].keys())):
            old_count = original_usage.get(model, 0)
            new_count = results['model_usage'].get(model, 0)
            change = new_count - old_count
            print(f"   {model.upper()}: {old_count} → {new_count} ({change:+d})")
    
    except FileNotFoundError:
        print(f"\n⚠️ Original results file not found for comparison")
    
    return results

def main():
    """Main function"""
    test_agentic_system_corrected()

if __name__ == "__main__":
    main()