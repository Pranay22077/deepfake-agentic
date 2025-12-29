#!/usr/bin/env python3
"""
Full Agentic System Evaluation
Test with 50 real + 50 fake videos to measure complete system performance
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from eraksha_agent import ErakshAgent

def find_test_videos():
    """Find the test video directories"""
    test_data_paths = [
        "test-data/test-data/raw",
        "test-data/raw", 
        "test_data/raw",
        "data/test-data/raw"
    ]
    
    for path in test_data_paths:
        if os.path.exists(path):
            real_path = os.path.join(path, "real")
            fake_path = os.path.join(path, "fake")
            
            if os.path.exists(real_path) and os.path.exists(fake_path):
                return real_path, fake_path
    
    return None, None

def get_video_files(directory, max_count=50):
    """Get video files from directory"""
    if not os.path.exists(directory):
        return []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(directory, file))
    
    return video_files[:max_count]

def evaluate_agentic_system():
    """Comprehensive evaluation of the agentic system"""
    print("🎯 E-RAKSHA AGENTIC SYSTEM - FULL EVALUATION")
    print("=" * 70)
    print("Testing with 50 real + 50 fake videos")
    print("=" * 70)
    
    # Find test videos
    real_dir, fake_dir = find_test_videos()
    
    if not real_dir or not fake_dir:
        print("❌ Test video directories not found!")
        print("Expected structure: test-data/test-data/raw/real/ and test-data/test-data/raw/fake/")
        return False
    
    # Get video files
    real_videos = get_video_files(real_dir, 50)
    fake_videos = get_video_files(fake_dir, 50)
    
    print(f"📁 Found {len(real_videos)} real videos in: {real_dir}")
    print(f"📁 Found {len(fake_videos)} fake videos in: {fake_dir}")
    
    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("❌ No videos found in test directories!")
        return False
    
    # Initialize agentic system
    print(f"\n🚀 Initializing E-Raksha Agentic System...")
    try:
        agent = ErakshAgent()
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return False
    
    # Test results storage
    results = {
        'real': [],
        'fake': [],
        'model_usage': defaultdict(int),
        'processing_times': [],
        'confidence_levels': defaultdict(int),
        'routing_decisions': defaultdict(int)
    }
    
    total_videos = len(real_videos) + len(fake_videos)
    processed = 0
    
    print(f"\n🎬 Processing {total_videos} videos...")
    print("=" * 70)
    
    # Process real videos
    print(f"\n📹 Processing {len(real_videos)} REAL videos...")
    for i, video_path in enumerate(real_videos):
        processed += 1
        video_name = os.path.basename(video_path)
        
        print(f"[{processed:3d}/{total_videos}] Real: {video_name[:30]}...", end=" ")
        
        try:
            start_time = time.time()
            result = agent.predict(video_path)
            processing_time = time.time() - start_time
            
            if result['success']:
                prediction = result['prediction']
                confidence = result['confidence']
                best_model = result['best_model']
                specialists_used = result.get('specialists_used', [])
                
                # Store result
                results['real'].append({
                    'video': video_name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'correct': prediction == 'REAL',
                    'best_model': best_model,
                    'specialists_used': specialists_used,
                    'processing_time': processing_time
                })
                
                # Update statistics
                results['model_usage'][best_model] += 1
                results['processing_times'].append(processing_time)
                results['confidence_levels'][result['confidence_level']] += 1
                results['routing_decisions'][len(specialists_used)] += 1
                
                # Print result
                status = "✅" if prediction == 'REAL' else "❌"
                print(f"{status} {prediction} ({confidence:.1%}) via {best_model.upper()} [{processing_time:.2f}s]")
                
            else:
                print(f"❌ ERROR: {result.get('error', 'Unknown')}")
                results['real'].append({
                    'video': video_name,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'correct': False,
                    'best_model': 'none',
                    'specialists_used': [],
                    'processing_time': 0.0
                })
        
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)[:50]}...")
            results['real'].append({
                'video': video_name,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'correct': False,
                'best_model': 'none',
                'specialists_used': [],
                'processing_time': 0.0
            })
    
    # Process fake videos
    print(f"\n📹 Processing {len(fake_videos)} FAKE videos...")
    for i, video_path in enumerate(fake_videos):
        processed += 1
        video_name = os.path.basename(video_path)
        
        print(f"[{processed:3d}/{total_videos}] Fake: {video_name[:30]}...", end=" ")
        
        try:
            start_time = time.time()
            result = agent.predict(video_path)
            processing_time = time.time() - start_time
            
            if result['success']:
                prediction = result['prediction']
                confidence = result['confidence']
                best_model = result['best_model']
                specialists_used = result.get('specialists_used', [])
                
                # Store result
                results['fake'].append({
                    'video': video_name,
                    'prediction': prediction,
                    'confidence': confidence,
                    'correct': prediction == 'FAKE',
                    'best_model': best_model,
                    'specialists_used': specialists_used,
                    'processing_time': processing_time
                })
                
                # Update statistics
                results['model_usage'][best_model] += 1
                results['processing_times'].append(processing_time)
                results['confidence_levels'][result['confidence_level']] += 1
                results['routing_decisions'][len(specialists_used)] += 1
                
                # Print result
                status = "✅" if prediction == 'FAKE' else "❌"
                print(f"{status} {prediction} ({confidence:.1%}) via {best_model.upper()} [{processing_time:.2f}s]")
                
            else:
                print(f"❌ ERROR: {result.get('error', 'Unknown')}")
                results['fake'].append({
                    'video': video_name,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'correct': False,
                    'best_model': 'none',
                    'specialists_used': [],
                    'processing_time': 0.0
                })
        
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)[:50]}...")
            results['fake'].append({
                'video': video_name,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'correct': False,
                'best_model': 'none',
                'specialists_used': [],
                'processing_time': 0.0
            })
    
    # Calculate comprehensive metrics
    print(f"\n📊 CALCULATING PERFORMANCE METRICS...")
    print("=" * 70)
    
    # Overall accuracy
    real_correct = sum(1 for r in results['real'] if r['correct'])
    fake_correct = sum(1 for r in results['fake'] if r['correct'])
    total_correct = real_correct + fake_correct
    total_processed = len(results['real']) + len(results['fake'])
    
    overall_accuracy = (total_correct / total_processed) * 100 if total_processed > 0 else 0
    
    # Class-specific metrics
    real_accuracy = (real_correct / len(results['real'])) * 100 if results['real'] else 0
    fake_accuracy = (fake_correct / len(results['fake'])) * 100 if results['fake'] else 0
    
    # Precision, Recall, F1
    true_positives = fake_correct  # Correctly identified fakes
    false_positives = len(results['real']) - real_correct  # Real videos classified as fake
    false_negatives = len(results['fake']) - fake_correct  # Fake videos classified as real
    true_negatives = real_correct  # Correctly identified reals
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confidence analysis
    all_confidences = [r['confidence'] for r in results['real'] + results['fake'] if r['confidence'] > 0]
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    # Processing time analysis
    avg_processing_time = np.mean(results['processing_times']) if results['processing_times'] else 0
    
    # Print comprehensive results
    print(f"\n🎯 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Videos Processed: {total_processed}")
    print(f"   Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"   Real Video Accuracy: {real_accuracy:.2f}% ({real_correct}/{len(results['real'])})")
    print(f"   Fake Video Accuracy: {fake_accuracy:.2f}% ({fake_correct}/{len(results['fake'])})")
    
    print(f"\n🎯 CLASSIFICATION METRICS:")
    print(f"   Precision (Fake Detection): {precision:.3f}")
    print(f"   Recall (Fake Detection): {recall:.3f}")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    
    print(f"\n⏱️ PERFORMANCE METRICS:")
    print(f"   Average Processing Time: {avg_processing_time:.2f}s per video")
    print(f"   Total Processing Time: {sum(results['processing_times']):.1f}s")
    
    print(f"\n🤖 MODEL USAGE STATISTICS:")
    for model, count in sorted(results['model_usage'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_processed) * 100
        print(f"   {model.upper()}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n🎚️ CONFIDENCE LEVEL DISTRIBUTION:")
    for level, count in sorted(results['confidence_levels'].items()):
        percentage = (count / total_processed) * 100
        print(f"   {level.upper()}: {count} videos ({percentage:.1f}%)")
    
    print(f"\n🔄 ROUTING DECISIONS:")
    for num_specialists, count in sorted(results['routing_decisions'].items()):
        percentage = (count / total_processed) * 100
        if num_specialists == 0:
            print(f"   Baseline Only: {count} videos ({percentage:.1f}%)")
        else:
            print(f"   {num_specialists} Specialists: {count} videos ({percentage:.1f}%)")
    
    # Detailed model performance
    print(f"\n🔬 DETAILED MODEL PERFORMANCE:")
    model_performance = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    for result_list in [results['real'], results['fake']]:
        for result in result_list:
            model = result['best_model']
            model_performance[model]['total'] += 1
            if result['correct']:
                model_performance[model]['correct'] += 1
            if result['confidence'] > 0:
                model_performance[model]['confidences'].append(result['confidence'])
    
    for model, stats in sorted(model_performance.items()):
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
            print(f"   {model.upper()}: {accuracy:.1f}% accuracy ({stats['correct']}/{stats['total']}) | Avg Conf: {avg_conf:.1%}")
    
    # Save detailed results
    detailed_results = {
        'summary': {
            'total_videos': total_processed,
            'overall_accuracy': overall_accuracy,
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time
        },
        'model_usage': dict(results['model_usage']),
        'confidence_levels': dict(results['confidence_levels']),
        'routing_decisions': dict(results['routing_decisions']),
        'detailed_results': results
    }
    
    # Save to file
    results_file = f"agentic_evaluation_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Final assessment
    print(f"\n🏆 FINAL ASSESSMENT")
    print("=" * 70)
    
    if overall_accuracy >= 80:
        print("🎉 EXCELLENT: System performance exceeds 80% accuracy!")
    elif overall_accuracy >= 70:
        print("✅ GOOD: System performance is above 70% accuracy")
    elif overall_accuracy >= 60:
        print("⚠️ FAIR: System performance is acceptable but could be improved")
    else:
        print("❌ POOR: System performance needs significant improvement")
    
    print(f"\n📈 KEY INSIGHTS:")
    
    # Best performing model
    best_model = max(model_performance.items(), key=lambda x: x[1]['correct'] if x[1]['total'] > 0 else 0)
    if best_model[1]['total'] > 0:
        best_accuracy = (best_model[1]['correct'] / best_model[1]['total']) * 100
        print(f"   🥇 Best Model: {best_model[0].upper()} ({best_accuracy:.1f}% accuracy)")
    
    # Specialist usage effectiveness
    specialist_videos = sum(count for specialists, count in results['routing_decisions'].items() if specialists > 0)
    if specialist_videos > 0:
        specialist_percentage = (specialist_videos / total_processed) * 100
        print(f"   🔬 Specialist Usage: {specialist_percentage:.1f}% of videos used specialist models")
    
    # Confidence distribution
    high_conf_videos = results['confidence_levels'].get('high', 0)
    if high_conf_videos > 0:
        high_conf_percentage = (high_conf_videos / total_processed) * 100
        print(f"   🎯 High Confidence: {high_conf_percentage:.1f}% of predictions were high confidence")
    
    return overall_accuracy >= 70

def main():
    """Main evaluation function"""
    success = evaluate_agentic_system()
    
    if success:
        print(f"\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        print("✅ E-Raksha Agentic System demonstrates good performance")
    else:
        print(f"\n⚠️ EVALUATION COMPLETED WITH CONCERNS")
        print("🔧 System may need tuning or additional training")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)