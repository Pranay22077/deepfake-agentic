#!/usr/bin/env python3
"""
Comprehensive Model Evaluation
Test the current model on 100 real test videos (50 real + 50 fake)
Calculate accuracy, precision, recall, and detailed statistics
"""

import os
import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluate model performance on test dataset"""
    
    def __init__(self, test_data_path="test-data/test-data/raw", api_url="http://localhost:8000"):
        self.test_data_path = Path(test_data_path)
        self.api_url = api_url
        self.results = []
        
    def find_test_videos(self):
        """Find all test videos and their labels"""
        videos = []
        
        # Real videos
        real_path = self.test_data_path / "real"
        if real_path.exists():
            for video_file in real_path.glob("*.mp4"):
                videos.append({
                    'path': str(video_file),
                    'filename': video_file.name,
                    'true_label': 'real',
                    'true_class': 0  # 0 = real
                })
        
        # Fake videos  
        fake_path = self.test_data_path / "fake"
        if fake_path.exists():
            for video_file in fake_path.glob("*.mp4"):
                videos.append({
                    'path': str(video_file),
                    'filename': video_file.name,
                    'true_label': 'fake',
                    'true_class': 1  # 1 = fake
                })
        
        print(f"Found {len(videos)} test videos:")
        real_count = sum(1 for v in videos if v['true_label'] == 'real')
        fake_count = sum(1 for v in videos if v['true_label'] == 'fake')
        print(f"  - Real videos: {real_count}")
        print(f"  - Fake videos: {fake_count}")
        
        return videos
    
    def test_api_connection(self):
        """Test if API is running"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] API is running - Model loaded: {data.get('model_loaded', False)}")
                return True
            else:
                print(f"[ERROR] API returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] API connection failed: {e}")
            return False
    
    def predict_video(self, video_path):
        """Get prediction for a single video"""
        try:
            with open(video_path, 'rb') as video_file:
                files = {'file': (os.path.basename(video_path), video_file, 'video/mp4')}
                
                response = requests.post(
                    f"{self.api_url}/predict", 
                    files=files,
                    timeout=30  # 30 second timeout per video
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'prediction': result.get('prediction', 'unknown'),
                        'confidence': result.get('confidence', 0.0),
                        'faces_analyzed': result.get('faces_analyzed', 0),
                        'fake_votes': result.get('fake_votes', 0),
                        'total_votes': result.get('total_votes', 0),
                        'analysis': result.get('analysis', {}),
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text[:200]}"
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_evaluation(self):
        """Run complete evaluation on all test videos"""
        print("[TEST] Starting Model Evaluation")
        print("=" * 60)
        
        # Check API connection
        if not self.test_api_connection():
            print("[ERROR] Cannot connect to API. Make sure backend is running on port 8000")
            return None
        
        # Find test videos
        videos = self.find_test_videos()
        if not videos:
            print("[ERROR] No test videos found. Check the path: test data/test data/raw/")
            return None
        
        print(f"\n[RUN] Testing {len(videos)} videos...")
        print("=" * 60)
        
        # Test each video
        results = []
        failed_videos = []
        
        for i, video in enumerate(tqdm(videos, desc="Testing videos")):
            print(f"\n[{i+1}/{len(videos)}] Testing: {video['filename']}")
            print(f"True label: {video['true_label']}")
            
            # Get prediction
            prediction = self.predict_video(video['path'])
            
            if prediction['success']:
                # Convert prediction to class
                pred_class = 0 if prediction['prediction'] == 'real' else 1
                
                result = {
                    'filename': video['filename'],
                    'true_label': video['true_label'],
                    'true_class': video['true_class'],
                    'predicted_label': prediction['prediction'],
                    'predicted_class': pred_class,
                    'confidence': prediction['confidence'],
                    'faces_analyzed': prediction['faces_analyzed'],
                    'fake_votes': prediction['fake_votes'],
                    'total_votes': prediction['total_votes'],
                    'correct': video['true_class'] == pred_class
                }
                
                results.append(result)
                
                # Print result
                status = "[OK] CORRECT" if result['correct'] else "[X] WRONG"
                print(f"Predicted: {prediction['prediction']} ({prediction['confidence']:.3f}) | {status}")
                
            else:
                failed_videos.append({
                    'filename': video['filename'],
                    'error': prediction['error']
                })
                print(f"[ERROR] FAILED: {prediction['error']}")
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        print(f"\n[STATS] Evaluation Complete!")
        print(f"Successful predictions: {len(results)}")
        print(f"Failed predictions: {len(failed_videos)}")
        
        # Save results
        self.results = results
        self.failed_videos = failed_videos
        
        return results
    
    def calculate_metrics(self):
        """Calculate detailed performance metrics"""
        if not self.results:
            print("[ERROR] No results to analyze")
            return None
        
        print("\n[METRICS] PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Extract predictions and true labels
        y_true = [r['true_class'] for r in self.results]
        y_pred = [r['predicted_class'] for r in self.results]
        confidences = [r['confidence'] for r in self.results]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Detailed breakdown
        correct_predictions = sum(1 for r in self.results if r['correct'])
        total_predictions = len(self.results)
        
        # By class analysis
        real_videos = [r for r in self.results if r['true_label'] == 'real']
        fake_videos = [r for r in self.results if r['true_label'] == 'fake']
        
        real_correct = sum(1 for r in real_videos if r['correct'])
        fake_correct = sum(1 for r in fake_videos if r['correct'])
        
        real_accuracy = real_correct / len(real_videos) if real_videos else 0
        fake_accuracy = fake_correct / len(fake_videos) if fake_videos else 0
        
        # Confidence analysis
        avg_confidence = np.mean(confidences)
        correct_confidences = [r['confidence'] for r in self.results if r['correct']]
        wrong_confidences = [r['confidence'] for r in self.results if not r['correct']]
        
        avg_correct_conf = np.mean(correct_confidences) if correct_confidences else 0
        avg_wrong_conf = np.mean(wrong_confidences) if wrong_confidences else 0
        
        # Print results
        print(f"[STATS] OVERALL METRICS:")
        print(f"  Accuracy:  {accuracy:.3f} ({correct_predictions}/{total_predictions})")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1-Score:  {f1:.3f}")
        
        print(f"\n[MATRIX] CONFUSION MATRIX:")
        print(f"                Predicted")
        print(f"                Real  Fake")
        print(f"  Actual Real   {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"  Actual Fake   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        print(f"\n[CLASS] CLASS-WISE ACCURACY:")
        print(f"  Real videos: {real_accuracy:.3f} ({real_correct}/{len(real_videos)})")
        print(f"  Fake videos: {fake_accuracy:.3f} ({fake_correct}/{len(fake_videos)})")
        
        print(f"\n[CONFIDENCE] CONFIDENCE ANALYSIS:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Correct predictions: {avg_correct_conf:.3f}")
        print(f"  Wrong predictions:   {avg_wrong_conf:.3f}")
        
        # Model behavior analysis
        print(f"\n[BEHAVIOR] MODEL BEHAVIOR:")
        real_as_real = sum(1 for r in real_videos if r['predicted_label'] == 'real')
        real_as_fake = sum(1 for r in real_videos if r['predicted_label'] == 'fake')
        fake_as_real = sum(1 for r in fake_videos if r['predicted_label'] == 'real')
        fake_as_fake = sum(1 for r in fake_videos if r['predicted_label'] == 'fake')
        
        print(f"  Real videos -> Predicted Real: {real_as_real}")
        print(f"  Real videos -> Predicted Fake: {real_as_fake}")
        print(f"  Fake videos -> Predicted Real: {fake_as_real}")
        print(f"  Fake videos -> Predicted Fake: {fake_as_fake}")
        
        # Save detailed results
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'avg_confidence': avg_confidence,
            'avg_correct_confidence': avg_correct_conf,
            'avg_wrong_confidence': avg_wrong_conf,
            'total_tested': total_predictions,
            'correct_predictions': correct_predictions,
            'failed_predictions': len(self.failed_videos)
        }
        
        return metrics
    
    def save_results(self, filename="evaluation_results.json"):
        """Save detailed results to file"""
        if not self.results:
            return
        
        output = {
            'evaluation_summary': {
                'total_videos': len(self.results),
                'failed_videos': len(self.failed_videos),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': self.results,
            'failed_videos': self.failed_videos,
            'metrics': self.calculate_metrics()
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[SAVED] Results saved to: {filename}")
    
    def print_summary(self):
        """Print final summary"""
        if not self.results:
            return
        
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("[SUMMARY] FINAL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model Accuracy: {metrics['accuracy']:.1%}")
        print(f"Real Video Accuracy: {metrics['real_accuracy']:.1%}")
        print(f"Fake Video Accuracy: {metrics['fake_accuracy']:.1%}")
        print(f"Average Confidence: {metrics['avg_confidence']:.3f}")
        print(f"Videos Tested: {metrics['total_tested']}")
        print(f"Failed Tests: {metrics['failed_predictions']}")
        
        # Model assessment
        if metrics['accuracy'] > 0.8:
            print("[EXCELLENT] Model performs very well!")
        elif metrics['accuracy'] > 0.6:
            print("[GOOD] Model performs reasonably well")
        elif metrics['accuracy'] > 0.5:
            print("[FAIR] Model is better than random guessing")
        else:
            print("[POOR] Model needs significant improvement")
        
        print("=" * 60)

def main():
    """Run the complete evaluation"""
    print("[TEST] E-Raksha Model Evaluation")
    print("Testing model on real test dataset")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        # Calculate and display metrics
        evaluator.calculate_metrics()
        
        # Save results
        evaluator.save_results()
        
        # Print final summary
        evaluator.print_summary()
    else:
        print("[ERROR] Evaluation failed. Check the setup and try again.")

if __name__ == "__main__":
    main()
