#!/usr/bin/env python3
"""
Test the E-Raksha API with sample videos
"""

import requests
import json
import os

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("API Health Check:")
            print(f"  Status: {data['status']}")
            print(f"  Model Loaded: {data['model_loaded']}")
            print(f"  Device: {data['device']}")
            print(f"  Timestamp: {data['timestamp']}")
            return True
        else:
            print(f"API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"API not accessible: {e}")
        return False

def test_video_prediction(video_path):
    """Test video prediction"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    print(f"\nTesting video: {video_path}")
    print(f"File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {'file': (os.path.basename(video_path), video_file, 'video/mp4')}
            
            print("Sending request to API...")
            response = requests.post("http://localhost:8000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("Prediction Results:")
                print(f"  Prediction: {result['prediction'].upper()}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Faces Analyzed: {result['faces_analyzed']}")
                
                if 'fake_votes' in result:
                    print(f"  Fake Votes: {result['fake_votes']}/{result['total_votes']}")
                
                if 'model_info' in result:
                    model_info = result['model_info']
                    print(f"  Model: {model_info.get('architecture', 'Unknown')}")
                    print(f"  Model Accuracy: {model_info.get('accuracy', 'Unknown')}")
                
                print(f"  Processing Time: {result.get('timestamp', 'Unknown')}")
                return True
            else:
                print(f"API request failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
    except Exception as e:
        print(f"Error testing video: {e}")
        return False

def main():
    print("E-Raksha API Test Suite")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        print("API is not running. Please start the backend first.")
        return
    
    print("\n" + "=" * 50)
    
    # Test videos
    test_videos = [
        "test_video_short.mp4",
        "test_video_long.mp4"
    ]
    
    success_count = 0
    for video in test_videos:
        if test_video_prediction(video):
            success_count += 1
        print("-" * 30)
    
    print(f"\nTest Summary:")
    print(f"  Total tests: {len(test_videos)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(test_videos) - success_count}")
    
    if success_count == len(test_videos):
        print("\n✅ All tests passed! Your E-Raksha API is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()