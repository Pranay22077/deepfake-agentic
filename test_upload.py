#!/usr/bin/env python3
"""
Test the agentic API with a real video upload
"""

import requests
import os

def test_api():
    """Test the API with a video file"""
    
    # Check if test video exists
    test_videos = ["test_video_short.mp4", "test_video_long.mp4"]
    test_video = None
    
    for video in test_videos:
        if os.path.exists(video):
            test_video = video
            break
    
    if not test_video:
        print("❌ No test video found")
        return
    
    print(f"🎬 Testing API with {test_video}")
    
    try:
        # Test health endpoint
        health_response = requests.get('http://localhost:8000/health')
        print(f"✅ Health check: {health_response.status_code}")
        
        # Test prediction endpoint
        with open(test_video, 'rb') as f:
            files = {'file': (test_video, f, 'video/mp4')}
            
            print("📤 Uploading video...")
            response = requests.post('http://localhost:8000/predict', files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction successful!")
                print(f"   Result: {result['prediction']}")
                print(f"   Confidence: {result['confidence']}%")
                print(f"   Best Model: {result['details']['best_model']}")
                print(f"   Processing Time: {result['details']['processing_time']}s")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()