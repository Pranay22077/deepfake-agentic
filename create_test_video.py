#!/usr/bin/env python3
"""
Create a simple test video for testing the deepfake detection API
"""

import cv2
import numpy as np
import os

def create_test_video(filename="test_video.mp4", duration=3, fps=30):
    """Create a simple test video with a moving circle (simulates a face)"""
    
    # Video properties
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    print(f"Creating test video: {filename}")
    print(f"Duration: {duration}s, FPS: {fps}, Frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create a frame with a moving circle (simulates a face)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(height):
            frame[y, :] = [50 + y//4, 100 + y//8, 150 + y//6]
        
        # Moving circle (simulates a face)
        center_x = int(width/2 + 100 * np.sin(frame_num * 0.1))
        center_y = int(height/2 + 50 * np.cos(frame_num * 0.1))
        radius = 80
        
        # Draw face-like circle
        cv2.circle(frame, (center_x, center_y), radius, (200, 180, 160), -1)
        
        # Add some facial features
        # Eyes
        eye1_x, eye1_y = center_x - 25, center_y - 20
        eye2_x, eye2_y = center_x + 25, center_y - 20
        cv2.circle(frame, (eye1_x, eye1_y), 8, (50, 50, 50), -1)
        cv2.circle(frame, (eye2_x, eye2_y), 8, (50, 50, 50), -1)
        
        # Mouth
        mouth_x, mouth_y = center_x, center_y + 30
        cv2.ellipse(frame, (mouth_x, mouth_y), (20, 10), 0, 0, 180, (100, 50, 50), -1)
        
        # Add some text
        cv2.putText(frame, f"Test Video - Frame {frame_num+1}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test video created: {filename}")
    print(f"File size: {os.path.getsize(filename) / 1024 / 1024:.2f} MB")
    
    return filename

if __name__ == "__main__":
    # Create test videos
    test_files = []
    
    # Create a short test video
    test_files.append(create_test_video("test_video_short.mp4", duration=2, fps=15))
    
    # Create a longer test video
    test_files.append(create_test_video("test_video_long.mp4", duration=5, fps=24))
    
    print("\nTest videos created:")
    for file in test_files:
        print(f"  - {file}")
    
    print("\nYou can now test these videos with the E-Raksha API!")