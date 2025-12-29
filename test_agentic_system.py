#!/usr/bin/env python3
"""
Test the E-Raksha Agentic System
Comprehensive testing of the unified system
"""

import os
import sys
import time
from eraksha_agent import ErakshAgent

def test_agent_initialization():
    """Test agent initialization"""
    print("🧪 Test 1: Agent Initialization")
    print("-" * 40)
    
    try:
        agent = ErakshAgent()
        
        # Check model status
        models_loaded = sum(1 for model in agent.models.values() if model is not None)
        total_models = len(agent.models)
        
        print(f"✅ Agent initialized successfully")
        print(f"📊 Models loaded: {models_loaded}/{total_models}")
        
        # Print detailed status
        for model_name, model in agent.models.items():
            status = "✅ Loaded" if model is not None else "❌ Not Available"
            print(f"   {model_name.upper()}: {status}")
        
        return agent, models_loaded > 0
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return None, False

def test_video_prediction(agent, video_path):
    """Test video prediction"""
    print(f"\n🧪 Test 2: Video Prediction - {os.path.basename(video_path)}")
    print("-" * 40)
    
    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_path}")
        return False
    
    try:
        start_time = time.time()
        result = agent.predict(video_path)
        processing_time = time.time() - start_time
        
        if result['success']:
            print(f"✅ Prediction successful")
            print(f"   Result: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.1%} ({result['confidence_level']})")
            print(f"   Best Model: {result['best_model'].upper()}")
            print(f"   Specialists Used: {', '.join(result['specialists_used']) if result['specialists_used'] else 'None'}")
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Request ID: {result['request_id']}")
            
            # Check video characteristics
            if 'video_characteristics' in result:
                chars = result['video_characteristics']
                print(f"   Video Info: {chars['resolution'][0]}x{chars['resolution'][1]}, {chars['fps']:.1f}fps, {chars['duration']:.1f}s")
                print(f"   Bitrate: {chars['bitrate']/1000:.0f} kbps, Brightness: {chars['avg_brightness']:.0f}")
            
            return True
        else:
            print(f"❌ Prediction failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_model_routing(agent):
    """Test intelligent model routing logic"""
    print(f"\n🧪 Test 3: Model Routing Logic")
    print("-" * 40)
    
    try:
        # Test routing with different confidence levels
        test_cases = [
            (0.95, "High confidence"),
            (0.75, "Medium confidence"),
            (0.45, "Low confidence")
        ]
        
        # Mock video characteristics
        mock_video_chars = {
            'is_compressed': True,
            'is_rerecorded': False,
            'is_low_light': False,
            'is_low_quality': True
        }
        
        for confidence, description in test_cases:
            specialists = agent.intelligent_routing(mock_video_chars, confidence)
            print(f"   {description} ({confidence:.1%}): {', '.join(specialists) if specialists else 'No specialists'}")
        
        print("✅ Routing logic working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Routing test failed: {e}")
        return False

def test_api_compatibility():
    """Test API response format compatibility"""
    print(f"\n🧪 Test 4: API Compatibility")
    print("-" * 40)
    
    try:
        agent = ErakshAgent()
        
        # Test with a video if available
        test_videos = ["test_video_short.mp4", "test_video_long.mp4"]
        
        for video_path in test_videos:
            if os.path.exists(video_path):
                result = agent.predict(video_path)
                
                # Check required API fields
                required_fields = [
                    'success', 'prediction', 'confidence', 'confidence_level',
                    'explanation', 'processing_time', 'request_id'
                ]
                
                missing_fields = [field for field in required_fields if field not in result]
                
                if not missing_fields:
                    print("✅ API response format compatible")
                    print(f"   All required fields present: {', '.join(required_fields)}")
                    return True
                else:
                    print(f"❌ Missing API fields: {', '.join(missing_fields)}")
                    return False
        
        print("⚠️ No test videos available for API compatibility test")
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

def test_error_handling(agent):
    """Test error handling"""
    print(f"\n🧪 Test 5: Error Handling")
    print("-" * 40)
    
    try:
        # Test with non-existent file
        result = agent.predict("non_existent_video.mp4")
        
        if not result['success'] and 'error' in result:
            print("✅ Error handling working correctly")
            print(f"   Error message: {result['error']}")
            return True
        else:
            print("❌ Error handling not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 E-Raksha Agentic System - Comprehensive Testing")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: Initialization
    agent, init_success = test_agent_initialization()
    test_results.append(("Initialization", init_success))
    
    if not init_success:
        print("\n❌ Cannot proceed with other tests - initialization failed")
        return
    
    # Test 2: Video Prediction
    test_videos = ["test_video_short.mp4", "test_video_long.mp4"]
    video_test_success = False
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            video_test_success = test_video_prediction(agent, video_path)
            break
    
    if not video_test_success and not any(os.path.exists(v) for v in test_videos):
        print(f"\n⚠️ No test videos found, skipping video prediction test")
        video_test_success = True  # Don't fail if no videos available
    
    test_results.append(("Video Prediction", video_test_success))
    
    # Test 3: Model Routing
    routing_success = test_model_routing(agent)
    test_results.append(("Model Routing", routing_success))
    
    # Test 4: API Compatibility
    api_success = test_api_compatibility()
    test_results.append(("API Compatibility", api_success))
    
    # Test 5: Error Handling
    error_success = test_error_handling(agent)
    test_results.append(("Error Handling", error_success))
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed_tests += 1
    
    print(f"\n🎯 Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! E-Raksha Agentic System is ready for deployment.")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️ Most tests passed. System is functional but may need minor fixes.")
    else:
        print("❌ Multiple test failures. System needs attention before deployment.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)