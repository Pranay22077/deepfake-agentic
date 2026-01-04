#!/usr/bin/env python3
"""
Docker Deployment Test Script
Tests if E-Raksha is running correctly in Docker
"""

import requests
import time
import sys

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("[OK] Backend Health Check: PASSED")
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   Device: {data.get('device', 'unknown')}")
            return True
        else:
            print(f"[ERROR] Backend Health Check: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"[ERROR] Backend Health Check: FAILED ({e})")
        return False

def test_frontend():
    """Test frontend accessibility"""
    try:
        response = requests.get("http://localhost:3001", timeout=10)
        if response.status_code == 200:
            print("[OK] Frontend Access: PASSED")
            return True
        else:
            print(f"[ERROR] Frontend Access: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"[ERROR] Frontend Access: FAILED ({e})")
        return False

def test_api_docs():
    """Test API documentation"""
    try:
        response = requests.get("http://localhost:8000/docs", timeout=10)
        if response.status_code == 200:
            print("[OK] API Documentation: PASSED")
            return True
        else:
            print(f"[ERROR] API Documentation: FAILED (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"[ERROR] API Documentation: FAILED ({e})")
        return False

def main():
    """Run all tests"""
    print("[TEST] E-Raksha Docker Deployment Test")
    print("=" * 40)
    
    print("[WAIT] Waiting for services to start...")
    time.sleep(5)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("Frontend Access", test_frontend),
        ("API Documentation", test_api_docs)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[CHECK] Testing {test_name}...")
        if test_func():
            passed += 1
        time.sleep(1)
    
    print("\n" + "=" * 40)
    print(f"[STATS] Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[DONE] All tests passed! E-Raksha is running correctly.")
        print("\n[ACCESS] Access the application:")
        print("   Frontend: http://localhost:3001")
        print("   Backend API: http://localhost:8000")
        print("   API Docs: http://localhost:8000/docs")
        return 0
    else:
        print("[ERROR] Some tests failed. Check the Docker logs:")
        print("   docker-compose logs")
        return 1

if __name__ == "__main__":
    sys.exit(main())
