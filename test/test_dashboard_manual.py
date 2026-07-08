#!/usr/bin/env python3
"""
Manual test script for Model Manager Dashboard
Tests basic functionality without Playwright
"""

import requests
import time
import subprocess
import sys
import signal

def test_dashboard():
    """Test dashboard endpoints."""
    base_url = "http://127.0.0.1:8899"
    
    print("Testing MCP Dashboard endpoints...")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Dashboard loads
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/mcp", timeout=5)
        if response.status_code == 200 and "Model Browser" in response.text:
            print("✅ Test 1: Dashboard loads with Model Browser tab")
            tests_passed += 1
        else:
            print(f"❌ Test 1: Dashboard loaded but Model Browser tab not found")
            if response.status_code == 200:
                print(f"   Response contains: {response.text[:200]}...")
    except Exception as e:
        print(f"❌ Test 1: Dashboard failed to load: {e}")
    
    # Test 2: Model stats API
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/api/mcp/models/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'total_cached_models' in data:
                print(f"✅ Test 2: Model stats API works (Total models: {data['total_cached_models']})")
                tests_passed += 1
            else:
                print("❌ Test 2: Model stats API returned unexpected format")
        else:
            print(f"❌ Test 2: Model stats API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 2: Model stats API failed: {e}")
    
    # Test 3: Model search API
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/api/mcp/models/search?q=bert&limit=5", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                print(f"✅ Test 3: Model search API works (Found {len(data['results'])} results)")
                tests_passed += 1
            else:
                print("❌ Test 3: Model search API returned unexpected format")
        else:
            print(f"❌ Test 3: Model search API returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 3: Model search API failed: {e}")
    
    # Test 4: Static JS file serves
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/static/js/model-manager.js", timeout=5)
        if response.status_code == 200 and "ModelManager" in response.text:
            print("✅ Test 4: model-manager.js serves correctly")
            tests_passed += 1
        else:
            print(f"❌ Test 4: model-manager.js returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 4: model-manager.js failed: {e}")
    
    # Test 5: Static CSS file serves  
    tests_total += 1
    try:
        response = requests.get(f"{base_url}/static/css/dashboard.css", timeout=5)
        if response.status_code == 200 and "model-card" in response.text:
            print("✅ Test 5: dashboard.css with Model Browser styles serves correctly")
            tests_passed += 1
        else:
            print(f"❌ Test 5: dashboard.css returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Test 5: dashboard.css failed: {e}")
    
    print("=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    
    return tests_passed == tests_total

if __name__ == '__main__':
    print("Note: This script expects the MCP dashboard to be running on http://127.0.0.1:8899")
    print("Start it with: python3 -m ipfs_accelerate_py.mcp_dashboard")
    print()
    
    success = test_dashboard()
    sys.exit(0 if success else 1)
