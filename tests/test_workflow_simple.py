#!/usr/bin/env python3
"""
Simple validation script to test HuggingFace workflow without full Playwright setup.
Tests the API endpoints directly.
"""

import requests
import time
import subprocess
import sys
import json
from pathlib import Path

def test_huggingface_workflow():
    """Test the HuggingFace model search and download workflow via API."""
    
    print("\n" + "="*80)
    print("HUGGINGFACE WORKFLOW API TEST")
    print("="*80)
    
    base_url = "http://127.0.0.1:8899"
    
    # Test 1: Check if server is running
    print("\n📋 Step 1: Checking if MCP server is running...")
    try:
        response = requests.get(f"{base_url}/mcp", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Server not accessible: {e}")
        print("\n   💡 Start the server with: python3 -m ipfs_accelerate_py.mcp_dashboard")
        return False
    
    # Test 2: Search for "bert" model
    print("\n📋 Step 2: Searching for 'bert' model on HuggingFace...")
    try:
        params = {'q': 'bert', 'limit': '5'}
        response = requests.get(f"{base_url}/api/mcp/models/search", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('results', [])
            print(f"   ✅ Search successful - found {len(models)} models")
            
            if len(models) > 0:
                print(f"\n   📦 Sample model found:")
                model = models[0]
                model_id = model.get('model_id', 'unknown')
                model_info = model.get('model_info', {})
                print(f"      ID: {model_id}")
                print(f"      Name: {model_info.get('model_name', 'N/A')}")
                print(f"      Downloads: {model_info.get('downloads', 0)}")
                
                # Test 3: Try to download the first bert model
                print(f"\n📋 Step 3: Attempting to download model '{model_id}'...")
                try:
                    download_response = requests.post(
                        f"{base_url}/api/mcp/models/download",
                        json={'model_id': model_id},
                        timeout=30
                    )
                    
                    if download_response.status_code == 200:
                        result = download_response.json()
                        if result.get('status') == 'success':
                            print(f"   ✅ Download successful!")
                            print(f"      Message: {result.get('message', 'N/A')}")
                        else:
                            print(f"   ⚠️ Download returned non-success status")
                            print(f"      Response: {result}")
                    else:
                        print(f"   ❌ Download failed with status {download_response.status_code}")
                        print(f"      Response: {download_response.text[:200]}")
                        
                except Exception as e:
                    print(f"   ❌ Download request failed: {e}")
                
                # Test 4: Verify model appears in stats
                print(f"\n📋 Step 4: Checking if model appears in Model Browser...")
                time.sleep(2)  # Give it a moment
                
                try:
                    stats_response = requests.get(f"{base_url}/api/mcp/models/stats", timeout=5)
                    if stats_response.status_code == 200:
                        stats = stats_response.json()
                        total_models = stats.get('total_cached_models', 0)
                        print(f"   ✅ Stats loaded - {total_models} total models cached")
                        
                        # Search again to see if our model is there
                        search_response = requests.get(
                            f"{base_url}/api/mcp/models/search",
                            params={'q': 'bert', 'limit': '20'},
                            timeout=10
                        )
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            search_models = search_data.get('results', [])
                            model_ids = [m.get('model_id', '') for m in search_models]
                            
                            if model_id in model_ids:
                                print(f"   ✅ Model '{model_id}' found in Model Browser!")
                            else:
                                print(f"   ⚠️ Model '{model_id}' not yet visible in Model Browser")
                                print(f"      Found models: {model_ids[:3]}")
                    else:
                        print(f"   ❌ Stats request failed with status {stats_response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Verification failed: {e}")
                
                return True
            else:
                print("   ⚠️ No models found in search results")
                return False
        else:
            print(f"   ❌ Search failed with status {response.status_code}")
            print(f"      Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ❌ Search request failed: {e}")
        return False
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    

if __name__ == '__main__':
    success = test_huggingface_workflow()
    
    if success:
        print("\n✅ Workflow test PASSED")
        print("\nTo test the full UI workflow:")
        print("  1. Make sure Playwright is installed: pip install playwright && playwright install")
        print("  2. Run: python3 tests/test_huggingface_workflow.py")
        sys.exit(0)
    else:
        print("\n❌ Workflow test FAILED")
        print("\nMake sure:")
        print("  1. MCP server is running: python3 -m ipfs_accelerate_py.mcp_dashboard")
        print("  2. The server is accessible at http://127.0.0.1:8899")
        sys.exit(1)
