#!/usr/bin/env python3
"""
Test script for the Model Discovery system.
"""

import sys
import os
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_manager():
    """Test basic model manager functionality."""
    print("Testing Model Manager...")
    try:
        from ipfs_accelerate_py.model_manager import ModelManager, BanditModelRecommender
        
        # Test model manager
        manager = ModelManager()
        print(f"✅ Model Manager initialized")
        
        # Test bandit recommender
        bandit = BanditModelRecommender()
        print(f"✅ Bandit Recommender initialized")
        
        return True
    except Exception as e:
        print(f"❌ Model Manager test failed: {e}")
        return False

def test_hf_scanner():
    """Test HuggingFace Hub scanner."""
    print("Testing HuggingFace Hub Scanner...")
    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
        
        # Test scanner initialization
        scanner = HuggingFaceHubScanner(cache_dir="./test_cache")
        print(f"✅ HuggingFace Hub Scanner initialized")
        
        # Test small scan
        print("Running small test scan (limit: 5)...")
        results = scanner.scan_all_models(limit=5)
        print(f"✅ Test scan completed: {results['models_processed']} models processed")
        
        # Test search
        if scanner.model_cache:
            search_results = scanner.search_models("text", limit=3)
            print(f"✅ Search completed: {len(search_results)} results")
        
        return True
    except Exception as e:
        print(f"❌ HuggingFace Scanner test failed: {e}")
        return False

def test_mcp_dashboard():
    """Test MCP Dashboard."""
    print("Testing MCP Dashboard...")
    try:
        from ipfs_accelerate_py.mcp_dashboard import MCPDashboard
        
        # Test dashboard initialization
        dashboard = MCPDashboard(port=8901)  # Use different port
        print(f"✅ MCP Dashboard initialized")
        
        # Test scanner access
        scanner = dashboard._get_hub_scanner()
        if scanner:
            print(f"✅ Hub scanner accessible from dashboard")
        else:
            print("⚠️  Hub scanner not available (imports may be missing)")
        
        return True
    except Exception as e:
        print(f"❌ MCP Dashboard test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Model Discovery System")
    print("=" * 50)
    
    tests = [
        test_model_manager,
        test_hf_scanner,
        test_mcp_dashboard
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
            print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())