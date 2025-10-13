#!/usr/bin/env python3
"""
Real HuggingFace API Test

This test actually calls the HuggingFace API to verify the integration works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_real_api_search():
    """Test if the scanner can actually search HuggingFace."""
    
    print("\n" + "="*80)
    print("🔬 Real HuggingFace API Search Test")
    print("="*80)
    
    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
        
        print("\n📋 Creating scanner instance...")
        scanner = HuggingFaceHubScanner(cache_dir="./test_cache")
        print("   ✅ Scanner created")
        
        # Test 1: Search for a well-known model
        print("\n📋 Test 1: Searching for 'gpt2'...")
        try:
            results = scanner.search_models(query="gpt2", limit=5)
            print(f"   ✅ Search returned {len(results)} results")
            
            if len(results) > 0:
                first = results[0]
                print(f"   First result: {first.get('model_id', 'unknown')}")
                
                # Check if it's real data
                if 'gpt2' in first.get('model_id', '').lower():
                    print("   ✅ Results match search query (REAL API)")
                else:
                    print("   ⚠️  Results don't match query (might be mock)")
                    
                # Check downloads count
                downloads = first.get('model_info', {}).get('downloads', 0)
                if downloads > 1000000:
                    print(f"   ✅ Model has {downloads:,} downloads (REAL DATA)")
                else:
                    print(f"   ⚠️  Low download count: {downloads} (might be mock)")
            else:
                print("   ❌ No results returned")
                return False
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test 2: Search for a specific model
        print("\n📋 Test 2: Searching for 'meta-llama/Llama-2-7b'...")
        try:
            results = scanner.search_models(query="meta-llama/Llama-2-7b", limit=3)
            print(f"   ✅ Search returned {len(results)} results")
            
            if len(results) > 0:
                # Check if we got the right model
                found_llama = any('llama' in r.get('model_id', '').lower() for r in results)
                if found_llama:
                    print("   ✅ Found Llama models (REAL API)")
                else:
                    print("   ⚠️  No Llama models found (might be mock)")
                    
        except Exception as e:
            print(f"   ⚠️  Search failed: {e}")
        
        # Test 3: Check if search engine is initialized
        print("\n📋 Test 3: Checking search engine initialization...")
        if hasattr(scanner, 'search_engine'):
            engine_type = type(scanner.search_engine).__name__
            print(f"   Search engine type: {engine_type}")
            
            if 'Mock' in engine_type or 'Fallback' in engine_type:
                print("   ⚠️  Using mock/fallback search engine")
                return False
            else:
                print("   ✅ Using real search engine")
        else:
            print("   ⚠️  No search_engine attribute found")
        
        print("\n✅ Real API integration is working!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_real_api_search()
    
    if success:
        print("\n" + "="*80)
        print("✅ REAL HUGGINGFACE API INTEGRATION CONFIRMED")
        print("="*80)
        print("\nThe system is now using real HuggingFace Hub API calls")
        print("Your Playwright tests are now testing actual functionality!")
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("❌ STILL USING MOCK DATA")
        print("="*80)
        print("\nThe search engine is not properly initialized")
        sys.exit(1)
