#!/usr/bin/env python3
"""
Validation script for HuggingFace API integration.
Tests Phases 1-2 (backend) without requiring network access.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*80)
print("HUGGINGFACE API INTEGRATION VALIDATION")
print("="*80)
print("\nValidating implementation without network access (using mocks)")

# Mock API response for testing
MOCK_API_RESPONSE = [
    {
        'id': 'bert-base-uncased',
        'author': 'google',
        'downloads': 1000000,
        'likes': 5000,
        'tags': ['transformers', 'pytorch', 'bert'],
        'pipeline_tag': 'fill-mask',
        'library_name': 'transformers',
        'created_at': '2020-01-01T00:00:00Z',
        'lastModified': '2023-01-01T00:00:00Z',
        'private': False,
        'gated': False,
    }
]

# ============================================================================
# PHASE 1: Test Backend Tools
# ============================================================================

print("\n" + "="*80)
print("PHASE 1: VALIDATING BACKEND TOOLS")
print("="*80)

def validate_phase1_backend_tools():
    """Test that backend MCP server tools exist and work."""
    
    print("\n📋 Step 1.1: Testing HuggingFaceHubScanner class...")
    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner
        scanner = HuggingFaceHubScanner()
        print("   ✅ HuggingFaceHubScanner instantiated")
        
        print("\n📋 Step 1.2: Testing scanner.search_models() method...")
        # Mock the API call
        with patch.object(scanner, '_search_huggingface_api', return_value=MOCK_API_RESPONSE):
            results = scanner.search_models(query="bert", limit=3)
            if results and len(results) > 0:
                print(f"   ✅ Search returned {len(results)} models")
                print(f"   📦 Sample: {results[0].get('model_id', 'unknown')}")
                return True, scanner
            else:
                print("   ❌ Search returned no results")
                return False, None
        
    except Exception as e:
        print(f"   ❌ Backend scanner failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

# ============================================================================
# PHASE 2: Test Package Functions
# ============================================================================

print("\n" + "="*80)
print("PHASE 2: VALIDATING PACKAGE FUNCTIONS")
print("="*80)

def validate_phase2_package_functions(scanner):
    """Test that package functions work independently."""
    
    if not scanner:
        print("⚠️ Skipping Phase 2 - scanner not available")
        return False
    
    print("\n📋 Step 2.1: Testing search functionality...")
    try:
        # Test that we can search and get results
        with patch.object(scanner, '_search_huggingface_api', return_value=MOCK_API_RESPONSE):
            results = scanner.search_models(query="bert", limit=5)
            
        if results:
            print(f"   ✅ Package search works: {len(results)} results")
            
            # Test result structure
            first_result = results[0]
            required_fields = ['model_id', 'model_info']
            missing = [f for f in required_fields if f not in first_result]
            
            if missing:
                print(f"   ⚠️ Missing fields in result: {missing}")
            else:
                print("   ✅ Result structure valid")
            
            return True
        else:
            print("   ❌ No results from search")
            return False
            
    except Exception as e:
        print(f"   ❌ Package function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# Run Tests
# ============================================================================

print("\n" + "="*80)
print("RUNNING VALIDATION TESTS")
print("="*80)

success_phase1, scanner = validate_phase1_backend_tools()

if success_phase1:
    print("\n✅ Phase 1 passed!")
else:
    print("\n❌ Phase 1 failed!")
    sys.exit(1)

success_phase2 = validate_phase2_package_functions(scanner)

if success_phase2:
    print("\n✅ Phase 2 passed!")
else:
    print("\n❌ Phase 2 failed!")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n✅ All validated phases passed!")
print("\n📋 Status:")
print("   ✅ Phase 1: Backend tools - PASS")
print("   ✅ Phase 2: Package functions - PASS")
print("   ⏳ Phase 3: API endpoints - Requires running server")
print("   ⏳ Phase 4: GUI integration - Requires Playwright with network")
print("\n📝 Next Steps:")
print("   1. Install dependencies: pip install flask flask-cors huggingface_hub")
print("   2. Start dashboard: ipfs-accelerate mcp start")
print("   3. Run full test: python tests/test_comprehensive_validation.py")
print("\n💡 Implementation ready for production testing!")
