#!/usr/bin/env python3
"""
HuggingFace Integration Validation Test

This test checks whether the system has real HuggingFace Hub integration
or if it's using only the mock/fallback scanner.

Usage:
    python3 tests/test_huggingface_integration_check.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_scanner_availability():
    """Test if real HuggingFace scanners are available."""

    print("\n" + "=" * 80)
    print("🔍 HuggingFace Integration Validation")
    print("=" * 80)

    results = {
        "enhanced_scanner": False,
        "standard_scanner": False,
        "huggingface_hub": False,
        "using_mock": False,
    }

    # Test 1: Enhanced Scanner
    print("\n📋 Test 1: Checking Enhanced HuggingFace Scanner...")
    try:
        from ipfs_accelerate_py.enhanced_huggingface_scanner import EnhancedHuggingFaceScanner

        print("   ✅ EnhancedHuggingFaceScanner is available")
        results["enhanced_scanner"] = True

        # Try to instantiate
        try:
            scanner = EnhancedHuggingFaceScanner(cache_dir="./test_cache")
            print("   ✅ EnhancedHuggingFaceScanner can be instantiated")

            # Check if it has required methods
            required_methods = ["search_models", "download_model", "scan_all_models"]
            for method in required_methods:
                if hasattr(scanner, method):
                    print(f"   ✅ Has method: {method}")
                else:
                    print(f"   ❌ Missing method: {method}")
                    results["enhanced_scanner"] = False
        except Exception as e:
            print(f"   ⚠️  Cannot instantiate: {e}")
            results["enhanced_scanner"] = False

    except ImportError as e:
        print(f"   ❌ Not available: {e}")

    # Test 2: Standard Scanner
    print("\n📋 Test 2: Checking Standard HuggingFace Scanner...")
    try:
        from ipfs_accelerate_py.huggingface_hub_scanner import HuggingFaceHubScanner

        print("   ✅ HuggingFaceHubScanner is available")
        results["standard_scanner"] = True

        # Try to instantiate
        try:
            scanner = HuggingFaceHubScanner(cache_dir="./test_cache")
            print("   ✅ HuggingFaceHubScanner can be instantiated")

            # Check if it has required methods
            required_methods = ["search_models", "download_model"]
            for method in required_methods:
                if hasattr(scanner, method):
                    print(f"   ✅ Has method: {method}")
                else:
                    print(f"   ❌ Missing method: {method}")
                    results["standard_scanner"] = False
        except Exception as e:
            print(f"   ⚠️  Cannot instantiate: {e}")
            results["standard_scanner"] = False

    except ImportError as e:
        print(f"   ❌ Not available: {e}")

    # Test 3: HuggingFace Hub Library
    print("\n📋 Test 3: Checking HuggingFace Hub Library...")
    try:
        from huggingface_hub import HfApi

        print("   ✅ huggingface_hub library is installed")
        results["huggingface_hub"] = True

        # Try to access API
        try:
            api = HfApi()
            print("   ✅ HfApi can be instantiated")

            # Test API endpoint (no auth required)
            try:
                # This is a lightweight call to check if API is accessible
                model_info = api.model_info("gpt2")
                print(f"   ✅ API accessible (test call successful)")
                print(f"   Model: {model_info.modelId}")
                print(f"   Downloads: {model_info.downloads}")
            except Exception as e:
                print(f"   ⚠️  API call failed: {e}")
                print("   (This might be a network issue, not a code issue)")
        except Exception as e:
            print(f"   ⚠️  Cannot instantiate HfApi: {e}")

    except ImportError as e:
        print(f"   ❌ Not installed: {e}")
        print("   Install with: pip install huggingface_hub")

    # Test 4: Check what dashboard actually uses
    print("\n📋 Test 4: Checking what MCP Dashboard uses...")
    try:
        from ipfs_accelerate_py.mcp_dashboard import MCPDashboard

        dashboard = MCPDashboard(port=9999, host="127.0.0.1")  # Different port to avoid conflicts

        scanner = dashboard._get_hub_scanner()
        scanner_type = type(scanner).__name__

        print(f"   Scanner type: {scanner_type}")

        if scanner_type == "WorkingMockScanner":
            print("   ⚠️  Dashboard is using WorkingMockScanner (fallback/mock data)")
            results["using_mock"] = True
        elif scanner_type in ["EnhancedHuggingFaceScanner", "HuggingFaceHubScanner"]:
            print(f"   ✅ Dashboard is using real scanner: {scanner_type}")
        else:
            print(f"   ❓ Unknown scanner type: {scanner_type}")

        # Check if scanner has real data or mock data
        if hasattr(scanner, "model_cache"):
            cache_size = len(scanner.model_cache)
            print(f"   Model cache size: {cache_size}")

            if cache_size <= 10:
                print("   ⚠️  Very small cache - likely mock data")
                results["using_mock"] = True
            else:
                print(f"   ✅ Substantial cache - likely real data")

    except Exception as e:
        print(f"   ❌ Error checking dashboard: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    print(f"\nEnhanced Scanner Available: {'✅ YES' if results['enhanced_scanner'] else '❌ NO'}")
    print(f"Standard Scanner Available: {'✅ YES' if results['standard_scanner'] else '❌ NO'}")
    print(f"HuggingFace Hub Library:    {'✅ YES' if results['huggingface_hub'] else '❌ NO'}")
    print(f"Using Mock/Fallback Data:   {'⚠️  YES' if results['using_mock'] else '✅ NO'}")

    print("\n" + "-" * 80)

    if results["enhanced_scanner"] or results["standard_scanner"]:
        if not results["using_mock"]:
            print("✅ REAL INTEGRATION: System has real HuggingFace integration")
            print("   The tests are validating actual HuggingFace Hub functionality")
            return True
        else:
            print("⚠️  MIXED STATE: Scanner exists but using mock data")
            print("   Real scanner is available but not being used")
            print("   Check imports and dependencies")
            return False
    else:
        print("❌ MOCK ONLY: System is using mock/fallback data")
        print("   Tests are only validating UI and mock data")
        print("   Real HuggingFace integration is NOT tested")
        print("\n   To enable real integration:")
        print("   1. pip install huggingface_hub")
        print("   2. Implement enhanced_huggingface_scanner.py")
        print("   3. Test with real API calls")
        return False


if __name__ == "__main__":
    success = test_scanner_availability()
    sys.exit(0 if success else 1)
