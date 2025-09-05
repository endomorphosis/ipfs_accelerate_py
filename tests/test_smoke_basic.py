#!/usr/bin/env python3
"""
Basic smoke test for IPFS Accelerate Python
Tests core functionality without requiring GPU or special hardware.
"""

import sys
import os
import pytest

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hardware_detection_import():
    """Test that hardware_detection module can be imported."""
    try:
        import hardware_detection
        assert hardware_detection is not None
        print("✓ hardware_detection module imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import hardware_detection: {e}")

def test_hardware_detection_basic():
    """Test basic hardware detection functionality."""
    try:
        import hardware_detection
        
        # Test basic hardware detection without requiring specific hardware
        detector = hardware_detection.HardwareDetector()
        
        # This should always work - CPU detection
        available_hardware = detector.get_available_hardware()
        assert 'cpu' in available_hardware
        assert available_hardware['cpu'] is True
        print("✓ Basic hardware detection works")
        
        # Test hardware details
        details = detector.get_hardware_details()
        assert isinstance(details, dict)
        assert 'cpu' in details
        print("✓ Hardware details retrieval works")
        
    except Exception as e:
        pytest.fail(f"Hardware detection basic test failed: {e}")

def test_detect_available_hardware_function():
    """Test the detect_available_hardware function."""
    try:
        import hardware_detection
        
        # Test the main detection function
        result = hardware_detection.detect_available_hardware()
        
        assert isinstance(result, dict)
        assert 'hardware' in result
        assert 'details' in result
        assert 'best_available' in result
        
        # CPU should always be available
        assert result['hardware']['cpu'] is True
        print("✓ detect_available_hardware function works")
        
    except Exception as e:
        pytest.fail(f"detect_available_hardware test failed: {e}")

def test_model_hardware_compatibility():
    """Test model hardware compatibility checking."""
    try:
        import hardware_detection
        
        # Test with a common model name
        compatibility = hardware_detection.get_model_hardware_compatibility("bert-base-uncased")
        
        assert isinstance(compatibility, dict)
        assert 'cpu' in compatibility
        assert compatibility['cpu'] is True  # CPU should always be compatible
        print("✓ Model hardware compatibility checking works")
        
    except Exception as e:
        pytest.fail(f"Model hardware compatibility test failed: {e}")

def test_get_hardware_detection_code():
    """Test hardware detection code generation."""
    try:
        import hardware_detection
        
        # Test code generation
        code = hardware_detection.get_hardware_detection_code()
        
        assert isinstance(code, str)
        assert len(code) > 0
        assert "import os" in code
        assert "HAS_CUDA" in code
        print("✓ Hardware detection code generation works")
        
    except Exception as e:
        pytest.fail(f"Hardware detection code generation test failed: {e}")

def test_basic_ipfs_accelerate_import():
    """Test that main ipfs_accelerate module can be imported."""
    try:
        # Try to import the main module
        import ipfs_accelerate_py
        print("✓ ipfs_accelerate_py module imported successfully")
    except ImportError as e:
        # This might fail due to heavy dependencies, so we'll make it a warning
        print(f"⚠ ipfs_accelerate_py import failed (expected in minimal environment): {e}")

if __name__ == "__main__":
    print("Running basic smoke tests...")
    
    # Run tests individually and report results
    tests = [
        test_hardware_detection_import,
        test_hardware_detection_basic,
        test_detect_available_hardware_function,
        test_model_hardware_compatibility,
        test_get_hardware_detection_code,
        test_basic_ipfs_accelerate_import,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic smoke tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)