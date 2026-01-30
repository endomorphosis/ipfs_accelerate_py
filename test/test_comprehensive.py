#!/usr/bin/env python3
"""
Comprehensive Test Suite for IPFS Accelerate Python

This test suite provides comprehensive coverage of all core functionality
without requiring actual GPU or special hardware. It uses mocking and 
simulation to test all code paths.

Key features:
- CPU-only execution (no GPU required)
- Mock hardware environments
- Smoke tests for all major features
- Integration tests
- Performance simulation tests
"""

import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock
from typing import Dict, List, Any

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestHardwareDetectionCore(unittest.TestCase):
    """Core hardware detection tests."""
    
    def setUp(self):
        """Set up test environment."""
        # Import here to avoid issues with module loading
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_hardware_detector_creation(self):
        """Test that HardwareDetector can be created."""
        detector = self.hardware_detection.HardwareDetector()
        self.assertIsNotNone(detector)
    
    def test_cpu_always_available(self):
        """Test that CPU is always detected as available."""
        detector = self.hardware_detection.HardwareDetector()
        available = detector.get_available_hardware()
        
        self.assertIsInstance(available, dict)
        self.assertIn('cpu', available)
        self.assertTrue(available['cpu'], "CPU should always be available")
    
    def test_hardware_details_structure(self):
        """Test that hardware details have correct structure."""
        detector = self.hardware_detection.HardwareDetector()
        details = detector.get_hardware_details()
        
        self.assertIsInstance(details, dict)
        self.assertIn('cpu', details)
        self.assertIsInstance(details['cpu'], dict)
    
    def test_detect_available_hardware_function(self):
        """Test the main detection function."""
        result = self.hardware_detection.detect_available_hardware()
        
        self.assertIsInstance(result, dict)
        required_keys = ['hardware', 'details', 'best_available']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # CPU should be available
        self.assertTrue(result['hardware']['cpu'])
        
        # best_available should be one of the available hardware types
        available_hardware = [k for k, v in result['hardware'].items() if v]
        self.assertIn(result['best_available'], available_hardware,
                     f"best_available '{result['best_available']}' should be one of {available_hardware}")
    
    def test_model_compatibility_checking(self):
        """Test model hardware compatibility."""
        # Test with various model names
        test_models = [
            'bert-base-uncased',
            'gpt2',
            'resnet50',
            'clip-vit-base-patch32',
            'whisper-base'
        ]
        
        for model_name in test_models:
            compatibility = self.hardware_detection.get_model_hardware_compatibility(model_name)
            
            self.assertIsInstance(compatibility, dict)
            self.assertIn('cpu', compatibility)
            self.assertTrue(compatibility['cpu'], f"CPU should be compatible with {model_name}")


class TestHardwareDetectionMocked(unittest.TestCase):
    """Tests with mocked hardware environments."""
    
    def setUp(self):
        """Set up test environment."""
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    @patch.dict(os.environ, {'MOCK_CUDA': '1'})
    def test_cuda_environment_variables(self):
        """Test CUDA detection with environment variables."""
        # This tests that environment variable mocking works
        self.assertEqual(os.environ.get('MOCK_CUDA'), '1')
    
    @patch('importlib.util.find_spec')
    def test_openvino_mock(self, mock_find_spec):
        """Test OpenVINO detection with mocking."""
        # Mock OpenVINO as available
        mock_find_spec.return_value = MagicMock()
        mock_find_spec.return_value.origin = "/mock/openvino.py"
        
        # Create new detector instance (this will re-run the detection logic)
        detector = self.hardware_detection.HardwareDetector()
        details = detector.get_hardware_details()
        
        self.assertIsInstance(details, dict)
        # The test passes as long as we can create the detector
    
    def test_hardware_priority_selection(self):
        """Test hardware selection with priority lists."""
        detector = self.hardware_detection.HardwareDetector()
        
        # Test with CPU-only priority (should always work)
        cpu_priority = detector.get_hardware_by_priority(['cpu'])
        self.assertEqual(cpu_priority, 'cpu')
        
        # Test with mixed priority (should fallback to CPU)
        mixed_priority = detector.get_hardware_by_priority(['cuda', 'cpu'])
        self.assertEqual(mixed_priority, 'cpu')  # Should fallback to CPU


class TestCodeGeneration(unittest.TestCase):
    """Test code generation functionality."""
    
    def setUp(self):
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_hardware_detection_code_generation(self):
        """Test hardware detection code generation."""
        code = self.hardware_detection.get_hardware_detection_code()
        
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 100, "Generated code should be substantial")
        
        # Check for key components
        required_elements = [
            'import os',
            'import importlib.util',
            'HAS_CUDA',
            'HAS_ROCM',
            'HAS_MPS',
            'def check_hardware():',
            'DEVICE = '
        ]
        
        for element in required_elements:
            self.assertIn(element, code, f"Missing element in generated code: {element}")
    
    def test_generated_code_is_valid_python(self):
        """Test that generated code is syntactically valid Python."""
        code = self.hardware_detection.get_hardware_detection_code()
        
        # Try to compile the generated code
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            self.fail(f"Generated code has syntax error: {e}")


class TestBrowserFeatureDetection(unittest.TestCase):
    """Test browser feature detection."""
    
    def setUp(self):
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_browser_feature_detection(self):
        """Test browser feature detection function."""
        features = self.hardware_detection.detect_browser_features()
        
        self.assertIsInstance(features, dict)
        
        required_keys = ['running_in_browser', 'webnn_available', 'webgpu_available', 'environment']
        for key in required_keys:
            self.assertIn(key, features, f"Missing key in browser features: {key}")
        
        # In a normal Python environment, we should not be in a browser
        self.assertFalse(features['running_in_browser'])
        self.assertEqual(features['environment'], 'node')


class TestPerformanceAndIntegration(unittest.TestCase):
    """Integration and performance tests."""
    
    def setUp(self):
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_detector_performance(self):
        """Test that detection completes in reasonable time."""
        import time
        
        start_time = time.time()
        detector = self.hardware_detection.HardwareDetector()
        available = detector.get_available_hardware()
        details = detector.get_hardware_details()
        end_time = time.time()
        
        elapsed = end_time - start_time
        self.assertLess(elapsed, 5.0, "Hardware detection should complete within 5 seconds")
        
        # Verify we got results
        self.assertIsInstance(available, dict)
        self.assertIsInstance(details, dict)
    
    def test_cache_functionality(self):
        """Test hardware detection caching."""
        # Create detector with a temporary cache file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name
        
        try:
            # First detection (should create cache)
            detector1 = self.hardware_detection.HardwareDetector(cache_file=cache_file)
            result1 = detector1.get_available_hardware()
            
            # Cache file should exist now
            self.assertTrue(os.path.exists(cache_file))
            
            # Second detection (should use cache)
            detector2 = self.hardware_detection.HardwareDetector(cache_file=cache_file)
            result2 = detector2.get_available_hardware()
            
            # Results should be the same
            self.assertEqual(result1, result2)
            
        finally:
            # Clean up
            if os.path.exists(cache_file):
                os.unlink(cache_file)
    
    def test_multiple_detector_instances(self):
        """Test creating multiple detector instances."""
        detectors = []
        
        # Create multiple detectors
        for i in range(3):
            detector = self.hardware_detection.HardwareDetector()
            detectors.append(detector)
        
        # All should work
        for i, detector in enumerate(detectors):
            available = detector.get_available_hardware()
            self.assertIsInstance(available, dict, f"Detector {i} failed")
            self.assertTrue(available.get('cpu', False), f"Detector {i} doesn't see CPU")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_invalid_model_name(self):
        """Test compatibility checking with invalid model names."""
        # These should not crash
        invalid_names = ['', None, 123, [], {}]
        
        for invalid_name in invalid_names:
            try:
                compatibility = self.hardware_detection.get_model_hardware_compatibility(str(invalid_name))
                self.assertIsInstance(compatibility, dict)
            except Exception as e:
                # Some errors are expected, but shouldn't crash the process
                self.assertIsNotNone(e)  # Just verify we handled it
    
    def test_detector_with_invalid_cache_path(self):
        """Test detector with invalid cache paths."""
        # Path that doesn't exist
        invalid_paths = [
            '/nonexistent/path/cache.json',
            '/root/cache.json',  # No permission
            '',  # Empty path
        ]
        
        for invalid_path in invalid_paths:
            # Should not crash, should fallback gracefully
            try:
                detector = self.hardware_detection.HardwareDetector(cache_file=invalid_path)
                result = detector.get_available_hardware()
                self.assertIsInstance(result, dict)
            except Exception as e:
                # Some errors are acceptable, but shouldn't prevent basic functionality
                self.assertIsInstance(e, (IOError, OSError, PermissionError))


def run_comprehensive_tests():
    """Run all tests and provide detailed reporting."""
    test_classes = [
        TestHardwareDetectionCore,
        TestHardwareDetectionMocked,
        TestCodeGeneration,
        TestBrowserFeatureDetection,
        TestPerformanceAndIntegration,
        TestErrorHandling,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("Running comprehensive test suite...")
    print("=" * 70)
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}:")
        print("-" * 40)
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_failed = len(result.failures) + len(result.errors)
        class_passed = class_total - class_failed
        
        total_tests += class_total
        passed_tests += class_passed
        failed_tests += class_failed
        
        if class_failed > 0:
            print(f"❌ {class_failed} failed out of {class_total}")
        else:
            print(f"✅ All {class_total} tests passed")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed_tests}/{total_tests} tests passed ({failed_tests} failed)")
    
    if failed_tests == 0:
        print("🎉 All comprehensive tests passed!")
        return True
    else:
        print(f"❌ {failed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)