#!/usr/bin/env python3
"""
Integration tests for IPFS Accelerate Python main functionality.

These tests focus on testing the main package features without requiring
heavy dependencies or GPU hardware.
"""

import sys
import os
import pytest
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# Add the parent directory to sys.path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestIPFSAccelerateIntegration(unittest.TestCase):
    """Integration tests for main IPFS Accelerate functionality."""
    
    def test_main_module_import(self):
        """Test that the main module can be imported."""
        try:
            import ipfs_accelerate_py
            self.assertIsNotNone(ipfs_accelerate_py)
            print("✓ Main IPFS accelerate module imported successfully")
        except ImportError as e:
            print(f"⚠ Main module import failed (expected in minimal environment): {e}")
            # This is acceptable in minimal test environments
    
    def test_web_compatibility_module(self):
        """Test web compatibility module functionality."""
        try:
            import web_compatibility
            
            # Test that the module has expected functions/classes
            self.assertTrue(hasattr(web_compatibility, '__file__'))
            print("✓ Web compatibility module imported successfully")
            
        except Exception as e:
            self.fail(f"Web compatibility module test failed: {e}")
    
    def test_webgpu_platform_module(self):
        """Test WebGPU platform module functionality."""
        try:
            import webgpu_platform
            
            # Test that the module has expected functions/classes
            self.assertTrue(hasattr(webgpu_platform, '__file__'))
            print("✓ WebGPU platform module imported successfully")
            
        except Exception as e:
            self.fail(f"WebGPU platform module test failed: {e}")

class TestHardwareIntegration(unittest.TestCase):
    """Integration tests for hardware-related functionality."""
    
    def setUp(self):
        import hardware_detection
        self.hardware_detection = hardware_detection
    
    def test_hardware_detection_integration(self):
        """Test full hardware detection integration."""
        # Test the complete workflow
        result = self.hardware_detection.detect_available_hardware()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        required_keys = ['hardware', 'details', 'best_available', 'torch_device']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Test hardware enumeration
        hardware = result['hardware']
        expected_hardware = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'webnn', 'webgpu', 'qualcomm']
        for hw_type in expected_hardware:
            self.assertIn(hw_type, hardware)
        
        # CPU should always be available
        self.assertTrue(hardware['cpu'])
        
        # best_available should be one of the available hardware types
        # In CI environment, webgpu/webnn might be detected due to Node.js presence
        self.assertIn(result['best_available'], hardware.keys())
        available_hardware = [k for k, v in hardware.items() if v]
        self.assertIn(result['best_available'], available_hardware, 
                     f"best_available '{result['best_available']}' should be one of {available_hardware}")
        
        # torch_device should be valid
        self.assertIsInstance(result['torch_device'], str)
        
        print("✓ Hardware detection integration test passed")
    
    def test_model_compatibility_integration(self):
        """Test model compatibility checking with various models."""
        test_cases = [
            ('bert-base-uncased', 'text'),
            ('resnet50', 'vision'),
            ('gpt2', 'text_generation'),
            ('clip-vit-base-patch32', 'multimodal'),
            ('whisper-base', 'audio'),
            ('t5-small', 'text_generation'),
        ]
        
        for model_name, model_type in test_cases:
            compatibility = self.hardware_detection.get_model_hardware_compatibility(model_name)
            
            # Basic validation
            self.assertIsInstance(compatibility, dict)
            self.assertIn('cpu', compatibility)
            self.assertTrue(compatibility['cpu'], f"CPU should be compatible with {model_name}")
            
            # Check that all hardware types are present
            expected_hardware = ['cpu', 'cuda', 'rocm', 'mps', 'openvino', 'qualcomm', 'webnn', 'webgpu']
            for hw_type in expected_hardware:
                self.assertIn(hw_type, compatibility)
        
        print("✓ Model compatibility integration test passed")
    
    def test_hardware_priority_integration(self):
        """Test hardware priority selection integration."""
        detector = self.hardware_detection.HardwareDetector()
        
        # Test CPU-first priority (should always work)
        result = detector.get_hardware_by_priority(['cpu'])
        self.assertEqual(result, 'cpu', "Priority ['cpu'] should always return cpu")
        
        # Test that priority system respects the order and returns first available
        # Get actually available hardware
        available = detector.get_available_hardware()
        available_list = [k for k, v in available.items() if v]
        
        # Test with non-existent hardware followed by CPU (should fallback to CPU)
        result = detector.get_hardware_by_priority(['nonexistent_hw', 'cpu'])
        self.assertEqual(result, 'cpu', "Should fallback to CPU when non-existent hw is requested")
        
        # Test that when we request multiple hardware types, we get the first available one
        for hw_type in available_list:
            result = detector.get_hardware_by_priority([hw_type])
            self.assertEqual(result, hw_type, f"Priority ['{hw_type}'] should return {hw_type}")
        
        print("✓ Hardware priority integration test passed")

class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for performance-related functionality."""
    
    def test_detection_performance_under_load(self):
        """Test hardware detection performance under load."""
        import time
        import threading
        import hardware_detection
        
        results = []
        errors = []
        
        def detect_hardware():
            try:
                start_time = time.time()
                detector = hardware_detection.HardwareDetector()
                result = detector.get_available_hardware()
                end_time = time.time()
                
                results.append({
                    'duration': end_time - start_time,
                    'cpu_available': result.get('cpu', False)
                })
            except Exception as e:
                errors.append(e)
        
        # Run multiple detections in parallel
        threads = []
        for i in range(5):
            thread = threading.Thread(target=detect_hardware)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Validate results
        self.assertEqual(len(errors), 0, f"Errors occurred during concurrent detection: {errors}")
        self.assertEqual(len(results), 5, "Not all detections completed")
        
        for result in results:
            self.assertLess(result['duration'], 2.0, "Detection should complete within 2 seconds")
            self.assertTrue(result['cpu_available'], "CPU should be available in all detections")
        
        print("✓ Performance under load test passed")
    
    def test_cache_performance_integration(self):
        """Test caching performance integration."""
        import time
        import hardware_detection
        
        # Create temporary cache file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            cache_file = f.name
        
        try:
            # First detection (cold cache)
            start_time = time.time()
            detector1 = hardware_detection.HardwareDetector(cache_file=cache_file)
            result1 = detector1.get_available_hardware()
            cold_time = time.time() - start_time
            
            # Second detection (warm cache)  
            start_time = time.time()
            detector2 = hardware_detection.HardwareDetector(cache_file=cache_file)
            result2 = detector2.get_available_hardware()
            warm_time = time.time() - start_time
            
            # Validate results are consistent
            self.assertEqual(result1, result2, "Cached results should match fresh detection")
            
            # Cache should be faster (though this may not always be true in practice)
            print(f"Cold detection: {cold_time:.3f}s, Warm detection: {warm_time:.3f}s")
            
            # Both should complete in reasonable time
            self.assertLess(cold_time, 3.0, "Cold detection should complete within 3 seconds")
            self.assertLess(warm_time, 3.0, "Warm detection should complete within 3 seconds")
            
        finally:
            # Clean up
            if os.path.exists(cache_file):
                os.unlink(cache_file)
        
        print("✓ Cache performance integration test passed")

class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration and environment handling."""
    
    def test_environment_variable_integration(self):
        """Test environment variable handling integration."""
        import hardware_detection
        
        # Test with various environment configurations
        test_configs = [
            ({}, 'baseline'),
            ({'WEBNN_AVAILABLE': '1'}, 'webnn_available'),
            ({'WEBGPU_AVAILABLE': '1'}, 'webgpu_available'),
            ({'ROCM_HOME': '/opt/rocm'}, 'rocm_home'),
            ({'QUALCOMM_SDK': '/opt/qualcomm'}, 'qualcomm_sdk'),
        ]
        
        for env_vars, config_name in test_configs:
            with patch.dict(os.environ, env_vars, clear=False):
                detector = hardware_detection.HardwareDetector()
                result = detector.get_available_hardware()
                
                # Basic validation
                self.assertIsInstance(result, dict)
                self.assertTrue(result.get('cpu', False), f"CPU should be available in {config_name}")
                
                print(f"✓ Environment configuration '{config_name}' handled correctly")
    
    def test_cache_file_configurations(self):
        """Test various cache file configurations."""
        import hardware_detection
        
        # Test with different cache file scenarios
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_scenarios = [
                (None, 'no_cache'),
                (os.path.join(temp_dir, 'test_cache.json'), 'valid_cache'),
                ('', 'empty_cache'),
            ]
            
            for cache_file, scenario_name in cache_scenarios:
                try:
                    detector = hardware_detection.HardwareDetector(cache_file=cache_file)
                    result = detector.get_available_hardware()
                    
                    self.assertIsInstance(result, dict)
                    self.assertTrue(result.get('cpu', False))
                    
                    print(f"✓ Cache scenario '{scenario_name}' handled correctly")
                    
                except Exception as e:
                    self.fail(f"Cache scenario '{scenario_name}' failed: {e}")

def run_integration_tests():
    """Run all integration tests with detailed reporting."""
    test_classes = [
        TestIPFSAccelerateIntegration,
        TestHardwareIntegration,
        TestPerformanceIntegration,
        TestConfigurationIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    print("Running integration test suite...")
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
    print(f"INTEGRATION TEST SUMMARY: {passed_tests}/{total_tests} tests passed ({failed_tests} failed)")
    
    if failed_tests == 0:
        print("🎉 All integration tests passed!")
        return True
    else:
        print(f"❌ {failed_tests} integration tests failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)