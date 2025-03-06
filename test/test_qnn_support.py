#!/usr/bin/env python
"""
Test script for QNN (Qualcomm Neural Network) support module.

This script tests the functionality of the QNN support module, including:
- Hardware detection
- Power monitoring
- Model optimization recommendations

Run this script to verify the QNN support implementation.
"""

import os
import sys
import json
import time
import unittest
from pathlib import Path

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import QNN support module
from hardware_detection.qnn_support import (
    QNNCapabilityDetector,
    QNNPowerMonitor,
    QNNModelOptimizer
)

class TestQNNSupport(unittest.TestCase):
    """Test cases for QNN support module"""
    
    def setUp(self):
        """Set up test environment"""
        self.detector = QNNCapabilityDetector()
        self.monitor = QNNPowerMonitor()
        self.optimizer = QNNModelOptimizer()
        
        # Create a dummy model file for testing
        self.test_model_path = os.path.join(os.path.dirname(__file__), "test_model.onnx")
        with open(self.test_model_path, "w") as f:
            f.write("DUMMY_MODEL_CONTENT")
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove test model file
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
    
    def test_capability_detection(self):
        """Test QNN hardware capability detection"""
        # Test availability
        self.assertTrue(self.detector.is_available())
        
        # Test device selection
        self.assertTrue(self.detector.select_device())
        
        # Test capability summary
        summary = self.detector.get_capability_summary()
        self.assertIsNotNone(summary)
        self.assertIn("device_name", summary)
        self.assertIn("compute_units", summary)
        self.assertIn("memory_mb", summary)
        self.assertIn("precision_support", summary)
        self.assertIn("recommended_models", summary)
        
        # Test model compatibility
        compat = self.detector.test_model_compatibility(self.test_model_path)
        self.assertIsNotNone(compat)
        self.assertIn("compatible", compat)
    
    def test_power_monitoring(self):
        """Test QNN power monitoring"""
        # Test monitoring start
        self.assertTrue(self.monitor.start_monitoring())
        
        # Run for a short period
        time.sleep(1)
        
        # Test stop and results
        results = self.monitor.stop_monitoring()
        self.assertIsNotNone(results)
        self.assertIn("device_name", results)
        self.assertIn("average_power_watts", results)
        self.assertIn("peak_power_watts", results)
        self.assertIn("thermal_throttling_detected", results)
        self.assertIn("estimated_battery_impact_percent", results)
        
        # Test monitoring data
        data = self.monitor.get_monitoring_data()
        self.assertGreater(len(data), 0)
        
        # Test battery life estimation
        battery_est = self.monitor.estimate_battery_life(1.5)
        self.assertIn("estimated_runtime_hours", battery_est)
        self.assertIn("battery_percent_per_hour", battery_est)
        self.assertIn("efficiency_score", battery_est)
    
    def test_model_optimization(self):
        """Test QNN model optimization recommendations"""
        # Test getting supported optimizations
        opts = self.optimizer.get_supported_optimizations()
        self.assertIsNotNone(opts)
        self.assertIn("quantization", opts)
        self.assertIn("pruning", opts)
        
        # Test optimization recommendations
        rec_bert = self.optimizer.recommend_optimizations("bert-base-uncased.onnx")
        self.assertIsNotNone(rec_bert)
        self.assertIn("recommended_optimizations", rec_bert)
        
        rec_llama = self.optimizer.recommend_optimizations("llama-7b.onnx")
        self.assertIsNotNone(rec_llama)
        self.assertIn("recommended_optimizations", rec_llama)
        
        rec_whisper = self.optimizer.recommend_optimizations("whisper-small.onnx")
        self.assertIsNotNone(rec_whisper)
        self.assertIn("recommended_optimizations", rec_whisper)
        
        # Test optimization simulation
        sim = self.optimizer.simulate_optimization(
            self.test_model_path, 
            ["quantization:int8", "pruning:magnitude"]
        )
        self.assertIsNotNone(sim)
        self.assertIn("original_size_bytes", sim)
        self.assertIn("optimized_size_bytes", sim)
        self.assertIn("speedup_factor", sim)


def run_simple_tests():
    """Run simple tests without unittest framework"""
    print("Running simple QNN support tests...\n")
    
    # Create test objects
    detector = QNNCapabilityDetector()
    monitor = QNNPowerMonitor()
    optimizer = QNNModelOptimizer()
    
    # Test capability detection
    print("Testing QNN capability detection...")
    if detector.is_available():
        detector.select_device()
        summary = detector.get_capability_summary()
        print(f"  Detected device: {summary['device_name']}")
        print(f"  Compute units: {summary['compute_units']}")
        print(f"  Memory: {summary['memory_mb']} MB")
        print(f"  Supported precisions: {', '.join(summary['precision_support'])}")
        print(f"  Recommended models: {len(summary['recommended_models'])} models")
    else:
        print("  QNN hardware not detected")
    
    # Test power monitoring
    print("\nTesting QNN power monitoring...")
    monitor.start_monitoring()
    print("  Monitoring for 2 seconds...")
    time.sleep(2)
    results = monitor.stop_monitoring()
    print(f"  Average power: {results['average_power_watts']} W")
    print(f"  Battery impact: {results['estimated_battery_impact_percent']}%")
    print(f"  Power efficiency score: {results['power_efficiency_score']}/100")
    
    # Test model optimization
    print("\nTesting QNN model optimization...")
    models = ["bert-base-uncased.onnx", "whisper-tiny.onnx", "llama-7b.onnx"]
    for model in models:
        print(f"  Optimizing {model}...")
        recommendations = optimizer.recommend_optimizations(model)
        if recommendations["compatible"]:
            print(f"    Compatible: Yes")
            print(f"    Recommended optimizations: {', '.join(recommendations['recommended_optimizations'])}")
            print(f"    Estimated memory reduction: {recommendations['estimated_memory_reduction']}")
            print(f"    Power efficiency score: {recommendations['estimated_power_efficiency_score']}/100")
        else:
            print(f"    Compatible: No ({recommendations['reason']})")
    
    print("\nAll tests completed successfully.")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test QNN support module")
    parser.add_argument("--unittest", action="store_true", help="Run unittest framework")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()
    
    if args.unittest:
        # Run unittest framework
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        # Run simple tests
        run_simple_tests()