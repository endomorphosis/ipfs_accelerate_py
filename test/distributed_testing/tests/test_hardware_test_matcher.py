#!/usr/bin/env python3
"""
Unit tests for the Hardware-Test Matching Algorithms module.

This module tests the functionality of the hardware_test_matcher.py module,
which provides intelligent algorithms for matching tests to appropriate
hardware resources in a distributed testing environment.
"""

import unittest
import sys
import os
import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import uuid

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from hardware_test_matcher import (
    TestRequirementType, TestType, TestRequirement, TestProfile,
    HardwareTestMatch, TestPerformanceRecord, TestHardwareMatcher
)

from enhanced_hardware_capability import (
    HardwareCapability, WorkerHardwareCapabilities, 
    HardwareCapabilityDetector, HardwareCapabilityComparator,
    HardwareType, PrecisionType, CapabilityScore, HardwareVendor
)

from distributed_error_handler import (
    DistributedErrorHandler, ErrorType, ErrorSeverity
)


class TestHardwareTestMatcherInit(unittest.TestCase):
    """Test the initialization of the TestHardwareMatcher class."""
    
    def test_init_default(self):
        """Test default initialization."""
        matcher = TestHardwareMatcher()
        self.assertIsNotNone(matcher.hardware_detector)
        self.assertIsNotNone(matcher.hardware_comparator)
        self.assertEqual(matcher.test_profiles, {})
        self.assertIsInstance(matcher.performance_history, dict)
        self.assertEqual(matcher.worker_capabilities, {})
        self.assertTrue(all(w > 0 for w in matcher.match_factor_weights.values()))
        self.assertTrue(matcher.enable_adaptive_weights)
    
    def test_init_with_params(self):
        """Test initialization with parameters."""
        detector = HardwareCapabilityDetector()
        error_handler = DistributedErrorHandler()
        db_connection = MagicMock()
        
        matcher = TestHardwareMatcher(
            hardware_capability_detector=detector,
            error_handler=error_handler,
            db_connection=db_connection
        )
        
        self.assertEqual(matcher.hardware_detector, detector)
        self.assertEqual(matcher.error_handler, error_handler)
        self.assertEqual(matcher.db_connection, db_connection)


class TestHardwareTestMatcherBasic(unittest.TestCase):
    """Test basic functionality of the TestHardwareMatcher class."""
    
    def setUp(self):
        """Set up test environment."""
        self.matcher = TestHardwareMatcher()
        
        # Create test profiles
        self.compute_test = TestProfile(
            test_id="compute_test",
            test_type=TestType.COMPUTE_INTENSIVE,
            estimated_duration_seconds=60,
            estimated_memory_mb=500,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.COMPUTE,
                    value="high",
                    importance=0.9
                ),
                TestRequirement(
                    requirement_type=TestRequirementType.HARDWARE_TYPE,
                    value=HardwareType.GPU,
                    importance=0.8
                )
            ],
            tags=["gpu", "compute"]
        )
        
        self.memory_test = TestProfile(
            test_id="memory_test",
            test_type=TestType.MEMORY_INTENSIVE,
            estimated_duration_seconds=120,
            estimated_memory_mb=4000,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.MEMORY,
                    value=3000,
                    importance=0.9
                )
            ],
            tags=["memory"]
        )
        
        # Create mock worker capabilities
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        
        # Create CPU capability
        self.cpu_capability = HardwareCapability(
            hardware_type=HardwareType.CPU,
            vendor=HardwareVendor.INTEL,
            model="Intel Core i7",
            cores=8,
            memory_gb=16.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP64,
                PrecisionType.INT32
            ],
            scores={
                "compute": CapabilityScore.GOOD,
                "memory": CapabilityScore.GOOD,
                "overall": CapabilityScore.GOOD
            }
        )
        
        # Create GPU capability
        self.gpu_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA RTX 3080",
            compute_units=68,
            memory_gb=10.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP16,
                PrecisionType.INT32,
                PrecisionType.INT8
            ],
            scores={
                "compute": CapabilityScore.EXCELLENT,
                "memory": CapabilityScore.GOOD,
                "overall": CapabilityScore.EXCELLENT
            },
            capabilities={
                "tensor_cores": True,
                "cuda_cores": 8704
            }
        )
        
        # Create worker capabilities
        self.worker_capabilities = WorkerHardwareCapabilities(
            worker_id=self.worker_id,
            os_type="Linux",
            os_version="5.15.0",
            hostname="test-worker",
            cpu_count=8,
            total_memory_gb=32.0,
            hardware_capabilities=[self.cpu_capability, self.gpu_capability]
        )
    
    def test_register_test_profile(self):
        """Test registering a test profile."""
        result = self.matcher.register_test_profile(self.compute_test)
        self.assertTrue(result)
        self.assertIn("compute_test", self.matcher.test_profiles)
        self.assertEqual(self.matcher.test_profiles["compute_test"], self.compute_test)
    
    def test_register_worker_capabilities(self):
        """Test registering worker capabilities."""
        result = self.matcher.register_worker_capabilities(self.worker_capabilities)
        self.assertTrue(result)
        self.assertIn(self.worker_id, self.matcher.worker_capabilities)
        self.assertEqual(self.matcher.worker_capabilities[self.worker_id], self.worker_capabilities)
    
    def test_get_test_profile(self):
        """Test getting a test profile."""
        self.matcher.register_test_profile(self.compute_test)
        profile = self.matcher.get_test_profile("compute_test")
        self.assertEqual(profile, self.compute_test)
        
        # Test non-existent profile
        profile = self.matcher.get_test_profile("non_existent")
        self.assertIsNone(profile)
    
    def test_get_worker_capability(self):
        """Test getting worker capabilities."""
        self.matcher.register_worker_capabilities(self.worker_capabilities)
        capabilities = self.matcher.get_worker_capability(self.worker_id)
        self.assertEqual(capabilities, self.worker_capabilities)
        
        # Test non-existent worker
        capabilities = self.matcher.get_worker_capability("non_existent")
        self.assertIsNone(capabilities)
    
    def test_create_test_profile_from_dict(self):
        """Test creating a test profile from a dictionary."""
        profile_data = {
            "test_id": "dict_test",
            "test_type": "memory_intensive",
            "estimated_duration_seconds": 90,
            "estimated_memory_mb": 2000,
            "metadata": {"priority": "high"},
            "tags": ["memory", "database"],
            "requirements": [
                {
                    "requirement_type": "memory",
                    "value": 1500,
                    "importance": 0.8,
                    "description": "Requires 1.5GB memory"
                }
            ]
        }
        
        profile = self.matcher.create_test_profile_from_dict(profile_data)
        self.assertEqual(profile.test_id, "dict_test")
        self.assertEqual(profile.test_type, TestType.MEMORY_INTENSIVE)
        self.assertEqual(profile.estimated_duration_seconds, 90)
        self.assertEqual(profile.estimated_memory_mb, 2000)
        self.assertEqual(profile.metadata, {"priority": "high"})
        self.assertEqual(profile.tags, ["memory", "database"])
        self.assertEqual(len(profile.requirements), 1)
        self.assertEqual(profile.requirements[0].requirement_type, TestRequirementType.MEMORY)
        self.assertEqual(profile.requirements[0].value, 1500)
        self.assertEqual(profile.requirements[0].importance, 0.8)


class TestHardwareTestMatcherMatching(unittest.TestCase):
    """Test matching functionality of the TestHardwareMatcher class."""
    
    def setUp(self):
        """Set up test environment."""
        self.matcher = TestHardwareMatcher()
        
        # Create test profiles
        self.compute_test = TestProfile(
            test_id="compute_test",
            test_type=TestType.COMPUTE_INTENSIVE,
            estimated_duration_seconds=60,
            estimated_memory_mb=500,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.COMPUTE,
                    value="high",
                    importance=0.9
                ),
                TestRequirement(
                    requirement_type=TestRequirementType.HARDWARE_TYPE,
                    value=HardwareType.GPU,
                    importance=0.8
                )
            ],
            tags=["gpu", "compute"]
        )
        
        self.memory_test = TestProfile(
            test_id="memory_test",
            test_type=TestType.MEMORY_INTENSIVE,
            estimated_duration_seconds=120,
            estimated_memory_mb=4000,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.MEMORY,
                    value=3000,
                    importance=0.9
                )
            ],
            tags=["memory"]
        )
        
        self.precision_test = TestProfile(
            test_id="precision_test",
            test_type=TestType.PRECISION_SENSITIVE,
            estimated_duration_seconds=30,
            estimated_memory_mb=1000,
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.PRECISION,
                    value=PrecisionType.FP16,
                    importance=1.0
                )
            ],
            tags=["precision"]
        )
        
        # Register test profiles
        self.matcher.register_test_profile(self.compute_test)
        self.matcher.register_test_profile(self.memory_test)
        self.matcher.register_test_profile(self.precision_test)
        
        # Create worker capabilities
        self.worker_id = f"worker_{uuid.uuid4().hex[:8]}"
        
        # Create CPU capability
        self.cpu_capability = HardwareCapability(
            hardware_type=HardwareType.CPU,
            vendor=HardwareVendor.INTEL,
            model="Intel Core i7",
            cores=8,
            memory_gb=16.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP64,
                PrecisionType.INT32
            ],
            scores={
                "compute": CapabilityScore.GOOD,
                "memory": CapabilityScore.GOOD,
                "overall": CapabilityScore.GOOD
            }
        )
        
        # Create GPU capability
        self.gpu_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA RTX 3080",
            compute_units=68,
            memory_gb=10.0,
            supported_precisions=[
                PrecisionType.FP32, 
                PrecisionType.FP16,
                PrecisionType.INT32,
                PrecisionType.INT8
            ],
            scores={
                "compute": CapabilityScore.EXCELLENT,
                "memory": CapabilityScore.GOOD,
                "overall": CapabilityScore.EXCELLENT
            },
            capabilities={
                "tensor_cores": True,
                "cuda_cores": 8704
            }
        )
        
        # Create worker capabilities
        self.worker_capabilities = WorkerHardwareCapabilities(
            worker_id=self.worker_id,
            os_type="Linux",
            os_version="5.15.0",
            hostname="test-worker",
            cpu_count=8,
            total_memory_gb=32.0,
            hardware_capabilities=[self.cpu_capability, self.gpu_capability]
        )
        
        # Register worker capabilities
        self.matcher.register_worker_capabilities(self.worker_capabilities)
    
    def test_match_test_to_hardware(self):
        """Test matching a test to hardware."""
        # Test compute-intensive test (should prefer GPU)
        match = self.matcher.match_test_to_hardware("compute_test")
        self.assertIsNotNone(match)
        self.assertEqual(match.test_id, "compute_test")
        self.assertEqual(match.worker_id, self.worker_id)
        self.assertEqual(match.hardware_type, HardwareType.GPU)
        
        # Test memory-intensive test (should work with either CPU or GPU)
        match = self.matcher.match_test_to_hardware("memory_test")
        self.assertIsNotNone(match)
        self.assertEqual(match.test_id, "memory_test")
        
        # Test precision-sensitive test (should prefer GPU with FP16 support)
        match = self.matcher.match_test_to_hardware("precision_test")
        self.assertIsNotNone(match)
        self.assertEqual(match.test_id, "precision_test")
        self.assertEqual(match.hardware_type, HardwareType.GPU)
    
    def test_match_test_to_hardware_no_match(self):
        """Test matching a test to hardware with no suitable match."""
        # Create a test with impossible requirements
        impossible_test = TestProfile(
            test_id="impossible_test",
            test_type=TestType.MEMORY_INTENSIVE,
            estimated_memory_mb=1000000,  # 1TB memory requirement
            requirements=[
                TestRequirement(
                    requirement_type=TestRequirementType.MEMORY,
                    value=1000000,
                    importance=1.0
                )
            ]
        )
        
        self.matcher.register_test_profile(impossible_test)
        
        # Try to match
        match = self.matcher.match_test_to_hardware("impossible_test")
        self.assertIsNone(match)
    
    def test_match_tests_to_workers(self):
        """Test matching multiple tests to workers."""
        matches = self.matcher.match_tests_to_workers(["compute_test", "memory_test", "precision_test"])
        self.assertEqual(len(matches), 3)
        self.assertIn("compute_test", matches)
        self.assertIn("memory_test", matches)
        self.assertIn("precision_test", matches)
    
    def test_match_with_performance_history(self):
        """Test matching with performance history."""
        # Register performance history for compute test on GPU
        gpu_hardware_id = f"{self.worker_id}:{HardwareType.GPU.value}:{self.gpu_capability.model}"
        self.matcher.register_test_performance(TestPerformanceRecord(
            test_id="compute_test",
            worker_id=self.worker_id,
            hardware_id=gpu_hardware_id,
            execution_time_seconds=45.2,
            memory_usage_mb=450,
            success=True
        ))
        
        # Match again (should use performance history)
        match = self.matcher.match_test_to_hardware("compute_test")
        self.assertIsNotNone(match)
        self.assertEqual(match.test_id, "compute_test")
        self.assertEqual(match.worker_id, self.worker_id)
        self.assertEqual(match.hardware_type, HardwareType.GPU)
        
        # Performance factor should be higher now
        self.assertGreater(match.match_factors.get("historical_performance", 0), 0.5)
    
    def test_match_with_error_history(self):
        """Test matching with error history."""
        # Register failed performance history for compute test on GPU
        gpu_hardware_id = f"{self.worker_id}:{HardwareType.GPU.value}:{self.gpu_capability.model}"
        for _ in range(3):  # Multiple failures
            self.matcher.register_test_performance(TestPerformanceRecord(
                test_id="compute_test",
                worker_id=self.worker_id,
                hardware_id=gpu_hardware_id,
                execution_time_seconds=60,
                memory_usage_mb=450,
                success=False,
                error_type="resource"
            ))
        
        # Match again (should consider error history)
        match = self.matcher.match_test_to_hardware("compute_test")
        
        # Error history factor should be lower now
        self.assertLess(match.match_factors.get("error_history", 1.0), 0.5)
    
    def test_get_specialized_matcher(self):
        """Test getting a specialized matcher."""
        # Get specialized matcher for compute-intensive tests
        specialized = self.matcher.get_specialized_matcher(TestType.COMPUTE_INTENSIVE)
        
        # Check that weights were adjusted
        self.assertEqual(specialized.match_factor_weights["compute_capability"], 1.0)
        self.assertEqual(specialized.match_factor_weights["historical_performance"], 0.9)
        self.assertEqual(specialized.match_factor_weights["memory_compatibility"], 0.7)
        
        # Check that instance is different but data is shared
        self.assertIsNot(specialized, self.matcher)
        self.assertIs(specialized.worker_capabilities, self.matcher.worker_capabilities)
        self.assertIs(specialized.performance_history, self.matcher.performance_history)
    
    def test_get_test_performance_history(self):
        """Test getting test performance history."""
        # Register performance history
        gpu_hardware_id = f"{self.worker_id}:{HardwareType.GPU.value}:{self.gpu_capability.model}"
        record = TestPerformanceRecord(
            test_id="compute_test",
            worker_id=self.worker_id,
            hardware_id=gpu_hardware_id,
            execution_time_seconds=45.2,
            memory_usage_mb=450,
            success=True
        )
        self.matcher.register_test_performance(record)
        
        # Get history for specific hardware
        history = self.matcher.get_test_performance_history("compute_test", gpu_hardware_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].execution_time_seconds, 45.2)
        
        # Get history for all hardware
        history = self.matcher.get_test_performance_history("compute_test")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].execution_time_seconds, 45.2)


# Run the tests
if __name__ == '__main__':
    unittest.main()