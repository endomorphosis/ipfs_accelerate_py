#!/usr/bin/env python3
"""
Unit tests for Hardware-Aware Scheduler integration with Load Balancer.
"""

import os
import sys
import logging
import unittest
from unittest.mock import patch, MagicMock
import random
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components to test
from distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, HardwareTaxonomy, WorkloadProfile, WorkloadType
)
from distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler
from distributed_testing.load_balancer_integration import (
    create_hardware_aware_load_balancer, register_type_specific_schedulers, shutdown_integration
)

# Import load balancer components
from duckdb_api.distributed_testing.load_balancer.models import (
    TestRequirements, WorkerCapabilities, WorkerLoad, WorkerPerformance
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_hardware_aware_scheduler")


class TestHardwareAwareScheduler(unittest.TestCase):
    """Test cases for hardware-aware scheduler."""
    
    def setUp(self):
        """Set up test environment."""
        # Create taxonomy and workload manager
        self.taxonomy = HardwareTaxonomy()
        self.workload_manager = HardwareWorkloadManager(self.taxonomy)
        
        # Create scheduler
        self.scheduler = HardwareAwareScheduler(self.workload_manager, self.taxonomy)
        
        # Sample worker capabilities
        self.worker_capabilities = {
            "worker1": self._create_test_worker_capabilities("worker1", "cpu"),
            "worker2": self._create_test_worker_capabilities("worker2", "gpu"),
            "worker3": self._create_test_worker_capabilities("worker3", "browser")
        }
        
        # Sample worker loads
        self.worker_loads = {
            "worker1": self._create_test_worker_load("worker1", 0.3),
            "worker2": self._create_test_worker_load("worker2", 0.5),
            "worker3": self._create_test_worker_load("worker3", 0.7)
        }
        
        # Sample performance data
        self.performance_data = {}
    
    def tearDown(self):
        """Clean up after tests."""
        self.workload_manager.stop()
    
    def _create_test_worker_capabilities(self, worker_id: str, worker_type: str) -> WorkerCapabilities:
        """Create test worker capabilities."""
        # Base worker capabilities
        capabilities = WorkerCapabilities(
            worker_id=worker_id,
            hostname=f"host-{worker_id}",
            hardware_specs={
                "cpu": {"cores": 8, "threads": 16},
                "memory": {"total_gb": 32}
            },
            software_versions={
                "python": "3.9.5"
            },
            supported_backends=["cpu"],
            available_accelerators={},
            available_memory=16.0,
            cpu_cores=8,
            cpu_threads=16
        )
        
        # Add specific capabilities based on worker type
        if worker_type == "gpu":
            capabilities.supported_backends.extend(["cuda", "gpu"])
            capabilities.available_accelerators["gpu"] = 1
        
        elif worker_type == "browser":
            capabilities.supported_backends.extend(["webgpu", "webnn"])
        
        return capabilities
    
    def _create_test_worker_load(self, worker_id: str, load_factor: float) -> WorkerLoad:
        """Create test worker load."""
        return WorkerLoad(
            worker_id=worker_id,
            active_tests=int(load_factor * 10),
            cpu_utilization=load_factor * 70.0,
            memory_utilization=load_factor * 60.0,
            gpu_utilization=load_factor * 80.0 if "worker2" in worker_id else 0.0,
            io_utilization=load_factor * 30.0,
            network_utilization=load_factor * 20.0
        )
    
    def _create_test_requirements(self, test_id: str, test_type: str) -> TestRequirements:
        """Create test requirements."""
        return TestRequirements(
            test_id=test_id,
            test_type=test_type,
            model_id="test-model",
            minimum_memory=2.0,
            expected_duration=60.0,
            priority=3
        )
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        self.assertIsNotNone(self.scheduler)
        self.assertEqual(self.scheduler.workload_manager, self.workload_manager)
        self.assertEqual(self.scheduler.hardware_taxonomy, self.taxonomy)
    
    def test_worker_capability_conversion(self):
        """Test conversion of worker capabilities to hardware profiles."""
        # Register a worker's capabilities
        worker_id = "worker2"  # GPU worker
        capabilities = self.worker_capabilities[worker_id]
        
        # Update worker hardware profiles
        self.scheduler._update_worker_hardware_profiles({worker_id: capabilities})
        
        # Check if worker is in cache
        self.assertIn(worker_id, self.scheduler.worker_hardware_cache)
        
        # Check if profiles were created
        hardware_profiles = self.scheduler.worker_hardware_cache[worker_id]
        self.assertTrue(len(hardware_profiles) > 0)
        
        # At least one profile should be GPU for worker2
        gpu_profiles = [p for p in hardware_profiles if p.hardware_class.value == "gpu"]
        self.assertTrue(len(gpu_profiles) > 0)
    
    def test_test_requirement_conversion(self):
        """Test conversion of test requirements to workload profile."""
        # Create test requirements
        test_id = "test123"
        test_type = "vision_classification"
        requirements = self._create_test_requirements(test_id, test_type)
        
        # Convert to workload profile
        workload_profile = self.scheduler._test_to_workload_profile(requirements)
        
        # Check workload profile attributes
        self.assertEqual(workload_profile.workload_id, test_id)
        self.assertEqual(workload_profile.workload_type, WorkloadType.VISION)
        self.assertEqual(workload_profile.min_memory_bytes, int(requirements.minimum_memory * 1024 * 1024 * 1024))
        self.assertEqual(workload_profile.priority, requirements.priority)
    
    def test_worker_selection(self):
        """Test worker selection for test requirements."""
        # Create test requirements
        test_id = "test123"
        test_type = "vision_classification"
        requirements = self._create_test_requirements(test_id, test_type)
        
        # Select worker
        selected_worker = self.scheduler.select_worker(
            requirements,
            self.worker_capabilities,
            self.worker_loads,
            self.performance_data
        )
        
        # Check if a worker was selected
        self.assertIsNotNone(selected_worker)
        self.assertIn(selected_worker, self.worker_capabilities.keys())
        
        # For vision classification, worker2 (GPU) should be preferred
        # But this depends on the efficiency scoring algorithm
        logger.info(f"Selected worker for {test_type}: {selected_worker}")
    
    def test_efficiency_adjustment(self):
        """Test efficiency score adjustment based on load and thermal state."""
        # Base efficiency
        base_efficiency = 0.8
        
        # Worker with high load and cooling state
        worker_id = "worker3"
        load = self.worker_loads[worker_id]
        load.cooling_state = True
        load.performance_level = 0.7
        
        # Get hardware profile
        self.scheduler._update_worker_hardware_profiles({worker_id: self.worker_capabilities[worker_id]})
        hardware_profile = self.scheduler.worker_hardware_cache[worker_id][0]
        
        # Calculate adjusted efficiency
        adjusted_efficiency = self.scheduler._adjust_efficiency_for_load_and_thermal(
            base_efficiency, worker_id, load, hardware_profile
        )
        
        # Should be lower than base efficiency due to high load and cooling state
        self.assertLess(adjusted_efficiency, base_efficiency)
        logger.info(f"Base efficiency: {base_efficiency}, Adjusted: {adjusted_efficiency}")
    
    def test_integration_creation(self):
        """Test creation of integrated system."""
        # Create load balancer with hardware-aware scheduling
        load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer()
        
        # Check if components were created correctly
        self.assertIsNotNone(load_balancer)
        self.assertIsNotNone(workload_manager)
        self.assertIsNotNone(scheduler)
        
        # Check if load balancer is using the hardware-aware scheduler
        self.assertEqual(load_balancer.default_scheduler, scheduler)
        
        # Clean up
        shutdown_integration(load_balancer, workload_manager)
    
    def test_composite_scheduler(self):
        """Test creation of composite scheduler."""
        # Create load balancer with composite scheduler
        load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer(
            use_composite=True,
            hardware_scheduler_weight=0.8
        )
        
        # Check if composite scheduler was created
        self.assertIsNotNone(load_balancer.default_scheduler)
        self.assertEqual(load_balancer.default_scheduler.__class__.__name__, "CompositeScheduler")
        
        # Clean up
        shutdown_integration(load_balancer, workload_manager)
    
    def test_type_specific_schedulers(self):
        """Test registration of type-specific schedulers."""
        # Create load balancer with hardware-aware scheduling
        load_balancer, workload_manager, scheduler = create_hardware_aware_load_balancer()
        
        # Create mock schedulers for specific test types
        mock_vision_scheduler = MagicMock()
        mock_nlp_scheduler = MagicMock()
        
        # Register type-specific schedulers
        type_scheduler_map = {
            "vision_classification": mock_vision_scheduler,
            "nlp_text_classification": mock_nlp_scheduler
        }
        
        register_type_specific_schedulers(load_balancer, scheduler, type_scheduler_map)
        
        # Check if type-specific schedulers were registered
        self.assertEqual(load_balancer.test_type_schedulers["vision_classification"], mock_vision_scheduler)
        self.assertEqual(load_balancer.test_type_schedulers["nlp_text_classification"], mock_nlp_scheduler)
        
        # Clean up
        shutdown_integration(load_balancer, workload_manager)


if __name__ == "__main__":
    unittest.main()