"""
Test the integration between Enhanced Hardware Taxonomy and Heterogeneous Scheduler.

This test verifies that the integration between the Enhanced Hardware Taxonomy
and the Heterogeneous Scheduler works correctly, with proper capability-based
worker and task matching.
"""

import unittest
import logging
import time
import uuid
from typing import Dict, Any, List, Set
from unittest.mock import MagicMock, patch

from data.duckdb.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareVendor,
    HardwareArchitecture
)
from data.duckdb.distributed_testing.enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy,
    HardwareCapabilityProfile,
    CapabilityDefinition
)
from data.duckdb.distributed_testing.heterogeneous_scheduler import (
    HeterogeneousScheduler,
    WorkerState,
    TestTask,
    WorkloadProfile
)
from data.duckdb.distributed_testing.hardware_taxonomy_integrator import (
    HardwareTaxonomyIntegrator
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestHardwareTaxonomyIntegration(unittest.TestCase):
    """Test the integration between the Enhanced Hardware Taxonomy and Heterogeneous Scheduler."""

    def setUp(self):
        """Set up test resources."""
        # Create a taxonomy instance
        self.taxonomy = EnhancedHardwareTaxonomy()
        
        # Register some test capabilities
        self.taxonomy.register_capability(
            CapabilityDefinition(
                capability_id="matrix_multiplication",
                name="Matrix Multiplication",
                description="Basic matrix multiplication support",
                performance_impact=0.5
            )
        )
        
        self.taxonomy.register_capability(
            CapabilityDefinition(
                capability_id="tensor_core_acceleration",
                name="Tensor Core Acceleration",
                description="Hardware acceleration for tensor operations",
                performance_impact=0.8,
                prerequisites={"matrix_multiplication"}
            )
        )
        
        self.taxonomy.register_capability(
            CapabilityDefinition(
                capability_id="conv_acceleration",
                name="Convolution Acceleration",
                description="Hardware acceleration for convolution operations",
                performance_impact=0.7,
                prerequisites={"matrix_multiplication"}
            )
        )
        
        # Create the taxonomy integrator
        self.integrator = HardwareTaxonomyIntegrator(taxonomy=self.taxonomy)
        
        # Create the heterogeneous scheduler with taxonomy enabled
        self.scheduler = HeterogeneousScheduler(
            strategy="adaptive",
            thermal_management=False,  # Disable for simplicity in testing
            enable_workload_learning=True,
            use_enhanced_taxonomy=True
        )
        
        # Use our test integrator
        self.scheduler.taxonomy_integrator = self.integrator
    
    def create_test_worker(self, worker_id: str, hardware_class: str, 
                          capabilities: Set[str] = None) -> Dict[str, Any]:
        """Create a test worker with specified hardware class and capabilities."""
        hardware_profile = {
            "hardware_class": hardware_class,
            "vendor": "test_vendor",
            "architecture": "test_arch",
            "model_name": f"Test {hardware_class.upper()} Model",
            "memory_gb": 16.0,
            "compute_units": 8,
            "features": ["avx2", "fma"] if hardware_class == "cpu" else ["tensor_cores"] if hardware_class == "gpu" else [],
            "supported_backends": ["pytorch", "onnx"],
            "memory_available_gb": 12.0
        }
        
        return {
            "worker_id": worker_id,
            "capabilities": {
                "hardware_profiles": [hardware_profile],
                "optimal_hardware": {
                    "nlp": {"hardware_class": hardware_class, "effectiveness_score": 0.8},
                    "vision": {"hardware_class": hardware_class, "effectiveness_score": 0.7},
                    "audio": {"hardware_class": hardware_class, "effectiveness_score": 0.6}
                }
            },
            "hardware_profiles": [hardware_profile]
        }
    
    def create_test_task(self, task_id: str, workload_type: str, 
                        required_capabilities: Set[str] = None,
                        preferred_capabilities: Set[str] = None) -> TestTask:
        """Create a test task with specified workload type and capabilities."""
        # Create workload profile
        profile = WorkloadProfile(
            workload_type=workload_type,
            operation_types=["matmul", "softmax"] if workload_type == "nlp" else ["conv", "pool"] if workload_type == "vision" else ["fft"],
            precision_types=["fp16", "fp32"],
            min_memory_gb=2.0,
            preferred_memory_gb=4.0,
            required_features=["tensor_cores"] if workload_type == "nlp" else [],
            batch_size_options=[1, 4, 8, 16]
        )
        
        # Add capabilities if provided
        if required_capabilities:
            for cap in required_capabilities:
                profile.add_required_capability(cap)
        
        if preferred_capabilities:
            for cap in preferred_capabilities:
                profile.add_preferred_capability(cap)
        
        # Create task
        return TestTask(
            task_id=task_id,
            workload_profile=profile,
            priority=2,
            batch_size=8
        )
    
    def test_register_worker_with_capabilities(self):
        """Test that registering a worker enhances it with capabilities."""
        # Create a test worker
        worker_data = self.create_test_worker("worker1", "gpu")
        
        # Register worker with scheduler
        worker = self.scheduler.register_worker(
            worker_data["worker_id"], 
            worker_data["capabilities"]
        )
        
        # Check that worker has been enhanced
        self.assertTrue(hasattr(worker, "capability_profiles"), 
                      "Worker should have capability_profiles attribute")
        
        # GPU workers should automatically have matrix_multiplication capability
        has_matrix_mult = False
        for profile in worker.capability_profiles:
            if "matrix_multiplication" in profile.capabilities:
                has_matrix_mult = True
                break
        
        self.assertTrue(has_matrix_mult, 
                      "GPU worker should have matrix_multiplication capability")
    
    def test_submit_task_with_capabilities(self):
        """Test that submitting a task enhances it with capabilities."""
        # Create a test task
        task = self.create_test_task("task1", "nlp")
        
        # Submit task to scheduler
        self.scheduler.submit_task(task)
        
        # Check that task profile has been enhanced with capabilities
        self.assertTrue(len(task.workload_profile.required_capabilities) > 0 or 
                      len(task.workload_profile.preferred_capabilities) > 0,
                      "Task profile should have capabilities after submission")
    
    def test_capability_based_scheduling(self):
        """Test that scheduling considers capability-based matching."""
        # Create multiple workers with different capabilities
        gpu_worker = self.create_test_worker("gpu_worker", "gpu")
        cpu_worker = self.create_test_worker("cpu_worker", "cpu")
        
        # Register workers
        gpu_worker_state = self.scheduler.register_worker(
            gpu_worker["worker_id"], 
            gpu_worker["capabilities"]
        )
        
        cpu_worker_state = self.scheduler.register_worker(
            cpu_worker["worker_id"], 
            cpu_worker["capabilities"]
        )
        
        # Auto-assign tensor core capability to GPU worker
        for profile in gpu_worker_state.capability_profiles:
            profile.capabilities.add("tensor_core_acceleration")
        
        # Create NLP task that benefits from tensor cores
        nlp_task = self.create_test_task(
            "nlp_task", 
            "nlp", 
            required_capabilities={"matrix_multiplication"},
            preferred_capabilities={"tensor_core_acceleration"}
        )
        
        # Create vision task
        vision_task = self.create_test_task(
            "vision_task", 
            "vision",
            required_capabilities={"matrix_multiplication"},
            preferred_capabilities={"conv_acceleration"}
        )
        
        # Submit tasks
        self.scheduler.submit_task(nlp_task)
        self.scheduler.submit_task(vision_task)
        
        # Schedule tasks
        self.scheduler.schedule_tasks()
        
        # Check that NLP task was assigned to GPU worker (has tensor cores)
        assigned_worker_id = nlp_task.assigned_worker_id
        self.assertEqual(assigned_worker_id, "gpu_worker", 
                       "NLP task should be assigned to GPU worker due to tensor core capability")
    
    def test_enhanced_vs_standard_affinity(self):
        """Test that enhanced affinity calculation differs from standard calculation."""
        # Create a worker with specific capabilities
        worker_data = self.create_test_worker("worker1", "gpu")
        worker = self.scheduler.register_worker(
            worker_data["worker_id"], 
            worker_data["capabilities"]
        )
        
        # Add tensor core capability
        for profile in worker.capability_profiles:
            profile.capabilities.add("tensor_core_acceleration")
        
        # Create a task that benefits from tensor cores
        task = self.create_test_task(
            "task1", 
            "nlp", 
            required_capabilities={"matrix_multiplication"},
            preferred_capabilities={"tensor_core_acceleration"}
        )
        
        # Get standard affinity score
        standard_score = self.scheduler._calculate_standard_affinity(worker, task)
        
        # Get enhanced affinity score
        enhanced_score = self.integrator.calculate_enhanced_affinity(worker, task)
        
        # The enhanced score should differ from the standard score
        self.assertNotEqual(standard_score, enhanced_score, 
                          "Enhanced affinity score should differ from standard score")
        
        # In this case, enhanced score should be higher due to matching tensor core capability
        self.assertGreater(enhanced_score, standard_score, 
                         "Enhanced score should be higher due to matching capabilities")
    
    def test_capability_breakdown(self):
        """Test the capability breakdown functionality."""
        # Create a worker with specific capabilities
        worker_data = self.create_test_worker("worker1", "gpu")
        worker = self.scheduler.register_worker(
            worker_data["worker_id"], 
            worker_data["capabilities"]
        )
        
        # Add several capabilities
        for profile in worker.capability_profiles:
            profile.capabilities.add("matrix_multiplication")
            profile.capabilities.add("tensor_core_acceleration")
            profile.capabilities.add("conv_acceleration")
        
        # Get capability breakdown
        breakdown = self.integrator.get_capability_breakdown(worker)
        
        # Check that breakdown contains information for NLP workload
        self.assertIn("nlp", breakdown, "Breakdown should contain NLP workload")
        
        # Check that tensor_core_acceleration has high impact for NLP
        nlp_impacts = {cap_id: impact for cap_id, impact in breakdown["nlp"]}
        self.assertIn("tensor_core_acceleration", nlp_impacts, 
                    "tensor_core_acceleration should be in NLP impact list")
        self.assertGreaterEqual(nlp_impacts["tensor_core_acceleration"], 0.5, 
                              "tensor_core_acceleration should have high impact for NLP")


if __name__ == "__main__":
    unittest.main()