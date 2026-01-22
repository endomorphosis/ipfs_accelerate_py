#!/usr/bin/env python3
"""
Test Fault Tolerance Visualization

This script tests the fault tolerance visualization system, generating
sample visualizations and reports to verify its functionality.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import components to test
from duckdb_api.distributed_testing.hardware_aware_fault_tolerance import (
    HardwareAwareFaultToleranceManager,
    FailureType,
    RecoveryStrategy,
    FailureContext,
    RecoveryAction,
    visualize_fault_tolerance
)
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    HardwareCapabilityProfile,
    MemoryProfile
)
from duckdb_api.distributed_testing.ml_pattern_detection import (
    MLPatternDetector
)


class TestFaultToleranceVisualization(unittest.TestCase):
    """Test case for fault tolerance visualization."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for visualizations
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mocks
        self.db_manager = MagicMock()
        self.coordinator = MagicMock()
        
        # Create a fault tolerance manager with ML detection
        self.manager = HardwareAwareFaultToleranceManager(
            db_manager=self.db_manager,
            coordinator=self.coordinator,
            enable_ml=True
        )
        
        # Create hardware profiles for testing
        self.cpu_profile = HardwareCapabilityProfile(
            hardware_class=HardwareClass.CPU,
            architecture=HardwareArchitecture.X86_64,
            vendor=HardwareVendor.INTEL,
            memory=MemoryProfile(
                total_bytes=8 * 1024 * 1024 * 1024,
                available_bytes=6 * 1024 * 1024 * 1024
            ),
            compute_units=8
        )
        
        self.gpu_profile = HardwareCapabilityProfile(
            hardware_class=HardwareClass.GPU,
            architecture=HardwareArchitecture.GPU_CUDA,
            vendor=HardwareVendor.NVIDIA,
            memory=MemoryProfile(
                total_bytes=16 * 1024 * 1024 * 1024,
                available_bytes=14 * 1024 * 1024 * 1024
            ),
            compute_units=128
        )
        
        # Add sample failure contexts
        self._add_sample_failures()
        
        # Add sample recovery actions
        self._add_sample_recovery_actions()
    
    def tearDown(self):
        """Clean up after the test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def _add_sample_failures(self):
        """Add sample failure contexts to the manager."""
        # CPU failures
        for i in range(5):
            failure_context = FailureContext(
                task_id=f"task_cpu_{i}",
                worker_id=f"worker_cpu_{i % 2}",
                hardware_profile=self.cpu_profile,
                error_message=f"CPU error {i}",
                error_type=FailureType.SOFTWARE_ERROR,
                timestamp=datetime.now() - timedelta(hours=i),
                attempt=1
            )
            self.manager.failure_history.append(failure_context)
        
        # GPU failures
        for i in range(3):
            failure_context = FailureContext(
                task_id=f"task_gpu_{i}",
                worker_id=f"worker_gpu_{i % 2}",
                hardware_profile=self.gpu_profile,
                error_message=f"CUDA error {i}",
                error_type=FailureType.HARDWARE_ERROR,
                timestamp=datetime.now() - timedelta(hours=i*2),
                attempt=1
            )
            self.manager.failure_history.append(failure_context)
        
        # OOM errors
        for i in range(2):
            failure_context = FailureContext(
                task_id=f"task_oom_{i}",
                worker_id=f"worker_gpu_{i}",
                hardware_profile=self.gpu_profile,
                error_message="CUDA out of memory",
                error_type=FailureType.RESOURCE_EXHAUSTION,
                timestamp=datetime.now() - timedelta(hours=i*3),
                attempt=1
            )
            self.manager.failure_history.append(failure_context)
    
    def _add_sample_recovery_actions(self):
        """Add sample recovery actions to the manager."""
        # CPU task recoveries
        for i in range(5):
            task_id = f"task_cpu_{i}"
            
            # First attempt: delayed retry
            action1 = RecoveryAction(
                strategy=RecoveryStrategy.DELAYED_RETRY,
                worker_id=f"worker_cpu_{i % 2}",
                message="First attempt: retry with delay",
                delay=2.0 ** i  # Exponential backoff
            )
            
            # Second attempt: different worker
            action2 = RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                worker_id=None,
                message="Second attempt: try different worker"
            )
            
            # Add to recovery history
            self.manager.recovery_history[task_id] = [action1, action2]
        
        # GPU task recoveries
        for i in range(3):
            task_id = f"task_gpu_{i}"
            
            # First attempt: different worker
            action1 = RecoveryAction(
                strategy=RecoveryStrategy.DIFFERENT_WORKER,
                worker_id=None,
                message="Hardware error: try different worker"
            )
            
            # Add to recovery history
            self.manager.recovery_history[task_id] = [action1]
        
        # OOM task recoveries
        for i in range(2):
            task_id = f"task_oom_{i}"
            
            # First attempt: reduced batch size
            action1 = RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_BATCH_SIZE,
                worker_id=f"worker_gpu_{i}",
                message="OOM error: reduce batch size"
            )
            
            # Second attempt: reduced precision
            action2 = RecoveryAction(
                strategy=RecoveryStrategy.REDUCED_PRECISION,
                worker_id=f"worker_gpu_{i}",
                message="OOM error: reduce precision"
            )
            
            # Third attempt: fallback to CPU
            action3 = RecoveryAction(
                strategy=RecoveryStrategy.FALLBACK_CPU,
                worker_id=None,
                message="OOM error: fallback to CPU"
            )
            
            # Add to recovery history
            self.manager.recovery_history[task_id] = [action1, action2, action3]
    
    def test_visualization_generation(self):
        """Test that visualizations can be generated correctly."""
        # Call the visualize function
        output_path = visualize_fault_tolerance(self.manager, self.temp_dir)
        
        # Check that the output file exists
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Check the content of the HTML file
        with open(output_path, 'r') as f:
            content = f.read()
            
        # Check that the HTML file contains expected elements
        self.assertIn("Hardware-Aware Fault Tolerance Report", content)
        self.assertIn("Total Failures", content)
        self.assertIn("Detected Patterns", content)
        
        # Check that image files were created
        image_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertGreaterEqual(len(image_files), 1)
    
    def test_manager_create_visualization(self):
        """Test that the manager can create visualizations."""
        # Create visualizations using the manager method
        output_path = self.manager.create_visualization(self.temp_dir)
        
        # Check that the output file exists
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Check that image files were created
        image_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.png')]
        self.assertGreaterEqual(len(image_files), 1)


if __name__ == "__main__":
    unittest.main()