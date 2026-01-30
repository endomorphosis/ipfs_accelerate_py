#!/usr/bin/env python3
"""
Tests for the Hardware Capability Detector.

This module contains tests for the hardware_capability_detector.py module, which is used
to detect hardware capabilities on worker nodes in the distributed testing framework.
"""

import os
import sys
import unittest
import tempfile
import shutil
import uuid
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

pytest.importorskip("psutil")

try:
    # Import hardware capability detector
    from .hardware_capability_detector import (
        HardwareCapabilityDetector,
        HardwareType,
        HardwareVendor,
        PrecisionType,
        CapabilityScore,
        HardwareCapability,
        WorkerHardwareCapabilities
    )
except ImportError:
    from test.distributed_testing.hardware_capability_detector import (
        HardwareCapabilityDetector,
        HardwareType,
        HardwareVendor,
        PrecisionType,
        CapabilityScore,
        HardwareCapability,
        WorkerHardwareCapabilities
    )


class TestHardwareCapabilityDetector(unittest.TestCase):
    """Tests for the HardwareCapabilityDetector class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, 'test_hardware_capabilities.duckdb')
        
        # Create a detector with test database
        self.detector = HardwareCapabilityDetector(
            worker_id=f"test_worker_{uuid.uuid4().hex[:8]}",
            db_path=self.db_path,
            enable_browser_detection=False
        )
    
    def tearDown(self):
        """Clean up after tests."""
        # Close database connection
        if hasattr(self.detector, 'db_connection') and self.detector.db_connection:
            self.detector.db_connection.close()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test detector initialization."""
        # Verify worker ID
        self.assertIsNotNone(self.detector.worker_id)
        self.assertTrue(self.detector.worker_id.startswith("test_worker_"))
        
        # Verify database path
        self.assertEqual(self.detector.db_path, self.db_path)
        
        # Verify database connection
        self.assertIsNotNone(self.detector.db_connection)
    
    @patch('psutil.cpu_count', return_value=8)
    @patch('psutil.virtual_memory')
    def test_detect_all_capabilities(self, mock_virtual_memory, mock_cpu_count):
        """Test detecting all hardware capabilities."""
        # Mock virtual memory
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_virtual_memory.return_value = mock_memory
        
        # Detect capabilities
        capabilities = self.detector.detect_all_capabilities()
        
        # Verify basic info
        self.assertEqual(capabilities.worker_id, self.detector.worker_id)
        self.assertEqual(capabilities.cpu_count, 8)
        self.assertEqual(capabilities.total_memory_gb, 16.0)
        
        # Verify capabilities (at least CPU should be detected)
        self.assertGreaterEqual(len(capabilities.hardware_capabilities), 1)
        
        # Verify CPU capabilities
        cpu_capabilities = [cap for cap in capabilities.hardware_capabilities 
                          if cap.hardware_type == HardwareType.CPU]
        self.assertEqual(len(cpu_capabilities), 1)
        self.assertEqual(cpu_capabilities[0].cores, 8)
    
    def test_generate_hardware_fingerprint(self):
        """Test generating hardware fingerprint."""
        # Create test capabilities
        capabilities = WorkerHardwareCapabilities(
            worker_id="test_worker",
            hostname="test-host",
            os_type="Linux",
            os_version="Test OS 1.0",
            cpu_count=8,
            total_memory_gb=16.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Test CPU",
                    cores=8,
                    memory_gb=16.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.GPU,
                    vendor=HardwareVendor.NVIDIA,
                    model="Test GPU",
                    memory_gb=8.0
                )
            ]
        )
        
        # Generate fingerprint
        fingerprint = self.detector.generate_hardware_fingerprint(capabilities)
        
        # Verify fingerprint properties
        self.assertIsInstance(fingerprint, str)
        self.assertEqual(len(fingerprint), 64)  # SHA-256 is 64 hex chars
        
        # Verify fingerprint consistency
        fingerprint2 = self.detector.generate_hardware_fingerprint(capabilities)
        self.assertEqual(fingerprint, fingerprint2)
        
        # Verify fingerprint changes with different hardware
        capabilities.hardware_capabilities.append(
            HardwareCapability(
                hardware_type=HardwareType.TPU,
                vendor=HardwareVendor.GOOGLE,
                model="Test TPU",
                memory_gb=4.0
            )
        )
        fingerprint3 = self.detector.generate_hardware_fingerprint(capabilities)
        self.assertNotEqual(fingerprint, fingerprint3)
    
    def test_store_and_retrieve_capabilities(self):
        """Test storing and retrieving hardware capabilities."""
        # Create test capabilities
        worker_id = f"test_worker_{uuid.uuid4().hex[:8]}"
        capabilities = WorkerHardwareCapabilities(
            worker_id=worker_id,
            hostname="test-host",
            os_type="Linux",
            os_version="Test OS 1.0",
            cpu_count=8,
            total_memory_gb=16.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Test CPU",
                    cores=8,
                    memory_gb=16.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.INT32
                    ],
                    scores={
                        "compute": CapabilityScore.GOOD,
                        "memory": CapabilityScore.AVERAGE
                    }
                ),
                HardwareCapability(
                    hardware_type=HardwareType.GPU,
                    vendor=HardwareVendor.NVIDIA,
                    model="Test GPU",
                    memory_gb=8.0,
                    supported_precisions=[
                        PrecisionType.FP32,
                        PrecisionType.FP16,
                        PrecisionType.INT8
                    ],
                    scores={
                        "compute": CapabilityScore.EXCELLENT,
                        "memory": CapabilityScore.GOOD
                    }
                )
            ]
        )
        
        # Store capabilities
        self.assertTrue(self.detector.store_capabilities(capabilities))
        
        # Retrieve capabilities
        retrieved = self.detector.get_worker_capabilities(worker_id)
        
        # Verify retrieved capabilities
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.worker_id, worker_id)
        self.assertEqual(retrieved.hostname, "test-host")
        self.assertEqual(retrieved.os_type, "Linux")
        self.assertEqual(retrieved.cpu_count, 8)
        self.assertEqual(retrieved.total_memory_gb, 16.0)
        
        # Verify hardware capabilities
        self.assertEqual(len(retrieved.hardware_capabilities), 2)
        
        # Verify CPU capabilities
        cpu_capabilities = [cap for cap in retrieved.hardware_capabilities 
                          if cap.hardware_type == HardwareType.CPU]
        self.assertEqual(len(cpu_capabilities), 1)
        self.assertEqual(cpu_capabilities[0].vendor, HardwareVendor.INTEL)
        self.assertEqual(cpu_capabilities[0].model, "Test CPU")
        self.assertEqual(cpu_capabilities[0].cores, 8)
        self.assertEqual(cpu_capabilities[0].memory_gb, 16.0)
        
        # Verify GPU capabilities
        gpu_capabilities = [cap for cap in retrieved.hardware_capabilities 
                          if cap.hardware_type == HardwareType.GPU]
        self.assertEqual(len(gpu_capabilities), 1)
        self.assertEqual(gpu_capabilities[0].vendor, HardwareVendor.NVIDIA)
        self.assertEqual(gpu_capabilities[0].model, "Test GPU")
        self.assertEqual(gpu_capabilities[0].memory_gb, 8.0)
        
        # Verify precisions
        gpu_precisions = [p.value for p in gpu_capabilities[0].supported_precisions]
        self.assertIn(PrecisionType.FP32.value, gpu_precisions)
        self.assertIn(PrecisionType.FP16.value, gpu_precisions)
        self.assertIn(PrecisionType.INT8.value, gpu_precisions)
        
        # Verify scores
        self.assertEqual(gpu_capabilities[0].scores.get('compute'), CapabilityScore.EXCELLENT)
        self.assertEqual(gpu_capabilities[0].scores.get('memory'), CapabilityScore.GOOD)
    
    def test_find_compatible_workers(self):
        """Test finding compatible workers."""
        # Create and store test workers
        worker_ids = []
        
        # Worker with NVIDIA GPU
        worker_gpu = WorkerHardwareCapabilities(
            worker_id=f"worker_gpu_{uuid.uuid4().hex[:6]}",
            hostname="gpu-worker",
            os_type="Linux",
            os_version="Test OS 1.0",
            cpu_count=16,
            total_memory_gb=64.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Intel Xeon",
                    cores=16,
                    memory_gb=64.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.GPU,
                    vendor=HardwareVendor.NVIDIA,
                    model="NVIDIA Test GPU",
                    memory_gb=16.0
                )
            ]
        )
        self.detector.store_capabilities(worker_gpu)
        worker_ids.append(worker_gpu.worker_id)
        
        # Worker with CPU only
        worker_cpu = WorkerHardwareCapabilities(
            worker_id=f"worker_cpu_{uuid.uuid4().hex[:6]}",
            hostname="cpu-worker",
            os_type="Linux",
            os_version="Test OS 1.0",
            cpu_count=32,
            total_memory_gb=128.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.AMD,
                    model="AMD EPYC",
                    cores=32,
                    memory_gb=128.0
                )
            ]
        )
        self.detector.store_capabilities(worker_cpu)
        worker_ids.append(worker_cpu.worker_id)
        
        # Worker with WebGPU
        worker_web = WorkerHardwareCapabilities(
            worker_id=f"worker_web_{uuid.uuid4().hex[:6]}",
            hostname="web-worker",
            os_type="Linux",
            os_version="Test OS 1.0",
            cpu_count=4,
            total_memory_gb=8.0,
            hardware_capabilities=[
                HardwareCapability(
                    hardware_type=HardwareType.CPU,
                    vendor=HardwareVendor.INTEL,
                    model="Intel Core i5",
                    cores=4,
                    memory_gb=8.0
                ),
                HardwareCapability(
                    hardware_type=HardwareType.WEBGPU,
                    vendor=HardwareVendor.UNKNOWN,
                    model="Chrome WebGPU",
                    memory_gb=2.0
                )
            ]
        )
        self.detector.store_capabilities(worker_web)
        worker_ids.append(worker_web.worker_id)
        
        # Test finding workers by hardware type
        gpu_workers = self.detector.get_workers_by_hardware_type(HardwareType.GPU)
        self.assertEqual(len(gpu_workers), 1)
        self.assertEqual(gpu_workers[0], worker_gpu.worker_id)
        
        webgpu_workers = self.detector.get_workers_by_hardware_type(HardwareType.WEBGPU)
        self.assertEqual(len(webgpu_workers), 1)
        self.assertEqual(webgpu_workers[0], worker_web.worker_id)
        
        cpu_workers = self.detector.get_workers_by_hardware_type(HardwareType.CPU)
        self.assertEqual(len(cpu_workers), 3)  # All workers have CPU
        
        # Test finding compatible workers with hardware requirements
        gpu_compatible = self.detector.find_compatible_workers(
            hardware_requirements={"hardware_type": HardwareType.GPU}
        )
        self.assertEqual(len(gpu_compatible), 1)
        self.assertEqual(gpu_compatible[0], worker_gpu.worker_id)
        
        # Test finding compatible workers with memory requirements
        high_memory = self.detector.find_compatible_workers(
            hardware_requirements={"hardware_type": HardwareType.CPU},
            min_memory_gb=100.0
        )
        self.assertEqual(len(high_memory), 1)
        self.assertEqual(high_memory[0], worker_cpu.worker_id)
        
        # Test finding compatible workers with preferred hardware types
        preferred = self.detector.find_compatible_workers(
            hardware_requirements={},
            preferred_hardware_types=[HardwareType.GPU, HardwareType.CPU]
        )
        self.assertEqual(len(preferred), 3)  # All workers are compatible
        # GPU worker should be first due to preference
        self.assertEqual(preferred[0], worker_gpu.worker_id)


if __name__ == '__main__':
    unittest.main()