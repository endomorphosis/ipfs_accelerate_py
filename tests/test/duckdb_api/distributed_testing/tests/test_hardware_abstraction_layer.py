"""
Test script for the Hardware Abstraction Layer.

This script demonstrates integration between the enhanced hardware taxonomy
and the hardware abstraction layer.
"""

import unittest
from typing import Dict, Set, Any

from ..hardware_taxonomy import (
    HardwareClass,
    HardwareArchitecture,
    HardwareVendor,
    SoftwareBackend,
    PrecisionType,
    AcceleratorFeature,
    HardwareCapabilityProfile,
    create_cpu_profile,
    create_gpu_profile,
    create_npu_profile,
    create_browser_profile
)
from ..enhanced_hardware_taxonomy import (
    EnhancedHardwareTaxonomy,
    CapabilityScope
)
from ..hardware_abstraction_layer import (
    HardwareAbstractionLayer,
    OperationContext,
    HardwareBackend
)


class TestHardwareAbstractionLayer(unittest.TestCase):
    """Test cases for the HardwareAbstractionLayer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a taxonomy with enhanced capability registry
        self.taxonomy = EnhancedHardwareTaxonomy()
        
        # Create a hardware abstraction layer
        self.hal = HardwareAbstractionLayer(taxonomy=self.taxonomy)
        
        # Create test hardware profiles
        self.cpu_profile = create_cpu_profile(
            model_name="Intel Core i9-12900K",
            vendor=HardwareVendor.INTEL,
            cores=16,
            memory_gb=64.0,
            clock_speed_mhz=5200,
            has_avx=True,
            has_avx2=True,
            has_avx512=True
        )
        
        self.gpu_profile = create_gpu_profile(
            model_name="NVIDIA RTX 4090",
            vendor=HardwareVendor.NVIDIA,
            compute_units=128,
            memory_gb=24.0,
            clock_speed_mhz=2520,
            has_tensor_cores=True,
            has_ray_tracing=True,
            compute_capability="8.9",
            memory_bandwidth_gbps=1008.0,
            tdp_w=450.0
        )
        
        self.npu_profile = create_npu_profile(
            model_name="Qualcomm Hexagon NPU",
            vendor=HardwareVendor.QUALCOMM,
            compute_units=8,
            memory_gb=8.0,
            clock_speed_mhz=1000,
            has_quantization=True,
            tdp_w=5.0
        )
        
        self.browser_profile = create_browser_profile(
            browser_name="Chrome",
            supports_webgpu=True,
            supports_webnn=True,
            gpu_profile=self.gpu_profile
        )
        
        # Register hardware with the HAL
        self.hal.register_hardware(self.cpu_profile)
        self.hal.register_hardware(self.gpu_profile)
        self.hal.register_hardware(self.npu_profile)
        self.hal.register_hardware(self.browser_profile)
        
        # Manually assign capabilities to all hardware profiles
        # First assign matrix_multiplication to both GPU and NPU
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="matrix_multiplication"
        )
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.npu_profile,
            capability_id="matrix_multiplication"
        )
        
        # Then assign tensor_core_acceleration to GPU
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.gpu_profile,
            capability_id="tensor_core_acceleration"
        )
        
        # Assign low_precision_computation to NPU
        self.taxonomy.assign_capability_to_hardware(
            hardware_profile=self.npu_profile,
            capability_id="low_precision_computation"
        )
    
    def test_backend_creation(self):
        """Test backend creation for different hardware types."""
        # Get backends for each hardware profile
        cpu_backend = self.hal.get_backend(self.cpu_profile)
        gpu_backend = self.hal.get_backend(self.gpu_profile)
        npu_backend = self.hal.get_backend(self.npu_profile)
        browser_backend = self.hal.get_backend(self.browser_profile)
        
        # Verify all backends were created
        self.assertIsNotNone(cpu_backend)
        self.assertIsNotNone(gpu_backend)
        self.assertIsNotNone(npu_backend)
        self.assertIsNotNone(browser_backend)
        
        # Verify backend types
        from ..hardware_abstraction_layer import CPUBackend, GPUBackend, NPUBackend, BrowserBackend
        self.assertIsInstance(cpu_backend, CPUBackend)
        self.assertIsInstance(gpu_backend, GPUBackend)
        self.assertIsInstance(npu_backend, NPUBackend)
        self.assertIsInstance(browser_backend, BrowserBackend)
    
    def test_simple_hardware_execution(self):
        """Test simple hardware execution checks."""
        # Create operation contexts without special requirements
        fp16_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.FP16,
            memory_requirement_bytes=1024*1024*1024,  # 1GB
            batch_size=16
        )
        
        int8_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.INT8,
            memory_requirement_bytes=256*1024*1024,  # 256MB
            batch_size=32
        )
        
        # Get backends
        gpu_backend = self.hal.get_backend(self.gpu_profile)
        cpu_backend = self.hal.get_backend(self.cpu_profile)
        npu_backend = self.hal.get_backend(self.npu_profile)
        
        # Test basic execution
        # GPU can handle both FP16 and INT8
        self.assertTrue(gpu_backend.can_execute(fp16_context))
        self.assertTrue(gpu_backend.can_execute(int8_context))
        
        # CPU can handle FP32 but might not support FP16
        fp32_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.FP32,
            memory_requirement_bytes=1024*1024*1024,  # 1GB
            batch_size=16
        )
        self.assertTrue(cpu_backend.can_execute(fp32_context))
        
        # Create a context with unrealistic memory requirements
        large_memory_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.FP32,
            memory_requirement_bytes=1000*1024*1024*1024,  # 1000 GB
            batch_size=16
        )
        
        # No hardware should be able to execute this
        self.assertFalse(gpu_backend.can_execute(large_memory_context))
        self.assertFalse(cpu_backend.can_execute(large_memory_context))
        self.assertFalse(npu_backend.can_execute(large_memory_context))
    
    def test_find_best_backend(self):
        """Test finding the best backend for operations."""
        # Create operation for matrix multiplication
        matrix_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.FP16,
            # No required capabilities so it can run on any hardware
            memory_requirement_bytes=1024*1024*1024,  # 1GB
            batch_size=16,
            prefer_throughput=True
        )
        
        # Find the best backend
        best_result = self.hal.find_best_backend_for_operation(matrix_context)
        self.assertIsNotNone(best_result)
        
        # Best backend should be the GPU for matrix operations
        best_backend, performance = best_result
        self.assertEqual(best_backend.hardware_profile.hardware_class, HardwareClass.GPU)
        
        # Create operation for small int8 operation
        quantized_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.INT8,
            # Don't require specific capabilities for this test
            memory_requirement_bytes=64*1024*1024,  # 64MB
            batch_size=1
        )
        
        # Find the best backend
        best_result = self.hal.find_best_backend_for_operation(quantized_context)
        self.assertIsNotNone(best_result)
        
        # Based on our NPU backend implementation, it should be best for INT8 operations
        best_backend, performance = best_result
        self.assertEqual(best_backend.hardware_profile.hardware_class, HardwareClass.NPU)
    
    def test_browser_specific_optimization(self):
        """Test browser-specific optimizations."""
        # Create a browser-specific context for audio processing
        audio_context = OperationContext(
            operation_type="audio",
            precision=PrecisionType.FP32,
            memory_requirement_bytes=128*1024*1024,  # 128MB
            batch_size=1
        )
        
        # Create different browser profiles
        firefox_profile = create_browser_profile(
            browser_name="Firefox",
            supports_webgpu=True,
            supports_webnn=False,
            gpu_profile=self.gpu_profile
        )
        
        edge_profile = create_browser_profile(
            browser_name="Edge",
            supports_webgpu=True,
            supports_webnn=True,
            gpu_profile=self.gpu_profile
        )
        
        # Register browsers with HAL
        self.hal.register_hardware(firefox_profile)
        self.hal.register_hardware(edge_profile)
        
        # Get backends
        firefox_backend = self.hal.get_backend(firefox_profile)
        edge_backend = self.hal.get_backend(edge_profile)
        chrome_backend = self.hal.get_backend(self.browser_profile)
        
        # Get performance estimates
        firefox_perf = firefox_backend.get_estimated_performance(audio_context)
        edge_perf = edge_backend.get_estimated_performance(audio_context)
        chrome_perf = chrome_backend.get_estimated_performance(audio_context)
        
        # Firefox should be best for audio according to our browser factors
        self.assertGreater(firefox_perf, edge_perf)
        self.assertGreater(firefox_perf, chrome_perf)
        
        # Create a WebNN context
        webnn_context = OperationContext(
            operation_type="inference",
            precision=PrecisionType.FP32,
            memory_requirement_bytes=128*1024*1024,  # 128MB
            batch_size=1
        )
        
        # Edge should be best for WebNN inference
        edge_perf = edge_backend.get_estimated_performance(webnn_context)
        chrome_perf = chrome_backend.get_estimated_performance(webnn_context)
        
        # Edge should be better than Chrome for WebNN
        self.assertGreater(edge_perf, chrome_perf)
    
    def test_backend_specific_optimizations(self):
        """Test backend-specific optimizations for different hardware types."""
        # Create a common context for matrix multiplication
        matrix_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.FP16,
            memory_requirement_bytes=1024*1024*1024,  # 1GB
            batch_size=16
        )
        
        # Get backends
        cpu_backend = self.hal.get_backend(self.cpu_profile)
        gpu_backend = self.hal.get_backend(self.gpu_profile)
        
        # Get performance estimates
        cpu_perf = cpu_backend.get_estimated_performance(matrix_context)
        gpu_perf = gpu_backend.get_estimated_performance(matrix_context)
        
        # GPU should be faster for matrix multiplication with tensor cores
        self.assertGreater(gpu_perf, cpu_perf)
        
        # Create a context for quantized operation
        int8_context = OperationContext(
            operation_type="matmul",
            precision=PrecisionType.INT8,
            memory_requirement_bytes=512*1024*1024,  # 512MB
            batch_size=32
        )
        
        # Get NPU backend
        npu_backend = self.hal.get_backend(self.npu_profile)
        
        # Get performance estimates for int8
        gpu_int8_perf = gpu_backend.get_estimated_performance(int8_context)
        npu_int8_perf = npu_backend.get_estimated_performance(int8_context)
        
        # NPU should excel at int8 operations
        self.assertGreater(npu_int8_perf, gpu_int8_perf)


if __name__ == "__main__":
    unittest.main()