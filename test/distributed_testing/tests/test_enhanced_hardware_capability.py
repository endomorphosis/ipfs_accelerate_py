#!/usr/bin/env python3
"""
Unit tests for the Enhanced Hardware Capability module.

This module contains comprehensive tests for the HardwareCapabilityDetector
and HardwareCapabilityComparator classes to ensure proper detection, comparison,
and compatibility checking of hardware capabilities.
"""

import unittest
import sys
import os
import logging
from unittest.mock import patch, MagicMock
import importlib.util
from typing import Dict, List, Set, Optional, Tuple, Any, Union

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from enhanced_hardware_capability import (
    HardwareCapabilityDetector, HardwareCapabilityComparator,
    HardwareCapability, WorkerHardwareCapabilities,
    HardwareType, HardwareVendor, PrecisionType, CapabilityScore
)


class HardwareCapabilityDetectorTests(unittest.TestCase):
    """Unit tests for the HardwareCapabilityDetector class."""
    
    def setUp(self):
        """Set up a test detector before each test."""
        # Disable logging for tests
        logging.disable(logging.CRITICAL)
        
        # Create a fresh detector with a fixed worker ID for testing
        self.detector = HardwareCapabilityDetector(worker_id="test_worker")
    
    def tearDown(self):
        """Clean up after each test."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.worker_id, "test_worker")
        self.assertIsNotNone(self.detector.os_info)
        self.assertIsNotNone(self.detector.hostname)
    
    def test_generate_worker_id(self):
        """Test worker ID generation."""
        worker_id = self.detector._generate_worker_id()
        self.assertTrue(worker_id.startswith("worker_"))
        self.assertGreaterEqual(len(worker_id), 12)
    
    def test_get_os_info(self):
        """Test OS info retrieval."""
        os_type, os_version = self.detector._get_os_info()
        self.assertIsInstance(os_type, str)
        self.assertIsInstance(os_version, str)
        self.assertGreater(len(os_type), 0)
        self.assertGreater(len(os_version), 0)
    
    def test_get_hostname(self):
        """Test hostname retrieval."""
        hostname = self.detector._get_hostname()
        self.assertIsInstance(hostname, str)
        self.assertGreater(len(hostname), 0)
    
    @unittest.skipUnless(
        importlib.util.find_spec('psutil') is not None and importlib.util.find_spec('cpuinfo') is not None,
        'psutil and py-cpuinfo are required for this test'
    )
    @patch('cpuinfo.get_cpu_info')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_detect_cpu_capabilities(self, mock_vm, mock_cpu_count, mock_cpu_info):
        """Test CPU capabilities detection."""
        # Mock CPU info
        mock_cpu_info.return_value = {
            'vendor_id': 'GenuineIntel',
            'brand_raw': 'Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz',
            'model': '158',
            'hz_advertised_raw': (3600000000, 0),
            'arch': 'X86_64',
            'l1_data_cache_size': 32768,
            'l2_cache_size': 262144,
            'l3_cache_size': 16777216,
            'flags': ['fpu', 'vme', 'de', 'pse', 'tsc', 'msr', 'pae', 'mce', 'cx8', 'apic',
                     'sep', 'mtrr', 'pge', 'mca', 'cmov', 'pat', 'pse36', 'clflush', 'dts',
                     'acpi', 'mmx', 'fxsr', 'sse', 'sse2', 'ss', 'ht', 'tm', 'pbe', 'sse3',
                     'pclmulqdq', 'dtes64', 'monitor', 'ds_cpl', 'vmx', 'est', 'tm2', 'ssse3',
                     'sdbg', 'fma', 'cx16', 'xtpr', 'pdcm', 'pcid', 'dca', 'sse4_1', 'sse4_2',
                     'x2apic', 'movbe', 'popcnt', 'tsc_deadline_timer', 'aes', 'xsave',
                     'avx', 'f16c', 'rdrand', 'lahf_lm', 'abm', '3dnowprefetch', 'cpuid_fault',
                     'invpcid_single', 'pti', 'ssbd', 'ibrs', 'ibpb', 'stibp', 'l1tf', 'avx2',
                     'fsgsbase', 'bmi1', 'hle', 'smep', 'bmi2', 'erms', 'invpcid', 'rtm', 'mpx',
                     'rdseed', 'adx', 'smap', 'clflushopt', 'xsaveopt', 'xsavec', 'xgetbv1',
                     'xsaves', 'dtherm', 'ida', 'arat', 'pln', 'pts', 'hwp', 'hwp_notify',
                     'hwp_act_window', 'hwp_epp', 'md_clear']
        }
        
        # Mock CPU count and memory
        mock_cpu_count.side_effect = lambda logical: 8 if not logical else 16
        memory_mock = MagicMock()
        memory_mock.total = 16 * 1024**3  # 16 GB
        mock_vm.return_value = memory_mock
        
        # Get CPU capabilities
        capability = self.detector.detect_cpu_capabilities()
        
        # Verify capability properties
        self.assertEqual(capability.hardware_type, HardwareType.CPU)
        self.assertEqual(capability.vendor, HardwareVendor.INTEL)
        self.assertEqual(capability.model, 'Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz')
        self.assertEqual(capability.version, '158')
        self.assertEqual(capability.cores, 8)
        self.assertEqual(capability.memory_gb, 16.0)
        
        # Verify supported precisions
        self.assertIn(PrecisionType.FP64, capability.supported_precisions)
        self.assertIn(PrecisionType.FP32, capability.supported_precisions)
        self.assertIn(PrecisionType.INT64, capability.supported_precisions)
        self.assertIn(PrecisionType.INT32, capability.supported_precisions)
        
        # Verify capabilities
        self.assertEqual(capability.capabilities['threads'], 16)
        self.assertEqual(capability.capabilities['architecture'], 'X86_64')
        self.assertEqual(capability.capabilities['frequency_mhz'], 3600.0)
        self.assertEqual(capability.capabilities['l1_cache_kb'], 32.0)
        self.assertEqual(capability.capabilities['l2_cache_kb'], 256.0)
        self.assertEqual(capability.capabilities['l3_cache_kb'], 16384.0)
        self.assertTrue(capability.capabilities['avx'])
        self.assertTrue(capability.capabilities['avx2'])
        self.assertTrue(capability.capabilities['sse4'])
        
        # Verify scores
        self.assertIn('compute', capability.scores)
        self.assertIn('memory', capability.scores)
        self.assertIn('vector', capability.scores)
        self.assertIn('precision', capability.scores)
        self.assertIn('overall', capability.scores)
    
    @unittest.skipUnless(
        importlib.util.find_spec('pynvml') is not None,
        'pynvml is required for this test'
    )
    @patch('pynvml.nvmlInit')
    @patch('pynvml.nvmlDeviceGetCount')
    @patch('pynvml.nvmlDeviceGetHandleByIndex')
    @patch('pynvml.nvmlDeviceGetName')
    @patch('pynvml.nvmlDeviceGetMemoryInfo')
    @patch('pynvml.nvmlDeviceGetCudaComputeCapability')
    @patch('pynvml.nvmlSystemGetDriverVersion')
    @patch('pynvml.nvmlDeviceGetTotalEccErrors')
    @patch('pynvml.nvmlShutdown')
    def test_detect_nvidia_gpus(self, mock_shutdown, mock_ecc, mock_driver, 
                              mock_cc, mock_memory, mock_name, mock_handle, 
                              mock_count, mock_init):
        """Test NVIDIA GPU detection."""
        # Mock NVML
        mock_count.return_value = 2
        
        # Mock device handle
        def handle_by_index(idx):
            return f"handle_{idx}"
        mock_handle.side_effect = handle_by_index
        
        # Mock device info
        def device_name(handle):
            if handle == "handle_0":
                return "NVIDIA GeForce RTX 3080"
            else:
                return "NVIDIA GeForce RTX 3070"
        mock_name.side_effect = device_name
        
        # Mock memory info
        memory_info_0 = MagicMock()
        memory_info_0.total = 10 * 1024**3  # 10 GB
        memory_info_0.used = 2 * 1024**3    # 2 GB
        
        memory_info_1 = MagicMock()
        memory_info_1.total = 8 * 1024**3   # 8 GB
        memory_info_1.used = 1 * 1024**3    # 1 GB
        
        def get_memory_info(handle):
            if handle == "handle_0":
                return memory_info_0
            else:
                return memory_info_1
        mock_memory.side_effect = get_memory_info
        
        # Mock compute capability
        def get_compute_capability(handle):
            if handle == "handle_0":
                return (8, 6)  # RTX 3080
            else:
                return (8, 5)  # RTX 3070
        mock_cc.side_effect = get_compute_capability
        
        # Mock driver version
        mock_driver.return_value = "460.91.03"
        
        # Mock ECC errors (not available)
        mock_ecc.return_value = -1
        
        # Detect NVIDIA GPUs
        gpu_capabilities = self.detector._detect_nvidia_gpus()
        
        # Verify 2 GPUs detected
        self.assertEqual(len(gpu_capabilities), 2)
        
        # Verify first GPU
        gpu0 = gpu_capabilities[0]
        self.assertEqual(gpu0.hardware_type, HardwareType.GPU)
        self.assertEqual(gpu0.vendor, HardwareVendor.NVIDIA)
        self.assertEqual(gpu0.model, "NVIDIA GeForce RTX 3080")
        self.assertEqual(gpu0.version, "8.6")
        self.assertEqual(gpu0.driver_version, "460.91.03")
        self.assertAlmostEqual(gpu0.memory_gb, 10.0)
        
        # Verify supported precisions for Ampere GPU
        self.assertIn(PrecisionType.FP32, gpu0.supported_precisions)
        self.assertIn(PrecisionType.FP16, gpu0.supported_precisions)
        self.assertIn(PrecisionType.INT8, gpu0.supported_precisions)
        self.assertIn(PrecisionType.BF16, gpu0.supported_precisions)
        self.assertIn(PrecisionType.INT4, gpu0.supported_precisions)
        
        # Verify capabilities
        self.assertEqual(gpu0.capabilities['compute_capability'], "8.6")
        self.assertTrue(gpu0.capabilities['tensor_cores'])
        self.assertFalse(gpu0.capabilities['ecc_enabled'])
        self.assertAlmostEqual(gpu0.capabilities['memory_used_gb'], 2.0)
        
        # Verify scores
        self.assertIn('compute', gpu0.scores)
        self.assertIn('memory', gpu0.scores)
        self.assertIn('precision', gpu0.scores)
        self.assertIn('overall', gpu0.scores)
        
        # Verify compute score for Ampere GPU should be excellent
        self.assertEqual(gpu0.scores['compute'], CapabilityScore.EXCELLENT)
    
    @patch('subprocess.run')
    def test_detect_amd_gpus(self, mock_run):
        """Test AMD GPU detection."""
        # Mock rocm-smi not found
        mock_run.return_value.returncode = 1
        
        # Detect AMD GPUs - should be empty since rocm-smi failed
        gpu_capabilities = self.detector._detect_amd_gpus()
        self.assertEqual(len(gpu_capabilities), 0)
        
        # Mock rocm-smi found but failed to run
        mock_run.reset_mock()
        mock_run.side_effect = [
            MagicMock(returncode=0),  # which rocm-smi
            MagicMock(returncode=1)   # rocm-smi command
        ]
        
        # Detect AMD GPUs - should be empty since rocm-smi failed
        gpu_capabilities = self.detector._detect_amd_gpus()
        self.assertEqual(len(gpu_capabilities), 0)
        
        # Mock successful rocm-smi run
        mock_run.reset_mock()
        mock_run.side_effect = [
            MagicMock(returncode=0),  # which rocm-smi
            MagicMock(
                returncode=0,
                stdout=b'''{
                    "0": {
                        "Card series": "AMD Radeon RX 6800",
                        "Card model": "AMD Radeon RX 6800 XT",
                        "Driver version": "21.30.4",
                        "Compute Units": 72,
                        "Memory Total": "16 GB",
                        "Memory Used": "2 GB",
                        "Memory Clock": "1000 MHz",
                        "GPU Clock": "2250 MHz",
                        "Temperature": "65C",
                        "Power": "220W"
                    },
                    "1": {
                        "Card series": "AMD Radeon VII",
                        "Card model": "AMD Radeon VII",
                        "Driver version": "21.30.4",
                        "Compute Units": 60,
                        "Memory Total": "16 GB",
                        "Memory Used": "1 GB",
                        "Memory Clock": "1000 MHz",
                        "GPU Clock": "1800 MHz",
                        "Temperature": "70C",
                        "Power": "200W"
                    }
                }''',
                stderr=b''
            )
        ]
        
        # Detect AMD GPUs
        gpu_capabilities = self.detector._detect_amd_gpus()
        
        # Verify 2 GPUs detected
        self.assertEqual(len(gpu_capabilities), 2)
        
        # Verify first GPU
        gpu0 = gpu_capabilities[0]
        self.assertEqual(gpu0.hardware_type, HardwareType.GPU)
        self.assertEqual(gpu0.vendor, HardwareVendor.AMD)
        self.assertEqual(gpu0.model, "AMD Radeon RX 6800")
        self.assertEqual(gpu0.version, "AMD Radeon RX 6800 XT")
        self.assertEqual(gpu0.driver_version, "21.30.4")
        self.assertEqual(gpu0.compute_units, 72)
        self.assertEqual(gpu0.memory_gb, 16.0)
        
        # Verify capabilities
        self.assertEqual(gpu0.capabilities['memory_clock'], "1000 MHz")
        self.assertEqual(gpu0.capabilities['gpu_clock'], "2250 MHz")
        self.assertEqual(gpu0.capabilities['memory_used'], "2 GB")
        self.assertEqual(gpu0.capabilities['temperature'], "65C")
        self.assertEqual(gpu0.capabilities['power'], "220W")
    
    @patch('subprocess.run')
    def test_detect_apple_gpus(self, mock_run):
        """Test Apple GPU detection."""
        # Mock system_profiler run
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=b'''{
                "SPDisplaysDataType": [
                    {
                        "spdisplays_device_name": "Apple M1 Pro",
                        "spdisplays_metal": true,
                        "spdisplays_mtlgpufamilysupport": "MTLGPUFamilyApple7",
                        "spdisplays_vendor": "Apple"
                    }
                ]
            }''',
            stderr=b''
        )
        
        # Detect Apple GPUs
        gpu_capabilities = self.detector._detect_apple_gpus()
        
        # Check if test is running on macOS, if not, detection should return empty list
        import platform
        if platform.system() != 'Darwin':
            self.assertEqual(len(gpu_capabilities), 0)
            return
        
        # On macOS, verify GPU detected
        self.assertEqual(len(gpu_capabilities), 1)
        
        # Verify GPU info
        gpu0 = gpu_capabilities[0]
        self.assertEqual(gpu0.hardware_type, HardwareType.GPU)
        self.assertEqual(gpu0.vendor, HardwareVendor.APPLE)
        self.assertEqual(gpu0.model, "Apple M1 Pro")
        self.assertEqual(gpu0.version, "MTLGPUFamilyApple7")
        
        # Verify capabilities
        self.assertTrue(gpu0.capabilities['metal_support'])
        self.assertEqual(gpu0.capabilities['metal_family'], "MTLGPUFamilyApple7")
        self.assertTrue(gpu0.capabilities['apple_silicon'])
        
        # Verify supported precisions
        self.assertIn(PrecisionType.FP32, gpu0.supported_precisions)
        self.assertIn(PrecisionType.FP16, gpu0.supported_precisions)
        self.assertIn(PrecisionType.INT32, gpu0.supported_precisions)
        self.assertIn(PrecisionType.INT16, gpu0.supported_precisions)
        self.assertIn(PrecisionType.INT8, gpu0.supported_precisions)
        self.assertIn(PrecisionType.BF16, gpu0.supported_precisions)
    
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_cpu_capabilities')
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_gpu_capabilities')
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_tpu_capabilities')
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_npu_capabilities')
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_webgpu_capabilities')
    @patch('enhanced_hardware_capability.HardwareCapabilityDetector.detect_webnn_capabilities')
    def test_detect_all_capabilities(self, mock_webnn, mock_webgpu, mock_npu, 
                                   mock_tpu, mock_gpu, mock_cpu):
        """Test detecting all capabilities."""
        # Mock CPU detection
        cpu_capability = HardwareCapability(
            hardware_type=HardwareType.CPU,
            vendor=HardwareVendor.INTEL,
            model="Intel Core i9-9900K",
            cores=8,
            memory_gb=32.0
        )
        mock_cpu.return_value = cpu_capability
        
        # Mock GPU detection
        gpu_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA GeForce RTX 3080",
            memory_gb=10.0
        )
        mock_gpu.return_value = [gpu_capability]
        
        # Mock TPU detection (none)
        mock_tpu.return_value = []
        
        # Mock NPU detection (none)
        mock_npu.return_value = []
        
        # Mock WebGPU detection (none)
        mock_webgpu.return_value = None
        
        # Mock WebNN detection (none)
        mock_webnn.return_value = None
        
        # Detect all capabilities
        worker_capabilities = self.detector.detect_all_capabilities()
        
        # Verify worker capabilities
        self.assertEqual(worker_capabilities.worker_id, "test_worker")
        self.assertIsNotNone(worker_capabilities.os_type)
        self.assertIsNotNone(worker_capabilities.os_version)
        self.assertIsNotNone(worker_capabilities.hostname)
        self.assertIsNotNone(worker_capabilities.cpu_count)
        self.assertIsNotNone(worker_capabilities.total_memory_gb)
        self.assertIsNotNone(worker_capabilities.last_updated)
        
        # Verify hardware capabilities
        self.assertEqual(len(worker_capabilities.hardware_capabilities), 2)
        self.assertEqual(worker_capabilities.hardware_capabilities[0].hardware_type, HardwareType.CPU)
        self.assertEqual(worker_capabilities.hardware_capabilities[1].hardware_type, HardwareType.GPU)


class HardwareCapabilityComparatorTests(unittest.TestCase):
    """Unit tests for the HardwareCapabilityComparator class."""
    
    def setUp(self):
        """Set up test capabilities and comparator before each test."""
        # Disable logging for tests
        logging.disable(logging.CRITICAL)
        
        # Create a fresh comparator
        self.comparator = HardwareCapabilityComparator()
        
        # Create test capabilities
        self.cpu_capability = HardwareCapability(
            hardware_type=HardwareType.CPU,
            vendor=HardwareVendor.INTEL,
            model="Intel Core i9-9900K",
            cores=8,
            memory_gb=16.0,
            supported_precisions=[
                PrecisionType.FP64,
                PrecisionType.FP32,
                PrecisionType.INT64,
                PrecisionType.INT32
            ],
            scores={
                'compute': CapabilityScore.GOOD,
                'memory': CapabilityScore.AVERAGE,
                'vector': CapabilityScore.GOOD,
                'precision': CapabilityScore.AVERAGE,
                'overall': CapabilityScore.GOOD
            }
        )
        
        self.gpu_capability = HardwareCapability(
            hardware_type=HardwareType.GPU,
            vendor=HardwareVendor.NVIDIA,
            model="NVIDIA GeForce RTX 3080",
            memory_gb=10.0,
            supported_precisions=[
                PrecisionType.FP32,
                PrecisionType.FP16,
                PrecisionType.INT32,
                PrecisionType.INT16,
                PrecisionType.INT8
            ],
            scores={
                'compute': CapabilityScore.EXCELLENT,
                'memory': CapabilityScore.GOOD,
                'precision': CapabilityScore.EXCELLENT,
                'overall': CapabilityScore.EXCELLENT
            },
            capabilities={
                'compute_capability': "8.6",
                'tensor_cores': True
            }
        )
        
        self.tpu_capability = HardwareCapability(
            hardware_type=HardwareType.TPU,
            vendor=HardwareVendor.GOOGLE,
            model="Google TPU v3",
            memory_gb=16.0,
            supported_precisions=[
                PrecisionType.FP32,
                PrecisionType.BF16
            ],
            scores={
                'compute': CapabilityScore.EXCELLENT,
                'memory': CapabilityScore.GOOD,
                'precision': CapabilityScore.GOOD,
                'overall': CapabilityScore.GOOD
            }
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Re-enable logging
        logging.disable(logging.NOTSET)
    
    def test_compare_capabilities(self):
        """Test comparing two hardware capabilities."""
        # Compare CPU and GPU
        comparison = self.comparator.compare_capabilities(self.cpu_capability, self.gpu_capability)
        
        # Verify comparison results
        self.assertFalse(comparison['same_type'])
        self.assertFalse(comparison['same_vendor'])
        
        # Verify score comparisons
        self.assertIn('overall', comparison['score_comparisons'])
        self.assertEqual(comparison['score_comparisons']['overall']['first'], 'GOOD')
        self.assertEqual(comparison['score_comparisons']['overall']['second'], 'EXCELLENT')
        self.assertEqual(comparison['score_comparisons']['overall']['difference'], -1)
        
        # Verify memory ratio
        self.assertAlmostEqual(comparison['memory_ratio'], 1.6)
        
        # Verify precision comparison
        self.assertIn('FP64', comparison['precision_comparison']['first_only'])
        self.assertIn('INT64', comparison['precision_comparison']['first_only'])
        self.assertIn('FP16', comparison['precision_comparison']['second_only'])
        self.assertIn('INT16', comparison['precision_comparison']['second_only'])
        self.assertIn('INT8', comparison['precision_comparison']['second_only'])
        self.assertIn('FP32', comparison['precision_comparison']['common'])
        self.assertIn('INT32', comparison['precision_comparison']['common'])
        
        # Compare GPU and TPU (similar types for ML)
        comparison = self.comparator.compare_capabilities(self.gpu_capability, self.tpu_capability)
        
        # Verify comparison results
        self.assertFalse(comparison['same_type'])
        self.assertFalse(comparison['same_vendor'])
        
        # Verify precision comparison
        self.assertIn('BF16', comparison['precision_comparison']['second_only'])
        self.assertIn('FP32', comparison['precision_comparison']['common'])
    
    def test_is_compatible(self):
        """Test checking if a capability is compatible with requirements."""
        # Check GPU compatibility with matching requirements
        requirements = {
            'hardware_type': HardwareType.GPU,
            'min_memory_gb': 8.0,
            'required_precisions': [PrecisionType.FP16, PrecisionType.INT8],
            'min_scores': {
                'compute': CapabilityScore.GOOD
            },
            'required_capabilities': {
                'tensor_cores': True
            }
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertTrue(is_compatible)
        self.assertTrue(details['hardware_type']['compatible'])
        self.assertTrue(details['memory']['compatible'])
        self.assertTrue(details['precision']['compatible'])
        self.assertTrue(details['score_compute']['compatible'])
        self.assertTrue(details['capability_tensor_cores']['compatible'])
        
        # Check with incompatible hardware type
        requirements = {
            'hardware_type': HardwareType.TPU
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertFalse(is_compatible)
        self.assertFalse(details['hardware_type']['compatible'])
        
        # Check with incompatible memory requirement
        requirements = {
            'hardware_type': HardwareType.GPU,
            'min_memory_gb': 16.0
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertFalse(is_compatible)
        self.assertFalse(details['memory']['compatible'])
        
        # Check with incompatible precision requirement
        requirements = {
            'hardware_type': HardwareType.GPU,
            'required_precisions': [PrecisionType.FP64]
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertFalse(is_compatible)
        self.assertFalse(details['precision']['compatible'])
        
        # Check with incompatible score requirement
        requirements = {
            'hardware_type': HardwareType.GPU,
            'min_scores': {
                'memory': CapabilityScore.EXCELLENT
            }
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertFalse(is_compatible)
        self.assertFalse(details['score_memory']['compatible'])
        
        # Check with incompatible capability requirement
        requirements = {
            'hardware_type': HardwareType.GPU,
            'required_capabilities': {
                'tensor_cores': False
            }
        }
        
        is_compatible, details = self.comparator.is_compatible(self.gpu_capability, requirements)
        self.assertFalse(is_compatible)
        self.assertFalse(details['capability_tensor_cores']['compatible'])
    
    def test_estimate_performance(self):
        """Test estimating performance for a workload."""
        # Estimate inference performance for GPU
        estimate = self.comparator.estimate_performance(
            self.gpu_capability, 
            "inference", 
            {
                'precision': PrecisionType.FP16,
                'memory_gb': 4.0,
                'batch_size': 8
            }
        )
        
        # Verify estimate properties
        self.assertGreater(estimate['relative_score'], 0.0)
        self.assertGreater(estimate['confidence'], 0.0)
        self.assertIn(estimate['recommendation_level'], 
                     ['excellent', 'good', 'average', 'minimal', 'not_recommended'])
        
        # GPU should be recommended for inference
        self.assertIn(estimate['recommendation_level'], ['excellent', 'good'])
        
        # Estimate inference performance for CPU
        cpu_estimate = self.comparator.estimate_performance(
            self.cpu_capability, 
            "inference", 
            {
                'precision': PrecisionType.FP32,
                'memory_gb': 4.0,
                'batch_size': 1
            }
        )
        
        # Verify CPU should have lower score than GPU for inference
        self.assertLess(cpu_estimate['relative_score'], estimate['relative_score'])
        
        # Estimate training performance
        train_estimate = self.comparator.estimate_performance(
            self.gpu_capability, 
            "training", 
            {
                'precision': PrecisionType.FP16,
                'memory_gb': 8.0,
                'batch_size': 16
            }
        )
        
        # Verify estimate properties for training
        self.assertGreater(train_estimate['relative_score'], 0.0)
        
        # Estimate with incompatible precision
        incompatible_estimate = self.comparator.estimate_performance(
            self.gpu_capability, 
            "inference", 
            {
                'precision': PrecisionType.FP64,
                'memory_gb': 4.0
            }
        )
        
        # Score should be lower due to incompatible precision
        self.assertLess(incompatible_estimate['relative_score'], estimate['relative_score'])
        self.assertLess(incompatible_estimate['confidence'], estimate['confidence'])
    
    def test_find_best_hardware(self):
        """Test finding best hardware for a workload."""
        # Create a list of capabilities
        capabilities = [self.cpu_capability, self.gpu_capability, self.tpu_capability]
        
        # Find best hardware for inference
        best_capability, estimate = self.comparator.find_best_hardware(
            capabilities,
            "inference",
            {
                'precision': PrecisionType.FP32,
                'memory_gb': 4.0,
                'batch_size': 4
            }
        )
        
        # Verify GPU or TPU should be selected for inference
        self.assertIn(best_capability.hardware_type, [HardwareType.GPU, HardwareType.TPU])
        
        # Empty list should raise ValueError
        with self.assertRaises(ValueError):
            self.comparator.find_best_hardware([], "inference")
        
        # Find best hardware for inference with FP64 precision (only CPU supports)
        best_capability, estimate = self.comparator.find_best_hardware(
            capabilities,
            "inference",
            {
                'precision': PrecisionType.FP64,
                'memory_gb': 4.0
            }
        )
        
        # Only CPU supports FP64
        self.assertEqual(best_capability.hardware_type, HardwareType.CPU)


if __name__ == '__main__':
    unittest.main()