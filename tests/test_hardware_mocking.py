#!/usr/bin/env python3
"""
Hardware Mocking System for IPFS Accelerate Python

This module provides comprehensive mocking and simulation capabilities for all supported
hardware platforms, allowing tests to run without requiring actual hardware.

Features:
- Mock CUDA, ROCm, MPS, OpenVINO, WebNN, WebGPU, Qualcomm backends
- Realistic performance simulation
- Configurable hardware capabilities
- Environment variable controls
- Integration with pytest fixtures
"""

import os
import sys
import contextlib
from typing import Dict, Any, List, Optional, Union
from unittest.mock import MagicMock, patch
import random
import time

# Hardware type constants (matching hardware_detection.py)
CPU = "cpu"
CUDA = "cuda" 
ROCM = "rocm"
MPS = "mps"
OPENVINO = "openvino"
WEBNN = "webnn"
WEBGPU = "webgpu"
QUALCOMM = "qualcomm"

class MockHardwareConfig:
    """Configuration for mock hardware simulation."""
    
    def __init__(self):
        self.enabled_hardware = {
            CPU: True,  # Always enabled
            CUDA: os.environ.get('MOCK_CUDA', 'false').lower() in ('true', '1', 'yes'),
            ROCM: os.environ.get('MOCK_ROCM', 'false').lower() in ('true', '1', 'yes'),
            MPS: os.environ.get('MOCK_MPS', 'false').lower() in ('true', '1', 'yes'),
            OPENVINO: os.environ.get('MOCK_OPENVINO', 'false').lower() in ('true', '1', 'yes'),
            WEBNN: os.environ.get('MOCK_WEBNN', 'false').lower() in ('true', '1', 'yes'),
            WEBGPU: os.environ.get('MOCK_WEBGPU', 'false').lower() in ('true', '1', 'yes'),
            QUALCOMM: os.environ.get('MOCK_QUALCOMM', 'false').lower() in ('true', '1', 'yes'),
        }
        
        # Default hardware capabilities
        self.hardware_capabilities = {
            CPU: {
                "cores": 8,
                "memory_gb": 16,
                "supports_avx": True,
                "supports_avx2": True,
            },
            CUDA: {
                "device_count": 1,
                "memory_gb": 8,
                "compute_capability": (7, 5),
                "cuda_version": "11.8",
                "driver_version": "520.61.05",
            },
            ROCM: {
                "device_count": 1,
                "memory_gb": 8,
                "rocm_version": "5.4.0",
                "gfx_arch": "gfx900",
            },
            MPS: {
                "device_count": 1,
                "memory_gb": 16,  # Unified memory on Apple Silicon
                "metal_version": "3.0",
                "chip": "M2",
            },
            OPENVINO: {
                "version": "2023.0",
                "supported_devices": ["CPU", "GPU"],
                "inference_precision": ["FP32", "FP16", "INT8"],
            },
            WEBNN: {
                "supported_operations": ["conv2d", "matmul", "relu", "softmax"],
                "preferred_layout": "nchw",
                "max_tensor_size": 2**30,  # 1GB
            },
            WEBGPU: {
                "adapter_type": "discrete-gpu",
                "vendor": "mock-vendor",
                "max_buffer_size": 2**30,  # 1GB
                "supports_compute_shaders": True,
            },
            QUALCOMM: {
                "npu_version": "2.0",
                "dsp_arch": "v68",
                "supported_precisions": ["FP32", "FP16", "INT8"],
            }
        }

class MockTorch:
    """Mock PyTorch module for testing without torch dependency."""
    
    class cuda:
        @staticmethod
        def is_available():
            return MockHardwareConfig().enabled_hardware[CUDA]
        
        @staticmethod
        def device_count():
            if MockHardwareConfig().enabled_hardware[CUDA]:
                return MockHardwareConfig().hardware_capabilities[CUDA]["device_count"]
            return 0
        
        @staticmethod
        def get_device_name(device=0):
            return f"Mock CUDA Device {device}"
        
        @staticmethod  
        def get_device_properties(device=0):
            props = MagicMock()
            capabilities = MockHardwareConfig().hardware_capabilities[CUDA]
            props.name = f"Mock CUDA Device {device}"
            props.total_memory = capabilities["memory_gb"] * 1024 * 1024 * 1024
            props.major = capabilities["compute_capability"][0]
            props.minor = capabilities["compute_capability"][1]
            return props
    
    class mps:
        @staticmethod
        def is_available():
            return MockHardwareConfig().enabled_hardware[MPS]
    
    class version:
        cuda = MockHardwareConfig().hardware_capabilities[CUDA].get("cuda_version")
        
    @staticmethod
    def tensor(data, device='cpu'):
        """Mock tensor creation."""
        mock_tensor = MagicMock()
        mock_tensor.device = device
        mock_tensor.shape = getattr(data, 'shape', [])
        return mock_tensor

class MockImportSpec:
    """Mock importlib.util.find_spec results."""
    
    def __init__(self, name: str, available: bool = True):
        self.name = name
        self.origin = f"/mock/path/{name}.py" if available else None
    
    def __bool__(self):
        return self.origin is not None

class HardwareMocker:
    """Main hardware mocking class."""
    
    def __init__(self, config: Optional[MockHardwareConfig] = None):
        self.config = config or MockHardwareConfig()
        self.active_patches = []
        
    def mock_torch(self):
        """Mock PyTorch and related functionality."""
        # Mock torch import
        torch_patch = patch.dict('sys.modules', {'torch': MockTorch()})
        self.active_patches.append(torch_patch)
        return torch_patch
    
    def mock_importlib_find_spec(self):
        """Mock importlib.util.find_spec for various libraries."""
        def mock_find_spec(name):
            # Map package names to hardware types
            package_mapping = {
                'torch': CUDA,  # Will be handled separately
                'openvino': OPENVINO,
                'webnn': WEBNN,
                'webnn_js': WEBNN,
                'webgpu': WEBGPU,
                'wgpu': WEBGPU,
                'qnn_wrapper': QUALCOMM,
                'qti': QUALCOMM,
            }
            
            hardware_type = package_mapping.get(name)
            if hardware_type and self.config.enabled_hardware.get(hardware_type, False):
                return MockImportSpec(name, available=True)
            elif hardware_type:
                return None  # Not available
            else:
                # For unknown packages, return None (not available)
                return None
        
        import_patch = patch('importlib.util.find_spec', side_effect=mock_find_spec)
        self.active_patches.append(import_patch)
        return import_patch
    
    def mock_environment_variables(self):
        """Mock environment variables for hardware detection."""
        env_vars = {}
        
        # Add hardware-specific environment variables based on config
        if self.config.enabled_hardware[ROCM]:
            env_vars['ROCM_HOME'] = '/opt/rocm'
            
        if self.config.enabled_hardware[QUALCOMM]:
            env_vars['QUALCOMM_SDK'] = '/opt/qualcomm'
            
        if self.config.enabled_hardware[WEBNN]:
            env_vars['WEBNN_AVAILABLE'] = '1'
            
        if self.config.enabled_hardware[WEBGPU]:
            env_vars['WEBGPU_AVAILABLE'] = '1'
        
        # Set simulation flags for web platforms when enabled
        if self.config.enabled_hardware[WEBNN]:
            env_vars['WEBNN_SIMULATION'] = '1'
        if self.config.enabled_hardware[WEBGPU]:
            env_vars['WEBGPU_SIMULATION'] = '1'
        
        env_patch = patch.dict(os.environ, env_vars)
        self.active_patches.append(env_patch)
        return env_patch
    
    def mock_performance_simulation(self):
        """Add realistic performance simulation."""
        # This would add delays and resource usage simulation
        # For now, we'll keep it simple
        pass
    
    def enable_all_hardware(self):
        """Enable all hardware types for comprehensive testing."""
        for hardware_type in self.config.enabled_hardware:
            if hardware_type != CPU:  # CPU is always enabled
                self.config.enabled_hardware[hardware_type] = True
    
    def enable_specific_hardware(self, hardware_types: List[str]):
        """Enable only specific hardware types."""
        # First disable all except CPU
        for hardware_type in self.config.enabled_hardware:
            if hardware_type != CPU:
                self.config.enabled_hardware[hardware_type] = False
        
        # Then enable requested types
        for hardware_type in hardware_types:
            if hardware_type in self.config.enabled_hardware:
                self.config.enabled_hardware[hardware_type] = True
    
    @contextlib.contextmanager
    def mock_hardware_environment(self):
        """Context manager to set up complete mock hardware environment."""
        patches = [
            self.mock_torch(),
            self.mock_importlib_find_spec(),
            self.mock_environment_variables(),
        ]
        
        # Start all patches
        for patch_obj in patches:
            patch_obj.start()
        
        try:
            yield self.config
        finally:
            # Stop all patches
            for patch_obj in reversed(patches):
                patch_obj.stop()

# Convenience functions
def create_cpu_only_environment():
    """Create environment with only CPU available."""
    config = MockHardwareConfig()
    # CPU is always enabled, others are disabled by default
    return HardwareMocker(config)

def create_cuda_environment():
    """Create environment with CPU and CUDA available."""
    config = MockHardwareConfig()
    config.enabled_hardware[CUDA] = True
    return HardwareMocker(config)

def create_web_environment():
    """Create environment with CPU, WebNN, and WebGPU available."""
    config = MockHardwareConfig()
    config.enabled_hardware[WEBNN] = True
    config.enabled_hardware[WEBGPU] = True
    return HardwareMocker(config)

def create_comprehensive_environment():
    """Create environment with all hardware types available."""
    config = MockHardwareConfig()
    mocker = HardwareMocker(config)
    mocker.enable_all_hardware()
    return mocker

# Pytest fixtures
try:
    import pytest
    
    @pytest.fixture
    def mock_cpu_only():
        """Pytest fixture for CPU-only environment."""
        mocker = create_cpu_only_environment()
        with mocker.mock_hardware_environment() as config:
            yield config
    
    @pytest.fixture
    def mock_cuda():
        """Pytest fixture for CUDA environment."""
        mocker = create_cuda_environment()
        with mocker.mock_hardware_environment() as config:
            yield config
    
    @pytest.fixture
    def mock_web_hardware():
        """Pytest fixture for web hardware environment."""
        mocker = create_web_environment()
        with mocker.mock_hardware_environment() as config:
            yield config
    
    @pytest.fixture
    def mock_all_hardware():
        """Pytest fixture for comprehensive hardware environment."""
        mocker = create_comprehensive_environment()
        with mocker.mock_hardware_environment() as config:
            yield config

except ImportError:
    # pytest not available, skip fixtures
    pass

if __name__ == "__main__":
    # Test the mocking system
    print("Testing hardware mocking system...")
    
    # Test CPU-only environment
    print("\n1. Testing CPU-only environment:")
    mocker = create_cpu_only_environment()
    with mocker.mock_hardware_environment():
        import hardware_detection
        detector = hardware_detection.HardwareDetector()
        available = detector.get_available_hardware()
        print(f"Available hardware: {available}")
    
    # Test CUDA environment
    print("\n2. Testing CUDA environment:")
    mocker = create_cuda_environment()
    with mocker.mock_hardware_environment():
        # We need to reload the module to pick up the new mocks
        import importlib
        import hardware_detection
        importlib.reload(hardware_detection)
        detector = hardware_detection.HardwareDetector()
        available = detector.get_available_hardware()
        print(f"Available hardware: {available}")
    
    print("\n✓ Hardware mocking system test completed!")