"""
Test for WebGPU matmul operations.

This test verifies matrix multiplication operations on WebGPU.
"""

import pytest
import numpy as np
import time
import torch
import sys
from pathlib import Path

# Add the root directory to the Python path
test_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

from common.hardware_detection import (
    skip_if_no_webgpu,
    is_webgpu_available,
    get_webgpu_device
)


@pytest.fixture
def webgpu_device():
    """Get WebGPU device for testing."""
    if not is_webgpu_available():
        pytest.skip("WebGPU not available")
    return get_webgpu_device()


@pytest.mark.hardware
@pytest.mark.webgpu
@pytest.mark.compute_shaders
class TestWebGPUMatmul:
    """Test suite for WebGPU matmul operations."""

    @skip_if_no_webgpu
    def test_device_available(self, webgpu_device):
        """Test that WebGPU device is available."""
        assert webgpu_device is not None

    @skip_if_no_webgpu
    @pytest.mark.parametrize("matrix_size", [(32, 32), (64, 64), (128, 128), (256, 256)])
    def test_matmul_correctness(self, webgpu_device, matrix_size):
        """Test matrix multiplication correctness with different matrix sizes."""
        m, n = matrix_size
        k = m  # For simplicity, use square matrices
        
        # Create random matrices
        a = np.random.rand(m, k).astype(np.float32)
        b = np.random.rand(k, n).astype(np.float32)
        
        # CPU reference result
        expected = np.matmul(a, b)
        
        # WebGPU computation
        a_tensor = torch.tensor(a, device=webgpu_device)
        b_tensor = torch.tensor(b, device=webgpu_device)
        result_tensor = torch.matmul(a_tensor, b_tensor)
        result = result_tensor.cpu().numpy()
        
        # Check results
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    @skip_if_no_webgpu
    @pytest.mark.benchmark
    def test_matmul_performance(self, webgpu_device):
        """Benchmark matrix multiplication performance."""
        matrix_size = 1024
        
        # Create random matrices
        a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        
        # Create tensors
        a_tensor = torch.tensor(a, device=webgpu_device)
        b_tensor = torch.tensor(b, device=webgpu_device)
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(a_tensor, b_tensor)
        
        # Benchmark
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            _ = torch.matmul(a_tensor, b_tensor)
            webgpu_device.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        print(f"Average matmul time for {matrix_size}x{matrix_size}: {avg_time:.4f} seconds")
        
        # Calculate FLOPS
        flops = 2 * matrix_size**3  # For matrix multiplication
        gflops = flops / (avg_time * 1e9)
        print(f"Performance: {gflops:.2f} GFLOPS")

    @skip_if_no_webgpu
    def test_memory_usage(self, webgpu_device):
        """Test memory usage on WebGPU."""
        # Test with increasing matrix sizes to observe memory usage
        for size in [1024, 2048, 4096]:
            # Skip larger sizes if GPU memory is limited
            if size > 2048 and torch.cuda.get_device_properties(0).total_memory < 8e9:
                continue
                
            # Create random matrices
            a = np.random.rand(size, size).astype(np.float32)
            b = np.random.rand(size, size).astype(np.float32)
            
            # Move to device
            try:
                a_tensor = torch.tensor(a, device=webgpu_device)
                b_tensor = torch.tensor(b, device=webgpu_device)
                result = torch.matmul(a_tensor, b_tensor)
                
                # Check that result is correct shape
                assert result.shape == (size, size)
                
                # Clean up to free memory
                del a_tensor, b_tensor, result
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Out of memory for size {size}x{size}")
                    # This is not a test failure, just a limitation
                    continue
                else:
                    raise