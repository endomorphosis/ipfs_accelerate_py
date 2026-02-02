"""
Tests for the Quantization Manager module.

These are framework tests that run by default (no @pytest.mark.model_test).
Tests validate quantization logic without requiring actual models or GPUs.
"""

import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generators.skill_generator.hardware.quantization_manager import (
        QuantizationManager,
        QuantizationMethod
    )
    HAS_QUANTIZATION = True
except ImportError:
    HAS_QUANTIZATION = False


@pytest.mark.skipif(not HAS_QUANTIZATION, reason="Quantization manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestQuantizationManagerInit:
    """Test QuantizationManager initialization"""
    
    def test_init_with_valid_hardware(self):
        """Test initialization with valid hardware types"""
        for hardware in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]:
            manager = QuantizationManager(hardware)
            assert manager.hardware == hardware
            assert manager.model_arch in ["encoder-only", "decoder-only", "vision", "speech", "multimodal", "mixture-of-experts", "state-space"]
    
    def test_init_with_invalid_hardware(self):
        """Test initialization with invalid hardware"""
        with pytest.raises((ValueError, KeyError)):
            QuantizationManager("invalid_hardware")
    
    def test_init_with_model_arch(self):
        """Test initialization with specific model architecture"""
        manager = QuantizationManager("cuda", "decoder-only")
        assert manager.hardware == "cuda"
        assert manager.model_arch == "decoder-only"
    
    def test_init_with_invalid_model_arch(self):
        """Test initialization with invalid model architecture"""
        with pytest.raises((ValueError, KeyError)):
            QuantizationManager("cuda", "invalid_arch")
    
    def test_default_model_arch(self):
        """Test default model architecture"""
        manager = QuantizationManager("cpu")
        assert manager.model_arch in ["encoder-only", "decoder-only", "vision", "speech", "multimodal", "mixture-of-experts", "state-space"]
    
    def test_compatibility_matrix_exists(self):
        """Test that compatibility matrix is initialized"""
        manager = QuantizationManager("cuda")
        assert hasattr(manager, 'compatibility_matrix') or hasattr(manager, '_compatibility_matrix')


@pytest.mark.skipif(not HAS_QUANTIZATION, reason="Quantization manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestHardwareCompatibility:
    """Test hardware compatibility checking"""
    
    def test_check_compatibility_cuda(self):
        """Test CUDA compatibility for various methods"""
        manager = QuantizationManager("cuda", "decoder-only")
        
        # CUDA should support bitsandbytes, GPTQ, AWQ
        cuda_methods = [
            QuantizationMethod.BITSANDBYTES_4BIT,
            QuantizationMethod.BITSANDBYTES_8BIT,
            QuantizationMethod.GPTQ,
            QuantizationMethod.AWQ
        ]
        
        for method in cuda_methods:
            assert manager.check_compatibility(method) == True, f"CUDA should support {method}"
    
    def test_check_compatibility_cpu(self):
        """Test CPU compatibility"""
        manager = QuantizationManager("cpu", "decoder-only")
        
        # CPU should support GGUF and dynamic INT8
        assert manager.check_compatibility(QuantizationMethod.GGUF) == True
        assert manager.check_compatibility(QuantizationMethod.DYNAMIC_INT8) == True
        
        # CPU should NOT support bitsandbytes
        assert manager.check_compatibility(QuantizationMethod.BITSANDBYTES_4BIT) == False
    
    def test_check_compatibility_openvino(self):
        """Test OpenVINO compatibility"""
        manager = QuantizationManager("openvino", "encoder-only")
        
        # OpenVINO should support INT8
        assert manager.check_compatibility(QuantizationMethod.OPENVINO_INT8) == True
    
    def test_check_compatibility_qnn(self):
        """Test QNN compatibility"""
        manager = QuantizationManager("qnn", "encoder-only")
        
        # QNN should support 8-bit quantization
        assert manager.check_compatibility(QuantizationMethod.QNN_8BIT) == True
    
    def test_all_hardware_types(self):
        """Test that all hardware types have some compatible methods"""
        for hardware in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]:
            manager = QuantizationManager(hardware)
            
            # Should have at least one compatible method
            compatible_count = 0
            for method in QuantizationMethod:
                if method != QuantizationMethod.NONE:
                    if manager.check_compatibility(method):
                        compatible_count += 1
            
            assert compatible_count > 0, f"{hardware} should have at least one compatible method"


@pytest.mark.skipif(not HAS_QUANTIZATION, reason="Quantization manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMethodRecommendation:
    """Test quantization method recommendation"""
    
    def test_get_recommended_method_cuda(self):
        """Test method recommendation for CUDA"""
        manager = QuantizationManager("cuda", "decoder-only")
        method = manager.get_recommended_method()
        
        # Should recommend a CUDA-compatible method
        assert method in [
            QuantizationMethod.BITSANDBYTES_4BIT,
            QuantizationMethod.BITSANDBYTES_8BIT,
            QuantizationMethod.GPTQ,
            QuantizationMethod.AWQ,
            QuantizationMethod.DYNAMIC_INT8,
            QuantizationMethod.NONE
        ]
    
    def test_get_recommended_method_cpu(self):
        """Test method recommendation for CPU"""
        manager = QuantizationManager("cpu", "decoder-only")
        method = manager.get_recommended_method()
        
        # Should recommend a CPU-compatible method
        assert method in [
            QuantizationMethod.GGUF,
            QuantizationMethod.DYNAMIC_INT8,
            QuantizationMethod.NONE
        ]
    
    def test_get_recommended_method_with_memory_budget(self):
        """Test method recommendation with memory budget"""
        manager = QuantizationManager("cuda", "decoder-only")
        
        # Small memory budget should favor aggressive quantization
        method_small = manager.get_recommended_method(memory_budget_gb=4.0)
        assert method_small in [
            QuantizationMethod.BITSANDBYTES_4BIT,
            QuantizationMethod.GPTQ,
            QuantizationMethod.AWQ
        ]
        
        # Large memory budget might allow less quantization
        method_large = manager.get_recommended_method(memory_budget_gb=32.0)
        assert method_large in list(QuantizationMethod)
    
    def test_get_recommended_method_speed_priority(self):
        """Test method recommendation prioritizing speed"""
        manager = QuantizationManager("cuda", "decoder-only")
        method = manager.get_recommended_method(priority="speed")
        
        # Should return a valid method
        assert isinstance(method, QuantizationMethod)
    
    def test_get_recommended_method_memory_priority(self):
        """Test method recommendation prioritizing memory"""
        manager = QuantizationManager("cuda", "decoder-only")
        method = manager.get_recommended_method(priority="memory")
        
        # Should favor more aggressive quantization
        assert method in [
            QuantizationMethod.BITSANDBYTES_4BIT,
            QuantizationMethod.GPTQ,
            QuantizationMethod.AWQ
        ]
    
    def test_get_recommended_method_quality_priority(self):
        """Test method recommendation prioritizing quality"""
        manager = QuantizationManager("cuda", "decoder-only")
        method = manager.get_recommended_method(priority="quality")
        
        # Should favor less aggressive quantization or none
        assert isinstance(method, QuantizationMethod)
    
    def test_get_recommended_method_all_hardware(self):
        """Test method recommendation for all hardware types"""
        for hardware in ["cpu", "cuda", "rocm", "mps", "openvino", "qnn"]:
            manager = QuantizationManager(hardware)
            method = manager.get_recommended_method()
            
            # Should return a valid method
            assert isinstance(method, QuantizationMethod)
            
            # Method should be compatible with hardware
            if method != QuantizationMethod.NONE:
                assert manager.check_compatibility(method) == True
    
    def test_get_recommended_method_different_architectures(self):
        """Test method recommendation for different model architectures"""
        architectures = ["encoder-only", "decoder-only", "vision", "speech"]
        
        for arch in architectures:
            manager = QuantizationManager("cuda", arch)
            method = manager.get_recommended_method()
            
            # Should return a valid method
            assert isinstance(method, QuantizationMethod)


@pytest.mark.skipif(not HAS_QUANTIZATION, reason="Quantization manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMemorySavings:
    """Test memory savings estimation"""
    
    def test_estimate_memory_savings_4bit(self):
        """Test memory savings for 4-bit quantization"""
        manager = QuantizationManager("cuda")
        savings = manager.estimate_memory_savings(
            QuantizationMethod.BITSANDBYTES_4BIT,
            model_size_gb=10.0
        )
        
        # 4-bit should save ~75%
        assert 0.70 <= savings <= 0.80
    
    def test_estimate_memory_savings_8bit(self):
        """Test memory savings for 8-bit quantization"""
        manager = QuantizationManager("cuda")
        savings = manager.estimate_memory_savings(
            QuantizationMethod.BITSANDBYTES_8BIT,
            model_size_gb=10.0
        )
        
        # 8-bit should save ~50%
        assert 0.45 <= savings <= 0.55
    
    def test_estimate_memory_savings_none(self):
        """Test memory savings for no quantization"""
        manager = QuantizationManager("cuda")
        savings = manager.estimate_memory_savings(
            QuantizationMethod.NONE,
            model_size_gb=10.0
        )
        
        # No quantization = no savings
        assert savings == 0.0
    
    def test_estimate_memory_savings_different_sizes(self):
        """Test memory savings estimation for different model sizes"""
        manager = QuantizationManager("cuda")
        method = QuantizationMethod.BITSANDBYTES_4BIT
        
        for size_gb in [1.0, 5.0, 10.0, 50.0]:
            savings = manager.estimate_memory_savings(method, model_size_gb=size_gb)
            
            # Savings percentage should be consistent regardless of size
            assert 0.70 <= savings <= 0.80
    
    def test_estimate_memory_savings_all_methods(self):
        """Test memory savings for all quantization methods"""
        manager = QuantizationManager("cuda")
        
        for method in QuantizationMethod:
            savings = manager.estimate_memory_savings(method, model_size_gb=10.0)
            
            # Savings should be between 0 and 1
            assert 0.0 <= savings <= 1.0
    
    def test_memory_savings_consistency(self):
        """Test that more aggressive quantization saves more memory"""
        manager = QuantizationManager("cuda")
        model_size = 10.0
        
        savings_4bit = manager.estimate_memory_savings(
            QuantizationMethod.BITSANDBYTES_4BIT, model_size
        )
        savings_8bit = manager.estimate_memory_savings(
            QuantizationMethod.BITSANDBYTES_8BIT, model_size
        )
        savings_none = manager.estimate_memory_savings(
            QuantizationMethod.NONE, model_size
        )
        
        # 4-bit should save more than 8-bit, which saves more than none
        assert savings_4bit > savings_8bit > savings_none


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
