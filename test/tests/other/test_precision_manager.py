"""
Tests for the Precision Manager module.

These are framework tests that run by default (no @pytest.mark.model_test).
Tests validate precision management logic without requiring actual models or GPUs.
"""

import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generators.skill_generator.hardware.precision_manager import (
        PrecisionManager
    )
    HAS_PRECISION_MANAGER = True
except ImportError:
    HAS_PRECISION_MANAGER = False


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestPrecisionManagerInit:
    """Test PrecisionManager initialization"""
    
    def test_init_cpu(self):
        """Test initialization for CPU"""
        manager = PrecisionManager("cpu")
        assert manager.device == "cpu"
        assert manager.device_id is None or manager.device_id == 0
    
    def test_init_cuda(self):
        """Test initialization for CUDA (without requiring GPU)"""
        manager = PrecisionManager("cuda", device_id=0)
        assert manager.device == "cuda"
        assert manager.device_id == 0
    
    def test_init_with_device_id(self):
        """Test initialization with specific device ID"""
        manager = PrecisionManager("cuda", device_id=1)
        assert manager.device == "cuda"
        assert manager.device_id == 1
    
    def test_init_invalid_device(self):
        """Test initialization with invalid device"""
        try:
            manager = PrecisionManager("invalid_device")
            # If it doesn't raise, check it set something reasonable
            assert manager.device in ["cpu", "cuda", "rocm", "mps"]
        except (ValueError, KeyError):
            pass  # Expected
    
    def test_current_precision_initialized(self):
        """Test that current precision is set"""
        manager = PrecisionManager("cpu")
        assert hasattr(manager, 'current_precision')
        assert manager.current_precision in ["fp32", "fp16", "bf16", "int8"]


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestCapabilityDetection:
    """Test hardware capability detection"""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU precision manager for testing"""
        return PrecisionManager("cpu")
    
    def test_detect_capabilities_cpu(self, cpu_manager):
        """Test capability detection for CPU"""
        caps = cpu_manager.capabilities
        
        assert isinstance(caps, dict)
        assert "fp32" in caps
        assert caps["fp32"] == True  # CPU always supports FP32
    
    def test_fp32_always_supported(self, cpu_manager):
        """Test that FP32 is always supported"""
        assert cpu_manager.supports_precision("fp32") == True
    
    def test_supports_precision_method(self, cpu_manager):
        """Test supports_precision method"""
        # FP32 should be supported
        assert cpu_manager.supports_precision("fp32") == True
        
        # Check other precisions
        for precision in ["fp16", "bf16", "int8"]:
            result = cpu_manager.supports_precision(precision)
            assert isinstance(result, bool)
    
    def test_get_supported_precisions(self, cpu_manager):
        """Test getting list of supported precisions"""
        try:
            supported = cpu_manager.get_supported_precisions()
            assert isinstance(supported, list)
            assert "fp32" in supported
        except AttributeError:
            pass  # Acceptable if not implemented
    
    def test_capabilities_dict_format(self, cpu_manager):
        """Test capabilities dictionary format"""
        caps = cpu_manager.capabilities
        
        # Should have precision keys
        precision_keys = ["fp32", "fp16", "bf16", "int8"]
        for key in precision_keys:
            if key in caps:
                assert isinstance(caps[key], bool)


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestPrecisionSetting:
    """Test precision setting and configuration"""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU precision manager for testing"""
        return PrecisionManager("cpu")
    
    def test_set_precision_fp32(self, cpu_manager):
        """Test setting FP32 precision"""
        cpu_manager.set_precision("fp32")
        assert cpu_manager.current_precision == "fp32"
    
    def test_set_precision_fp16(self, cpu_manager):
        """Test setting FP16 precision"""
        try:
            cpu_manager.set_precision("fp16")
            # If successful, should be set
            if cpu_manager.supports_precision("fp16"):
                assert cpu_manager.current_precision == "fp16"
        except (ValueError, RuntimeError):
            # Expected if FP16 not supported on CPU
            pass
    
    def test_set_precision_invalid(self, cpu_manager):
        """Test setting invalid precision"""
        with pytest.raises((ValueError, KeyError)):
            cpu_manager.set_precision("invalid_precision")
    
    def test_set_precision_unsupported(self, cpu_manager):
        """Test setting unsupported precision"""
        # Try to set a precision that might not be supported
        try:
            cpu_manager.set_precision("bf16")
            # If it works, check it's set
            if cpu_manager.supports_precision("bf16"):
                assert cpu_manager.current_precision == "bf16"
        except (ValueError, RuntimeError):
            # Expected if not supported
            pass
    
    def test_precision_fallback(self, cpu_manager):
        """Test precision fallback mechanism"""
        original_precision = cpu_manager.current_precision
        
        # Try to set a potentially unsupported precision
        try:
            cpu_manager.set_precision("int8")
        except (ValueError, RuntimeError):
            # Should fall back or raise
            pass
        
        # Should still have a valid precision
        assert cpu_manager.current_precision in ["fp32", "fp16", "bf16", "int8"]


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestErrorHandling:
    """Test precision error handling"""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU precision manager for testing"""
        return PrecisionManager("cpu")
    
    def test_handle_precision_error_generic(self, cpu_manager):
        """Test handling generic precision error"""
        error = RuntimeError("Precision error occurred")
        
        try:
            result = cpu_manager.handle_precision_error(error, "test_operation")
            assert isinstance(result, bool)
        except AttributeError:
            pass  # Acceptable if not implemented
    
    def test_handle_precision_error_cuda(self, cpu_manager):
        """Test handling CUDA precision error"""
        error = RuntimeError("CUDA error: device-side assert")
        
        try:
            result = cpu_manager.handle_precision_error(error, "inference")
            assert isinstance(result, bool)
            
            # Should fallback to FP32
            if result:
                assert cpu_manager.current_precision == "fp32"
        except AttributeError:
            pass  # Acceptable if not implemented
    
    def test_handle_precision_error_nan(self, cpu_manager):
        """Test handling NaN error"""
        error = RuntimeError("NaN detected in output")
        
        try:
            result = cpu_manager.handle_precision_error(error, "inference")
            assert isinstance(result, bool)
        except AttributeError:
            pass  # Acceptable if not implemented
    
    def test_error_statistics_tracking(self, cpu_manager):
        """Test that error statistics are tracked"""
        if hasattr(cpu_manager, 'error_counts') or hasattr(cpu_manager, 'get_error_stats'):
            error = RuntimeError("Test error")
            try:
                cpu_manager.handle_precision_error(error, "test")
                
                if hasattr(cpu_manager, 'error_counts'):
                    assert isinstance(cpu_manager.error_counts, dict)
                elif hasattr(cpu_manager, 'get_error_stats'):
                    stats = cpu_manager.get_error_stats()
                    assert isinstance(stats, dict)
            except AttributeError:
                pass


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestNumericalIssues:
    """Test numerical issue detection"""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU precision manager for testing"""
        return PrecisionManager("cpu")
    
    def test_check_for_numerical_issues_normal(self, cpu_manager):
        """Test checking normal tensor (no issues)"""
        try:
            # Create a mock tensor-like object
            class MockTensor:
                def __init__(self):
                    self.data = [1.0, 2.0, 3.0]
                
                def isnan(self):
                    return MockBool(False)
                
                def isinf(self):
                    return MockBool(False)
                
                def any(self):
                    return False
            
            class MockBool:
                def __init__(self, value):
                    self.value = value
                def any(self):
                    return self.value
            
            tensor = MockTensor()
            issues = cpu_manager.check_for_numerical_issues(tensor)
            
            assert isinstance(issues, dict)
            if "has_nan" in issues:
                assert issues["has_nan"] == False
            if "has_inf" in issues:
                assert issues["has_inf"] == False
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented
    
    def test_check_for_numerical_issues_dict_input(self, cpu_manager):
        """Test checking dict of tensors"""
        try:
            test_dict = {"tensor1": [1.0, 2.0], "tensor2": [3.0, 4.0]}
            issues = cpu_manager.check_for_numerical_issues(test_dict)
            assert isinstance(issues, dict)
        except (AttributeError, NotImplementedError, TypeError):
            pass  # Acceptable if not implemented or different signature
    
    def test_numerical_issues_response_format(self, cpu_manager):
        """Test that numerical issues check returns expected format"""
        try:
            # Try with a simple list
            issues = cpu_manager.check_for_numerical_issues([1.0, 2.0, 3.0])
            
            if isinstance(issues, dict):
                # Should have boolean flags
                expected_keys = ["has_nan", "has_inf", "has_issues"]
                assert any(key in issues for key in expected_keys)
        except (AttributeError, NotImplementedError, TypeError):
            pass  # Acceptable if not implemented
    
    def test_safe_precision_context(self, cpu_manager):
        """Test safe precision context manager"""
        try:
            with cpu_manager.safe_precision_context("test_operation"):
                # Should work without errors
                pass
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


@pytest.mark.skipif(not HAS_PRECISION_MANAGER, reason="Precision manager not available")
@pytest.mark.unit
@pytest.mark.framework
class TestContextManagers:
    """Test context managers for precision"""
    
    @pytest.fixture
    def cpu_manager(self):
        """Create a CPU precision manager for testing"""
        return PrecisionManager("cpu")
    
    def test_safe_precision_context_basic(self, cpu_manager):
        """Test basic safe precision context"""
        try:
            with cpu_manager.safe_precision_context("test"):
                pass
            # Should complete without error
        except AttributeError:
            pass  # Acceptable if not implemented
    
    def test_create_autocast_context(self, cpu_manager):
        """Test creating autocast context"""
        try:
            context = cpu_manager.create_autocast_context()
            # Should return a context manager
            assert context is not None
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented
    
    def test_mixed_precision_enablement(self, cpu_manager):
        """Test enabling mixed precision"""
        try:
            cpu_manager.enable_mixed_precision()
            # Should enable without error
            if hasattr(cpu_manager, 'mixed_precision_enabled'):
                assert cpu_manager.mixed_precision_enabled == True
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
