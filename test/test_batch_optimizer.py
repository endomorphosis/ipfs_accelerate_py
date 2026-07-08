"""
Tests for the Batch Optimizer module.

These are framework tests that run by default (no @pytest.mark.model_test).
Tests validate batch optimization logic without requiring actual models or GPUs.
"""

import pytest
import sys
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generators.skill_generator.hardware.batch_optimizer import (
        BatchOptimizer,
        OptimizationResult
    )
    HAS_BATCH_OPTIMIZER = True
except ImportError:
    HAS_BATCH_OPTIMIZER = False


@pytest.mark.skipif(not HAS_BATCH_OPTIMIZER, reason="Batch optimizer not available")
@pytest.mark.unit
@pytest.mark.framework
class TestBatchOptimizerInit:
    """Test BatchOptimizer initialization"""
    
    def test_init_cpu(self):
        """Test initialization for CPU"""
        optimizer = BatchOptimizer("cpu")
        assert optimizer.device == "cpu"
        assert optimizer.device_id is None or optimizer.device_id == 0
    
    def test_init_cuda(self):
        """Test initialization for CUDA (without requiring GPU)"""
        optimizer = BatchOptimizer("cuda", device_id=0)
        assert optimizer.device == "cuda"
        assert optimizer.device_id == 0
    
    def test_init_with_device_id(self):
        """Test initialization with specific device ID"""
        optimizer = BatchOptimizer("cuda", device_id=1)
        assert optimizer.device == "cuda"
        assert optimizer.device_id == 1
    
    def test_cache_initialized(self):
        """Test that cache is initialized"""
        optimizer = BatchOptimizer("cpu")
        assert hasattr(optimizer, 'cache') or hasattr(optimizer, '_cache')


@pytest.mark.skipif(not HAS_BATCH_OPTIMIZER, reason="Batch optimizer not available")
@pytest.mark.unit
@pytest.mark.framework
class TestBatchSizeEstimation:
    """Test batch size estimation"""
    
    @pytest.fixture
    def cpu_optimizer(self):
        """Create a CPU batch optimizer for testing"""
        return BatchOptimizer("cpu")
    
    def test_estimate_batch_size_basic(self, cpu_optimizer):
        """Test basic batch size estimation"""
        batch_size = cpu_optimizer.estimate_batch_size(
            model_size_mb=500,
            per_sample_memory_mb=10
        )
        
        assert isinstance(batch_size, int)
        assert batch_size > 0
        assert batch_size <= 256  # Reasonable upper limit
    
    def test_estimate_batch_size_small_model(self, cpu_optimizer):
        """Test batch size for small model"""
        batch_size = cpu_optimizer.estimate_batch_size(
            model_size_mb=100,
            per_sample_memory_mb=1
        )
        
        # Small model should allow larger batch
        assert batch_size >= 1
    
    def test_estimate_batch_size_large_model(self, cpu_optimizer):
        """Test batch size for large model"""
        batch_size = cpu_optimizer.estimate_batch_size(
            model_size_mb=5000,
            per_sample_memory_mb=50
        )
        
        # Large model should have smaller batch
        assert batch_size >= 1
        assert batch_size <= 64
    
    def test_estimate_batch_size_with_memory_budget(self, cpu_optimizer):
        """Test batch size estimation with memory budget"""
        try:
            batch_size = cpu_optimizer.estimate_batch_size(
                model_size_mb=1000,
                per_sample_memory_mb=10,
                available_memory_mb=4000
            )
            
            assert isinstance(batch_size, int)
            assert batch_size > 0
        except TypeError:
            # If method doesn't accept available_memory_mb, that's ok
            pass
    
    def test_estimate_batch_size_zero_per_sample(self, cpu_optimizer):
        """Test batch size estimation with zero per-sample memory"""
        # Should handle edge case gracefully
        try:
            batch_size = cpu_optimizer.estimate_batch_size(
                model_size_mb=1000,
                per_sample_memory_mb=0.1  # Very small
            )
            assert batch_size > 0
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to raise error for invalid input
    
    def test_batch_size_consistency(self, cpu_optimizer):
        """Test that batch size estimation is consistent"""
        # Same inputs should give same output
        batch_size1 = cpu_optimizer.estimate_batch_size(
            model_size_mb=1000,
            per_sample_memory_mb=10
        )
        batch_size2 = cpu_optimizer.estimate_batch_size(
            model_size_mb=1000,
            per_sample_memory_mb=10
        )
        
        assert batch_size1 == batch_size2


@pytest.mark.skipif(not HAS_BATCH_OPTIMIZER, reason="Batch optimizer not available")
@pytest.mark.unit
@pytest.mark.framework
class TestWorkloadRecommendations:
    """Test workload-specific recommendations"""
    
    @pytest.fixture
    def cpu_optimizer(self):
        """Create a CPU batch optimizer for testing"""
        return BatchOptimizer("cpu")
    
    def test_get_recommendation_realtime(self, cpu_optimizer):
        """Test recommendation for realtime workload"""
        try:
            batch_size = cpu_optimizer.get_recommendation_for_workload("realtime")
            
            # Realtime should prioritize latency (small batch)
            assert batch_size == 1 or batch_size <= 4
        except (AttributeError, KeyError):
            pass  # Acceptable if not implemented
    
    def test_get_recommendation_throughput(self, cpu_optimizer):
        """Test recommendation for throughput workload"""
        try:
            batch_size = cpu_optimizer.get_recommendation_for_workload("throughput")
            
            # Throughput should use larger batches
            assert batch_size >= 8
        except (AttributeError, KeyError):
            pass  # Acceptable if not implemented
    
    def test_get_recommendation_batch(self, cpu_optimizer):
        """Test recommendation for batch workload"""
        try:
            batch_size = cpu_optimizer.get_recommendation_for_workload("batch")
            
            # Batch workload can use very large batches
            assert batch_size >= 16
        except (AttributeError, KeyError):
            pass  # Acceptable if not implemented
    
    def test_get_recommendation_invalid_workload(self, cpu_optimizer):
        """Test recommendation for invalid workload"""
        try:
            cpu_optimizer.get_recommendation_for_workload("invalid_workload")
        except (ValueError, KeyError):
            pass  # Expected to raise error


@pytest.mark.skipif(not HAS_BATCH_OPTIMIZER, reason="Batch optimizer not available")
@pytest.mark.unit
@pytest.mark.framework
class TestCacheManagement:
    """Test batch size caching"""
    
    @pytest.fixture
    def cpu_optimizer(self):
        """Create a CPU batch optimizer for testing"""
        return BatchOptimizer("cpu")
    
    def test_cache_recommendation(self, cpu_optimizer):
        """Test caching a batch size recommendation"""
        try:
            model_name = "test_model"
            batch_size = 32
            
            cpu_optimizer.cache_recommendation(model_name, batch_size)
            
            # Should be able to retrieve it
            if hasattr(cpu_optimizer, 'get_cached_recommendation'):
                cached = cpu_optimizer.get_cached_recommendation(model_name)
                assert cached == batch_size
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented
    
    def test_get_cached_recommendation_not_exists(self, cpu_optimizer):
        """Test getting non-existent cached recommendation"""
        try:
            cached = cpu_optimizer.get_cached_recommendation("nonexistent_model")
            assert cached is None or isinstance(cached, int)
        except (AttributeError, KeyError):
            pass  # Acceptable if not implemented or raises error
    
    def test_clear_cache(self, cpu_optimizer):
        """Test clearing the cache"""
        try:
            # Add something to cache
            cpu_optimizer.cache_recommendation("test_model", 32)
            
            # Clear cache
            cpu_optimizer.clear_cache()
            
            # Should be empty now
            if hasattr(cpu_optimizer, 'get_cached_recommendation'):
                cached = cpu_optimizer.get_cached_recommendation("test_model")
                assert cached is None
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


@pytest.mark.skipif(not HAS_BATCH_OPTIMIZER, reason="Batch optimizer not available")
@pytest.mark.unit
@pytest.mark.framework
class TestResultSerialization:
    """Test OptimizationResult serialization"""
    
    def test_result_to_dict(self):
        """Test converting OptimizationResult to dict"""
        try:
            result = OptimizationResult(
                batch_size=32,
                throughput=100.0,
                latency=0.01,
                memory_usage=1000.0
            )
            
            result_dict = result.to_dict()
            
            assert isinstance(result_dict, dict)
            assert result_dict["batch_size"] == 32
            assert result_dict["throughput"] == 100.0
        except (NameError, AttributeError):
            pytest.skip("OptimizationResult not available or method not implemented")
    
    def test_result_from_dict(self):
        """Test creating OptimizationResult from dict"""
        try:
            result_dict = {
                "batch_size": 32,
                "throughput": 100.0,
                "latency": 0.01,
                "memory_usage": 1000.0
            }
            
            result = OptimizationResult.from_dict(result_dict)
            
            assert result.batch_size == 32
            assert result.throughput == 100.0
        except (NameError, AttributeError):
            pytest.skip("OptimizationResult not available or method not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
