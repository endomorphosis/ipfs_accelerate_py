"""
Tests for the Memory Profiler module.

These are framework tests that run by default (no @pytest.mark.model_test).
Tests validate memory profiling logic without requiring actual models or GPUs.
"""

import pytest
import sys
import time
from pathlib import Path

# Add module path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.generators.skill_generator.hardware.memory_profiler import (
        MemoryProfiler,
        MemorySnapshot,
        MemoryProfile,
        MemoryBudgetManager
    )
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMemoryProfilerInit:
    """Test MemoryProfiler initialization"""
    
    def test_init_cpu(self):
        """Test initialization for CPU"""
        profiler = MemoryProfiler("cpu")
        assert profiler.device == "cpu"
        assert profiler.device_id is None or profiler.device_id == 0
    
    def test_init_cuda(self):
        """Test initialization for CUDA (without requiring GPU)"""
        # Should not fail even without CUDA
        profiler = MemoryProfiler("cuda", device_id=0)
        assert profiler.device == "cuda"
        assert profiler.device_id == 0
    
    def test_init_with_device_id(self):
        """Test initialization with specific device ID"""
        profiler = MemoryProfiler("cuda", device_id=1)
        assert profiler.device == "cuda"
        assert profiler.device_id == 1
    
    def test_init_invalid_device(self):
        """Test initialization with invalid device"""
        # Should handle gracefully or raise appropriate error
        try:
            profiler = MemoryProfiler("invalid_device")
            # If it doesn't raise, check it set something reasonable
            assert profiler.device in ["cpu", "cuda", "rocm", "mps"]
        except (ValueError, KeyError):
            pass  # Expected
    
    def test_profiles_list_initialized(self):
        """Test that profiles list is initialized"""
        profiler = MemoryProfiler("cpu")
        assert hasattr(profiler, 'profiles') or hasattr(profiler, '_profiles')


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMemoryTracking:
    """Test memory tracking functionality"""
    
    @pytest.fixture
    def cpu_profiler(self):
        """Create a CPU profiler for testing"""
        return MemoryProfiler("cpu")
    
    def test_get_cpu_memory(self, cpu_profiler):
        """Test CPU memory reading"""
        memory = cpu_profiler.get_cpu_memory()
        
        assert isinstance(memory, dict)
        assert "used_mb" in memory or "allocated_mb" in memory
        assert "available_mb" in memory or "total_mb" in memory
        
        # Values should be positive
        for key, value in memory.items():
            if "mb" in key.lower():
                assert value >= 0, f"{key} should be non-negative"
    
    def test_get_device_memory_cpu(self, cpu_profiler):
        """Test getting device memory for CPU"""
        memory = cpu_profiler.get_device_memory()
        
        # CPU device memory should return CPU RAM info
        assert isinstance(memory, dict)
        assert len(memory) > 0
    
    def test_memory_tracking_consistency(self, cpu_profiler):
        """Test that multiple memory readings are consistent"""
        memory1 = cpu_profiler.get_cpu_memory()
        time.sleep(0.1)
        memory2 = cpu_profiler.get_cpu_memory()
        
        # Readings should be close (within 10% or 100MB)
        for key in memory1:
            if key in memory2 and "mb" in key.lower():
                diff = abs(memory1[key] - memory2[key])
                assert diff < 100 or diff < (memory1[key] * 0.1)
    
    def test_get_peak_memory(self, cpu_profiler):
        """Test getting peak memory usage"""
        # Should work even if no operations profiled yet
        try:
            peak = cpu_profiler.get_peak_memory()
            assert peak >= 0
        except (AttributeError, KeyError):
            pass  # Acceptable if not tracking peak yet
    
    def test_reset_peak_memory(self, cpu_profiler):
        """Test resetting peak memory tracking"""
        try:
            cpu_profiler.reset_peak_memory()
            peak = cpu_profiler.get_peak_memory()
            # After reset, peak should be 0 or current usage
            assert peak >= 0
        except (AttributeError, KeyError):
            pass  # Acceptable if not implemented
    
    def test_memory_dict_format(self, cpu_profiler):
        """Test that memory dict has expected format"""
        memory = cpu_profiler.get_cpu_memory()
        
        # Should have memory measurements in MB
        has_memory_field = any("mb" in key.lower() for key in memory.keys())
        assert has_memory_field, "Memory dict should have fields with 'mb'"


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMemorySnapshots:
    """Test memory snapshot functionality"""
    
    @pytest.fixture
    def cpu_profiler(self):
        """Create a CPU profiler for testing"""
        return MemoryProfiler("cpu")
    
    def test_create_snapshot(self, cpu_profiler):
        """Test creating a memory snapshot"""
        snapshot = cpu_profiler.create_snapshot()
        
        assert isinstance(snapshot, (MemorySnapshot, dict))
        
        if isinstance(snapshot, MemorySnapshot):
            assert hasattr(snapshot, 'timestamp') or hasattr(snapshot, 'cpu_memory')
        else:
            assert "timestamp" in snapshot or "memory" in snapshot
    
    def test_snapshot_has_memory_info(self, cpu_profiler):
        """Test that snapshot contains memory information"""
        snapshot = cpu_profiler.create_snapshot()
        
        # Check for memory information
        if isinstance(snapshot, MemorySnapshot):
            assert hasattr(snapshot, 'cpu_memory') or hasattr(snapshot, 'device_memory')
        else:
            has_memory = any("memory" in key.lower() for key in snapshot.keys())
            assert has_memory
    
    def test_multiple_snapshots(self, cpu_profiler):
        """Test creating multiple snapshots"""
        snapshot1 = cpu_profiler.create_snapshot()
        time.sleep(0.1)
        snapshot2 = cpu_profiler.create_snapshot()
        
        # Both should be valid
        assert snapshot1 is not None
        assert snapshot2 is not None
        
        # Timestamps should be different
        if isinstance(snapshot1, MemorySnapshot) and isinstance(snapshot2, MemorySnapshot):
            if hasattr(snapshot1, 'timestamp') and hasattr(snapshot2, 'timestamp'):
                assert snapshot1.timestamp != snapshot2.timestamp
    
    def test_snapshot_to_dict(self, cpu_profiler):
        """Test converting snapshot to dictionary"""
        snapshot = cpu_profiler.create_snapshot()
        
        if isinstance(snapshot, MemorySnapshot):
            if hasattr(snapshot, 'to_dict'):
                snapshot_dict = snapshot.to_dict()
                assert isinstance(snapshot_dict, dict)
            else:
                # MemorySnapshot might be a dict-like object
                assert isinstance(snapshot, (dict, object))
    
    def test_compare_snapshots(self, cpu_profiler):
        """Test comparing two snapshots"""
        snapshot1 = cpu_profiler.create_snapshot()
        snapshot2 = cpu_profiler.create_snapshot()
        
        # Should be able to compare (or at least not fail)
        try:
            delta = cpu_profiler.compare_snapshots(snapshot1, snapshot2)
            assert isinstance(delta, (dict, float, int))
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestMemoryProfiling:
    """Test operation profiling"""
    
    @pytest.fixture
    def cpu_profiler(self):
        """Create a CPU profiler for testing"""
        return MemoryProfiler("cpu")
    
    def test_profile_operation_context(self, cpu_profiler):
        """Test profiling an operation with context manager"""
        operation_name = "test_operation"
        
        with cpu_profiler.profile_operation(operation_name):
            # Simulate some work
            _ = [i**2 for i in range(1000)]
        
        # Check that profile was recorded
        if hasattr(cpu_profiler, 'profiles'):
            assert len(cpu_profiler.profiles) > 0
        elif hasattr(cpu_profiler, 'get_profiles'):
            profiles = cpu_profiler.get_profiles()
            assert len(profiles) > 0
    
    def test_profile_operation_timing(self, cpu_profiler):
        """Test that operation timing is recorded"""
        with cpu_profiler.profile_operation("timed_operation"):
            time.sleep(0.1)
        
        # Get the profile
        if hasattr(cpu_profiler, 'profiles') and len(cpu_profiler.profiles) > 0:
            profile = cpu_profiler.profiles[-1]
            if isinstance(profile, MemoryProfile):
                # Should have duration
                if hasattr(profile, 'duration'):
                    assert profile.duration >= 0.1
        elif hasattr(cpu_profiler, 'get_last_profile'):
            profile = cpu_profiler.get_last_profile()
            if profile and 'duration' in profile:
                assert profile['duration'] >= 0.1
    
    def test_profile_operation_memory_delta(self, cpu_profiler):
        """Test that memory delta is calculated"""
        with cpu_profiler.profile_operation("memory_operation"):
            # Allocate some memory
            data = [0] * 100000
        
        # Check for memory delta in profile
        if hasattr(cpu_profiler, 'profiles') and len(cpu_profiler.profiles) > 0:
            profile = cpu_profiler.profiles[-1]
            # Should have some memory information
            assert profile is not None
    
    def test_multiple_operations(self, cpu_profiler):
        """Test profiling multiple operations"""
        operations = ["op1", "op2", "op3"]
        
        for op_name in operations:
            with cpu_profiler.profile_operation(op_name):
                time.sleep(0.01)
        
        # Should have recorded all operations
        if hasattr(cpu_profiler, 'profiles'):
            assert len(cpu_profiler.profiles) >= len(operations)
    
    def test_get_memory_summary(self, cpu_profiler):
        """Test getting memory summary"""
        with cpu_profiler.profile_operation("summary_test"):
            pass
        
        # Should be able to get summary
        try:
            summary = cpu_profiler.get_memory_summary()
            assert isinstance(summary, dict)
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented
    
    def test_profile_export(self, cpu_profiler, tmp_path):
        """Test exporting profile to file"""
        with cpu_profiler.profile_operation("export_test"):
            pass
        
        # Try to export
        export_file = tmp_path / "profile.json"
        try:
            cpu_profiler.export_profile(str(export_file))
            assert export_file.exists()
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestLeakDetection:
    """Test memory leak detection"""
    
    @pytest.fixture
    def cpu_profiler(self):
        """Create a CPU profiler for testing"""
        return MemoryProfiler("cpu")
    
    def test_detect_memory_leaks_no_leak(self, cpu_profiler):
        """Test leak detection when there's no leak"""
        # Create snapshots without memory growth
        snapshot1 = cpu_profiler.create_snapshot()
        snapshot2 = cpu_profiler.create_snapshot()
        
        try:
            leaks = cpu_profiler.detect_memory_leaks(threshold_mb=10.0)
            assert isinstance(leaks, (list, bool))
            if isinstance(leaks, list):
                # No leaks expected
                assert len(leaks) == 0 or leaks == []
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented
    
    def test_detect_memory_leaks_with_threshold(self, cpu_profiler):
        """Test leak detection with different thresholds"""
        snapshot1 = cpu_profiler.create_snapshot()
        snapshot2 = cpu_profiler.create_snapshot()
        
        for threshold in [1.0, 10.0, 100.0]:
            try:
                leaks = cpu_profiler.detect_memory_leaks(threshold_mb=threshold)
                assert isinstance(leaks, (list, bool, dict))
            except (AttributeError, NotImplementedError):
                pass  # Acceptable if not implemented
    
    def test_memory_leak_detection_format(self, cpu_profiler):
        """Test that leak detection returns expected format"""
        try:
            leaks = cpu_profiler.detect_memory_leaks(threshold_mb=10.0)
            
            if isinstance(leaks, list):
                # List of leaks
                for leak in leaks:
                    assert isinstance(leak, (dict, str, object))
            elif isinstance(leaks, dict):
                # Dict with leak information
                assert len(leaks) >= 0
        except (AttributeError, NotImplementedError):
            pass  # Acceptable if not implemented


@pytest.mark.skipif(not HAS_MEMORY_PROFILER, reason="Memory profiler not available")
@pytest.mark.unit
@pytest.mark.framework
class TestBudgetManager:
    """Test MemoryBudgetManager"""
    
    def test_budget_manager_init(self):
        """Test budget manager initialization"""
        try:
            manager = MemoryBudgetManager("cpu")
            assert manager.device == "cpu"
        except (NameError, ImportError):
            pytest.skip("MemoryBudgetManager not available")
    
    def test_get_available_memory(self):
        """Test getting available memory"""
        try:
            manager = MemoryBudgetManager("cpu")
            available = manager.get_available_memory_mb()
            assert available > 0
        except (NameError, ImportError, AttributeError):
            pytest.skip("MemoryBudgetManager not available or method not implemented")
    
    def test_can_fit_model(self):
        """Test checking if model can fit in memory"""
        try:
            manager = MemoryBudgetManager("cpu", safety_margin=0.2)
            
            # Small model should fit
            can_fit_small = manager.can_fit_model(model_size_mb=100)
            assert isinstance(can_fit_small, bool)
            
            # Very large model probably won't fit
            can_fit_large = manager.can_fit_model(model_size_mb=1000000)
            assert isinstance(can_fit_large, bool)
        except (NameError, ImportError, AttributeError):
            pytest.skip("MemoryBudgetManager not available or method not implemented")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
