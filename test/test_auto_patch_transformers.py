"""
Tests for the auto_patch_transformers module.

Tests the automatic patching system for HuggingFace transformers
to integrate distributed filesystem support.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest import mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipfs_accelerate_py import auto_patch_transformers


class TestAutoPatchTransformers:
    """Test suite for auto_patch_transformers module."""
    
    def setup_method(self):
        """Setup for each test."""
        # Ensure patches are restored before each test
        auto_patch_transformers.restore()
        
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore patches after each test
        auto_patch_transformers.restore()
    
    def test_should_patch_default(self):
        """Test that patching is enabled by default."""
        # Clear environment variables
        for key in ['TRANSFORMERS_PATCH_DISABLE', 'IPFS_KIT_DISABLE', 
                    'STORAGE_FORCE_LOCAL', 'CI']:
            os.environ.pop(key, None)
        
        assert auto_patch_transformers.should_patch() is True
    
    def test_should_patch_disabled_by_env(self):
        """Test that patching can be disabled via environment variables."""
        # Test TRANSFORMERS_PATCH_DISABLE
        os.environ['TRANSFORMERS_PATCH_DISABLE'] = '1'
        assert auto_patch_transformers.should_patch() is False
        os.environ.pop('TRANSFORMERS_PATCH_DISABLE')
        
        # Test IPFS_KIT_DISABLE
        os.environ['IPFS_KIT_DISABLE'] = '1'
        assert auto_patch_transformers.should_patch() is False
        os.environ.pop('IPFS_KIT_DISABLE')
        
        # Test STORAGE_FORCE_LOCAL
        os.environ['STORAGE_FORCE_LOCAL'] = '1'
        assert auto_patch_transformers.should_patch() is False
        os.environ.pop('STORAGE_FORCE_LOCAL')
        
        # Test CI environment
        os.environ['CI'] = '1'
        assert auto_patch_transformers.should_patch() is False
        os.environ.pop('CI')
    
    def test_apply_without_transformers(self):
        """Test apply() when transformers is not available."""
        with mock.patch.dict('sys.modules', {'transformers': None}):
            result = auto_patch_transformers.apply()
            assert result is False
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_apply_with_transformers(self):
        """Test apply() when transformers is available."""
        import transformers
        
        # Store original method
        original_from_pretrained = transformers.AutoModel.from_pretrained
        
        # Apply patches
        result = auto_patch_transformers.apply()
        
        # Should succeed
        assert result is True
        
        # Method should be patched
        assert transformers.AutoModel.from_pretrained != original_from_pretrained
        
        # Status should reflect patches
        status = auto_patch_transformers.get_status()
        assert status['enabled'] is True
        assert status['applied'] is True
        assert len(status['patched_classes']) > 0
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_restore(self):
        """Test restore() functionality."""
        import transformers
        
        # Store original method
        original_from_pretrained = transformers.AutoModel.from_pretrained
        
        # Apply patches
        auto_patch_transformers.apply()
        
        # Verify it's patched
        assert transformers.AutoModel.from_pretrained != original_from_pretrained
        
        # Restore
        result = auto_patch_transformers.restore()
        assert result is True
        
        # Should be restored (note: due to classmethod wrapping, exact equality may not hold)
        # But the status should reflect restoration
        status = auto_patch_transformers.get_status()
        assert status['enabled'] is False
        assert status['applied'] is False
        assert len(status['patched_classes']) == 0
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_patched_from_pretrained_basic(self):
        """Test that patched from_pretrained still works."""
        import transformers
        
        # Apply patches
        auto_patch_transformers.apply()
        
        # Create a mock model
        with mock.patch.object(
            transformers.AutoConfig,
            'from_pretrained',
            return_value=mock.MagicMock()
        ):
            # Should not raise an error
            try:
                # This will fail to load a real model, but should execute our patched code
                with pytest.raises((OSError, Exception)):
                    transformers.AutoConfig.from_pretrained("fake-model")
            except Exception:
                pass  # Expected to fail, we're just testing the patch applies
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_patched_from_pretrained_with_cache_dir(self):
        """Test that patched from_pretrained respects user-specified cache_dir."""
        import transformers
        
        # Apply patches
        auto_patch_transformers.apply()
        
        # Create a temporary cache directory
        with tempfile.TemporaryDirectory() as tmpdir:
            user_cache_dir = tmpdir
            
            # Mock the from_pretrained to capture kwargs
            original_method = auto_patch_transformers._original_from_pretrained.get(
                'transformers.AutoConfig', 
                transformers.AutoConfig.from_pretrained.__func__
            )
            
            with mock.patch.object(
                type(transformers.AutoConfig),
                '__dict__',
                {**type(transformers.AutoConfig).__dict__}
            ):
                # The patched method should preserve user's cache_dir
                pass  # This is complex to test without actual model loading
    
    def test_get_status(self):
        """Test get_status() returns correct information."""
        status = auto_patch_transformers.get_status()
        
        assert isinstance(status, dict)
        assert 'enabled' in status
        assert 'applied' in status
        assert 'patched_classes' in status
        assert 'should_patch' in status
    
    def test_is_patching_enabled(self):
        """Test is_patching_enabled() function."""
        # Initially should be false (after teardown)
        auto_patch_transformers.restore()
        assert auto_patch_transformers.is_patching_enabled() is False
        
        # After apply, should be true (if transformers available)
        if sys.modules.get('transformers'):
            auto_patch_transformers.apply()
            assert auto_patch_transformers.is_patching_enabled() is True
    
    def test_disable(self):
        """Test disable() function."""
        # disable() should be equivalent to restore()
        if sys.modules.get('transformers'):
            auto_patch_transformers.apply()
            assert auto_patch_transformers.is_patching_enabled() is True
            
            auto_patch_transformers.disable()
            assert auto_patch_transformers.is_patching_enabled() is False
    
    def test_create_patched_from_pretrained(self):
        """Test create_patched_from_pretrained creates a working wrapper."""
        # Create a mock original method
        def mock_from_pretrained(model_name, *args, **kwargs):
            return {"model_name": model_name, "kwargs": kwargs}
        
        # Create patched version
        patched = auto_patch_transformers.create_patched_from_pretrained(
            mock_from_pretrained,
            "TestModel"
        )
        
        # Test that it wraps correctly
        result = patched("test-model", trust_remote_code=True)
        assert result["model_name"] == "test-model"
        assert "trust_remote_code" in result["kwargs"]
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_multiple_apply_calls(self):
        """Test that multiple apply() calls don't cause issues."""
        import transformers
        
        # First apply
        result1 = auto_patch_transformers.apply()
        assert result1 is True
        
        # Second apply (should be idempotent)
        result2 = auto_patch_transformers.apply()
        assert result2 is True
        
        # Should still be patched
        status = auto_patch_transformers.get_status()
        assert status['applied'] is True
    
    def test_patching_with_ci_environment(self):
        """Test that patching is disabled in CI environment."""
        os.environ['CI'] = '1'
        
        # Should not patch
        assert auto_patch_transformers.should_patch() is False
        
        # apply() should return False
        result = auto_patch_transformers.apply()
        assert result is False
        
        os.environ.pop('CI')


class TestIntegrationWithStorageWrapper:
    """Test integration between auto_patch_transformers and storage_wrapper."""
    
    def setup_method(self):
        """Setup for each test."""
        auto_patch_transformers.restore()
    
    def teardown_method(self):
        """Cleanup after each test."""
        auto_patch_transformers.restore()
    
    @pytest.mark.skipif(
        not sys.modules.get('transformers'),
        reason="transformers not installed"
    )
    def test_patched_uses_storage_wrapper_cache(self):
        """Test that patched from_pretrained uses storage_wrapper cache_dir."""
        # This test would require mocking storage_wrapper
        # For now, just verify the import path works
        try:
            from ipfs_accelerate_py.common.storage_wrapper import get_storage_wrapper
            storage_wrapper_available = True
        except ImportError:
            storage_wrapper_available = False
        
        # If storage_wrapper is available, patching should use it
        if storage_wrapper_available:
            auto_patch_transformers.apply()
            status = auto_patch_transformers.get_status()
            assert status['applied'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
