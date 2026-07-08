"""
Tests for the IPFS backend router.

This test suite validates:
- Backend registration and selection
- ipfs_kit_py backend (preferred)
- HuggingFace cache backend (fallback)
- Kubo CLI backend (fallback)
- Backend fallback mechanisms
"""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from ipfs_accelerate_py import ipfs_backend_router


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return b"Hello, IPFS! This is test data."


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment for testing."""
    # Clear environment variables that affect backend selection
    env_vars = [
        "IPFS_BACKEND",
        "ENABLE_IPFS_KIT",
        "IPFS_KIT_DISABLE",
        "ENABLE_HF_CACHE",
        "KUBO_CMD",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    
    # Clear backend cache
    ipfs_backend_router._get_default_backend_cached.cache_clear()
    ipfs_backend_router._DEFAULT_BACKEND_OVERRIDE = None


class TestBackendProtocol:
    """Test the IPFSBackend protocol."""
    
    def test_backend_protocol(self):
        """Test that backends implement the required protocol."""
        # Check that backends have required methods
        required_methods = [
            'add_bytes', 'cat', 'pin', 'unpin',
            'block_put', 'block_get', 'add_path',
            'get_to_path', 'ls', 'dag_export'
        ]
        
        # Test HF cache backend (always available)
        backend = ipfs_backend_router.HuggingFaceCacheBackend()
        for method in required_methods:
            assert hasattr(backend, method), f"Backend missing method: {method}"
            assert callable(getattr(backend, method))


class TestHuggingFaceCacheBackend:
    """Test HuggingFace cache backend."""
    
    def test_init(self, temp_dir):
        """Test backend initialization."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        assert backend._cache_dir == Path(temp_dir)
        assert backend._ipfs_cache.exists()
    
    def test_add_and_cat_bytes(self, temp_dir, sample_data):
        """Test adding and retrieving bytes."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        # Add data
        cid = backend.add_bytes(sample_data, pin=True)
        assert cid is not None
        assert isinstance(cid, str)
        assert cid.startswith("bafy")
        
        # Retrieve data
        retrieved = backend.cat(cid)
        assert retrieved == sample_data
    
    def test_block_put_and_get(self, temp_dir, sample_data):
        """Test block operations."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        # Store block
        cid = backend.block_put(sample_data, codec="raw")
        assert cid is not None
        
        # Retrieve block
        retrieved = backend.block_get(cid)
        assert retrieved == sample_data
    
    def test_pin_unpin(self, temp_dir, sample_data):
        """Test pinning and unpinning."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        cid = backend.add_bytes(sample_data, pin=False)
        
        # Pin
        backend.pin(cid)
        meta_path = backend._ipfs_cache / f"{cid}.meta"
        assert meta_path.exists()
        
        # Unpin
        backend.unpin(cid)
        assert not meta_path.exists()
    
    def test_add_path(self, temp_dir, sample_data):
        """Test adding a file."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_bytes(sample_data)
        
        # Add file
        cid = backend.add_path(str(test_file), pin=True)
        assert cid is not None
        
        # Verify content
        retrieved = backend.cat(cid)
        assert retrieved == sample_data
    
    def test_get_to_path(self, temp_dir, sample_data):
        """Test retrieving to path."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        # Add data
        cid = backend.add_bytes(sample_data, pin=True)
        
        # Retrieve to path
        output_path = Path(temp_dir) / "output.txt"
        backend.get_to_path(cid, output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.read_bytes() == sample_data
    
    def test_cat_nonexistent(self, temp_dir):
        """Test retrieving nonexistent CID."""
        backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        
        with pytest.raises(RuntimeError, match="CID not found"):
            backend.cat("bafy_nonexistent")


class TestKuboCLIBackend:
    """Test Kubo CLI backend."""
    
    def test_init(self):
        """Test backend initialization."""
        backend = ipfs_backend_router.KuboCLIBackend(cmd="ipfs")
        assert backend._cmd == "ipfs"
    
    @patch('subprocess.run')
    def test_add_bytes(self, mock_run, sample_data):
        """Test adding bytes via CLI."""
        mock_run.return_value = Mock(returncode=0, stdout=b"QmTest123\n", stderr=b"")
        
        backend = ipfs_backend_router.KuboCLIBackend()
        cid = backend.add_bytes(sample_data, pin=True)
        
        assert cid == "QmTest123"
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0][0] == "ipfs"
        assert "add" in call_args[0][0]
    
    @patch('subprocess.run')
    def test_cat(self, mock_run, sample_data):
        """Test cat via CLI."""
        mock_run.return_value = Mock(returncode=0, stdout=sample_data, stderr=b"")
        
        backend = ipfs_backend_router.KuboCLIBackend()
        result = backend.cat("QmTest123")
        
        assert result == sample_data
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert "cat" in call_args[0][0]
        assert "QmTest123" in call_args[0][0]
    
    @patch('subprocess.run')
    def test_command_failure(self, mock_run):
        """Test handling of CLI command failure."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout=b"",
            stderr=b"Error: command failed"
        )
        
        backend = ipfs_backend_router.KuboCLIBackend()
        
        with pytest.raises(RuntimeError, match="Error: command failed"):
            backend.cat("QmTest123")


class TestIPFSKitBackend:
    """Test ipfs_kit_py backend."""
    
    def test_init_with_mock_storage(self):
        """Test initialization with mocked storage."""
        with patch('ipfs_accelerate_py.ipfs_backend_router.IPFSKitBackend._init_storage'):
            backend = ipfs_backend_router.IPFSKitBackend()
            assert backend is not None
    
    @patch('ipfs_accelerate_py.ipfs_kit_integration.get_storage')
    def test_init_success(self, mock_get_storage):
        """Test successful initialization."""
        mock_storage = Mock()
        mock_get_storage.return_value = mock_storage
        
        backend = ipfs_backend_router.IPFSKitBackend()
        assert backend._storage == mock_storage
        mock_get_storage.assert_called_once()
    
    @patch('ipfs_accelerate_py.ipfs_kit_integration.get_storage')
    def test_add_bytes(self, mock_get_storage, sample_data):
        """Test adding bytes."""
        mock_storage = Mock()
        mock_storage.store.return_value = "bafy_test_cid"
        mock_get_storage.return_value = mock_storage
        
        backend = ipfs_backend_router.IPFSKitBackend()
        cid = backend.add_bytes(sample_data, pin=True)
        
        assert cid == "bafy_test_cid"
        mock_storage.store.assert_called_once_with(sample_data, pin=True)
    
    @patch('ipfs_accelerate_py.ipfs_kit_integration.get_storage')
    def test_cat(self, mock_get_storage, sample_data):
        """Test retrieving bytes."""
        mock_storage = Mock()
        mock_storage.retrieve.return_value = sample_data
        mock_get_storage.return_value = mock_storage
        
        backend = ipfs_backend_router.IPFSKitBackend()
        result = backend.cat("bafy_test_cid")
        
        assert result == sample_data
        mock_storage.retrieve.assert_called_once_with("bafy_test_cid")


class TestBackendSelection:
    """Test backend selection and fallback logic."""
    
    def test_default_backend_selection(self, clean_env):
        """Test default backend selection."""
        backend = ipfs_backend_router.get_backend()
        assert backend is not None
        assert isinstance(backend, ipfs_backend_router.IPFSBackend)
    
    def test_explicit_backend(self, temp_dir):
        """Test using explicit backend."""
        explicit_backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        backend = ipfs_backend_router.get_backend(backend=explicit_backend)
        assert backend is explicit_backend
    
    def test_global_backend_override(self, temp_dir, clean_env):
        """Test global backend override."""
        override_backend = ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        ipfs_backend_router.set_default_ipfs_backend(override_backend)
        
        backend = ipfs_backend_router.get_backend()
        assert backend is override_backend
        
        # Clean up
        ipfs_backend_router.set_default_ipfs_backend(None)
    
    @patch.dict(os.environ, {'IPFS_KIT_DISABLE': '1'})
    def test_ipfs_kit_disabled(self, clean_env):
        """Test that ipfs_kit is disabled when env var is set."""
        backend = ipfs_backend_router._get_ipfs_kit_backend()
        assert backend is None
    
    @patch.dict(os.environ, {'ENABLE_HF_CACHE': 'false'})
    def test_hf_cache_disabled(self, clean_env):
        """Test that HF cache is disabled when env var is set."""
        backend = ipfs_backend_router._get_hf_cache_backend()
        assert backend is None
    
    def test_backend_caching(self, clean_env):
        """Test that backend is cached."""
        # Clear cache
        ipfs_backend_router._get_default_backend_cached.cache_clear()
        
        # Get backend twice
        backend1 = ipfs_backend_router.get_backend()
        backend2 = ipfs_backend_router.get_backend()
        
        # Should be the same instance due to caching
        assert backend1 is backend2


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_add_bytes_and_cat(self, temp_dir, sample_data, clean_env, monkeypatch):
        """Test add_bytes and cat convenience functions."""
        # Force HF cache backend for testing
        monkeypatch.setenv("IPFS_BACKEND", "hf_cache")
        monkeypatch.setenv("HF_HOME", temp_dir)
        ipfs_backend_router._get_default_backend_cached.cache_clear()
        
        # Register HF cache backend
        ipfs_backend_router.register_ipfs_backend(
            "hf_cache",
            lambda: ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        )
        
        # Add data
        cid = ipfs_backend_router.add_bytes(sample_data, pin=True)
        assert cid is not None
        
        # Retrieve data
        retrieved = ipfs_backend_router.cat(cid)
        assert retrieved == sample_data
    
    def test_block_put_and_get(self, temp_dir, sample_data, clean_env, monkeypatch):
        """Test block_put and block_get convenience functions."""
        monkeypatch.setenv("IPFS_BACKEND", "hf_cache")
        monkeypatch.setenv("HF_HOME", temp_dir)
        ipfs_backend_router._get_default_backend_cached.cache_clear()
        
        ipfs_backend_router.register_ipfs_backend(
            "hf_cache",
            lambda: ipfs_backend_router.HuggingFaceCacheBackend(cache_dir=temp_dir)
        )
        
        # Store block
        cid = ipfs_backend_router.block_put(sample_data, codec="raw")
        assert cid is not None
        
        # Get block
        retrieved = ipfs_backend_router.block_get(cid)
        assert retrieved == sample_data


class TestBackendRegistry:
    """Test backend registration."""
    
    def test_register_backend(self):
        """Test registering a custom backend."""
        def factory():
            return ipfs_backend_router.HuggingFaceCacheBackend()
        
        ipfs_backend_router.register_ipfs_backend("test_backend", factory)
        assert "test_backend" in ipfs_backend_router._PROVIDER_REGISTRY
    
    def test_register_empty_name(self):
        """Test that empty names are rejected."""
        with pytest.raises(ValueError, match="Backend name must be non-empty"):
            ipfs_backend_router.register_ipfs_backend("", lambda: None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
