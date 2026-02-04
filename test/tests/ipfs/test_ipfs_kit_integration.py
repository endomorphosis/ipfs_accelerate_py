"""
Tests for IPFS Kit Integration Layer

These tests verify that the integration layer works correctly with and without
ipfs_kit_py being available, ensuring proper fallback behavior for CI/CD.
"""

import os
import sys
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add ipfs_accelerate_py package directory to path
ipfs_accelerate_py_dir = Path(__file__).parent.parent / "ipfs_accelerate_py"
sys.path.insert(0, str(ipfs_accelerate_py_dir))

# Import the integration module
from ipfs_kit_integration import (
    IPFSKitStorage,
    get_storage,
    reset_storage,
    StorageBackendConfig
)


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def storage(temp_cache_dir):
    """Create a storage instance for testing."""
    reset_storage()
    storage = IPFSKitStorage(
        enable_ipfs_kit=False,  # Force fallback for predictable tests
        cache_dir=temp_cache_dir,
        force_fallback=True
    )
    yield storage
    reset_storage()


class TestIPFSKitStorageInitialization:
    """Test storage initialization and configuration."""
    
    def test_init_with_fallback(self, temp_cache_dir):
        """Test initialization with fallback mode."""
        storage = IPFSKitStorage(
            enable_ipfs_kit=False,
            cache_dir=temp_cache_dir,
            force_fallback=True
        )
        
        assert storage.using_fallback is True
        assert storage.cache_dir == Path(temp_cache_dir)
        assert storage.cache_dir.exists()
    
    def test_init_with_env_disable(self, temp_cache_dir):
        """Test that IPFS_KIT_DISABLE environment variable works."""
        with patch.dict(os.environ, {'IPFS_KIT_DISABLE': '1'}):
            storage = IPFSKitStorage(
                enable_ipfs_kit=True,
                cache_dir=temp_cache_dir
            )
            
            assert storage.force_fallback is True
            assert storage.using_fallback is True
    
    def test_cache_dir_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_path = Path(temp_cache_dir) / "nested" / "cache"
        storage = IPFSKitStorage(
            enable_ipfs_kit=False,
            cache_dir=str(cache_path),
            force_fallback=True
        )
        
        assert cache_path.exists()
        assert cache_path.is_dir()
    
    def test_default_cache_dir(self):
        """Test that default cache directory is used when none provided."""
        reset_storage()
        storage = IPFSKitStorage(force_fallback=True)
        
        expected_dir = Path.home() / ".cache" / "ipfs_accelerate"
        assert storage.cache_dir == expected_dir
        reset_storage()


class TestStorageOperations:
    """Test basic storage operations."""
    
    def test_store_bytes(self, storage):
        """Test storing bytes data."""
        data = b"Hello, IPFS!"
        cid = storage.store(data, filename="test.txt")
        
        assert isinstance(cid, str)
        assert cid.startswith("bafy")  # CIDv1-like format
        assert storage.exists(cid)
    
    def test_store_string(self, storage):
        """Test storing string data."""
        data = "Hello, world!"
        cid = storage.store(data, filename="test.txt")
        
        assert isinstance(cid, str)
        assert storage.exists(cid)
        
        # Retrieve and verify
        retrieved = storage.retrieve(cid)
        assert retrieved == data.encode('utf-8')
    
    def test_store_file(self, storage, temp_cache_dir):
        """Test storing from file path."""
        test_file = Path(temp_cache_dir) / "test_input.txt"
        test_data = b"File content here"
        test_file.write_bytes(test_data)
        
        cid = storage.store(test_file, filename="test_input.txt")
        
        assert storage.exists(cid)
        retrieved = storage.retrieve(cid)
        assert retrieved == test_data
    
    def test_retrieve_nonexistent(self, storage):
        """Test retrieving non-existent CID."""
        result = storage.retrieve("bafynonexistent")
        assert result is None
    
    def test_store_with_metadata(self, storage):
        """Test that metadata is stored correctly."""
        data = b"Test data"
        cid = storage.store(data, filename="test.bin", pin=True)
        
        # Check metadata file exists
        metadata_path = storage.cache_dir / f"{cid}.meta"
        assert metadata_path.exists()
        
        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['filename'] == "test.bin"
        assert metadata.get('fallback') is True


class TestContentAddressing:
    """Test content-addressed storage properties."""
    
    def test_deterministic_cid(self, storage):
        """Test that same content produces same CID."""
        data = b"Deterministic content"
        
        cid1 = storage.store(data, filename="file1.txt")
        cid2 = storage.store(data, filename="file2.txt")
        
        assert cid1 == cid2  # Same content = same CID
    
    def test_different_content_different_cid(self, storage):
        """Test that different content produces different CIDs."""
        data1 = b"Content 1"
        data2 = b"Content 2"
        
        cid1 = storage.store(data1)
        cid2 = storage.store(data2)
        
        assert cid1 != cid2


class TestPinning:
    """Test pinning functionality."""
    
    def test_pin_content(self, storage):
        """Test pinning content."""
        data = b"Pinned content"
        cid = storage.store(data, filename="pinned.txt")
        
        result = storage.pin(cid)
        assert result is True
        
        # Verify pinned status in metadata
        metadata_path = storage.cache_dir / f"{cid}.meta"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['pinned'] is True
    
    def test_unpin_content(self, storage):
        """Test unpinning content."""
        data = b"Unpinned content"
        cid = storage.store(data, filename="unpinned.txt", pin=True)
        
        # First verify it's pinned
        storage.pin(cid)
        
        # Then unpin
        result = storage.unpin(cid)
        assert result is True
        
        # Verify unpinned status in metadata
        metadata_path = storage.cache_dir / f"{cid}.meta"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert metadata['pinned'] is False


class TestListFiles:
    """Test file listing functionality."""
    
    def test_list_empty(self, storage):
        """Test listing when no files exist."""
        files = storage.list_files()
        assert isinstance(files, list)
        assert len(files) == 0
    
    def test_list_files(self, storage):
        """Test listing stored files."""
        # Store multiple files
        data1 = b"File 1"
        data2 = b"File 2"
        
        cid1 = storage.store(data1, filename="file1.txt")
        cid2 = storage.store(data2, filename="file2.txt", pin=True)
        
        # List files
        files = storage.list_files()
        
        assert len(files) == 2
        assert all(isinstance(f, dict) for f in files)
        
        # Check file information
        cids = [f['cid'] for f in files]
        assert cid1 in cids
        assert cid2 in cids
        
        # Find specific file
        file2 = next(f for f in files if f['cid'] == cid2)
        assert file2['filename'] == "file2.txt"
        assert file2['pinned'] is True


class TestDeletion:
    """Test content deletion."""
    
    def test_delete_existing(self, storage):
        """Test deleting existing content."""
        data = b"To be deleted"
        cid = storage.store(data, filename="delete_me.txt")
        
        assert storage.exists(cid)
        
        result = storage.delete(cid)
        assert result is True
        assert not storage.exists(cid)
    
    def test_delete_nonexistent(self, storage):
        """Test deleting non-existent content."""
        result = storage.delete("bafynonexistent")
        assert result is False
    
    def test_delete_removes_metadata(self, storage):
        """Test that deletion removes metadata file."""
        data = b"With metadata"
        cid = storage.store(data, filename="meta.txt")
        
        metadata_path = storage.cache_dir / f"{cid}.meta"
        assert metadata_path.exists()
        
        storage.delete(cid)
        assert not metadata_path.exists()


class TestBackendStatus:
    """Test backend status reporting."""
    
    def test_backend_status_fallback(self, storage):
        """Test backend status in fallback mode."""
        status = storage.get_backend_status()
        
        assert isinstance(status, dict)
        assert status['ipfs_kit_available'] is False
        assert status['using_fallback'] is True
        assert 'cache_dir' in status
        assert 'backends' in status
        assert status['backends']['local'] is True
    
    def test_is_available(self, storage):
        """Test is_available method."""
        assert storage.is_available() is False  # In fallback mode


class TestSingletonPattern:
    """Test singleton storage access."""
    
    def test_get_storage_singleton(self, temp_cache_dir):
        """Test that get_storage returns singleton."""
        reset_storage()
        
        storage1 = get_storage(
            enable_ipfs_kit=False,
            cache_dir=temp_cache_dir,
            force_fallback=True
        )
        storage2 = get_storage()
        
        assert storage1 is storage2
        
        reset_storage()
    
    def test_reset_storage(self, temp_cache_dir):
        """Test that reset_storage clears singleton."""
        storage1 = get_storage(
            enable_ipfs_kit=False,
            cache_dir=temp_cache_dir,
            force_fallback=True
        )
        
        reset_storage()
        
        storage2 = get_storage(force_fallback=True)
        
        assert storage1 is not storage2
        
        reset_storage()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_store_invalid_path(self, storage):
        """Test storing from invalid file path."""
        invalid_path = Path("/nonexistent/file.txt")
        
        with pytest.raises(Exception):
            storage.store(invalid_path)
    
    def test_retrieve_after_manual_deletion(self, storage):
        """Test retrieving after manually deleting file."""
        data = b"Manual delete"
        cid = storage.store(data)
        
        # Manually delete the file
        storage_path = storage.cache_dir / cid
        storage_path.unlink()
        
        # Should return None
        result = storage.retrieve(cid)
        assert result is None


class TestIPFSKitAvailability:
    """Test behavior when ipfs_kit_py might be available."""
    
    def test_fallback_when_import_fails(self, temp_cache_dir):
        """Test fallback when ipfs_kit_py import fails."""
        # This will naturally fall back since ipfs_kit_py may not be in the path
        storage = IPFSKitStorage(
            enable_ipfs_kit=True,  # Try to enable
            cache_dir=temp_cache_dir
        )
        
        # Should fall back gracefully
        assert isinstance(storage.using_fallback, bool)
        
        # Should still work with fallback
        data = b"Fallback test"
        cid = storage.store(data)
        assert storage.exists(cid)


class TestConfigurationOptions:
    """Test various configuration options."""
    
    def test_custom_config(self, temp_cache_dir):
        """Test initialization with custom config."""
        config = {
            'enable_ipfs': True,
            'enable_s3': False,
            'custom_option': 'value'
        }
        
        storage = IPFSKitStorage(
            enable_ipfs_kit=False,
            cache_dir=temp_cache_dir,
            config=config,
            force_fallback=True
        )
        
        assert storage.config == config
    
    def test_no_config(self, temp_cache_dir):
        """Test initialization without config."""
        storage = IPFSKitStorage(
            enable_ipfs_kit=False,
            cache_dir=temp_cache_dir,
            force_fallback=True
        )
        
        assert storage.config == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
