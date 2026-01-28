"""
Tests for ipfs_datasets_py integration

These tests verify that the integration works correctly both with and without
ipfs_datasets_py available, ensuring graceful fallbacks.
"""

import os
import pytest
import tempfile
from pathlib import Path


def test_availability_check():
    """Test that availability checking works."""
    from ipfs_accelerate_py.datasets_integration import (
        is_datasets_available,
        get_datasets_status
    )
    
    # Should return bool
    available = is_datasets_available()
    assert isinstance(available, bool)
    
    # Should return dict with status
    status = get_datasets_status()
    assert isinstance(status, dict)
    assert 'available' in status
    assert 'enabled' in status
    assert status['available'] == available


def test_datasets_manager_init():
    """Test that DatasetsManager initializes correctly."""
    from ipfs_accelerate_py.datasets_integration import DatasetsManager
    
    manager = DatasetsManager()
    assert isinstance(manager.enabled, bool)
    assert manager.cache_dir.exists()
    
    status = manager.get_status()
    assert isinstance(status, dict)
    assert 'enabled' in status


def test_datasets_manager_logging():
    """Test event logging functionality."""
    from ipfs_accelerate_py.datasets_integration import DatasetsManager
    
    manager = DatasetsManager()
    
    # Should not raise error
    result = manager.log_event("test_event", {"key": "value"})
    assert isinstance(result, bool)


def test_datasets_manager_provenance():
    """Test provenance tracking functionality."""
    from ipfs_accelerate_py.datasets_integration import DatasetsManager
    
    manager = DatasetsManager()
    
    # Should not raise error
    cid = manager.track_provenance("test_operation", {"key": "value"})
    # CID is optional (None if IPFS unavailable)
    assert cid is None or isinstance(cid, str)


def test_filesystem_handler_init():
    """Test that FilesystemHandler initializes correctly."""
    from ipfs_accelerate_py.datasets_integration import FilesystemHandler
    
    fs = FilesystemHandler()
    assert isinstance(fs.enabled, bool)
    assert fs.cache_dir.exists()
    
    status = fs.get_status()
    assert isinstance(status, dict)
    assert 'ipfs_enabled' in status


def test_filesystem_handler_add_file():
    """Test file addition functionality."""
    from ipfs_accelerate_py.datasets_integration import FilesystemHandler
    
    fs = FilesystemHandler()
    
    # Create a test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Test content")
        test_file = f.name
    
    try:
        # Should not raise error
        cid = fs.add_file(test_file)
        # CID is optional (None if IPFS unavailable)
        assert cid is None or isinstance(cid, str)
    finally:
        os.unlink(test_file)


def test_filesystem_handler_add_directory():
    """Test directory addition functionality."""
    from ipfs_accelerate_py.datasets_integration import FilesystemHandler
    
    fs = FilesystemHandler()
    
    # Create a test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Add a file to the directory
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content")
        
        # Should not raise error
        cid = fs.add_directory(temp_dir)
        # CID is optional (None if IPFS unavailable)
        assert cid is None or isinstance(cid, str)


def test_provenance_logger_init():
    """Test that ProvenanceLogger initializes correctly."""
    from ipfs_accelerate_py.datasets_integration import ProvenanceLogger
    
    logger = ProvenanceLogger()
    assert isinstance(logger.enabled, bool)
    assert logger.log_dir.exists()
    
    status = logger.get_status()
    assert isinstance(status, dict)
    assert 'ipfs_enabled' in status


def test_provenance_logger_inference():
    """Test inference logging functionality."""
    from ipfs_accelerate_py.datasets_integration import ProvenanceLogger
    
    logger = ProvenanceLogger()
    
    # Should not raise error
    cid = logger.log_inference("test-model", {
        "input": "test",
        "output": "result"
    })
    # CID is optional (None if IPFS unavailable)
    assert cid is None or isinstance(cid, str)


def test_provenance_logger_transformation():
    """Test transformation logging functionality."""
    from ipfs_accelerate_py.datasets_integration import ProvenanceLogger
    
    logger = ProvenanceLogger()
    
    # Should not raise error
    cid = logger.log_transformation("test_op", {"key": "value"})
    # CID is optional (None if IPFS unavailable)
    assert cid is None or isinstance(cid, str)


def test_provenance_logger_query():
    """Test log querying functionality."""
    from ipfs_accelerate_py.datasets_integration import ProvenanceLogger
    
    logger = ProvenanceLogger()
    
    # Log something
    logger.log_inference("test-model", {"test": "data"})
    
    # Query logs
    logs = logger.query_logs({"type": "inference"})
    assert isinstance(logs, list)


def test_workflow_coordinator_init():
    """Test that WorkflowCoordinator initializes correctly."""
    from ipfs_accelerate_py.datasets_integration import WorkflowCoordinator
    
    coordinator = WorkflowCoordinator()
    assert isinstance(coordinator.enabled, bool)
    assert coordinator.cache_dir.exists()
    
    status = coordinator.get_status()
    assert isinstance(status, dict)
    assert 'p2p_enabled' in status


def test_workflow_coordinator_task_submission():
    """Test task submission functionality."""
    from ipfs_accelerate_py.datasets_integration import WorkflowCoordinator
    
    coordinator = WorkflowCoordinator()
    
    # Should not raise error
    result = coordinator.submit_task(
        task_id="test-task",
        task_type="test",
        data={"key": "value"}
    )
    assert isinstance(result, bool)
    assert result is True


def test_workflow_coordinator_task_lifecycle():
    """Test full task lifecycle."""
    from ipfs_accelerate_py.datasets_integration import WorkflowCoordinator
    
    coordinator = WorkflowCoordinator()
    
    # Submit task
    coordinator.submit_task("lifecycle-task", "test", {"key": "value"})
    
    # Get next task
    task = coordinator.get_next_task("worker-001")
    # Task may be None if queue is empty or P2P unavailable
    if task:
        assert 'task_id' in task
        
        # Complete task
        result = coordinator.complete_task(task['task_id'], {"result": "success"})
        assert isinstance(result, bool)


def test_environment_variable_disable():
    """Test that IPFS_DATASETS_ENABLED=0 disables integration."""
    # Set environment variable
    os.environ['IPFS_DATASETS_ENABLED'] = '0'
    
    try:
        # Re-import to pick up environment change
        import importlib
        import ipfs_accelerate_py.datasets_integration as ds_int
        importlib.reload(ds_int)
        
        # Should be disabled
        available = ds_int.is_datasets_available()
        assert available is False
        
        status = ds_int.get_datasets_status()
        assert status['available'] is False
        
    finally:
        # Clean up
        del os.environ['IPFS_DATASETS_ENABLED']


def test_graceful_fallback():
    """Test that all operations work even when ipfs_datasets_py unavailable."""
    from ipfs_accelerate_py.datasets_integration import (
        DatasetsManager,
        FilesystemHandler,
        ProvenanceLogger,
        WorkflowCoordinator
    )
    
    # All should initialize without errors
    manager = DatasetsManager()
    fs = FilesystemHandler()
    logger = ProvenanceLogger()
    coordinator = WorkflowCoordinator()
    
    # All should have valid status
    assert isinstance(manager.get_status(), dict)
    assert isinstance(fs.get_status(), dict)
    assert isinstance(logger.get_status(), dict)
    assert isinstance(coordinator.get_status(), dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
