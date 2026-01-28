"""
Example: Basic usage of ipfs_datasets_py integration

This example demonstrates basic usage of the datasets integration layer,
showing how to use all four main components with graceful fallbacks.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipfs_accelerate_py.datasets_integration import (
    is_datasets_available,
    get_datasets_status,
    DatasetsManager,
    FilesystemHandler,
    ProvenanceLogger,
    WorkflowCoordinator
)


def main():
    """Main example function."""
    
    # Check availability
    print("=" * 60)
    print("IPFS Datasets Integration Status")
    print("=" * 60)
    
    status = get_datasets_status()
    print(f"Available: {status['available']}")
    print(f"Enabled: {status['enabled']}")
    print(f"Path: {status.get('path', 'N/A')}")
    if 'reason' in status:
        print(f"Reason: {status['reason']}")
    print()
    
    # Example 1: Datasets Manager
    print("=" * 60)
    print("Example 1: Datasets Manager")
    print("=" * 60)
    
    manager = DatasetsManager({
        'enable_audit': True,
        'enable_provenance': True,
        'enable_p2p': False
    })
    
    print(f"Manager enabled: {manager.enabled}")
    
    # Log an event
    if manager.log_event("example_started", {"component": "datasets_manager"}):
        print("✓ Event logged successfully")
    else:
        print("✓ Event logged locally (IPFS unavailable)")
    
    # Track provenance
    cid = manager.track_provenance("example_operation", {
        "operation": "test",
        "data": "example"
    })
    if cid:
        print(f"✓ Provenance tracked with CID: {cid}")
    else:
        print("✓ Provenance tracked locally (IPFS unavailable)")
    
    # Get status
    mgr_status = manager.get_status()
    print(f"✓ Manager status: {mgr_status}")
    print()
    
    # Example 2: Filesystem Handler
    print("=" * 60)
    print("Example 2: Filesystem Handler")
    print("=" * 60)
    
    fs = FilesystemHandler()
    
    print(f"Filesystem enabled: {fs.enabled}")
    
    # Create a test file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, IPFS Datasets Integration!")
        test_file = f.name
    
    try:
        # Add file
        cid = fs.add_file(test_file, pin=True)
        if cid:
            print(f"✓ File added to IPFS with CID: {cid}")
        else:
            print("✓ File cached locally (IPFS unavailable)")
        
        # Get filesystem status
        fs_status = fs.get_status()
        print(f"✓ Filesystem status: {fs_status}")
        
    finally:
        # Clean up
        os.unlink(test_file)
    
    print()
    
    # Example 3: Provenance Logger
    print("=" * 60)
    print("Example 3: Provenance Logger")
    print("=" * 60)
    
    logger = ProvenanceLogger()
    
    print(f"Logger enabled: {logger.enabled}")
    
    # Log inference
    logger.log_inference("bert-base-uncased", {
        "input": "Example text",
        "output_dim": 768,
        "duration_ms": 100
    })
    print("✓ Inference logged")
    
    # Log transformation
    logger.log_transformation("tokenization", {
        "tokenizer": "bert-tokenizer",
        "max_length": 512
    })
    print("✓ Transformation logged")
    
    # Log worker execution
    logger.log_worker_execution("worker-001", "task-001", {
        "status": "completed",
        "duration_ms": 5000
    })
    print("✓ Worker execution logged")
    
    # Query logs
    logs = logger.query_logs({"type": "inference"}, limit=10)
    print(f"✓ Found {len(logs)} inference logs")
    
    # Get logger status
    log_status = logger.get_status()
    print(f"✓ Logger status: {log_status}")
    print()
    
    # Example 4: Workflow Coordinator
    print("=" * 60)
    print("Example 4: Workflow Coordinator")
    print("=" * 60)
    
    coordinator = WorkflowCoordinator({
        'enable_p2p': False  # Disabled for safety in examples
    })
    
    print(f"Coordinator enabled: {coordinator.enabled}")
    
    # Submit task
    coordinator.submit_task(
        task_id="example-task-001",
        task_type="inference",
        data={"model": "bert-base", "input": "test"},
        priority=5
    )
    print("✓ Task submitted")
    
    # Get next task
    task = coordinator.get_next_task("worker-001", ["CPU"])
    if task:
        print(f"✓ Got task: {task['task_id']}")
        
        # Complete task
        coordinator.complete_task(task['task_id'], {
            "status": "success",
            "output": "result"
        })
        print("✓ Task completed")
    
    # List pending tasks
    pending = coordinator.list_pending_tasks()
    print(f"✓ Pending tasks: {len(pending)}")
    
    # Get coordinator status
    coord_status = coordinator.get_status()
    print(f"✓ Coordinator status: {coord_status}")
    print()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
