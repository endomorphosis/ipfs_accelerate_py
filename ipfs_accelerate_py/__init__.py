"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration. Key features include:

- Hardware acceleration (CPU, GPU, OpenVINO, WebNN, WebGPU)
- IPFS-based content distribution and caching
- Browser integration for client-side inference
- Model type detection and optimization
- Cross-platform support
"""

# Import original components
try:
    from .container_backends import backends
except ImportError:
    backends = None

try:
    from .install_depends import install_depends
except ImportError:
    install_depends = None

import os

# Optionally skip importing the heavy core (avoids ipfs_kit_py import at import-time)
if os.environ.get("IPFS_ACCEL_SKIP_CORE", "0") != "1":
    try:
        from .ipfs_accelerate import ipfs_accelerate_py as original_ipfs_accelerate_py
    except ImportError:
        original_ipfs_accelerate_py = None
else:
    original_ipfs_accelerate_py = None

try:
    from .ipfs_multiformats import ipfs_multiformats_py
except ImportError:
    ipfs_multiformats_py = None

try:
    from .worker import worker
except ImportError:
    worker = None

try:
    from .config import config
except ImportError:
    config = None

SKIP_CORE = os.environ.get("IPFS_ACCEL_SKIP_CORE", "0") == "1"

# Import WebNN/WebGPU integration (skip when core is disabled)
if not SKIP_CORE:
    try:
        from .webnn_webgpu_integration import (
            accelerate_with_browser,
            WebNNWebGPUAccelerator,
            get_accelerator
        )
        webnn_webgpu_available = True
    except ImportError:
        webnn_webgpu_available = False
        
        # Create stubs if not available
        def accelerate_with_browser(*args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is not available")
        
        def get_accelerator(*args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is not available")
        
        class WebNNWebGPUAccelerator:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("WebNN/WebGPU integration is not available")
else:
    webnn_webgpu_available = False
    def accelerate_with_browser(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")
    def get_accelerator(*args, **kwargs):
        raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")
    class WebNNWebGPUAccelerator:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("WebNN/WebGPU integration is disabled (IPFS_ACCEL_SKIP_CORE=1)")

# Import Model Manager (skip by default to avoid heavy optional deps at import time)
if os.environ.get("IPFS_ACCEL_IMPORT_EAGER", "0") == "1":
    try:
        from .model_manager import (
            ModelManager, ModelMetadata, IOSpec, ModelType, DataType,
            create_model_from_huggingface, get_default_model_manager
        )
        model_manager_available = True
    except ImportError:
        model_manager_available = False
        def get_default_model_manager(*args, **kwargs):
            raise NotImplementedError("Model Manager is not available")
        class ModelManager:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("Model Manager is not available")
else:
    model_manager_available = False
    def get_default_model_manager(*args, **kwargs):
        raise NotImplementedError("Model Manager is not imported by default. Set IPFS_ACCEL_IMPORT_EAGER=1 to enable.")
    class ModelManager:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("Model Manager is not imported by default. Set IPFS_ACCEL_IMPORT_EAGER=1 to enable.")

# Import our new implementation
try:
    import sys
    import os
    
    # Add the parent directory to the path to import from top-level module
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from ipfs_accelerate_py import ipfs_accelerate_py, get_instance
except ImportError:
    # Fall back to original implementation if it exists
    if original_ipfs_accelerate_py is not None:
        ipfs_accelerate_py = original_ipfs_accelerate_py
        get_instance = lambda: None
    else:
        # Create stub if neither is available
        def ipfs_accelerate_py(*args, **kwargs):
            raise NotImplementedError("IPFS Accelerate Python is not available")
        
        def get_instance():
            raise NotImplementedError("IPFS Accelerate Python is not available")

# Export all components
export = {
    "backends": backends,
    "config": config,
    "install_depends": install_depends,
    "ipfs_accelerate_py": ipfs_accelerate_py,
    "worker": worker,
    "ipfs_multiformats_py": ipfs_multiformats_py,
    "get_instance": get_instance,
    "accelerate_with_browser": accelerate_with_browser,
    "WebNNWebGPUAccelerator": WebNNWebGPUAccelerator,
    "get_accelerator": get_accelerator,
    "webnn_webgpu_available": webnn_webgpu_available,
    "ModelManager": ModelManager,
    "get_default_model_manager": get_default_model_manager,
    "model_manager_available": model_manager_available
}

# Add CLI entry point for package access
try:
    from .cli_entry import main as cli_main
    export["cli_main"] = cli_main
except ImportError:
    cli_main = None

# Add system logs access
try:
    from .logs import get_system_logs, SystemLogs
    export["get_system_logs"] = get_system_logs
    export["SystemLogs"] = SystemLogs
except ImportError:
    get_system_logs = None
    SystemLogs = None

# Add P2P workflow scheduler access
try:
    from .p2p_workflow_scheduler import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag,
        MerkleClock,
        FibonacciHeap,
        calculate_hamming_distance
    )
    export["P2PWorkflowScheduler"] = P2PWorkflowScheduler
    export["P2PTask"] = P2PTask
    export["WorkflowTag"] = WorkflowTag
    export["MerkleClock"] = MerkleClock
    export["FibonacciHeap"] = FibonacciHeap
    export["calculate_hamming_distance"] = calculate_hamming_distance
except ImportError:
    P2PWorkflowScheduler = None
    P2PTask = None
    WorkflowTag = None
    MerkleClock = None
    FibonacciHeap = None
    calculate_hamming_distance = None

__all__ = [
    'ipfs_accelerate_py', 'get_instance', 'backends', 'config', 
    'install_depends', 'worker', 'ipfs_multiformats_py',
    'accelerate_with_browser', 'WebNNWebGPUAccelerator', 'get_accelerator',
    'webnn_webgpu_available', 'ModelManager', 'get_default_model_manager',
    'model_manager_available', 'cli_main', 'get_system_logs', 'SystemLogs',
    'P2PWorkflowScheduler', 'P2PTask', 'WorkflowTag', 'MerkleClock',
    'FibonacciHeap', 'calculate_hamming_distance'
]

# Package version
__version__ = "0.4.0"