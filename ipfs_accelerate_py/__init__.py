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
except Exception:
    backends = None

try:
    from .install_depends import install_depends
except Exception:
    install_depends = None

import os
import sys
from pathlib import Path

def _add_external_package(package_name: str) -> None:
    """Ensure external bundled packages are importable without pip install."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "external" / package_name
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

# Optionally skip importing the heavy core (avoids ipfs_kit_py import at import-time)
if os.environ.get("IPFS_ACCEL_SKIP_CORE", "0") != "1":
    try:
        _add_external_package("ipfs_kit_py")
        _add_external_package("ipfs_model_manager_py")
        _add_external_package("ipfs_transformers_py")
        from .ipfs_accelerate import ipfs_accelerate_py as original_ipfs_accelerate_py
    except Exception:
        original_ipfs_accelerate_py = None
else:
    original_ipfs_accelerate_py = None

try:
    from .ipfs_multiformats import ipfs_multiformats_py
except Exception:
    ipfs_multiformats_py = None

try:
    from .worker import worker
except Exception:
    worker = None

try:
    from .config import config
except Exception:
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
    except Exception:
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
    except Exception:
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

_global_instance = None

# Public constructor/entrypoint (may be unavailable when core is disabled or missing deps)
if original_ipfs_accelerate_py is not None:
    ipfs_accelerate_py = original_ipfs_accelerate_py

    def get_instance(**kwargs):
        """Get or create a process-wide singleton instance of ipfs_accelerate_py.

        Accepts optional dependency injections (e.g., ``deps``, ``ipfs_kit``) and
        forwards them to the constructor on first creation.
        """
        global _global_instance
        if _global_instance is None:
            _global_instance = ipfs_accelerate_py(**kwargs)
        elif kwargs:
            # Best-effort: attach injected deps to existing singleton.
            for k, v in kwargs.items():
                try:
                    setattr(_global_instance, k, v)
                except Exception:
                    pass
        return _global_instance
else:
    def ipfs_accelerate_py(*args, **kwargs):
        raise NotImplementedError(
            "IPFS Accelerate core is not available (missing deps) or disabled. "
            "Set IPFS_ACCEL_SKIP_CORE=0 and install core dependencies to enable."
        )

    def get_instance():
        raise NotImplementedError(
            "IPFS Accelerate core is not available (missing deps) or disabled. "
            "Set IPFS_ACCEL_SKIP_CORE=0 and install core dependencies to enable."
        )

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

# Add IPFS Kit integration
try:
    from .ipfs_kit_integration import (
        IPFSKitStorage,
        get_storage,
        reset_storage,
        StorageBackendConfig
    )
    export["IPFSKitStorage"] = IPFSKitStorage
    export["get_storage"] = get_storage
    export["reset_storage"] = reset_storage
    export["StorageBackendConfig"] = StorageBackendConfig
except ImportError:
    IPFSKitStorage = None
    get_storage = None
    reset_storage = None
    StorageBackendConfig = None

# Add auto-patching for transformers (applies automatically on import if enabled)
try:
    from . import auto_patch_transformers
    export["auto_patch_transformers"] = auto_patch_transformers
except ImportError:
    auto_patch_transformers = None

# Add LLM router functionality
try:
    from .llm_router import (
        generate_text,
        get_llm_provider,
        register_llm_provider,
        clear_llm_router_caches,
        LLMProvider
    )
    from .router_deps import (
        RouterDeps,
        get_default_router_deps,
        set_default_router_deps
    )
    export["generate_text"] = generate_text
    export["get_llm_provider"] = get_llm_provider
    export["register_llm_provider"] = register_llm_provider
    export["clear_llm_router_caches"] = clear_llm_router_caches
    export["LLMProvider"] = LLMProvider
    export["RouterDeps"] = RouterDeps
    export["get_default_router_deps"] = get_default_router_deps
    export["set_default_router_deps"] = set_default_router_deps
    llm_router_available = True
except ImportError:
    generate_text = None
    get_llm_provider = None
    register_llm_provider = None
    clear_llm_router_caches = None
    LLMProvider = None
    RouterDeps = None
    get_default_router_deps = None
    set_default_router_deps = None
    llm_router_available = False

__all__ = [
    'ipfs_accelerate_py', 'get_instance', 'backends', 'config', 
    'install_depends', 'worker', 'ipfs_multiformats_py',
    'accelerate_with_browser', 'WebNNWebGPUAccelerator', 'get_accelerator',
    'webnn_webgpu_available', 'ModelManager', 'get_default_model_manager',
    'model_manager_available', 'cli_main', 'get_system_logs', 'SystemLogs',
    'P2PWorkflowScheduler', 'P2PTask', 'WorkflowTag', 'MerkleClock',
    'FibonacciHeap', 'calculate_hamming_distance',
    'IPFSKitStorage', 'get_storage', 'reset_storage', 'StorageBackendConfig',
    'auto_patch_transformers',
    'generate_text', 'get_llm_provider', 'register_llm_provider',
    'clear_llm_router_caches', 'LLMProvider', 'RouterDeps',
    'get_default_router_deps', 'set_default_router_deps', 'llm_router_available'
]

# Package version
__version__ = "0.4.0"