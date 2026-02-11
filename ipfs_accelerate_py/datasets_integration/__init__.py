"""
IPFS Datasets Integration for ipfs_accelerate_py

This module provides integration with ipfs_datasets_py for distributed dataset
manipulation services. It performs filesystem operations in a local-first and
decentralized fashion using IPFS, with graceful fallbacks when the package is
not present or disabled (as in CI/CD environments).

Key features:
- Distributed dataset management via IPFS
- Event and provenance logging for decentralized execution
- P2P workflow scheduling for worker coordination
- UnixFS-based filesystem operations
- Model manager integration for IPFS storage
- GitHub Copilot logs and pull request data tracking

Environment Variables:
    IPFS_DATASETS_ENABLED: Enable/disable ipfs_datasets_py integration (default: auto-detect)
    IPFS_DATASETS_PATH: Path to ipfs_datasets_py submodule (default: ipfs_datasets_py)

Example:
    >>> from ipfs_accelerate_py.datasets_integration import DatasetsManager
    >>> # Works with or without ipfs_datasets_py - graceful fallback
    >>> manager = DatasetsManager()
    >>> if manager.enabled:
    ...     # IPFS features available
    ...     manager.log_event("inference_started", {"model": "bert-base"})
    ... else:
    ...     # Local fallback mode
    ...     pass
"""

import os
import sys
from typing import Optional, Any, Dict

# Check if ipfs_datasets_py is available
_DATASETS_AVAILABLE = None
_DATASETS_PATH = None


def _check_datasets_availability() -> bool:
    """Check if ipfs_datasets_py is available and enabled."""
    global _DATASETS_AVAILABLE, _DATASETS_PATH
    
    if _DATASETS_AVAILABLE is not None:
        return _DATASETS_AVAILABLE
    
    # Check environment variable
    env_enabled = os.environ.get('IPFS_DATASETS_ENABLED', 'auto').lower()
    if env_enabled in ('0', 'false', 'no', 'off', 'disabled'):
        _DATASETS_AVAILABLE = False
        return False
    
    # Try to find and import ipfs_datasets_py
    try:
        # Check for submodule path
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        submodule_path = os.path.join(current_dir, 'external', 'ipfs_datasets_py')
        custom_path = os.environ.get('IPFS_DATASETS_PATH')
        
        if custom_path and os.path.isdir(custom_path):
            _DATASETS_PATH = custom_path
        elif os.path.isdir(submodule_path):
            _DATASETS_PATH = submodule_path
        
        # Add to path if found
        if _DATASETS_PATH and _DATASETS_PATH not in sys.path:
            sys.path.insert(0, _DATASETS_PATH)
        
        # Try to import
        import ipfs_datasets_py
        _DATASETS_AVAILABLE = True
        return True
        
    except (ImportError, Exception) as e:
        # If auto mode, silently disable; if explicitly enabled, warn
        if env_enabled == 'auto':
            _DATASETS_AVAILABLE = False
        else:
            import warnings
            warnings.warn(
                f"ipfs_datasets_py enabled but not available: {e}. "
                "Falling back to local operations.",
                RuntimeWarning
            )
            _DATASETS_AVAILABLE = False
        return False


def is_datasets_available() -> bool:
    """
    Check if ipfs_datasets_py integration is available.
    
    Returns:
        bool: True if ipfs_datasets_py is available and enabled
    
    Example:
        >>> if is_datasets_available():
        ...     # Use distributed features
        ...     pass
        ... else:
        ...     # Use local fallback
        ...     pass
    """
    return _check_datasets_availability()


def get_datasets_status() -> Dict[str, Any]:
    """
    Get detailed status of ipfs_datasets_py integration.
    
    Returns:
        Dict with status information including:
        - available: bool - Whether ipfs_datasets_py is available and working
        - path: Optional[str] - Path to ipfs_datasets_py if found
        - enabled: bool - Whether integration is enabled (not explicitly disabled)
        - mode: str - Configuration mode ('auto', 'enabled', 'disabled')
        - reason: str - Explanation if unavailable
    
    Example:
        >>> status = get_datasets_status()
        >>> print(f"Datasets available: {status['available']}")
    """
    available = is_datasets_available()
    env_val = os.environ.get('IPFS_DATASETS_ENABLED', 'auto').lower()
    
    # Determine if enabled (not explicitly disabled)
    is_enabled = env_val not in ('0', 'false', 'no', 'off', 'disabled')
    
    # Determine mode
    if env_val in ('0', 'false', 'no', 'off', 'disabled'):
        mode = 'disabled'
    elif env_val in ('1', 'true', 'yes', 'on', 'enabled'):
        mode = 'enabled'
    else:
        mode = 'auto'
    
    status = {
        'available': available,
        'path': _DATASETS_PATH,
        'enabled': is_enabled,
        'mode': mode,
    }
    
    if not available:
        if not is_enabled:
            status['reason'] = 'Explicitly disabled via IPFS_DATASETS_ENABLED'
        else:
            status['reason'] = 'Package not found or import failed'
    
    return status


# Public API: always expose integration classes
# Their internal flags handle availability, enabling graceful fallback
from .manager import DatasetsManager
from .filesystem import FilesystemHandler
from .provenance import ProvenanceLogger
from .workflow import WorkflowCoordinator

__all__ = [
    'is_datasets_available',
    'get_datasets_status',
    'DatasetsManager',
    'FilesystemHandler',
    'ProvenanceLogger',
    'WorkflowCoordinator',
]
