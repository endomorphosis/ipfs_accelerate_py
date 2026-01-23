"""
IPFS Accelerate Python package.

This package provides a framework for hardware-accelerated machine learning inference
with IPFS network-based distribution and acceleration.
"""

# Import from new implementation
from .ipfs_accelerate_py import ipfs_accelerate_py, get_instance

# Re-export common subpackages from the internal implementation package.
# The repo layout has an outer package (this file) and an inner implementation
# package at `ipfs_accelerate_py/`. Many callers import `ipfs_accelerate_py.github_cli`.
try:  # pragma: no cover
    from .ipfs_accelerate_py import github_cli as github_cli  # type: ignore
    import sys as _sys

    _sys.modules[__name__ + ".github_cli"] = github_cli
except Exception:
    github_cli = None  # type: ignore

# Try to import from package if available
try:
    from ipfs_accelerate_py import export
except ImportError:
    # Create export if not available from package
    export = {"ipfs_accelerate_py": ipfs_accelerate_py, "get_instance": get_instance}

__all__ = ['ipfs_accelerate_py', 'get_instance', 'export', 'github_cli']