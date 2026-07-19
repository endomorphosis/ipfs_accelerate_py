"""Backward-compatible import wrapper for MCP++ libp2p compatibility.

The canonical implementation lives in
``ipfs_accelerate_py.mcplusplus_module.p2p.libp2p_runtime``.
"""

from ipfs_accelerate_py.mcplusplus_module.p2p.libp2p_runtime import (  # noqa: F401
    ensure_libp2p_compatible,
    patch_libp2p_compatibility,
)
