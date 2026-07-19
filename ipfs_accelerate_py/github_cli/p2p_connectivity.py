"""Backward-compatible wrapper for MCP++ P2P connectivity.

The canonical implementation lives in
``ipfs_accelerate_py.mcplusplus_module.p2p.connectivity``. Keeping this module
as a shim preserves older GitHub cache imports without maintaining a second
libp2p discovery implementation.
"""

from ipfs_accelerate_py.mcplusplus_module.p2p.connectivity import (
    DEFAULT_BOOTSTRAP_PEERS,
    ConnectivityConfig,
    UniversalConnectivity,
    get_universal_connectivity,
)

__all__ = [
    "DEFAULT_BOOTSTRAP_PEERS",
    "ConnectivityConfig",
    "UniversalConnectivity",
    "get_universal_connectivity",
]
