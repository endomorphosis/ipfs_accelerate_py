"""P2P networking compatibility layer for MCP++.

This package preserves the historical ``mcplusplus_module.p2p`` import surface,
while delegating to canonical implementations in
``ipfs_accelerate_py.mcp_server.mcplusplus`` where available.
"""

from .. import _missing_dependency_stub

# Task queue compatibility layer (canonical implementation lives under
# ipfs_accelerate_py.mcp_server.mcplusplus.task_queue).
try:
    from .taskqueue import P2PTaskQueue, RemoteQueue
except ImportError:
    P2PTaskQueue = _missing_dependency_stub("P2PTaskQueue")
    RemoteQueue = _missing_dependency_stub("RemoteQueue")

try:
    from .workflow import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag,
        MerkleClock,
        get_scheduler,
        HAVE_P2P_SCHEDULER,
    )
except ImportError:
    P2PWorkflowScheduler = _missing_dependency_stub("P2PWorkflowScheduler")
    P2PTask = _missing_dependency_stub("P2PTask")
    WorkflowTag = _missing_dependency_stub("WorkflowTag")
    MerkleClock = _missing_dependency_stub("MerkleClock")

    def get_scheduler():
        return None

    HAVE_P2P_SCHEDULER = False

try:
    from .peer_registry import P2PPeerRegistry
except ImportError:
    P2PPeerRegistry = _missing_dependency_stub("P2PPeerRegistry")

try:
    from .bootstrap import SimplePeerBootstrap
except ImportError:
    SimplePeerBootstrap = _missing_dependency_stub("SimplePeerBootstrap")

try:
    from .connectivity import ConnectivityConfig, UniversalConnectivity

    # Backward-compatible aliases used by older call sites.
    DiscoveryConfig = ConnectivityConfig
    ConnectivityHelper = UniversalConnectivity
except ImportError:
    ConnectivityConfig = _missing_dependency_stub("ConnectivityConfig")
    UniversalConnectivity = _missing_dependency_stub("UniversalConnectivity")
    DiscoveryConfig = _missing_dependency_stub("DiscoveryConfig")
    ConnectivityHelper = _missing_dependency_stub("ConnectivityHelper")

__all__ = [
    # Task queue
    "P2PTaskQueue",
    "RemoteQueue",
    # Workflow scheduler
    "P2PWorkflowScheduler",
    "P2PTask",
    "WorkflowTag",
    "MerkleClock",
    "get_scheduler",
    "HAVE_P2P_SCHEDULER",
    # Peer discovery
    "P2PPeerRegistry",
    "SimplePeerBootstrap",
    # Connectivity
    "ConnectivityConfig",
    "UniversalConnectivity",
    "DiscoveryConfig",
    "ConnectivityHelper",
]
