"""P2P networking compatibility layer for MCP++.

This package preserves the historical ``mcplusplus_module.p2p`` import surface,
while delegating to canonical implementations in
``ipfs_accelerate_py.mcp_server.mcplusplus`` where available.
"""

# Task queue compatibility layer (canonical implementation lives under
# ipfs_accelerate_py.mcp_server.mcplusplus.task_queue).
try:
    from .taskqueue import P2PTaskQueue, RemoteQueue
except ImportError:
    P2PTaskQueue = None
    RemoteQueue = None

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
    P2PWorkflowScheduler = None
    P2PTask = None
    WorkflowTag = None
    MerkleClock = None
    get_scheduler = None
    HAVE_P2P_SCHEDULER = False

try:
    from .peer_registry import P2PPeerRegistry
except ImportError:
    P2PPeerRegistry = None

try:
    from .bootstrap import SimplePeerBootstrap
except ImportError:
    SimplePeerBootstrap = None

try:
    from .connectivity import ConnectivityConfig, UniversalConnectivity

    # Backward-compatible aliases used by older call sites.
    DiscoveryConfig = ConnectivityConfig
    ConnectivityHelper = UniversalConnectivity
except ImportError:
    ConnectivityConfig = None
    UniversalConnectivity = None
    DiscoveryConfig = None
    ConnectivityHelper = None

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
