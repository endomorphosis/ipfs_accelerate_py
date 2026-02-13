"""
P2P networking layer for MCP++

This module provides peer-to-peer networking capabilities using libp2p,
refactored from the original MCP implementation to work natively with Trio.

Components:
-----------
- taskqueue: P2P task queue client for distributed task execution
- workflow: P2P workflow scheduler for multi-peer coordination
- peer_registry: Peer discovery and registry management
- bootstrap: Bootstrap helpers for peer discovery

All components are designed to work natively with Trio's structured concurrency.
"""

# Import P2P components (will be copied and refactored)
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

__all__ = [
    "P2PTaskQueue",
    "RemoteQueue",
    "P2PWorkflowScheduler",
    "P2PTask",
    "WorkflowTag",
    "MerkleClock",
    "get_scheduler",
    "HAVE_P2P_SCHEDULER",
    "P2PPeerRegistry",
    "SimplePeerBootstrap",
]
