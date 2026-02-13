"""
P2P Workflow Scheduler for MCP++ (refactored from original MCP module).

This module provides P2P workflow scheduling capabilities, allowing workflows
to bypass GitHub API and execute across the peer-to-peer network using Trio.

Module: ipfs_accelerate_py.mcplusplus_module.p2p.workflow

Refactored from: ipfs_accelerate_py/mcp/tools/p2p_workflow_tools.py
Key changes:
- Trio-native async operations
- Improved error handling and logging
- Full type hints throughout
"""

from __future__ import annotations

import logging
import socket
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.p2p.workflow")

# Import P2P scheduler components with fallback
try:
    from ipfs_accelerate_py.p2p_workflow_scheduler import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag,
        MerkleClock,
    )
    HAVE_P2P_SCHEDULER = True
except ImportError as e:
    logger.warning(f"P2P workflow scheduler not available: {e}")
    HAVE_P2P_SCHEDULER = False
    P2PWorkflowScheduler = None
    P2PTask = None
    WorkflowTag = None
    MerkleClock = None


# Global scheduler instance
_scheduler_instance: Optional[P2PWorkflowScheduler] = None


def get_scheduler() -> Optional[P2PWorkflowScheduler]:
    """Get or create the global P2P workflow scheduler instance.
    
    Returns:
        P2PWorkflowScheduler instance or None if not available
    """
    global _scheduler_instance
    
    if not HAVE_P2P_SCHEDULER:
        return None
    
    if _scheduler_instance is None:
        # Generate a peer ID (in production, this should come from IPFS)
        peer_id = f"peer-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        _scheduler_instance = P2PWorkflowScheduler(peer_id=peer_id)
        logger.info(f"Created P2P scheduler with peer_id: {peer_id}")
    
    return _scheduler_instance


def reset_scheduler() -> None:
    """Reset the global scheduler instance (mainly for testing)."""
    global _scheduler_instance
    _scheduler_instance = None


__all__ = [
    "HAVE_P2P_SCHEDULER",
    "P2PWorkflowScheduler",
    "P2PTask",
    "WorkflowTag",
    "MerkleClock",
    "get_scheduler",
    "reset_scheduler",
]
