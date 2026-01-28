"""
P2P Workflow Scheduler Tools for MCP Server

This module provides MCP tools for P2P workflow scheduling,
allowing workflows to bypass GitHub API and execute across
the peer-to-peer network.
"""

import logging
import os
import time
import uuid
from typing import Dict, List, Any, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.p2p_workflow")

def _is_pytest() -> bool:
    return os.environ.get("PYTEST_CURRENT_TEST") is not None


# Try imports with fallbacks
try:
    if _is_pytest():
        raise ImportError("Using mock MCP under pytest")
    from fastmcp import FastMCP
except ImportError:
    try:
        from mcp.mock_mcp import FastMCP
    except ImportError:
        from ipfs_accelerate_py.mcp.mock_mcp import FastMCP

# Import P2P scheduler components
try:
    from ipfs_accelerate_py.p2p_workflow_scheduler import (
        P2PWorkflowScheduler,
        P2PTask,
        WorkflowTag,
        MerkleClock
    )
    HAVE_P2P_SCHEDULER = True
except ImportError as e:
    logger.warning(f"P2P workflow scheduler not available: {e}")
    HAVE_P2P_SCHEDULER = False

# Global scheduler instance
_scheduler_instance: Optional[P2PWorkflowScheduler] = None


def get_scheduler() -> Optional[P2PWorkflowScheduler]:
    """Get or create the global P2P scheduler instance"""
    global _scheduler_instance
    
    if not HAVE_P2P_SCHEDULER:
        return None
    
    if _scheduler_instance is None:
        # Generate a peer ID (in production, this should come from IPFS)
        import socket
        peer_id = f"peer-{socket.gethostname()}-{uuid.uuid4().hex[:8]}"
        _scheduler_instance = P2PWorkflowScheduler(peer_id=peer_id)
        logger.info(f"Created P2P scheduler with peer_id: {peer_id}")
    
    return _scheduler_instance


def register_p2p_workflow_tools(mcp: FastMCP) -> None:
    """Register P2P workflow scheduler tools with the MCP server."""
    logger.info("Registering P2P workflow scheduler tools")
    
    if not HAVE_P2P_SCHEDULER:
        logger.warning("P2P workflow scheduler not available, skipping registration")
        return
    
    @mcp.tool()
    def p2p_scheduler_status() -> Dict[str, Any]:
        """
        Get P2P workflow scheduler status
        
        Returns:
            Scheduler status including queue size, peer count, and task counts
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_scheduler_status",
                    "timestamp": time.time()
                }
            
            status = scheduler.get_status()
            status["tool"] = "p2p_scheduler_status"
            status["timestamp"] = time.time()
            return status
            
        except Exception as e:
            logger.error(f"Error in p2p_scheduler_status: {e}")
            return {
                "error": str(e),
                "tool": "p2p_scheduler_status",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_submit_task(
        task_id: str,
        workflow_id: str,
        name: str,
        tags: List[str],
        priority: int = 5
    ) -> Dict[str, Any]:
        """
        Submit a task to the P2P workflow scheduler
        
        Args:
            task_id: Unique task identifier
            workflow_id: Workflow this task belongs to
            name: Human-readable task name
            tags: List of tags (e.g., "p2p-only", "code-generation", "web-scraping")
            priority: Task priority (1-10, higher = more important)
        
        Returns:
            Submission status and task details
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_submit_task",
                    "timestamp": time.time()
                }
            
            # Convert string tags to WorkflowTag enums
            workflow_tags = []
            for tag_str in tags:
                try:
                    # Convert kebab-case to enum format
                    enum_name = tag_str.upper().replace('-', '_')
                    workflow_tags.append(WorkflowTag[enum_name])
                except (KeyError, AttributeError):
                    logger.warning(f"Unknown tag: {tag_str}, skipping")
            
            # Create task
            task = P2PTask(
                task_id=task_id,
                workflow_id=workflow_id,
                name=name,
                tags=workflow_tags,
                priority=priority,
                created_at=time.time()
            )
            
            # Submit task
            success = scheduler.submit_task(task)
            
            return {
                "success": success,
                "task_id": task_id,
                "task_hash": task.task_hash,
                "priority": priority,
                "tool": "p2p_submit_task",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in p2p_submit_task: {e}")
            return {
                "error": str(e),
                "tool": "p2p_submit_task",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_get_next_task() -> Dict[str, Any]:
        """
        Get the next task to execute from the P2P scheduler
        
        This uses the merkle clock + hamming distance to determine if
        this peer should handle the task.
        
        Returns:
            Next task details or None if no tasks available
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_get_next_task",
                    "timestamp": time.time()
                }
            
            task = scheduler.get_next_task()
            
            if task is None:
                return {
                    "task": None,
                    "message": "No tasks available for this peer",
                    "tool": "p2p_get_next_task",
                    "timestamp": time.time()
                }
            
            return {
                "task": {
                    "task_id": task.task_id,
                    "workflow_id": task.workflow_id,
                    "name": task.name,
                    "tags": [tag.value for tag in task.tags],
                    "priority": task.priority,
                    "task_hash": task.task_hash,
                    "assigned_peer": task.assigned_peer
                },
                "tool": "p2p_get_next_task",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in p2p_get_next_task: {e}")
            return {
                "error": str(e),
                "tool": "p2p_get_next_task",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_mark_task_complete(task_id: str) -> Dict[str, Any]:
        """
        Mark a task as completed in the P2P scheduler
        
        Args:
            task_id: Task identifier
        
        Returns:
            Completion status
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_mark_task_complete",
                    "timestamp": time.time()
                }
            
            success = scheduler.mark_task_complete(task_id)
            
            return {
                "success": success,
                "task_id": task_id,
                "tool": "p2p_mark_task_complete",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in p2p_mark_task_complete: {e}")
            return {
                "error": str(e),
                "tool": "p2p_mark_task_complete",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_check_workflow_tags(tags: List[str]) -> Dict[str, Any]:
        """
        Check if a workflow should bypass GitHub API based on tags
        
        Args:
            tags: List of workflow tags
        
        Returns:
            Information about whether workflow should use P2P
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_check_workflow_tags",
                    "timestamp": time.time()
                }
            
            # Convert string tags to WorkflowTag enums
            workflow_tags = []
            for tag_str in tags:
                try:
                    enum_name = tag_str.upper().replace('-', '_')
                    workflow_tags.append(WorkflowTag[enum_name])
                except (KeyError, AttributeError):
                    pass
            
            should_bypass = scheduler.should_bypass_github(workflow_tags)
            is_p2p_only = scheduler.is_p2p_only(workflow_tags)
            
            return {
                "should_bypass_github": should_bypass,
                "is_p2p_only": is_p2p_only,
                "tags": tags,
                "tool": "p2p_check_workflow_tags",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in p2p_check_workflow_tags: {e}")
            return {
                "error": str(e),
                "tool": "p2p_check_workflow_tags",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_update_peer_state(peer_id: str, clock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update state information for a peer in the network
        
        Args:
            peer_id: Peer identifier
            clock_data: Merkle clock data from peer
        
        Returns:
            Update status
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_update_peer_state",
                    "timestamp": time.time()
                }
            
            # Reconstruct merkle clock from data
            clock = MerkleClock.from_dict(clock_data)
            
            # Update peer state
            scheduler.update_peer_state(peer_id, clock)
            
            return {
                "success": True,
                "peer_id": peer_id,
                "tool": "p2p_update_peer_state",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in p2p_update_peer_state: {e}")
            return {
                "error": str(e),
                "tool": "p2p_update_peer_state",
                "timestamp": time.time()
            }
    
    @mcp.tool()
    def p2p_get_merkle_clock() -> Dict[str, Any]:
        """
        Get this peer's current merkle clock state
        
        Returns:
            Merkle clock data
        """
        try:
            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_get_merkle_clock",
                    "timestamp": time.time()
                }
            
            clock_data = scheduler.merkle_clock.to_dict()
            clock_data["tool"] = "p2p_get_merkle_clock"
            clock_data["timestamp"] = time.time()
            
            return clock_data
            
        except Exception as e:
            logger.error(f"Error in p2p_get_merkle_clock: {e}")
            return {
                "error": str(e),
                "tool": "p2p_get_merkle_clock",
                "timestamp": time.time()
            }
    
    logger.info("P2P workflow scheduler tools registered successfully")
