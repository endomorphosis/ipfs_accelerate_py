"""
P2P Workflow Scheduler Tools for MCP++ (refactored from original MCP module).

This module provides MCP tools for P2P workflow scheduling, allowing workflows
to bypass GitHub API and execute across the peer-to-peer network.

Module: ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools

Refactored from: ipfs_accelerate_py/mcp/tools/p2p_workflow_tools.py
Key changes:
- Uses mcplusplus_module.p2p.workflow for scheduler access
- Consistent error handling and response format
- Full type hints throughout
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from ..p2p.workflow import get_scheduler, HAVE_P2P_SCHEDULER, WorkflowTag, P2PTask, MerkleClock

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.tools.workflow")


def register_p2p_workflow_tools(mcp: Any) -> None:
    """Register P2P workflow scheduler tools with the MCP server.
    
    This is the MCP++ refactored version that uses the Trio-native workflow scheduler.
    
    Args:
        mcp: MCP server instance to register tools with
    """
    logger.info("Registering P2P workflow scheduler tools")
    
    if not HAVE_P2P_SCHEDULER:
        logger.warning("P2P workflow scheduler not available, skipping registration")
        return

    @mcp.tool()
    def p2p_scheduler_status() -> Dict[str, Any]:
        """Get P2P workflow scheduler status.
        
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
        """Submit a task to the P2P workflow scheduler.
        
        Args:
            task_id: Unique task identifier
            workflow_id: Workflow this task belongs to
            name: Human-readable task name
            tags: List of tags (e.g., "p2p-only", "code-generation", "web-scraping")
            priority: Task priority (1-10, higher = more important, default: 5)
        
        Returns:
            Submission status and task details including task_hash
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
                    # Convert kebab-case to enum format (e.g., "p2p-only" -> "P2P_ONLY")
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
        """Get the next task to execute from the P2P scheduler.
        
        Uses the merkle clock + hamming distance to determine if this peer
        should handle the task.
        
        Returns:
            Next task details or None if no tasks available for this peer
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
        """Mark a task as completed in the P2P scheduler.
        
        Args:
            task_id: Task identifier to mark as complete
        
        Returns:
            Completion status with success flag
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
        """Check if a workflow should bypass GitHub API based on tags.
        
        Args:
            tags: List of workflow tags to check
        
        Returns:
            Information about whether workflow should use P2P, including
            should_bypass_github and is_p2p_only flags
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
        """Update state information for a peer in the network.
        
        Args:
            peer_id: Peer identifier
            clock_data: Merkle clock data from peer as a dictionary
        
        Returns:
            Update status with success flag
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
        """Get this peer's current merkle clock state.
        
        Returns:
            Merkle clock data as a dictionary with clock state and metadata
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

    def _set_execution_context(tool_name: str, execution_context: str) -> None:
        tools = getattr(mcp, "tools", None)
        if not isinstance(tools, dict):
            return
        tool_entry = tools.get(tool_name)
        if not isinstance(tool_entry, dict):
            return
        tool_entry["execution_context"] = execution_context

    for _tool_name in [
        "p2p_scheduler_status",
        "p2p_submit_task",
        "p2p_get_next_task",
        "p2p_mark_task_complete",
        "p2p_check_workflow_tags",
        "p2p_update_peer_state",
        "p2p_get_merkle_clock",
    ]:
        _set_execution_context(_tool_name, "server")
    
    logger.info("P2P workflow scheduler tools registered successfully")


__all__ = ["register_p2p_workflow_tools"]
