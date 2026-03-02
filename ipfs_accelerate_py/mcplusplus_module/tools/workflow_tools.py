"""P2P workflow MCP tools for MCP++.

This module preserves the historical MCP++ workflow tool names while routing
core scheduler operations through canonical workflow adapters in
``ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools.native_p2p_workflow_tools``.
"""

from __future__ import annotations

from importlib import import_module
import logging
import time
from typing import Any, Dict, List

from ..p2p.workflow import HAVE_P2P_SCHEDULER, MerkleClock, WorkflowTag, get_scheduler

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.tools.workflow")


class _FallbackWorkflowAdapter:
    """Dependency-light fallback for canonical workflow adapter import failures."""

    @staticmethod
    async def get_p2p_scheduler_status() -> Dict[str, Any]:
        return {"success": False, "error": "canonical_p2p_workflow_adapter_unavailable"}

    @staticmethod
    async def schedule_p2p_workflow(**kwargs: Any) -> Dict[str, Any]:
        workflow_id = str(kwargs.get("workflow_id") or "")
        return {
            "success": False,
            "error": "canonical_p2p_workflow_adapter_unavailable",
            "workflow_id": workflow_id,
        }

    @staticmethod
    async def get_next_p2p_workflow() -> Dict[str, Any]:
        return {
            "success": False,
            "error": "canonical_p2p_workflow_adapter_unavailable",
            "workflow": None,
        }


def _resolve_canonical_workflow_adapter() -> Any:
    """Resolve canonical workflow adapter module with resilient fallback."""
    try:
        module = import_module(
            "ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools.native_p2p_workflow_tools"
        )
        return module
    except (ImportError, AttributeError) as exc:
        logger.debug("Falling back to local workflow unavailable adapter: %s", exc)
        return _FallbackWorkflowAdapter()


canonical = _resolve_canonical_workflow_adapter()


def register_p2p_workflow_tools(mcp: Any) -> None:
    """Register P2P workflow scheduler tools with MCP compatibility names."""
    logger.info("Registering P2P workflow scheduler tools")

    @mcp.tool()
    async def p2p_scheduler_status() -> Dict[str, Any]:
        """Get P2P workflow scheduler status via canonical workflow adapter."""
        try:
            status = await canonical.get_p2p_scheduler_status()
            if not isinstance(status, dict):
                status = {"success": False, "error": "invalid_response"}
            status["tool"] = "p2p_scheduler_status"
            status["timestamp"] = time.time()
            return status
        except Exception as e:
            logger.error("Error in p2p_scheduler_status: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_scheduler_status",
                "timestamp": time.time(),
            }

    @mcp.tool()
    async def p2p_submit_task(
        task_id: str,
        workflow_id: str,
        name: str,
        tags: List[str],
        priority: int = 5,
    ) -> Dict[str, Any]:
        """Submit a task by scheduling a canonical P2P workflow entry."""
        try:
            result = await canonical.schedule_p2p_workflow(
                workflow_id=workflow_id,
                name=name,
                tags=tags,
                priority=float(priority),
                metadata={"task_id": task_id},
            )
            success = bool(isinstance(result, dict) and result.get("success", True))
            return {
                "success": success,
                "task_id": task_id,
                "workflow_id": workflow_id,
                "result": result if isinstance(result, dict) else {"raw": result},
                "tool": "p2p_submit_task",
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Error in p2p_submit_task: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_submit_task",
                "timestamp": time.time(),
            }

    @mcp.tool()
    async def p2p_get_next_task() -> Dict[str, Any]:
        """Get next queued P2P workflow from canonical adapter."""
        try:
            result = await canonical.get_next_p2p_workflow()
            if not isinstance(result, dict):
                return {
                    "task": None,
                    "message": "No tasks available for this peer",
                    "tool": "p2p_get_next_task",
                    "timestamp": time.time(),
                }
            workflow = result.get("workflow")
            return {
                "task": workflow,
                "message": result.get("message"),
                "tool": "p2p_get_next_task",
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Error in p2p_get_next_task: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_get_next_task",
                "timestamp": time.time(),
            }

    @mcp.tool()
    def p2p_mark_task_complete(task_id: str) -> Dict[str, Any]:
        """Mark a task as complete in local Trio scheduler when available."""
        try:
            if not HAVE_P2P_SCHEDULER:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_mark_task_complete",
                    "timestamp": time.time(),
                }

            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_mark_task_complete",
                    "timestamp": time.time(),
                }

            success = bool(scheduler.mark_task_complete(task_id))
            return {
                "success": success,
                "task_id": task_id,
                "tool": "p2p_mark_task_complete",
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Error in p2p_mark_task_complete: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_mark_task_complete",
                "timestamp": time.time(),
            }

    @mcp.tool()
    def p2p_check_workflow_tags(tags: List[str]) -> Dict[str, Any]:
        """Check tag routing behavior against local Trio scheduler rules."""
        try:
            if not HAVE_P2P_SCHEDULER:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_check_workflow_tags",
                    "timestamp": time.time(),
                }

            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_check_workflow_tags",
                    "timestamp": time.time(),
                }

            workflow_tags = []
            for tag_str in tags:
                try:
                    workflow_tags.append(WorkflowTag[tag_str.upper().replace("-", "_")])
                except (KeyError, AttributeError):
                    continue

            should_bypass = scheduler.should_bypass_github(workflow_tags)
            is_p2p_only = scheduler.is_p2p_only(workflow_tags)
            return {
                "should_bypass_github": should_bypass,
                "is_p2p_only": is_p2p_only,
                "tags": tags,
                "tool": "p2p_check_workflow_tags",
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Error in p2p_check_workflow_tags: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_check_workflow_tags",
                "timestamp": time.time(),
            }

    @mcp.tool()
    def p2p_update_peer_state(peer_id: str, clock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update local scheduler peer state when Trio scheduler is available."""
        try:
            if not HAVE_P2P_SCHEDULER:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_update_peer_state",
                    "timestamp": time.time(),
                }

            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_update_peer_state",
                    "timestamp": time.time(),
                }

            clock = MerkleClock.from_dict(clock_data)
            scheduler.update_peer_state(peer_id, clock)
            return {
                "success": True,
                "peer_id": peer_id,
                "tool": "p2p_update_peer_state",
                "timestamp": time.time(),
            }
        except Exception as e:
            logger.error("Error in p2p_update_peer_state: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_update_peer_state",
                "timestamp": time.time(),
            }

    @mcp.tool()
    def p2p_get_merkle_clock() -> Dict[str, Any]:
        """Get local scheduler Merkle clock when Trio scheduler is available."""
        try:
            if not HAVE_P2P_SCHEDULER:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_get_merkle_clock",
                    "timestamp": time.time(),
                }

            scheduler = get_scheduler()
            if scheduler is None:
                return {
                    "error": "P2P scheduler not available",
                    "tool": "p2p_get_merkle_clock",
                    "timestamp": time.time(),
                }

            clock_data = scheduler.merkle_clock.to_dict()
            clock_data["tool"] = "p2p_get_merkle_clock"
            clock_data["timestamp"] = time.time()
            return clock_data
        except Exception as e:
            logger.error("Error in p2p_get_merkle_clock: %s", e)
            return {
                "error": str(e),
                "tool": "p2p_get_merkle_clock",
                "timestamp": time.time(),
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

    logger.info("P2P workflow scheduler tools registered successfully (canonical adapters)")


__all__ = ["register_p2p_workflow_tools"]
