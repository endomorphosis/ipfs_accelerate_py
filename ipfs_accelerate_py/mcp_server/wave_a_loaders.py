"""Wave A category loader bootstrap for unified mcp_server migration."""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable

from .hierarchical_tool_manager import HierarchicalToolManager
from .registration_adapter import LegacyCollectorMCP
from .tools.ipfs import register_native_ipfs_tools
from .tools.p2p import register_native_p2p_tools
from .tools.workflow import register_native_workflow_tools

logger = logging.getLogger(__name__)


def configure_wave_a_loaders(manager: HierarchicalToolManager) -> None:
    """Register initial Wave A loaders into a hierarchical manager."""
    manager.register_category_loader("ipfs", load_ipfs_tools)
    manager.register_category_loader("workflow", load_workflow_tools)
    manager.register_category_loader("p2p", load_p2p_tools)


def load_ipfs_tools(manager: HierarchicalToolManager) -> None:
    """Load IPFS tools, preferring native unified implementations when available."""
    collector = LegacyCollectorMCP()
    try:
        from ipfs_accelerate_py.mcp.tools.ipfs_files import register_files_tools

        register_files_tools(collector)
    except Exception as exc:
        logger.warning("Unable to load legacy IPFS tools: %s", exc)

    if collector.tools:
        _register_collected(manager, collector.tools, category="ipfs")

    # Native tools are registered last so they override legacy-captured tools
    # with the same names during Wave A migration.
    try:
        register_native_ipfs_tools(manager)
    except Exception as exc:
        logger.warning("Unable to load native IPFS tools: %s", exc)


def load_workflow_tools(manager: HierarchicalToolManager) -> None:
    """Load workflow tools, preferring native unified implementations when available."""
    collector = LegacyCollectorMCP()
    try:
        from ipfs_accelerate_py.mcp.tools.workflows import register_tools as register_workflow_tools

        register_workflow_tools(collector)
    except Exception as exc:
        logger.warning("Unable to load workflow tools: %s", exc)
        return

    if collector.tools:
        _register_collected(
            manager,
            collector.tools,
            category="workflow",
            skip_names={"get_workflow_templates", "list_workflows", "get_workflow", "create_workflow", "update_workflow", "delete_workflow", "start_workflow", "pause_workflow", "stop_workflow"},
        )

    # Native tools are registered last so they override legacy-captured tools
    # with the same names during Wave A migration.
    try:
        register_native_workflow_tools(manager)
    except Exception as exc:
        logger.warning("Unable to load native workflow tools: %s", exc)


def load_p2p_tools(manager: HierarchicalToolManager) -> None:
    """Load p2p tools, preferring native unified implementations when available."""
    collector = LegacyCollectorMCP()
    try:
        from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import register_tools as register_p2p_tools

        register_p2p_tools(collector)
    except Exception as exc:
        logger.warning("Unable to load p2p tools: %s", exc)
    
    if collector.tools:
        _register_collected(
            manager,
            collector.tools,
            category="p2p",
            skip_names={"p2p_taskqueue_status", "p2p_taskqueue_submit", "p2p_taskqueue_claim_next", "p2p_taskqueue_call_tool", "p2p_taskqueue_list_tasks", "p2p_taskqueue_get_task", "p2p_taskqueue_wait_task", "p2p_taskqueue_complete_task", "p2p_taskqueue_heartbeat", "p2p_taskqueue_cache_get", "p2p_taskqueue_cache_set", "p2p_taskqueue_submit_docker_hub", "p2p_taskqueue_submit_docker_github", "list_peers"},
        )

    # Native tools are registered last so they override legacy-captured tools
    # with the same names during Wave A migration.
    try:
        register_native_p2p_tools(manager)
    except Exception as exc:
        logger.warning("Unable to load native p2p tools: %s", exc)


def _register_collected(
    manager: HierarchicalToolManager,
    records: Dict[str, object],
    category: str,
    skip_names: Iterable[str] | None = None,
) -> None:
    """Register collected legacy tools into the manager under a fixed category."""
    skip = set(skip_names or ())
    for record in records.values():
        # record is LegacyToolRecord; keep duck typing to avoid strict import cycle.
        tool_name = getattr(record, "name")
        if tool_name in skip:
            continue

        tool_func = getattr(record, "function")
        tool_description = getattr(record, "description", "")
        tool_schema = getattr(record, "input_schema", None)
        execution_context = getattr(record, "execution_context", None)
        tags = getattr(record, "tags", None)

        runtime = None
        if execution_context == "worker":
            runtime = "trio"
        elif execution_context == "server":
            runtime = "fastapi"

        manager.register_tool(
            category=category,
            name=tool_name,
            func=tool_func,
            description=tool_description,
            input_schema=tool_schema,
            runtime=runtime,
            tags=tags,
        )


__all__ = [
    "configure_wave_a_loaders",
    "load_ipfs_tools",
    "load_workflow_tools",
    "load_p2p_tools",
]
