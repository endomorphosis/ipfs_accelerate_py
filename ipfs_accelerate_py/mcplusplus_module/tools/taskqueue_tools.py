"""P2P TaskQueue MCP tools for MCP++.

This module preserves the historical MCP++ taskqueue tool registration surface,
while delegating implementation to canonical native P2P tools in
``ipfs_accelerate_py.mcp_server.tools.p2p.native_p2p_tools``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools as canonical

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.tools.taskqueue")


def register_p2p_taskqueue_tools(mcp: Any) -> None:
    """Register P2P TaskQueue tools with canonical delegated behavior."""

    @mcp.tool()
    async def p2p_taskqueue_status(
        remote_multiaddr: str = "",
        peer_id: str = "",
        timeout_s: float = 10.0,
        detail: bool = False,
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_status(
            remote_multiaddr=remote_multiaddr,
            peer_id=peer_id,
            timeout_s=timeout_s,
            detail=detail,
        )

    @mcp.tool()
    async def p2p_taskqueue_submit(
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        remote_multiaddr: str = "",
        peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_submit(
            task_type=task_type,
            model_name=model_name,
            payload=payload,
            remote_multiaddr=remote_multiaddr,
            peer_id=peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_claim_next(
        worker_id: str,
        supported_task_types: Optional[List[str]] = None,
        peer_id: str = "",
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_claim_next(
            worker_id=worker_id,
            supported_task_types=supported_task_types,
            peer_id=peer_id,
            clock=clock,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_call_tool(
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_call_tool(
            tool_name=tool_name,
            args=args,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
            timeout_s=timeout_s,
        )

    @mcp.tool()
    async def p2p_taskqueue_list_tasks(
        status: str = "",
        limit: int = 50,
        task_types: Optional[List[str]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_list_tasks(
            status=status,
            limit=limit,
            task_types=task_types,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_get_task(
        task_id: str,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_get_task(
            task_id=task_id,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_wait_task(
        task_id: str,
        timeout_s: float = 60.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_wait_task(
            task_id=task_id,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_complete_task(
        task_id: str,
        status: str = "completed",
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_complete_task(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_heartbeat(
        peer_id: str,
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_heartbeat(
            peer_id=peer_id,
            clock=clock,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_cache_get(
        key: str,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_cache_get(
            key=key,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_cache_set(
        key: str,
        value: Any,
        ttl_s: Optional[float] = None,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_cache_set(
            key=key,
            value=value,
            ttl_s=ttl_s,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )

    @mcp.tool()
    async def p2p_taskqueue_submit_docker_hub(
        image: str,
        command: Any = None,
        entrypoint: Any = None,
        environment: Optional[Dict[str, Any]] = None,
        volumes: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_submit_docker_hub(
            image=image,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            volumes=volumes,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
            **kwargs,
        )

    @mcp.tool()
    async def p2p_taskqueue_submit_docker_github(
        repo_url: str,
        branch: str = "main",
        dockerfile_path: str = "Dockerfile",
        context_path: str = ".",
        command: Any = None,
        entrypoint: Any = None,
        environment: Optional[Dict[str, Any]] = None,
        build_args: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await canonical.p2p_taskqueue_submit_docker_github(
            repo_url=repo_url,
            branch=branch,
            dockerfile_path=dockerfile_path,
            context_path=context_path,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            build_args=build_args,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
            **kwargs,
        )

    @mcp.tool()
    async def list_peers(
        include_capabilities: bool = False,
        capabilities_timeout_s: float = 5.0,
        capabilities_detail: bool = False,
        discover: bool = True,
        discovery_timeout_s: float = 1.5,
        discovery_methods: Optional[List[str]] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        return await canonical.list_peers(
            include_capabilities=include_capabilities,
            capabilities_timeout_s=capabilities_timeout_s,
            capabilities_detail=capabilities_detail,
            discover=discover,
            discovery_timeout_s=discovery_timeout_s,
            discovery_methods=discovery_methods,
            limit=limit,
        )

    def _set_execution_context(tool_name: str, execution_context: str) -> None:
        tools = getattr(mcp, "tools", None)
        if not isinstance(tools, dict):
            return
        tool_entry = tools.get(tool_name)
        if not isinstance(tool_entry, dict):
            return
        tool_entry["execution_context"] = execution_context

    for _tool_name in [
        "p2p_taskqueue_status",
        "p2p_taskqueue_submit",
        "p2p_taskqueue_claim_next",
        "p2p_taskqueue_call_tool",
        "p2p_taskqueue_list_tasks",
        "p2p_taskqueue_get_task",
        "p2p_taskqueue_wait_task",
        "p2p_taskqueue_complete_task",
        "p2p_taskqueue_heartbeat",
        "p2p_taskqueue_cache_get",
        "p2p_taskqueue_cache_set",
        "p2p_taskqueue_submit_docker_hub",
        "p2p_taskqueue_submit_docker_github",
        "list_peers",
    ]:
        _set_execution_context(_tool_name, "server")

    logger.info("P2P taskqueue tools registered successfully (canonical adapters)")


__all__ = ["register_p2p_taskqueue_tools"]
