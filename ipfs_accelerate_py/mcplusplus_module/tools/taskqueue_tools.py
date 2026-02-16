"""P2P TaskQueue tools for MCP++ (refactored from original MCP module).

These tools provide a thin MCP surface over the libp2p TaskQueue client,
refactored to work natively with Trio without bridging overhead.

Module: ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools

Refactored from: ipfs_accelerate_py/mcp/tools/p2p_taskqueue.py
Key changes:
- Uses mcplusplus_module.trio.run_in_trio instead of inline _run_in_trio
- Consistent error handling and logging
- Full type hints throughout
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..trio import run_in_trio

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.tools.taskqueue")


def _remote_queue(*, peer_id: str = "", multiaddr: str = ""):
    """Create a RemoteQueue instance for P2P TaskQueue operations.
    
    Args:
        peer_id: Optional peer ID for the remote queue
        multiaddr: Optional multiaddress for the remote queue
        
    Returns:
        RemoteQueue instance configured with the provided parameters
    """
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue
    return RemoteQueue(peer_id=str(peer_id or "").strip(), multiaddr=str(multiaddr or "").strip())


def register_p2p_taskqueue_tools(mcp: Any) -> None:
    """Register all P2P TaskQueue tools with the MCP server.
    
    This is the MCP++ refactored version that uses run_in_trio from the trio module,
    providing Trio-native execution for all libp2p operations.
    
    Args:
        mcp: MCP server instance to register tools with
    """

    @mcp.tool()
    async def p2p_taskqueue_status(
        remote_multiaddr: str = "",
        peer_id: str = "",
        timeout_s: float = 10.0,
        detail: bool = False,
    ) -> Dict[str, Any]:
        """Get TaskQueue service status (with discovery if needed).
        
        If `remote_multiaddr` is empty, the client will try discovery in order:
        announce-file -> configured bootstrap endpoints -> rendezvous -> DHT -> mDNS.
        
        Args:
            remote_multiaddr: Optional explicit service multiaddr
            peer_id: Optional service peer ID hint
            timeout_s: Timeout in seconds (default: 10.0)
            detail: Include detailed status information (default: False)
            
        Returns:
            Dict with 'ok' status and service information, or error details
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import request_status

            remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(request_status, remote=remote, timeout_s=float(timeout_s), detail=bool(detail))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_status failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_submit(
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        remote_multiaddr: str = "",
        peer_id: str = "",
    ) -> Dict[str, Any]:
        """Submit a task to the TaskQueue service (with discovery if needed).
        
        Args:
            task_type: Type of task to submit (e.g., 'inference', 'docker.execute')
            model_name: Name of the model to use
            payload: Task-specific payload data
            remote_multiaddr: Optional explicit service multiaddr
            peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and task information including task_id
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_task_with_info

            remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
            info = await run_in_trio(
                submit_task_with_info,
                remote=remote,
                task_type=str(task_type),
                model_name=str(model_name),
                payload=(payload if isinstance(payload, dict) else {}),
            )
            out: Dict[str, Any] = {"ok": True}
            if isinstance(info, dict):
                out.update(info)
            return out
        except Exception as exc:
            logger.exception("p2p_taskqueue_submit failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_claim_next(
        worker_id: str,
        supported_task_types: Optional[List[str]] = None,
        peer_id: str = "",
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Claim the next available task from the TaskQueue service.
        
        Args:
            worker_id: Worker identifier (string)
            supported_task_types: Optional allowlist of task types this worker supports
            peer_id: Optional peer ID to advertise in the claim (for deterministic scheduling)
            clock: Optional Merkle clock dict for deterministic scheduling
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and claimed task details, or None if no tasks available
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import claim_next

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = await run_in_trio(
                claim_next,
                remote=remote,
                worker_id=str(worker_id),
                supported_task_types=[str(x) for x in (supported_task_types or []) if str(x).strip()],
                peer_id=str(peer_id) if peer_id else None,
                clock=(clock if isinstance(clock, dict) else None),
            )
            return {"ok": True, "task": task}
        except Exception as exc:
            logger.exception("p2p_taskqueue_claim_next failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_call_tool(
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Call a tool on the remote TaskQueue P2P service (op=call_tool).
        
        The remote service must have IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1.
        
        Args:
            tool_name: Name of the tool to call on the remote service
            args: Optional arguments to pass to the tool
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            timeout_s: Timeout in seconds (default: 30.0)
            
        Returns:
            Dict with 'ok' status and tool execution results
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import call_tool

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(
                call_tool,
                remote=remote,
                tool_name=str(tool_name),
                args=(args if isinstance(args, dict) else {}),
                timeout_s=float(timeout_s),
            )
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_call_tool failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_list_tasks(
        status: str = "",
        limit: int = 50,
        task_types: Optional[List[str]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """List tasks from the remote TaskQueue service.
        
        Args:
            status: Optional filter by task status (e.g., 'pending', 'claimed', 'completed')
            limit: Maximum number of tasks to return (default: 50)
            task_types: Optional filter by task types
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and list of tasks
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import list_tasks

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            parsed_status = str(status).strip().lower() if status else None
            parsed_task_types = None
            if task_types is not None:
                parsed_task_types = [str(x) for x in (task_types or []) if str(x).strip()]
            resp = await run_in_trio(
                list_tasks,
                remote=remote,
                status=parsed_status,
                limit=int(limit),
                task_types=parsed_task_types,
            )
            return {"ok": True, "tasks": resp}
        except Exception as exc:
            logger.exception("p2p_taskqueue_list_tasks failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_get_task(
        task_id: str,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Get a single task from the remote TaskQueue service.
        
        Args:
            task_id: Unique identifier of the task to retrieve
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and task details
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import get_task

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = await run_in_trio(
                get_task,
                remote=remote,
                task_id=str(task_id),
            )
            return {"ok": True, "task": task}
        except Exception as exc:
            logger.exception("p2p_taskqueue_get_task failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_wait_task(
        task_id: str,
        timeout_s: float = 60.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Wait for a task to finish on the remote TaskQueue service.
        
        Args:
            task_id: Unique identifier of the task to wait for
            timeout_s: Timeout in seconds (default: 60.0)
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and completed task details
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import wait_task

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = await run_in_trio(
                wait_task,
                remote=remote,
                task_id=str(task_id),
                timeout_s=float(timeout_s),
            )
            return {"ok": True, "task": task}
        except Exception as exc:
            logger.exception("p2p_taskqueue_wait_task failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_complete_task(
        task_id: str,
        status: str = "completed",
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Mark a task completed or failed on the remote TaskQueue service.
        
        Args:
            task_id: Unique identifier of the task to complete
            status: Task completion status (default: 'completed', or 'failed')
            result: Optional task result data
            error: Optional error message if status is 'failed'
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status confirming task completion
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import complete_task

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(
                complete_task,
                remote=remote,
                task_id=str(task_id),
                status=str(status),
                result=(result if isinstance(result, dict) else None),
                error=str(error) if error else None,
            )
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_complete_task failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_heartbeat(
        peer_id: str,
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Send a peer heartbeat to the remote TaskQueue service.
        
        Args:
            peer_id: Peer ID to advertise in the heartbeat
            clock: Optional Merkle clock dict for peer synchronization
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and heartbeat acknowledgment
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import heartbeat

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(
                heartbeat,
                remote=remote,
                peer_id=str(peer_id),
                clock=(clock if isinstance(clock, dict) else None),
            )
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_heartbeat failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_cache_get(
        key: str,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Read from the TaskQueue service's shared cache.
        
        Args:
            key: Cache key to retrieve
            timeout_s: Timeout in seconds (default: 10.0)
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status and cached value, or error if key not found
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_get

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(cache_get, remote=remote, key=str(key), timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_cache_get failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    async def p2p_taskqueue_cache_set(
        key: str,
        value: Any,
        ttl_s: Optional[float] = None,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Write to the TaskQueue service's shared cache.
        
        Args:
            key: Cache key to set
            value: Value to cache (will be JSON-serialized)
            ttl_s: Optional time-to-live in seconds
            timeout_s: Timeout in seconds (default: 10.0)
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            
        Returns:
            Dict with 'ok' status confirming cache write
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_set

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = await run_in_trio(
                cache_set,
                remote=remote,
                key=str(key),
                value=value,
                ttl_s=ttl_s,
                timeout_s=float(timeout_s),
            )
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_cache_set failed")
            return {"ok": False, "error": str(exc)}

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
        """Convenience wrapper: submit docker.execute task for a Docker Hub image.
        
        Args:
            image: Docker Hub image name (e.g., 'python:3.11')
            command: Optional command to run in the container
            entrypoint: Optional container entrypoint override
            environment: Optional environment variables dict
            volumes: Optional volume mounts dict
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            **kwargs: Additional task-specific arguments
            
        Returns:
            Dict with 'ok' status and task_id
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_docker_hub_task

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task_id = await run_in_trio(
                submit_docker_hub_task,
                remote=remote,
                image=str(image),
                command=command,
                entrypoint=entrypoint,
                environment=(environment if isinstance(environment, dict) else None),
                volumes=(volumes if isinstance(volumes, dict) else None),
                **kwargs,
            )
            return {"ok": True, "task_id": task_id}
        except Exception as exc:
            logger.exception("p2p_taskqueue_submit_docker_hub failed")
            return {"ok": False, "error": str(exc)}

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
        """Convenience wrapper: submit docker.github task for a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            branch: Git branch to use (default: 'main')
            dockerfile_path: Path to Dockerfile in repo (default: 'Dockerfile')
            context_path: Docker build context path (default: '.')
            command: Optional command to run in the container
            entrypoint: Optional container entrypoint override
            environment: Optional environment variables dict
            build_args: Optional Docker build arguments
            remote_multiaddr: Optional explicit service multiaddr
            remote_peer_id: Optional service peer ID hint
            **kwargs: Additional task-specific arguments
            
        Returns:
            Dict with 'ok' status and task_id
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_docker_github_task

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task_id = await run_in_trio(
                submit_docker_github_task,
                remote=remote,
                repo_url=str(repo_url),
                branch=str(branch),
                dockerfile_path=str(dockerfile_path),
                context_path=str(context_path),
                command=command,
                entrypoint=entrypoint,
                environment=(environment if isinstance(environment, dict) else None),
                build_args=(build_args if isinstance(build_args, dict) else None),
                **kwargs,
            )
            return {"ok": True, "task_id": task_id}
        except Exception as exc:
            logger.exception("p2p_taskqueue_submit_docker_github failed")
            return {"ok": False, "error": str(exc)}

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
        """List peers currently seen by this node's TaskQueue libp2p service.
        
        By default, returns peers that have recently contacted this server
        (heartbeat/claim/etc), within the peer timeout window.
        
        If `discover` is true and the seen peer list is empty, will try to
        discover peers (mDNS/rendezvous/DHT) and include them with discovered=true.
        
        If `include_capabilities` is true, will query each peer's TaskQueue status
        (capabilities) using discovery filtered by peer_id.
        
        Args:
            include_capabilities: Query peer capabilities (default: False)
            capabilities_timeout_s: Timeout for capability queries (default: 5.0)
            capabilities_detail: Include detailed capability info (default: False)
            discover: Enable peer discovery if no peers seen (default: True)
            discovery_timeout_s: Timeout for discovery operations (default: 1.5)
            discovery_methods: Discovery methods to try (default: ['mdns', 'rendezvous', 'dht'])
            limit: Maximum number of peers to return (default: 50)
            
        Returns:
            Dict with 'ok' status, self_peer_id, and list of peers with their info
        """
        try:
            from ipfs_accelerate_py.p2p_tasks.service import get_local_service_state, list_known_peers
            from ipfs_accelerate_py.p2p_tasks.client import (
                RemoteQueue,
                discover_peers_via_dht,
                discover_peers_via_mdns,
                discover_peers_via_rendezvous,
                get_capabilities,
            )

            state = get_local_service_state()
            if not bool(state.get("running")):
                return {"ok": False, "error": "p2p_service_not_running"}

            self_peer_id = str(state.get("peer_id") or "").strip()
            peers = list_known_peers(alive_only=True, limit=int(limit), exclude_peer_id=self_peer_id)

            if not peers and bool(discover):
                raw_methods = discovery_methods or ["mdns", "rendezvous", "dht"]
                methods = [str(x).strip().lower() for x in raw_methods if str(x).strip()]
                timeout_s = max(0.1, float(discovery_timeout_s))
                discovered: list[Dict[str, Any]] = []

                async def _add_found(found: list[RemoteQueue], method: str) -> None:
                    for rq in found or []:
                        try:
                            pid = str(getattr(rq, "peer_id", "") or "").strip()
                            ma = str(getattr(rq, "multiaddr", "") or "").strip()
                        except Exception:
                            pid, ma = "", ""
                        if pid and pid == self_peer_id:
                            continue
                        if not pid and not ma:
                            continue
                        discovered.append({"peer_id": pid, "multiaddr": ma, "discovered": True, "method": method})

                for method in methods:
                    try:
                        if method == "mdns":
                            await _add_found(
                                await run_in_trio(
                                    discover_peers_via_mdns,
                                    timeout_s=timeout_s,
                                    limit=int(limit),
                                    exclude_self=True,
                                ),
                                "mdns",
                            )
                        elif method == "rendezvous":
                            await _add_found(
                                await run_in_trio(
                                    discover_peers_via_rendezvous,
                                    timeout_s=timeout_s,
                                    limit=int(limit),
                                    exclude_self=True,
                                ),
                                "rendezvous",
                            )
                        elif method == "dht":
                            await _add_found(
                                await run_in_trio(
                                    discover_peers_via_dht,
                                    timeout_s=timeout_s,
                                    limit=int(limit),
                                    exclude_self=True,
                                ),
                                "dht",
                            )
                    except Exception:
                        # Best-effort; discovery can fail depending on environment
                        continue

                # Deduplicate by peer_id or multiaddr
                seen_keys: set[str] = set()
                deduped: list[Dict[str, Any]] = []
                for row in discovered:
                    pid = str(row.get("peer_id") or "").strip()
                    ma = str(row.get("multiaddr") or "").strip()
                    key = pid or ma
                    if not key or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    deduped.append(row)
                peers = deduped

            if include_capabilities:
                for row in peers:
                    pid = str(row.get("peer_id") or "").strip()
                    if not pid:
                        continue
                    try:
                        caps = await run_in_trio(
                            get_capabilities,
                            remote=RemoteQueue(peer_id=pid, multiaddr=str(row.get("multiaddr") or "")),
                            timeout_s=float(capabilities_timeout_s),
                            detail=bool(capabilities_detail),
                        )
                        row["capabilities"] = caps if isinstance(caps, dict) else {}
                    except Exception as exc:
                        row["capabilities_error"] = str(exc)

            return {
                "ok": True,
                "self_peer_id": self_peer_id,
                "listen_port": state.get("listen_port"),
                "started_at": state.get("started_at"),
                "count": len(peers),
                "peers": peers,
            }
        except Exception as exc:
            logger.exception("list_peers failed")
            return {"ok": False, "error": str(exc)}

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


__all__ = ["register_p2p_taskqueue_tools"]
