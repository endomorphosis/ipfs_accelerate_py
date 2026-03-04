"""Native p2p tool implementations for unified mcp_server."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _error_result(message: str) -> Dict[str, Any]:
    return {"ok": False, "error": message}


async def _request_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Any:
    """Request TaskQueue status using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import request_status

    remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        request_status,
        remote=remote,
        timeout_s=float(timeout_s),
        detail=bool(detail),
    )


async def p2p_taskqueue_status(
    remote_multiaddr: str = "",
    peer_id: str = "",
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Dict[str, Any]:
    """Get TaskQueue service status from native unified p2p tool path."""
    if not isinstance(timeout_s, (int, float)) or float(timeout_s) <= 0:
        return _error_result("timeout_s must be a number > 0")
    if not isinstance(detail, bool):
        return _error_result("detail must be a boolean")

    try:
        resp = await _request_status(
            remote_multiaddr=remote_multiaddr,
            peer_id=peer_id,
            timeout_s=timeout_s,
            detail=detail,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _submit_task_with_info(
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
    remote_multiaddr: str = "",
    peer_id: str = "",
) -> Any:
    """Submit a TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import submit_task_with_info

    remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        submit_task_with_info,
        remote=remote,
        task_type=str(task_type),
        model_name=str(model_name),
        payload=(payload if isinstance(payload, dict) else {}),
    )


async def p2p_taskqueue_submit(
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
    remote_multiaddr: str = "",
    peer_id: str = "",
) -> Dict[str, Any]:
    """Submit a task from native unified p2p tool path."""
    if not isinstance(task_type, str) or not task_type.strip():
        return _error_result("task_type must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        return _error_result("model_name must be a non-empty string")
    if not isinstance(payload, dict):
        return _error_result("payload must be an object")

    try:
        info = await _submit_task_with_info(
            task_type=task_type,
            model_name=model_name,
            payload=payload,
            remote_multiaddr=remote_multiaddr,
            peer_id=peer_id,
        )
        out: Dict[str, Any] = {"ok": True}
        if isinstance(info, dict):
            out.update(info)
        return out
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _claim_next(
    worker_id: str,
    supported_task_types: Optional[List[str]] = None,
    peer_id: str = "",
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Claim next task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import claim_next

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        claim_next,
        remote=remote,
        worker_id=str(worker_id),
        supported_task_types=[str(x) for x in (supported_task_types or []) if str(x).strip()],
        peer_id=str(peer_id) if peer_id else None,
        clock=(clock if isinstance(clock, dict) else None),
    )


async def p2p_taskqueue_claim_next(
    worker_id: str,
    supported_task_types: Optional[List[str]] = None,
    peer_id: str = "",
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Claim the next available task from native unified p2p tool path."""
    try:
        task = await _claim_next(
            worker_id=worker_id,
            supported_task_types=supported_task_types,
            peer_id=peer_id,
            clock=clock,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "task": task}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _call_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 30.0,
) -> Any:
    """Call remote p2p tool using legacy client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import call_tool

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        call_tool,
        remote=remote,
        tool_name=str(tool_name),
        args=(args if isinstance(args, dict) else {}),
        timeout_s=float(timeout_s),
    )


async def p2p_taskqueue_call_tool(
    tool_name: str,
    args: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """Call remote p2p tool from native unified tool path."""
    try:
        resp = await _call_tool(
            tool_name=tool_name,
            args=args,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
            timeout_s=timeout_s,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _list_tasks(
    status: str = "",
    limit: int = 50,
    task_types: Optional[List[str]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """List TaskQueue tasks using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import list_tasks

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    parsed_status = str(status).strip().lower() if status else None
    parsed_task_types = None
    if task_types is not None:
        parsed_task_types = [str(x) for x in (task_types or []) if str(x).strip()]

    return await _run_in_trio(
        list_tasks,
        remote=remote,
        status=parsed_status,
        limit=int(limit),
        task_types=parsed_task_types,
    )


async def p2p_taskqueue_list_tasks(
    status: str = "",
    limit: int = 50,
    task_types: Optional[List[str]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """List TaskQueue tasks from native unified p2p tool path."""
    try:
        tasks = await _list_tasks(
            status=status,
            limit=limit,
            task_types=task_types,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "tasks": tasks}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _get_task(
    task_id: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Get a single TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import get_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        get_task,
        remote=remote,
        task_id=str(task_id),
    )


async def p2p_taskqueue_get_task(
    task_id: str,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Get a single TaskQueue task from native unified p2p tool path."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result("task_id must be a non-empty string")

    try:
        task = await _get_task(
            task_id=task_id,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "task": task}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _wait_task(
    task_id: str,
    timeout_s: float = 60.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Wait for a TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import wait_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        wait_task,
        remote=remote,
        task_id=str(task_id),
        timeout_s=float(timeout_s),
    )


async def p2p_taskqueue_wait_task(
    task_id: str,
    timeout_s: float = 60.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Wait for a TaskQueue task from native unified p2p tool path."""
    if not isinstance(task_id, str) or not task_id.strip():
        return _error_result("task_id must be a non-empty string")
    if not isinstance(timeout_s, (int, float)) or float(timeout_s) <= 0:
        return _error_result("timeout_s must be a number > 0")

    try:
        task = await _wait_task(
            task_id=task_id,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return {"ok": True, "task": task}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _complete_task(
    task_id: str,
    status: str = "completed",
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Complete a TaskQueue task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import complete_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        complete_task,
        remote=remote,
        task_id=str(task_id),
        status=str(status),
        result=(result if isinstance(result, dict) else None),
        error=str(error) if error else None,
    )


async def p2p_taskqueue_complete_task(
    task_id: str,
    status: str = "completed",
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Complete a TaskQueue task from native unified p2p tool path."""
    try:
        resp = await _complete_task(
            task_id=task_id,
            status=status,
            result=result,
            error=error,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _heartbeat(
    peer_id: str,
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Send heartbeat using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import heartbeat

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        heartbeat,
        remote=remote,
        peer_id=str(peer_id),
        clock=(clock if isinstance(clock, dict) else None),
    )


async def p2p_taskqueue_heartbeat(
    peer_id: str,
    clock: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Send heartbeat from native unified p2p tool path."""
    try:
        resp = await _heartbeat(
            peer_id=peer_id,
            clock=clock,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _cache_get(
    key: str,
    timeout_s: float = 10.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Read shared cache using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import cache_get

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        cache_get,
        remote=remote,
        key=str(key),
        timeout_s=float(timeout_s),
    )


async def p2p_taskqueue_cache_get(
    key: str,
    timeout_s: float = 10.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Read shared cache from native unified p2p tool path."""
    try:
        resp = await _cache_get(
            key=key,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _cache_set(
    key: str,
    value: Any,
    ttl_s: Optional[float] = None,
    timeout_s: float = 10.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Any:
    """Write shared cache using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import cache_set

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        cache_set,
        remote=remote,
        key=str(key),
        value=value,
        ttl_s=ttl_s,
        timeout_s=float(timeout_s),
    )


async def p2p_taskqueue_cache_set(
    key: str,
    value: Any,
    ttl_s: Optional[float] = None,
    timeout_s: float = 10.0,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
) -> Dict[str, Any]:
    """Write shared cache from native unified p2p tool path."""
    try:
        resp = await _cache_set(
            key=key,
            value=value,
            ttl_s=ttl_s,
            timeout_s=timeout_s,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
        )
        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _submit_docker_hub(
    image: str,
    command: Any = None,
    entrypoint: Any = None,
    environment: Optional[Dict[str, Any]] = None,
    volumes: Optional[Dict[str, Any]] = None,
    remote_multiaddr: str = "",
    remote_peer_id: str = "",
    **kwargs: Any,
) -> Any:
    """Submit Docker Hub task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import submit_docker_hub_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
        submit_docker_hub_task,
        remote=remote,
        image=str(image),
        command=command,
        entrypoint=entrypoint,
        environment=(environment if isinstance(environment, dict) else None),
        volumes=(volumes if isinstance(volumes, dict) else None),
        **kwargs,
    )


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
    """Submit Docker Hub task from native unified p2p tool path."""
    try:
        task_id = await _submit_docker_hub(
            image=image,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            volumes=volumes,
            remote_multiaddr=remote_multiaddr,
            remote_peer_id=remote_peer_id,
            **kwargs,
        )
        return {"ok": True, "task_id": task_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _submit_docker_github(
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
) -> Any:
    """Submit Docker GitHub task using legacy p2p client helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _remote_queue, _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import submit_docker_github_task

    remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
    return await _run_in_trio(
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
    """Submit Docker GitHub task from native unified p2p tool path."""
    try:
        task_id = await _submit_docker_github(
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
        return {"ok": True, "task_id": task_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _list_peers(
    include_capabilities: bool = False,
    capabilities_timeout_s: float = 5.0,
    capabilities_detail: bool = False,
    discover: bool = True,
    discovery_timeout_s: float = 1.5,
    discovery_methods: Optional[List[str]] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List known/discovered peers using legacy p2p helpers lazily."""
    from ipfs_accelerate_py.mcp.tools.p2p_taskqueue import _run_in_trio
    from ipfs_accelerate_py.p2p_tasks.client import (
        RemoteQueue,
        discover_peers_via_dht,
        discover_peers_via_mdns,
        discover_peers_via_rendezvous,
        get_capabilities,
    )
    from ipfs_accelerate_py.p2p_tasks.service import get_local_service_state, list_known_peers

    state = get_local_service_state()
    if not bool(state.get("running")):
        return {"ok": False, "error": "p2p_service_not_running"}

    self_peer_id = str(state.get("peer_id") or "").strip()
    peers = list_known_peers(alive_only=True, limit=int(limit), exclude_peer_id=self_peer_id)

    if not peers and bool(discover):
        raw_methods = discovery_methods or ["mdns", "rendezvous", "dht"]
        methods = [str(x).strip().lower() for x in raw_methods if str(x).strip()]
        timeout_s = max(0.1, float(discovery_timeout_s))
        discovered: List[Dict[str, Any]] = []

        async def _add_found(found: List[RemoteQueue], method: str) -> None:
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
                        await _run_in_trio(
                            discover_peers_via_mdns,
                            timeout_s=timeout_s,
                            limit=int(limit),
                            exclude_self=True,
                        ),
                        "mdns",
                    )
                elif method == "rendezvous":
                    await _add_found(
                        await _run_in_trio(
                            discover_peers_via_rendezvous,
                            timeout_s=timeout_s,
                            limit=int(limit),
                            exclude_self=True,
                        ),
                        "rendezvous",
                    )
                elif method == "dht":
                    await _add_found(
                        await _run_in_trio(
                            discover_peers_via_dht,
                            timeout_s=timeout_s,
                            limit=int(limit),
                            exclude_self=True,
                        ),
                        "dht",
                    )
            except Exception:
                # Discovery is best-effort in mixed local/CI environments.
                continue

        seen_keys: set[str] = set()
        deduped: List[Dict[str, Any]] = []
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
                caps = await _run_in_trio(
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


async def list_peers(
    include_capabilities: bool = False,
    capabilities_timeout_s: float = 5.0,
    capabilities_detail: bool = False,
    discover: bool = True,
    discovery_timeout_s: float = 1.5,
    discovery_methods: Optional[List[str]] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List peers from native unified p2p tool path."""
    try:
        return await _list_peers(
            include_capabilities=include_capabilities,
            capabilities_timeout_s=capabilities_timeout_s,
            capabilities_detail=capabilities_detail,
            discover=discover,
            discovery_timeout_s=discovery_timeout_s,
            discovery_methods=discovery_methods,
            limit=limit,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def register_native_p2p_tools(manager: Any) -> None:
    """Register native p2p tools in the unified hierarchical manager."""
    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_status",
        func=p2p_taskqueue_status,
        description="Get p2p TaskQueue service status using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "remote_multiaddr": {"type": "string", "default": ""},
                "peer_id": {"type": "string", "default": ""},
                "timeout_s": {"type": "number", "default": 10.0, "minimum": 0.000001},
                "detail": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_submit",
        func=p2p_taskqueue_submit,
        description="Submit p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "minLength": 1},
                "model_name": {"type": "string", "minLength": 1},
                "payload": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_type", "model_name", "payload"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_claim_next",
        func=p2p_taskqueue_claim_next,
        description="Claim next p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "worker_id": {"type": "string"},
                "supported_task_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "peer_id": {"type": "string", "default": ""},
                "clock": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["worker_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_call_tool",
        func=p2p_taskqueue_call_tool,
        description="Call remote p2p TaskQueue tool using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "args": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
                "timeout_s": {"type": "number", "default": 30.0},
            },
            "required": ["tool_name"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_list_tasks",
        func=p2p_taskqueue_list_tasks,
        description="List p2p TaskQueue tasks using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "default": ""},
                "limit": {"type": "integer", "default": 50},
                "task_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": [],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_get_task",
        func=p2p_taskqueue_get_task,
        description="Get p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_wait_task",
        func=p2p_taskqueue_wait_task,
        description="Wait for p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "minLength": 1},
                "timeout_s": {"type": "number", "default": 60.0, "minimum": 0.000001},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_complete_task",
        func=p2p_taskqueue_complete_task,
        description="Complete p2p TaskQueue task using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {"type": "string", "default": "completed"},
                "result": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "error": {"type": "string", "default": ""},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["task_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_heartbeat",
        func=p2p_taskqueue_heartbeat,
        description="Send p2p TaskQueue heartbeat using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "peer_id": {"type": "string"},
                "clock": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["peer_id"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_cache_get",
        func=p2p_taskqueue_cache_get,
        description="Read p2p TaskQueue shared cache using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "timeout_s": {"type": "number", "default": 10.0},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["key"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_cache_set",
        func=p2p_taskqueue_cache_set,
        description="Write p2p TaskQueue shared cache using unified native implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {},
                "ttl_s": {"type": ["number", "null"], "default": None},
                "timeout_s": {"type": "number", "default": 10.0},
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["key", "value"],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_submit_docker_hub",
        func=p2p_taskqueue_submit_docker_hub,
        description="Submit Docker Hub task using unified native p2p implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "image": {"type": "string"},
                "command": {},
                "entrypoint": {},
                "environment": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "volumes": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["image"],
            "additionalProperties": True,
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="p2p_taskqueue_submit_docker_github",
        func=p2p_taskqueue_submit_docker_github,
        description="Submit Docker GitHub task using unified native p2p implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "repo_url": {"type": "string"},
                "branch": {"type": "string", "default": "main"},
                "dockerfile_path": {"type": "string", "default": "Dockerfile"},
                "context_path": {"type": "string", "default": "."},
                "command": {},
                "entrypoint": {},
                "environment": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "build_args": {
                    "type": ["object", "null"],
                    "additionalProperties": True,
                },
                "remote_multiaddr": {"type": "string", "default": ""},
                "remote_peer_id": {"type": "string", "default": ""},
            },
            "required": ["repo_url"],
            "additionalProperties": True,
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )

    manager.register_tool(
        category="p2p",
        name="list_peers",
        func=list_peers,
        description="List known/discovered peers using unified native p2p implementation.",
        input_schema={
            "type": "object",
            "properties": {
                "include_capabilities": {"type": "boolean", "default": False},
                "capabilities_timeout_s": {"type": "number", "default": 5.0},
                "capabilities_detail": {"type": "boolean", "default": False},
                "discover": {"type": "boolean", "default": True},
                "discovery_timeout_s": {"type": "number", "default": 1.5},
                "discovery_methods": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                },
                "limit": {"type": "integer", "default": 50},
            },
            "required": [],
        },
        runtime="trio",
        tags=["native", "wave-a", "p2p"],
    )
