"""P2P TaskQueue tools for the IPFS Accelerate MCP server.

These tools provide a thin MCP surface over the libp2p TaskQueue client.
They intentionally return structured `{ok, error, ...}` dicts rather than
raising, so MCP callers can handle partial connectivity/discovery.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ipfs_accelerate_mcp.tools.p2p_taskqueue")


def _remote_queue(*, peer_id: str = "", multiaddr: str = ""):
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue

    return RemoteQueue(peer_id=str(peer_id or "").strip(), multiaddr=str(multiaddr or "").strip())


def register_tools(mcp: Any) -> None:
    """Register p2p TaskQueue tools with the MCP server."""

    @mcp.tool()
    def p2p_taskqueue_status(
        remote_multiaddr: str = "",
        peer_id: str = "",
        timeout_s: float = 10.0,
        detail: bool = False,
    ) -> Dict[str, Any]:
        """Get TaskQueue service status (with discovery if needed).

        If `remote_multiaddr` is empty, the client will try:
        announce-file -> configured bootstrap endpoints (explicit only) -> rendezvous -> DHT -> mDNS.
        """

        try:
            from ipfs_accelerate_py.p2p_tasks.client import request_status_sync

            remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
            resp = request_status_sync(remote=remote, timeout_s=float(timeout_s), detail=bool(detail))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_status failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_submit(
        task_type: str,
        model_name: str,
        payload: Dict[str, Any],
        remote_multiaddr: str = "",
        peer_id: str = "",
    ) -> Dict[str, Any]:
        """Submit a task to the TaskQueue service (with discovery if needed)."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_task_with_info_sync

            remote = _remote_queue(peer_id=peer_id, multiaddr=remote_multiaddr)
            info = submit_task_with_info_sync(
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
    def p2p_taskqueue_claim_next(
        worker_id: str,
        supported_task_types: Optional[List[str]] = None,
        peer_id: str = "",
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Claim the next available task from the TaskQueue service.

        Args:
            worker_id: Worker identifier (string).
            supported_task_types: Optional allowlist of task types.
            peer_id: Optional peer ID to advertise in the claim call (used by deterministic scheduling).
            clock: Optional Merkle clock dict to help deterministic scheduling.
            remote_multiaddr: Optional explicit service multiaddr.
            remote_peer_id: Optional service peer id hint.
        """

        try:
            from ipfs_accelerate_py.p2p_tasks.client import claim_next_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = claim_next_sync(
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
    def p2p_taskqueue_call_tool(
        tool_name: str,
        args: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        timeout_s: float = 30.0,
    ) -> Dict[str, Any]:
        """Call a tool on the remote TaskQueue p2p service (op=call_tool).

        The remote service must have `IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS=1`.
        """

        try:
            from ipfs_accelerate_py.p2p_tasks.client import call_tool_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = call_tool_sync(remote=remote, tool_name=str(tool_name), args=(args if isinstance(args, dict) else {}), timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_call_tool failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_list_tasks(
        status: str = "",
        limit: int = 50,
        task_types: Optional[List[str]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """List tasks from the remote TaskQueue service."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import list_tasks_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = list_tasks_sync(
                remote=remote,
                status=str(status).strip().lower() if status else None,
                limit=int(limit),
                task_types=[str(x) for x in (task_types or []) if str(x).strip()] if task_types is not None else None,
            )
            return {"ok": True, "tasks": resp}
        except Exception as exc:
            logger.exception("p2p_taskqueue_list_tasks failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_get_task(
        task_id: str,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Get a single task from the remote TaskQueue service."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import get_task_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = get_task_sync(remote=remote, task_id=str(task_id))
            return {"ok": True, "task": task}
        except Exception as exc:
            logger.exception("p2p_taskqueue_get_task failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_wait_task(
        task_id: str,
        timeout_s: float = 60.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Wait for a task to finish on the remote TaskQueue service."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import wait_task_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task = wait_task_sync(remote=remote, task_id=str(task_id), timeout_s=float(timeout_s))
            return {"ok": True, "task": task}
        except Exception as exc:
            logger.exception("p2p_taskqueue_wait_task failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_complete_task(
        task_id: str,
        status: str = "completed",
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Mark a task completed/failed on the remote TaskQueue service."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import complete_task_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = complete_task_sync(
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
    def p2p_taskqueue_heartbeat(
        peer_id: str,
        clock: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Send a peer heartbeat to the remote TaskQueue service."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import heartbeat_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = heartbeat_sync(remote=remote, peer_id=str(peer_id), clock=(clock if isinstance(clock, dict) else None))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_heartbeat failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_cache_get(
        key: str,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Read from the TaskQueue service's shared cache."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_get_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = cache_get_sync(remote=remote, key=str(key), timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_cache_get failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_cache_set(
        key: str,
        value: Any,
        ttl_s: Optional[float] = None,
        timeout_s: float = 10.0,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
    ) -> Dict[str, Any]:
        """Write to the TaskQueue service's shared cache."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_set_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            resp = cache_set_sync(remote=remote, key=str(key), value=value, ttl_s=ttl_s, timeout_s=float(timeout_s))
            return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
        except Exception as exc:
            logger.exception("p2p_taskqueue_cache_set failed")
            return {"ok": False, "error": str(exc)}

    @mcp.tool()
    def p2p_taskqueue_submit_docker_hub(
        image: str,
        command: Any = None,
        entrypoint: Any = None,
        environment: Optional[Dict[str, Any]] = None,
        volumes: Optional[Dict[str, Any]] = None,
        remote_multiaddr: str = "",
        remote_peer_id: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience wrapper: submit `docker.execute` for a Docker Hub image."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_docker_hub_task_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task_id = submit_docker_hub_task_sync(
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
    def p2p_taskqueue_submit_docker_github(
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
        """Convenience wrapper: submit `docker.github` for a GitHub repo."""

        try:
            from ipfs_accelerate_py.p2p_tasks.client import submit_docker_github_task_sync

            remote = _remote_queue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)
            task_id = submit_docker_github_task_sync(
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
