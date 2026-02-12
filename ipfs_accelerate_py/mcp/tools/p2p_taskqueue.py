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
