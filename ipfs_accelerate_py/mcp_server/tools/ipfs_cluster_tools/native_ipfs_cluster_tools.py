"""Native IPFS-cluster tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_ipfs_cluster_api() -> Dict[str, Any]:
    """Resolve source IPFS-cluster APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.ipfs_cluster_tools.enhanced_ipfs_cluster_tools import (  # type: ignore
            manage_ipfs_cluster as _manage_ipfs_cluster,
            manage_ipfs_content as _manage_ipfs_content,
        )

        return {
            "manage_ipfs_cluster": _manage_ipfs_cluster,
            "manage_ipfs_content": _manage_ipfs_content,
        }
    except Exception:
        logger.warning("Source ipfs_cluster_tools import unavailable, using fallback cluster functions")

        async def _manage_cluster_fallback(
            action: str,
            node_id: Optional[str] = None,
            cid: Optional[str] = None,
            replication_factor: int = 3,
            cluster_config: Optional[Dict[str, Any]] = None,
            filters: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = node_id, cid, replication_factor, cluster_config, filters
            return {
                "action": action,
                "result": {"status": "fallback"},
                "status": "success",
                "cluster_operation": True,
            }

        async def _manage_content_fallback(
            action: str,
            cid: Optional[str] = None,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            pin: bool = True,
            content_type: str = "text/plain",
        ) -> Dict[str, Any]:
            _ = cid, content, metadata, pin, content_type
            return {
                "action": action,
                "result": {"status": "fallback"},
                "status": "success",
            }

        return {
            "manage_ipfs_cluster": _manage_cluster_fallback,
            "manage_ipfs_content": _manage_content_fallback,
        }


_API = _load_ipfs_cluster_api()


def _normalize_payload(result: Any) -> Dict[str, Any]:
    """Normalize backend output to deterministic status envelope."""
    payload = dict(result or {})
    if "error" in payload and payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    return payload


async def manage_ipfs_cluster(
    action: str,
    node_id: Optional[str] = None,
    cid: Optional[str] = None,
    replication_factor: int = 3,
    cluster_config: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute IPFS cluster-management operations."""
    normalized_action = str(action or "").strip().lower()
    valid_actions = {
        "status",
        "add_node",
        "remove_node",
        "pin_content",
        "unpin_content",
        "list_pins",
        "sync",
        "health_check",
        "rebalance",
        "backup_state",
    }
    if normalized_action not in valid_actions:
        return {
            "status": "error",
            "message": "action must be one of: status, add_node, remove_node, pin_content, unpin_content, list_pins, sync, health_check, rebalance, backup_state",
            "action": action,
        }

    normalized_node_id = str(node_id).strip() if node_id is not None else None
    if normalized_action == "remove_node" and not normalized_node_id:
        return {
            "status": "error",
            "message": "node_id is required for remove_node action",
            "node_id": node_id,
        }

    normalized_cid = str(cid).strip() if cid is not None else None
    if normalized_action in {"pin_content", "unpin_content"} and not normalized_cid:
        return {
            "status": "error",
            "message": f"cid is required for {normalized_action} action",
            "cid": cid,
        }

    if not isinstance(replication_factor, int) or replication_factor < 1:
        return {
            "status": "error",
            "message": "replication_factor must be an integer >= 1",
            "replication_factor": replication_factor,
        }
    if cluster_config is not None and not isinstance(cluster_config, dict):
        return {
            "status": "error",
            "message": "cluster_config must be an object when provided",
            "cluster_config": cluster_config,
        }
    if filters is not None and not isinstance(filters, dict):
        return {
            "status": "error",
            "message": "filters must be an object when provided",
            "filters": filters,
        }

    result = _API["manage_ipfs_cluster"](
        action=normalized_action,
        node_id=normalized_node_id,
        cid=normalized_cid,
        replication_factor=replication_factor,
        cluster_config=cluster_config,
        filters=filters,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("replication_factor", replication_factor)
    payload.setdefault("cluster_config", cluster_config or {})
    payload.setdefault("filters", filters or {})
    payload.setdefault("cluster_operation", True)
    payload.setdefault("result", {})
    if normalized_node_id is not None:
        payload.setdefault("node_id", normalized_node_id)
    if normalized_cid is not None:
        payload.setdefault("cid", normalized_cid)
    return payload


async def manage_ipfs_content(
    action: str,
    cid: Optional[str] = None,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    pin: bool = True,
    content_type: str = "text/plain",
) -> Dict[str, Any]:
    """Execute IPFS content operations such as upload, download, and verification."""
    normalized_action = str(action or "").strip().lower()
    valid_actions = {"upload", "download", "get_metadata", "verify_integrity", "list_content"}
    if normalized_action not in valid_actions:
        return {
            "status": "error",
            "message": "action must be one of: upload, download, get_metadata, verify_integrity, list_content",
            "action": action,
        }

    normalized_cid = str(cid).strip() if cid is not None else None
    if normalized_action in {"download", "get_metadata", "verify_integrity"} and not normalized_cid:
        return {
            "status": "error",
            "message": f"cid is required for {normalized_action} action",
            "cid": cid,
        }

    normalized_content = str(content).strip() if content is not None else None
    if normalized_action == "upload" and not normalized_content:
        return {
            "status": "error",
            "message": "content is required for upload action",
            "content": content,
        }
    if metadata is not None and not isinstance(metadata, dict):
        return {
            "status": "error",
            "message": "metadata must be an object when provided",
            "metadata": metadata,
        }
    if not isinstance(pin, bool):
        return {
            "status": "error",
            "message": "pin must be a boolean",
            "pin": pin,
        }
    normalized_content_type = str(content_type or "").strip()
    if not normalized_content_type:
        return {
            "status": "error",
            "message": "content_type must be a non-empty string",
            "content_type": content_type,
        }

    result = _API["manage_ipfs_content"](
        action=normalized_action,
        cid=normalized_cid,
        content=normalized_content,
        metadata=metadata,
        pin=pin,
        content_type=normalized_content_type,
    )
    if hasattr(result, "__await__"):
        payload = _normalize_payload(await result)
    else:
        payload = _normalize_payload(result)
    payload.setdefault("action", normalized_action)
    payload.setdefault("pin", pin)
    payload.setdefault("content_type", normalized_content_type)
    payload.setdefault("metadata", metadata or {})
    payload.setdefault("result", {})
    if normalized_cid is not None:
        payload.setdefault("cid", normalized_cid)
    if normalized_content is not None:
        payload.setdefault("content", normalized_content)
    return payload


def register_native_ipfs_cluster_tools(manager: Any) -> None:
    """Register native IPFS-cluster tools in unified hierarchical manager."""
    manager.register_tool(
        category="ipfs_cluster_tools",
        name="manage_ipfs_cluster",
        func=manage_ipfs_cluster,
        description="Manage IPFS cluster nodes, pin states, and cluster operations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status",
                        "add_node",
                        "remove_node",
                        "pin_content",
                        "unpin_content",
                        "list_pins",
                        "sync",
                        "health_check",
                        "rebalance",
                        "backup_state",
                    ],
                },
                "node_id": {"type": ["string", "null"]},
                "cid": {"type": ["string", "null"]},
                "replication_factor": {"type": "integer", "minimum": 1, "default": 3},
                "cluster_config": {"type": ["object", "null"]},
                "filters": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "ipfs-cluster"],
    )

    manager.register_tool(
        category="ipfs_cluster_tools",
        name="manage_ipfs_content",
        func=manage_ipfs_content,
        description="Manage IPFS content actions including upload, download, and integrity checks.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["upload", "download", "get_metadata", "verify_integrity", "list_content"],
                },
                "cid": {"type": ["string", "null"]},
                "content": {"type": ["string", "null"]},
                "metadata": {"type": ["object", "null"]},
                "pin": {"type": "boolean", "default": True},
                "content_type": {"type": "string", "default": "text/plain"},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "ipfs-cluster"],
    )
