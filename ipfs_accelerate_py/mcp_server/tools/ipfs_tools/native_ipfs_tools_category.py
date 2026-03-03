"""Native ipfs-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


def _load_ipfs_tools_api() -> Dict[str, Any]:
    """Resolve source ipfs-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.ipfs_tools import (  # type: ignore
            get_from_ipfs as _get_from_ipfs,
            pin_to_ipfs as _pin_to_ipfs,
        )

        return {
            "pin_to_ipfs": _pin_to_ipfs,
            "get_from_ipfs": _get_from_ipfs,
        }
    except Exception:
        logger.warning("Source ipfs_tools import unavailable, using fallback ipfs-tools functions")

        async def _pin_fallback(
            content_source: Union[str, Dict[str, Any]],
            recursive: bool = True,
            wrap_with_directory: bool = False,
            hash_algo: str = "sha2-256",
        ) -> Dict[str, Any]:
            _ = recursive, wrap_with_directory, hash_algo
            return {
                "status": "error",
                "message": "IPFS pin backend unavailable",
                "content_path": str(content_source),
            }

        async def _get_fallback(
            cid: str,
            output_path: Optional[str] = None,
            timeout_seconds: int = 60,
            gateway: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = output_path, timeout_seconds, gateway
            return {
                "status": "error",
                "message": "IPFS get backend unavailable",
                "cid": cid,
            }

        return {
            "pin_to_ipfs": _pin_fallback,
            "get_from_ipfs": _get_fallback,
        }


_API = _load_ipfs_tools_api()


async def pin_to_ipfs(
    content_source: Union[str, Dict[str, Any]],
    recursive: bool = True,
    wrap_with_directory: bool = False,
    hash_algo: str = "sha2-256",
) -> Dict[str, Any]:
    """Pin file/directory/content to IPFS."""
    if isinstance(content_source, str):
        normalized_source = content_source.strip()
        if not normalized_source:
            return {"status": "error", "message": "'content_source' must be a non-empty string or object."}
        content_source = normalized_source
    elif not isinstance(content_source, dict):
        return {"status": "error", "message": "'content_source' must be a string or object."}

    normalized_hash_algo = str(hash_algo or "sha2-256").strip() or "sha2-256"

    result = _API["pin_to_ipfs"](
        content_source=content_source,
        recursive=bool(recursive),
        wrap_with_directory=bool(wrap_with_directory),
        hash_algo=normalized_hash_algo,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def get_from_ipfs(
    cid: str,
    output_path: Optional[str] = None,
    timeout_seconds: int = 60,
    gateway: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve content from IPFS by CID."""
    normalized_cid = str(cid or "").strip()
    if not normalized_cid:
        return {"status": "error", "message": "'cid' is required."}

    normalized_timeout_seconds = int(timeout_seconds)
    if normalized_timeout_seconds <= 0:
        return {"status": "error", "message": "'timeout_seconds' must be greater than 0."}

    normalized_output_path = str(output_path).strip() if output_path is not None else None
    if output_path is not None and not normalized_output_path:
        return {"status": "error", "message": "'output_path' must be a non-empty string when provided."}

    normalized_gateway = str(gateway).strip() if gateway is not None else None
    if gateway is not None and not normalized_gateway:
        return {"status": "error", "message": "'gateway' must be a non-empty string when provided."}

    result = _API["get_from_ipfs"](
        cid=normalized_cid,
        output_path=normalized_output_path,
        timeout_seconds=normalized_timeout_seconds,
        gateway=normalized_gateway,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


def register_native_ipfs_tools_category(manager: Any) -> None:
    """Register native ipfs-tools category tools in unified manager."""
    manager.register_tool(
        category="ipfs_tools",
        name="pin_to_ipfs",
        func=pin_to_ipfs,
        description="Pin content to IPFS.",
        input_schema={
            "type": "object",
            "properties": {
                "content_source": {"type": ["string", "object"]},
                "recursive": {"type": "boolean", "default": True},
                "wrap_with_directory": {"type": "boolean", "default": False},
                "hash_algo": {"type": "string", "default": "sha2-256"},
            },
            "required": ["content_source"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "ipfs-tools"],
    )

    manager.register_tool(
        category="ipfs_tools",
        name="get_from_ipfs",
        func=get_from_ipfs,
        description="Get content from IPFS by CID.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string"},
                "output_path": {"type": ["string", "null"]},
                "timeout_seconds": {"type": "integer", "default": 60, "minimum": 1},
                "gateway": {"type": ["string", "null"]},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "ipfs-tools"],
    )
