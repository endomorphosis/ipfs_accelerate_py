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
    result = _API["pin_to_ipfs"](
        content_source=content_source,
        recursive=recursive,
        wrap_with_directory=wrap_with_directory,
        hash_algo=hash_algo,
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
    result = _API["get_from_ipfs"](
        cid=cid,
        output_path=output_path,
        timeout_seconds=timeout_seconds,
        gateway=gateway,
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
                "recursive": {"type": "boolean"},
                "wrap_with_directory": {"type": "boolean"},
                "hash_algo": {"type": "string"},
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
                "timeout_seconds": {"type": "integer"},
                "gateway": {"type": ["string", "null"]},
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "ipfs-tools"],
    )
