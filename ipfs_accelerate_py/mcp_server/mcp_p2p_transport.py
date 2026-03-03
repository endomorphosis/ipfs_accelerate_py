"""Canonical mcp+p2p transport facade for unified MCP runtime."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ipfs_accelerate_py.p2p_tasks.mcp_p2p_protocol import PROTOCOL_MCP_P2P_V1


def get_mcp_p2p_stats() -> dict[str, int]:
    """Return cumulative mcp+p2p transport counters."""
    from ipfs_accelerate_py.p2p_tasks.mcp_p2p import get_mcp_p2p_stats as _get

    return _get()


def reset_mcp_p2p_stats() -> None:
    """Reset cumulative mcp+p2p transport counters."""
    from ipfs_accelerate_py.p2p_tasks.mcp_p2p import reset_mcp_p2p_stats as _reset

    _reset()


async def read_u32_framed_json(
    stream: Any,
    *,
    max_frame_bytes: int = 1024 * 1024,
    chunk_size: int = 4096,
) -> Tuple[Optional[dict[str, Any]], Optional[str]]:
    """Read one u32-prefixed JSON-RPC message from stream."""
    from ipfs_accelerate_py.p2p_tasks.mcp_p2p import read_u32_framed_json as _read

    return await _read(stream, max_frame_bytes=max_frame_bytes, chunk_size=chunk_size)


async def write_u32_framed_json(
    stream: Any,
    obj: dict[str, Any],
    *,
    max_frame_bytes: int = 16 * 1024 * 1024,
) -> bool:
    """Write one u32-prefixed JSON-RPC message to stream."""
    from ipfs_accelerate_py.p2p_tasks.mcp_p2p import write_u32_framed_json as _write

    return await _write(stream, obj, max_frame_bytes=max_frame_bytes)


async def handle_mcp_p2p_stream(
    stream: Any,
    *,
    registry: Any | None = None,
    peer_id: str = "",
    local_peer_id: str = "",
    max_frame_bytes: int = 1024 * 1024,
) -> None:
    """Handle one mcp+p2p stream session using canonical transport bridge.

    Supports both `peer_id` and `local_peer_id` aliases for compatibility,
    and forwards max-frame limit controls to the canonical backend handler.
    """
    from ipfs_accelerate_py.p2p_tasks.mcp_p2p import handle_mcp_p2p_stream as _handle

    resolved_local_peer_id = str(local_peer_id or peer_id or "")
    await _handle(
        stream,
        local_peer_id=resolved_local_peer_id,
        registry=registry,
        max_frame_bytes=int(max_frame_bytes),
    )
