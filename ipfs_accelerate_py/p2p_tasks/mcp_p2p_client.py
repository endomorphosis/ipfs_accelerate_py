"""Client helpers for the MCP++ `/mcp+p2p/1.0.0` transport binding.

Intentionally small:
- u32 big-endian length-prefixed JSON framing
- JSON-RPC request/response helpers
- tiny convenience methods for initialize + tools/list + tools/call

This module is async-runtime agnostic (works under Trio or asyncio) as long
as the provided stream exposes `read`, `write`, and optionally `close`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from .mcp_p2p import read_u32_framed_json, write_u32_framed_json


@dataclass(frozen=True)
class MCPRemoteError(Exception):
    """Raised when the remote returns a JSON-RPC error."""

    code: int
    message: str
    id_value: Any = None

    def __str__(self) -> str:  # pragma: no cover
        return f"jsonrpc_error code={self.code} message={self.message!r} id={self.id_value!r}"


class MCPFramingError(Exception):
    """Raised when the stream yields deterministic framing/JSON errors."""


async def open_libp2p_stream_by_multiaddr(
    host: Any,
    *,
    peer_multiaddr: str,
    protocols: list[str],
) -> Any:
    """Dial a peer by multiaddr and open a stream for the given protocol ids.

    This helper is intentionally tiny and lazy-imports libp2p/multiaddr so the
    rest of the module stays importable without libp2p installed.
    """

    from multiaddr import Multiaddr
    from libp2p.peer.peerinfo import info_from_p2p_addr

    peer_info = info_from_p2p_addr(Multiaddr(str(peer_multiaddr)))
    await host.connect(peer_info)
    return await host.new_stream(peer_info.peer_id, list(protocols))


@asynccontextmanager
async def trio_libp2p_host_listen(*, listen_multiaddr: str = "/ip4/127.0.0.1/tcp/0"):
    """Create a libp2p host, start its network service, and listen.

    This is a Trio-specific helper (uses libp2p's `background_trio_service`).
    It is intended for integration tests and small scripts.

    Lazy-imports libp2p/multiaddr so the rest of the package can be imported
    without libp2p installed.
    """

    import inspect

    import libp2p
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    host_obj = libp2p.new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    try:
        async with background_trio_service(host.get_network()):
            await host.get_network().listen(Multiaddr(str(listen_multiaddr)))
            yield host
    finally:
        try:
            await host.close()
        except Exception:
            pass


class MCPP2PClient:
    def __init__(self, stream: Any, *, max_frame_bytes: int = 1024 * 1024) -> None:
        self._stream = stream
        self._max_frame_bytes = int(max_frame_bytes)
        self._next_id = 1

    def _alloc_id(self) -> int:
        i = int(self._next_id)
        self._next_id = i + 1
        return i

    async def request(self, method: str, params: Optional[dict[str, Any]] = None, *, id_value: Any = None) -> dict[str, Any]:
        if id_value is None:
            id_value = self._alloc_id()
        await write_u32_framed_json(
            self._stream,
            {
                "jsonrpc": "2.0",
                "id": id_value,
                "method": str(method),
                "params": params or {},
            },
        )

        resp, err = await read_u32_framed_json(self._stream, max_frame_bytes=self._max_frame_bytes)
        if resp is None:
            raise MCPFramingError(str(err or "invalid_message"))

        resp_id = resp.get("id")
        if resp_id is not None and id_value is not None and resp_id != id_value:
            raise MCPFramingError(f"mismatched_id expected={id_value!r} got={resp_id!r}")

        if "error" in resp and isinstance(resp.get("error"), dict):
            e = resp["error"]
            raise MCPRemoteError(
                code=int(e.get("code") if e.get("code") is not None else -32000),
                message=str(e.get("message") or "error"),
                id_value=resp.get("id"),
            )

        return resp

    async def request_raw(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Send a raw framed JSON-RPC message and await a single response."""

        request_msg = dict(msg)
        await write_u32_framed_json(self._stream, request_msg)
        resp, err = await read_u32_framed_json(self._stream, max_frame_bytes=self._max_frame_bytes)
        if resp is None:
            raise MCPFramingError(str(err or "invalid_message"))

        if "id" in request_msg:
            expected_id = request_msg.get("id")
            resp_id = resp.get("id")
            if resp_id is not None and expected_id is not None and resp_id != expected_id:
                raise MCPFramingError(f"mismatched_id expected={expected_id!r} got={resp_id!r}")

        if "error" in resp and isinstance(resp.get("error"), dict):
            e = resp["error"]
            raise MCPRemoteError(
                code=int(e.get("code") if e.get("code") is not None else -32000),
                message=str(e.get("message") or "error"),
                id_value=resp.get("id"),
            )
        return resp

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no `id`) and do not await a response."""

        await write_u32_framed_json(
            self._stream,
            {
                "jsonrpc": "2.0",
                "method": str(method),
                "params": params or {},
            },
        )

    async def notify_raw(self, msg: dict[str, Any]) -> None:
        """Send a raw JSON-RPC notification frame (must omit `id`)."""

        m = dict(msg)
        m.pop("id", None)
        await write_u32_framed_json(self._stream, m)

    async def initialize(self, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return await self.request("initialize", params or {}, id_value=1)

    async def tools_list(self) -> list[dict[str, Any]]:
        resp = await self.request("tools/list", {})
        tools = (resp.get("result") or {}).get("tools")
        if not isinstance(tools, list):
            return []
        return [t for t in tools if isinstance(t, dict)]

    async def tools_call(self, name: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        resp = await self.request(
            "tools/call",
            {"name": str(name), "arguments": arguments or {}},
        )
        return (resp.get("result") or {}).get("content")

    async def aclose(self) -> None:
        close = getattr(self._stream, "close", None)
        if callable(close):
            try:
                await close()
            except Exception:
                pass
