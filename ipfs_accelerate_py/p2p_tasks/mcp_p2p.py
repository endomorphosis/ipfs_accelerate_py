"""MCP++ mcp+p2p stream handler skeleton.

This implements a minimal libp2p stream handler for the draft MCP++ transport
binding protocol id `/mcp+p2p/1.0.0`.

Scope (intentional):
- Deterministic u32 big-endian length-prefixed framing for JSON messages.
- Enforce that the first JSON-RPC request is `initialize`.
- Reply with a minimal JSON-RPC response to `initialize`.

Non-goals (for now):
- Full MCP JSON-RPC method surface
- Capability negotiation beyond a tiny placeholder
- Concurrency/multiplexing of multiple in-flight requests
"""

from __future__ import annotations

import json
import struct
from typing import Any, Optional, Tuple


PROTOCOL_MCP_P2P_V1 = "/mcp+p2p/1.0.0"


def _env_int(name: str, default: int) -> int:
    try:
        import os

        raw = os.environ.get(name)
        if raw is None:
            return int(default)
        return int(str(raw).strip())
    except Exception:
        return int(default)


async def _read_exact(stream: Any, n: int, *, chunk_size: int = 4096) -> bytes:
    """Best-effort read exactly n bytes from a libp2p stream.

    Raises EOFError if the stream ends before n bytes are read.
    """

    remaining = int(n)
    if remaining <= 0:
        return b""

    parts: list[bytes] = []
    while remaining > 0:
        to_read = min(remaining, max(1, int(chunk_size)))
        chunk = await stream.read(to_read)
        if not chunk:
            raise EOFError("unexpected_eof")
        parts.append(bytes(chunk))
        remaining -= len(chunk)
    return b"".join(parts)


async def read_u32_framed_json(
    stream: Any,
    *,
    max_frame_bytes: int = 1024 * 1024,
    chunk_size: int = 4096,
) -> Tuple[Optional[dict[str, Any]], Optional[str]]:
    """Read a single u32 length-prefixed JSON object.

    Returns (obj, None) on success, else (None, error_code).

    error_code is one of: empty, frame_too_large, invalid_json, invalid_message, eof
    """

    max_b = int(max_frame_bytes)
    if max_b < 1:
        max_b = 1

    try:
        header = await _read_exact(stream, 4, chunk_size=chunk_size)
    except EOFError:
        return None, "empty"
    except Exception:
        return None, "eof"

    try:
        (length,) = struct.unpack(">I", header)
    except Exception:
        return None, "invalid_message"

    if int(length) > max_b:
        return None, "frame_too_large"

    try:
        payload = await _read_exact(stream, int(length), chunk_size=chunk_size)
    except EOFError:
        return None, "eof"
    except Exception:
        return None, "eof"

    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception:
        return None, "invalid_json"

    if not isinstance(obj, dict):
        return None, "invalid_message"

    return obj, None


async def write_u32_framed_json(stream: Any, obj: dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">I", len(data))
    await stream.write(header + data)


def _jsonrpc_error(*, id_value: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": id_value,
        "error": {
            "code": int(code),
            "message": str(message),
        },
    }


def _get_registry_tools(registry: Any | None) -> dict[str, dict[str, Any]]:
    if registry is None:
        return {}
    tools = getattr(registry, "tools", None)
    if isinstance(tools, dict):
        return tools
    # Some adapters expose tools via a property which can raise; best-effort.
    try:
        tools = registry.tools  # type: ignore[attr-defined]
        if isinstance(tools, dict):
            return tools
    except Exception:
        pass
    return {}


async def _maybe_validate(registry: Any | None, msg: dict[str, Any]) -> bool:
    if registry is None:
        return True
    fn = getattr(registry, "validate_p2p_message", None)
    if not callable(fn):
        return True
    try:
        res = fn(msg)
        if hasattr(res, "__await__"):
            return bool(await res)
        return bool(res)
    except Exception:
        return False


async def _call_tool(registry: Any | None, *, name: str, arguments: Any) -> tuple[bool, Any]:
    tools = _get_registry_tools(registry)
    desc = tools.get(str(name)) if isinstance(tools, dict) else None
    fn = None
    if isinstance(desc, dict):
        fn = desc.get("function")
    if not callable(fn):
        return False, {"error": "unknown_tool"}

    kwargs = arguments if isinstance(arguments, dict) else {}
    try:
        out = fn(**kwargs)
        if hasattr(out, "__await__"):
            out = await out
        return True, out
    except TypeError as exc:
        return False, {"error": "invalid_params", "detail": str(exc)}
    except Exception as exc:
        return False, {"error": "tool_error", "detail": str(exc)}


async def handle_mcp_p2p_stream(
    stream: Any,
    *,
    local_peer_id: str,
    registry: Any | None = None,
    max_frame_bytes: int = 1024 * 1024,
) -> None:
    """Minimal `/mcp+p2p/1.0.0` handler.

    Must not raise; libp2p swarm can stop listening if exceptions escape.
    """

    try:
        max_frames = _env_int("IPFS_ACCELERATE_PY_MCP_P2P_MAX_FRAMES", 128)
        if max_frames < 1:
            max_frames = 1

        initialized = False
        frames_seen = 0
        while True:
            frames_seen += 1
            if frames_seen > max_frames:
                await write_u32_framed_json(
                    stream,
                    _jsonrpc_error(id_value=None, code=-32010, message="rate_limited"),
                )
                break

            msg, err = await read_u32_framed_json(stream, max_frame_bytes=max_frame_bytes)
            if msg is None:
                # empty/eof -> end session
                if err in {None, "empty", "eof"}:
                    break
                # deterministic framing errors
                await write_u32_framed_json(
                    stream,
                    _jsonrpc_error(id_value=None, code=-32003, message=str(err or "invalid_message")),
                )
                break

            if not await _maybe_validate(registry, msg):
                await write_u32_framed_json(
                    stream,
                    _jsonrpc_error(id_value=msg.get("id"), code=-32001, message="unauthorized"),
                )
                break

            method = str(msg.get("method") or "")
            id_value = msg.get("id")
            jsonrpc = str(msg.get("jsonrpc") or "")

            if jsonrpc != "2.0":
                await write_u32_framed_json(
                    stream,
                    _jsonrpc_error(id_value=id_value, code=-32600, message="invalid_jsonrpc"),
                )
                break

            if not initialized:
                if method != "initialize":
                    await write_u32_framed_json(
                        stream,
                        _jsonrpc_error(id_value=id_value, code=-32000, message="init_required"),
                    )
                    break
                initialized = True
                await write_u32_framed_json(
                    stream,
                    {
                        "jsonrpc": "2.0",
                        "id": id_value,
                        "result": {
                            "ok": True,
                            "transport": PROTOCOL_MCP_P2P_V1,
                            "server": {"peer_id": str(local_peer_id or "")},
                        },
                    },
                )
                continue

            if method in {"tools/list", "tools.list"}:
                tools = _get_registry_tools(registry)
                items: list[dict[str, Any]] = []
                for name, desc in tools.items():
                    if not isinstance(desc, dict):
                        continue
                    items.append(
                        {
                            "name": str(name),
                            "description": str(desc.get("description") or ""),
                            "inputSchema": desc.get("input_schema") or {},
                        }
                    )
                items.sort(key=lambda x: x.get("name") or "")
                await write_u32_framed_json(
                    stream,
                    {"jsonrpc": "2.0", "id": id_value, "result": {"tools": items}},
                )
                continue

            if method in {"tools/call", "tools.call"}:
                params = msg.get("params")
                if not isinstance(params, dict):
                    await write_u32_framed_json(
                        stream,
                        _jsonrpc_error(id_value=id_value, code=-32602, message="invalid_params"),
                    )
                    continue
                tool_name = params.get("name")
                arguments = params.get("arguments")
                ok, out = await _call_tool(registry, name=str(tool_name or ""), arguments=arguments)
                if not ok:
                    await write_u32_framed_json(
                        stream,
                        _jsonrpc_error(id_value=id_value, code=-32002, message=str((out or {}).get("error") or "tool_error")),
                    )
                    continue
                await write_u32_framed_json(
                    stream,
                    {"jsonrpc": "2.0", "id": id_value, "result": {"content": out}},
                )
                continue

            await write_u32_framed_json(
                stream,
                _jsonrpc_error(id_value=id_value, code=-32601, message="method_not_found"),
            )
            continue

        try:
            await stream.close()
        except Exception:
            pass
    except Exception:
        try:
            await stream.close()
        except Exception:
            pass
