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

import threading
from typing import Any, Optional, Tuple

from ipfs_accelerate_py.mcp_server.mcplusplus.p2p_framing import (
    FrameSizeExceededError,
    FramingError,
    TokenBucketLimiter,
    decode_jsonrpc_frame,
    encode_jsonrpc_frame,
)


PROTOCOL_MCP_P2P_V1 = "/mcp+p2p/1.0.0"


_MCP_P2P_STATS_LOCK = threading.RLock()
_MCP_P2P_STATS: dict[str, int] = {
    "sessions_started": 0,
    "sessions_closed": 0,
    "initialized_sessions": 0,
    "frame_errors": 0,
    "rate_limited": 0,
    "unauthorized": 0,
    "internal_errors": 0,
}


def _resolve_profile_negotiation(registry: Any | None) -> tuple[list[str], dict[str, Any]]:
    """Resolve profile negotiation metadata from registry or unified defaults."""

    profiles: list[str] = []
    negotiation: dict[str, Any] = {
        "supports_profile_negotiation": True,
        "mode": "optional_additive",
        "profiles": profiles,
    }

    if registry is not None:
        try:
            raw_profiles = getattr(registry, "_unified_supported_profiles", None)
            if isinstance(raw_profiles, list):
                profiles = [str(p) for p in raw_profiles if str(p).strip()]
        except Exception:
            profiles = []

        try:
            raw_negotiation = getattr(registry, "_unified_profile_negotiation", None)
            if isinstance(raw_negotiation, dict):
                negotiation = dict(raw_negotiation)
        except Exception:
            pass

    if not profiles:
        # Lazy import to avoid pulling unified server at module import time.
        try:
            from ipfs_accelerate_py.mcp_server.server import get_unified_supported_profiles

            raw = get_unified_supported_profiles()
            if isinstance(raw, list):
                profiles = [str(p) for p in raw if str(p).strip()]
        except Exception:
            profiles = []

    # Normalize negotiation payload and keep it JSON-safe.
    negotiation.setdefault("supports_profile_negotiation", True)
    negotiation.setdefault("mode", "optional_additive")
    negotiation["profiles"] = list(profiles)
    return profiles, negotiation


def _select_profile(params: Any, supported_profiles: list[str]) -> str:
    """Select an active profile from client initialize params and supported list."""

    if not supported_profiles:
        return ""

    if isinstance(params, dict):
        requested = str(params.get("profile") or "").strip()
        if requested and requested in supported_profiles:
            return requested
        requested_many = params.get("profiles")
        if isinstance(requested_many, list):
            for candidate in requested_many:
                text = str(candidate or "").strip()
                if text and text in supported_profiles:
                    return text

    return str(supported_profiles[0])


def _inc_stat(key: str, amount: int = 1) -> None:
    with _MCP_P2P_STATS_LOCK:
        _MCP_P2P_STATS[key] = int(_MCP_P2P_STATS.get(key, 0)) + int(amount)


def get_mcp_p2p_stats() -> dict[str, int]:
    """Return cumulative transport handler counters."""
    with _MCP_P2P_STATS_LOCK:
        return dict(_MCP_P2P_STATS)


def reset_mcp_p2p_stats() -> None:
    """Reset cumulative transport handler counters."""
    with _MCP_P2P_STATS_LOCK:
        for key in list(_MCP_P2P_STATS.keys()):
            _MCP_P2P_STATS[key] = 0


def _env_int(name: str, default: int) -> int:
    try:
        import os

        raw = os.environ.get(name)
        if raw is None:
            return int(default)
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        import os

        raw = os.environ.get(name)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_int_compat(primary: str, compat: str, default: int) -> int:
    try:
        import os

        raw = os.environ.get(primary)
        if raw is None:
            raw = os.environ.get(compat)
        if raw is None:
            return int(default)
        return int(str(raw).strip())
    except Exception:
        return int(default)


def _env_float_compat(primary: str, compat: str, default: float) -> float:
    try:
        import os

        raw = os.environ.get(primary)
        if raw is None:
            raw = os.environ.get(compat)
        if raw is None:
            return float(default)
        return float(str(raw).strip())
    except Exception:
        return float(default)


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

    # Parse declared frame size first to avoid unbounded reads.
    try:
        declared = int.from_bytes(header, byteorder="big", signed=False)
    except Exception:
        return None, "invalid_message"

    if declared > max_b:
        return None, "frame_too_large"

    try:
        payload = await _read_exact(stream, declared, chunk_size=chunk_size)
    except EOFError:
        return None, "eof"
    except Exception:
        return None, "eof"

    try:
        obj, _consumed = decode_jsonrpc_frame(header + payload, max_frame_bytes=max_b)
        return obj, None
    except FrameSizeExceededError:
        return None, "frame_too_large"
    except FramingError as exc:
        code = str(exc)
        if code in {"incomplete_prefix", "incomplete_body"}:
            return None, "eof"
        if code == "payload_not_object":
            return None, "invalid_message"
        return None, "invalid_message"
    except Exception:
        return None, "invalid_json"


async def write_u32_framed_json(
    stream: Any,
    obj: dict[str, Any],
    *,
    max_frame_bytes: int = 16 * 1024 * 1024,
) -> None:
    await stream.write(encode_jsonrpc_frame(obj, max_frame_bytes=int(max_frame_bytes)))


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

    closed = False
    _inc_stat("sessions_started")

    try:
        max_frames = _env_int_compat(
            "IPFS_ACCELERATE_PY_MCP_P2P_MAX_FRAMES",
            "IPFS_DATASETS_PY_MCP_P2P_MAX_FRAMES",
            128,
        )
        if max_frames < 1:
            max_frames = 1
        rate_capacity = _env_int_compat(
            "IPFS_ACCELERATE_PY_MCP_P2P_RATE_CAPACITY",
            "IPFS_DATASETS_PY_MCP_P2P_RATE_CAPACITY",
            max_frames,
        )
        if rate_capacity < 1:
            rate_capacity = 1
        rate_refill_per_sec = _env_float_compat(
            "IPFS_ACCELERATE_PY_MCP_P2P_RATE_REFILL_PER_SEC",
            "IPFS_DATASETS_PY_MCP_P2P_RATE_REFILL_PER_SEC",
            float(max_frames),
        )
        if rate_refill_per_sec <= 0.0:
            rate_refill_per_sec = float(max_frames)
        limiter = TokenBucketLimiter(
            capacity=float(rate_capacity),
            refill_rate_per_sec=float(rate_refill_per_sec),
        )
        supported_profiles, profile_negotiation = _resolve_profile_negotiation(registry)

        initialized = False
        frames_seen = 0
        while True:
            msg, err = await read_u32_framed_json(stream, max_frame_bytes=max_frame_bytes)
            if msg is None:
                # empty/eof -> end session
                if err in {None, "empty", "eof"}:
                    break
                # deterministic framing errors
                _inc_stat("frame_errors")
                await write_u32_framed_json(
                    stream,
                    _jsonrpc_error(id_value=None, code=-32003, message=str(err or "invalid_message")),
                )
                break

            frames_seen += 1
            if frames_seen > max_frames:
                # Notifications must not receive a response.
                if "id" not in msg:
                    break
                _inc_stat("rate_limited")
                await write_u32_framed_json(stream, _jsonrpc_error(id_value=msg.get("id"), code=-32010, message="rate_limited"))
                break

            if not limiter.allow(cost=1.0):
                # Notifications must not receive a response.
                if "id" not in msg:
                    break
                _inc_stat("rate_limited")
                await write_u32_framed_json(stream, _jsonrpc_error(id_value=msg.get("id"), code=-32010, message="rate_limited"))
                break

            if not await _maybe_validate(registry, msg):
                # Notifications must not receive a response.
                if "id" not in msg:
                    break
                _inc_stat("unauthorized")
                await write_u32_framed_json(stream, _jsonrpc_error(id_value=msg.get("id"), code=-32001, message="unauthorized"))
                break

            method = str(msg.get("method") or "")
            id_value = msg.get("id")
            jsonrpc = str(msg.get("jsonrpc") or "")
            is_notification = "id" not in msg

            if jsonrpc != "2.0":
                # Ignore invalid notifications; deterministically error for requests.
                if is_notification:
                    continue
                await write_u32_framed_json(stream, _jsonrpc_error(id_value=id_value, code=-32600, message="invalid_jsonrpc"))
                break

            if not initialized:
                if method != "initialize":
                    # Notifications must not receive responses; ignore and keep waiting.
                    if is_notification:
                        continue
                    await write_u32_framed_json(stream, _jsonrpc_error(id_value=id_value, code=-32000, message="init_required"))
                    break
                # `initialize` as a notification is ignored; the session is not initialized.
                if is_notification:
                    continue
                initialized = True
                _inc_stat("initialized_sessions")
                await write_u32_framed_json(
                    stream,
                    {
                        "jsonrpc": "2.0",
                        "id": id_value,
                        "result": {
                            "ok": True,
                            "transport": PROTOCOL_MCP_P2P_V1,
                            "server": {"peer_id": str(local_peer_id or "")},
                            "profile_negotiation": dict(profile_negotiation),
                            "active_profile": _select_profile(msg.get("params"), supported_profiles),
                            "limits": {
                                "max_frame_bytes": int(max_frame_bytes),
                                "max_frames": int(max_frames),
                                "rate_capacity": int(rate_capacity),
                                "rate_refill_per_sec": float(rate_refill_per_sec),
                            },
                        },
                    },
                )
                continue

            if method in {"tools/list", "tools.list"}:
                if is_notification:
                    continue
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
                if is_notification:
                    continue
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

            if is_notification:
                continue

            await write_u32_framed_json(
                stream,
                _jsonrpc_error(id_value=id_value, code=-32601, message="method_not_found"),
            )
            continue

        try:
            await stream.close()
            closed = True
        except Exception:
            pass
    except Exception:
        _inc_stat("internal_errors")
        try:
            await stream.close()
            closed = True
        except Exception:
            pass
    finally:
        if closed:
            _inc_stat("sessions_closed")


__all__ = [
    "PROTOCOL_MCP_P2P_V1",
    "read_u32_framed_json",
    "write_u32_framed_json",
    "handle_mcp_p2p_stream",
    "get_mcp_p2p_stats",
    "reset_mcp_p2p_stats",
]
