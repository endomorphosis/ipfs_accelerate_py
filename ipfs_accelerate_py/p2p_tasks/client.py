"""libp2p client for the TaskQueue RPC service.

Supports:
- Explicit dialing by multiaddr.
- Bootstrap dialing using a configured list of peers.
- LAN discovery via mDNS (fallback when no multiaddr is provided).

Environment:
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS (comma-separated multiaddrs)
- IPFS_DATASETS_PY_TASK_P2P_DISCOVERY_TIMEOUT_S (compat, default: 5) / IPFS_ACCELERATE_PY_TASK_P2P_DISCOVERY_TIMEOUT_S
- IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT (compat, default: 9710) / IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT (used for mDNS)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .protocol import PROTOCOL_V1, get_shared_token


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class RemoteQueue:
    peer_id: str = ""
    multiaddr: str = ""


def _parse_bootstrap_peers() -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or ""
    )
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def _mdns_port() -> int:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT") or os.environ.get(
        "IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", "9710"
    )
    try:
        return int(str(raw).strip())
    except Exception:
        return 9710


async def _read_one_json_line(stream) -> Dict[str, Any]:
    raw = bytearray()
    max_bytes = 1024 * 1024
    while len(raw) < max_bytes:
        chunk = await stream.read(1024)
        if not chunk:
            break
        raw.extend(chunk)
        if b"\n" in chunk:
            break
    try:
        return json.loads((bytes(raw) or b"{}").rstrip(b"\n").decode("utf-8"))
    except Exception:
        return {"ok": False, "error": "invalid_json_response"}


async def _request_over_stream(*, stream, message: Dict[str, Any]) -> Dict[str, Any]:
    token = get_shared_token()
    if token and "token" not in message:
        message = dict(message)
        message["token"] = token
    await stream.write(json.dumps(message).encode("utf-8") + b"\n")
    return await _read_one_json_line(stream)


async def _try_peer_multiaddr(*, host, peer_multiaddr: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from multiaddr import Multiaddr
    from libp2p.peer.peerinfo import info_from_p2p_addr

    peer_info = info_from_p2p_addr(Multiaddr(peer_multiaddr))
    await host.connect(peer_info)
    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
    try:
        return await _request_over_stream(stream=stream, message=message)
    finally:
        try:
            await stream.close()
        except Exception:
            pass


async def _dial_via_bootstrap(*, host, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for addr in _parse_bootstrap_peers():
        try:
            resp = await _try_peer_multiaddr(host=host, peer_multiaddr=addr, message=message)
            if isinstance(resp, dict):
                return resp
        except Exception:
            continue
    return None


async def _dial_via_mdns(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Dict[str, Any]:
    import anyio

    try:
        from libp2p.discovery.mdns.mdns import MDNSDiscovery
        from libp2p.abc import PeerInfo
    except Exception as exc:
        return {"ok": False, "error": f"mdns_unavailable: {exc}"}

    discover_timeout_s = float(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DISCOVERY_TIMEOUT_S")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DISCOVERY_TIMEOUT_S", "5.0")
    )

    mdns = MDNSDiscovery(host.get_network(), port=_mdns_port())

    try:
        deadline = anyio.current_time() + max(0.1, discover_timeout_s)
        attempted: set[str] = set()

        while anyio.current_time() < deadline:
            discovered_peer_ids = list(mdns.listener.discovered_services.values())
            for pid in discovered_peer_ids:
                pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid)

                if pid_text in attempted:
                    continue
                if require_peer_id and pid_text != require_peer_id:
                    continue

                addrs = host.get_network().peerstore.addrs(pid)
                attempted.add(pid_text)
                if not addrs:
                    continue

                peer_info = PeerInfo(peer_id=pid, addrs=addrs)
                try:
                    await host.connect(peer_info)
                    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                    try:
                        resp = await _request_over_stream(stream=stream, message=message)
                        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass
                except Exception:
                    continue

            await anyio.sleep(0.1)

        return {"ok": False, "error": "discovery_timeout"}
    finally:
        try:
            try:
                mdns.listener.stop()
            except Exception:
                pass
            mdns.stop()
        except Exception:
            pass


async def _dial_and_request(*, remote: RemoteQueue, message: Dict[str, Any]) -> Dict[str, Any]:
    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect
    from libp2p import new_host
    from multiaddr import Multiaddr
    from libp2p.tools.async_service import background_trio_service

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    resp: Dict[str, Any]
    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr("/ip4/0.0.0.0/tcp/0"))

        with anyio.fail_after(20.0):
            if (remote.multiaddr or "").strip():
                resp = await _try_peer_multiaddr(host=host, peer_multiaddr=remote.multiaddr, message=message)  # type: ignore[assignment]
            else:
                # Try bootstrap peers first (cross-subnet), then LAN mDNS.
                boot = await _dial_via_bootstrap(host=host, message=message)
                if isinstance(boot, dict):
                    resp = boot
                else:
                    resp = await _dial_via_mdns(host=host, message=message, require_peer_id=(remote.peer_id or ""))

    try:
        await host.close()
    except Exception:
        pass

    return resp


async def submit_task(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> str:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "submit", "task_type": task_type, "model_name": model_name, "payload": payload},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit failed: {resp}")
    return str(resp.get("task_id"))


async def submit_task_with_info(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "submit", "task_type": task_type, "model_name": model_name, "payload": payload},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit failed: {resp}")
    return {"task_id": str(resp.get("task_id")), "peer_id": str(resp.get("peer_id") or "").strip()}


async def get_task(*, remote: RemoteQueue, task_id: str) -> Optional[Dict[str, Any]]:
    resp = await _dial_and_request(remote=remote, message={"op": "get", "task_id": task_id})
    if not resp.get("ok"):
        raise RuntimeError(f"get failed: {resp}")
    task = resp.get("task")
    return task if isinstance(task, dict) else None


async def wait_task(*, remote: RemoteQueue, task_id: str, timeout_s: float = 60.0) -> Optional[Dict[str, Any]]:
    resp = await _dial_and_request(remote=remote, message={"op": "wait", "task_id": task_id, "timeout_s": float(timeout_s)})
    if not resp.get("ok"):
        raise RuntimeError(f"wait failed: {resp}")
    task = resp.get("task")
    return task if isinstance(task, dict) else None


async def get_capabilities(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "status", "timeout_s": float(timeout_s), "detail": bool(detail)},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"status failed: {resp}")
    caps = resp.get("capabilities")
    return caps if isinstance(caps, dict) else {}


def get_capabilities_sync(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    """Synchronous wrapper around `get_capabilities`.

    Note: libp2p uses Trio internally; this wrapper runs a Trio event loop.
    """

    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await get_capabilities(remote=remote, timeout_s=timeout_s, detail=detail)

    trio.run(_main)
    return result


async def call_tool(*, remote: RemoteQueue, tool_name: str, args: Dict[str, Any] | None = None, timeout_s: float = 30.0) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "call_tool", "tool_name": str(tool_name), "args": (args if isinstance(args, dict) else {}), "timeout_s": float(timeout_s)},
    )
    if not isinstance(resp, dict):
        return {"ok": False, "tool": str(tool_name), "error": "invalid_response"}
    return resp


def call_tool_sync(*, remote: RemoteQueue, tool_name: str, args: Dict[str, Any] | None = None, timeout_s: float = 30.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await call_tool(remote=remote, tool_name=tool_name, args=args, timeout_s=timeout_s)

    trio.run(_main)
    return result


async def cache_get(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "cache.get", "key": str(key), "timeout_s": float(timeout_s)},
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_get_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_get(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result


async def cache_has(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "cache.has", "key": str(key), "timeout_s": float(timeout_s)},
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_has_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_has(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result


async def cache_set(*, remote: RemoteQueue, key: str, value: Any, ttl_s: float | None = None, timeout_s: float = 10.0) -> Dict[str, Any]:
    message: Dict[str, Any] = {"op": "cache.set", "key": str(key), "value": value, "timeout_s": float(timeout_s)}
    if ttl_s is not None:
        try:
            message["ttl_s"] = float(ttl_s)
        except Exception:
            pass

    resp = await _dial_and_request(remote=remote, message=message)
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_set_sync(*, remote: RemoteQueue, key: str, value: Any, ttl_s: float | None = None, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_set(remote=remote, key=str(key), value=value, ttl_s=ttl_s, timeout_s=float(timeout_s))

    trio.run(_main)
    return result


async def cache_delete(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "cache.delete", "key": str(key), "timeout_s": float(timeout_s)},
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_delete_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_delete(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result
