"""libp2p RPC service for the TaskQueue.

This is a thin transport wrapper around the local DuckDB-backed TaskQueue.
It enables other peers to submit tasks and wait for results.

Environment:
- IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT (compat) / IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT
- IPFS_DATASETS_PY_TASK_P2P_TOKEN (compat) / IPFS_ACCELERATE_PY_TASK_P2P_TOKEN
- IPFS_DATASETS_PY_TASK_P2P_MDNS (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_MDNS
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS (compat) / IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS
- IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP (compat) / IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP (for announce string)
- IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE (compat) / IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE (optional announce JSON)

Protocol:
- /ipfs-datasets/task-queue/1.0.0
- Newline-delimited JSON request/response
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .protocol import PROTOCOL_V1, auth_ok
from .task_queue import TaskQueue
from .cache_store import DiskTTLCache, cache_enabled as _cache_enabled, default_cache_dir


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ServiceConfig:
    listen_port: int = 9710


def _load_config() -> ServiceConfig:
    port = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT") or os.environ.get(
        "IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", "9710"
    )
    return ServiceConfig(listen_port=int(port))


def _parse_bootstrap_peers() -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or ""
    )
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            try:
                key = str(k)
            except Exception:
                continue
            out[key] = _jsonable(v)
        return out
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in list(value)]
    # Fallback: string representation.
    try:
        return repr(value)
    except Exception:
        return str(type(value))


def _accelerate_capabilities(accelerate_instance: object | None, *, detail: bool = False) -> Dict[str, Any]:
    if accelerate_instance is None:
        return {
            "task_types": ["text-generation"],
            "models": [],
            "endpoints_by_model": {},
            "endpoint_types_by_model": {},
            "hwtest": {},
            "resource_keys": [],
        }

    # Prefer core method when available to keep wrappers thin.
    try:
        get_caps = getattr(accelerate_instance, "get_capabilities", None)
        if callable(get_caps):
            try:
                import inspect

                sig = inspect.signature(get_caps)
                if "detail" in sig.parameters:
                    caps = get_caps(detail=bool(detail))
                else:
                    caps = get_caps()
            except Exception:
                caps = get_caps()
            if isinstance(caps, dict):
                # Ensure JSON-safe
                return _jsonable(caps)
    except Exception:
        pass

    # Fall back to status() when present (may include non-JSON values).
    try:
        import inspect

        status_fn = getattr(accelerate_instance, "status", None)
        if callable(status_fn):
            if inspect.iscoroutinefunction(status_fn):
                raw = None
                try:
                    # Trio can await a plain coroutine when it contains no
                    # framework-specific awaitables.
                    import trio

                    async def _call():
                        return await status_fn()

                    raw = trio.run(_call)
                except Exception:
                    raw = None
            else:
                raw = status_fn()

            if isinstance(raw, dict):
                # Summarize from known keys.
                endpoints_by_model: Dict[str, list[str]] = {}
                endpoint_types_by_model: Dict[str, list[str]] = {}
                models: set[str] = set()

                eh = raw.get("endpoint_handler")
                if isinstance(eh, dict):
                    for model, by_ep in eh.items():
                        m = str(model)
                        models.add(m)
                        if isinstance(by_ep, dict):
                            keys = sorted([str(k) for k in by_ep.keys()])
                            endpoints_by_model[m] = keys
                            endpoint_types_by_model[m] = keys

                eps = raw.get("endpoints")
                if isinstance(eps, dict):
                    for _group, by_model in eps.items():
                        if not isinstance(by_model, dict):
                            continue
                        for model, entries in by_model.items():
                            m = str(model)
                            models.add(m)
                            if isinstance(entries, (list, tuple)):
                                cur = set(endpoints_by_model.get(m, []))
                                cur.update([str(x) for x in entries])
                                endpoints_by_model[m] = sorted(cur)

                hw = raw.get("hwtest")
                hwtest = _jsonable(hw) if isinstance(hw, dict) else {}

                resource_keys: list[str] = []
                try:
                    res = getattr(accelerate_instance, "resources", None)
                    if isinstance(res, dict):
                        resource_keys = sorted([str(k) for k in res.keys()])
                except Exception:
                    pass

                return {
                    "task_types": ["text-generation"],
                    "models": sorted(models),
                    "endpoints_by_model": endpoints_by_model,
                    "endpoint_types_by_model": endpoint_types_by_model,
                    "hwtest": hwtest if isinstance(hwtest, dict) else {},
                    "resource_keys": resource_keys,
                }
    except Exception:
        pass

    models: list[str] = []
    endpoints_by_model: Dict[str, list[str]] = {}
    endpoint_types_by_model: Dict[str, list[str]] = {}
    hwtest: Dict[str, Any] = {}
    resource_keys: list[str] = []

    try:
        resources = getattr(accelerate_instance, "resources", None)
        if isinstance(resources, dict):
            eh = resources.get("endpoint_handler")
            if isinstance(eh, dict):
                for model, by_type in eh.items():
                    m = str(model)
                    models.append(m)
                    if isinstance(by_type, dict):
                        endpoints_by_model[m] = sorted([str(k) for k in by_type.keys()])
                        endpoint_types_by_model[m] = sorted([str(k) for k in by_type.keys()])
                    else:
                        endpoints_by_model[m] = []
                        endpoint_types_by_model[m] = []
    except Exception:
        pass

    try:
        ht = getattr(accelerate_instance, "hwtest", None)
        if isinstance(ht, dict):
            hwtest = _jsonable(ht)
    except Exception:
        pass

    try:
        resources = getattr(accelerate_instance, "resources", None)
        if isinstance(resources, dict):
            resource_keys = sorted([str(k) for k in resources.keys()])
    except Exception:
        pass

    return {
        "task_types": ["text-generation"],
        "models": sorted(set(models)),
        "endpoints_by_model": endpoints_by_model,
        "endpoint_types_by_model": endpoint_types_by_model,
        "hwtest": hwtest if isinstance(hwtest, dict) else {},
        "resource_keys": resource_keys,
    }


async def serve_task_queue(
    *,
    queue_path: str,
    listen_port: Optional[int] = None,
    accelerate_instance: object | None = None,
) -> None:
    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect
    from libp2p import new_host
    from multiaddr import Multiaddr
    from libp2p.tools.async_service import background_trio_service

    cfg = _load_config()
    if listen_port is not None:
        cfg.listen_port = int(listen_port)

    queue = TaskQueue(queue_path)
    cache_store = DiskTTLCache(default_cache_dir())

    print("ipfs_accelerate_py task queue p2p service: creating host...", file=sys.stderr, flush=True)
    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj
    peer_id = host.get_id().pretty()
    print("ipfs_accelerate_py task queue p2p service: host created", file=sys.stderr, flush=True)

    async def _handle(stream) -> None:
        try:
            raw = bytearray()
            max_bytes = 1024 * 1024
            while len(raw) < max_bytes:
                chunk = await stream.read(1024)
                if not chunk:
                    break
                raw.extend(chunk)
                if b"\n" in chunk:
                    break
            if not raw:
                return
            try:
                msg = json.loads(bytes(raw).rstrip(b"\n").decode("utf-8"))
            except Exception:
                await stream.write(json.dumps({"ok": False, "error": "invalid_json", "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if not isinstance(msg, dict):
                await stream.write(json.dumps({"ok": False, "error": "invalid_message", "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if not auth_ok(msg):
                await stream.write(json.dumps({"ok": False, "error": "unauthorized", "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            op = (msg.get("op") or "").strip().lower()

            if op == "submit":
                task_type = str(msg.get("task_type") or "text-generation")
                model_name = str(msg.get("model_name") or "")
                payload = msg.get("payload")
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                task_id = queue.submit(task_type=task_type, model_name=model_name, payload=payload)
                await stream.write(json.dumps({"ok": True, "task_id": task_id, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if op in {"status", "capabilities", "describe"}:
                detail = bool(msg.get("detail"))
                caps = _accelerate_capabilities(accelerate_instance, detail=detail)
                await stream.write(
                    json.dumps({"ok": True, "capabilities": caps, "peer_id": peer_id}).encode("utf-8") + b"\n"
                )
                return

            if op in {"tool", "call_tool", "tool.call"}:
                allow = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS", "")).lower() in {"1", "true", "yes"}
                if not allow:
                    await stream.write(json.dumps({"ok": False, "error": "tools_disabled", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                tool_name = str(msg.get("tool") or msg.get("tool_name") or msg.get("name") or "").strip()
                args = msg.get("args") or msg.get("arguments") or msg.get("params") or {}
                if not isinstance(args, dict):
                    args = {"value": args}

                if not tool_name:
                    await stream.write(json.dumps({"ok": False, "error": "missing_tool_name", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                try:
                    import inspect

                    if accelerate_instance is not None and hasattr(accelerate_instance, "call_tool"):
                        fn = getattr(accelerate_instance, "call_tool")
                        if inspect.iscoroutinefunction(fn):
                            resp = await fn(tool_name, args)
                        else:
                            resp = fn(tool_name, args)
                            if inspect.isawaitable(resp):
                                resp = await resp
                    else:
                        # Fallback: invoke on the global MCP server instance.
                        from ipfs_accelerate_py.mcp.server import get_mcp_server_instance
                        from ipfs_accelerate_py.tool_manifest import invoke_mcp_tool

                        mcp_like = get_mcp_server_instance()
                        resp = await invoke_mcp_tool(mcp_like, tool_name=tool_name, args=args, accelerate_instance=accelerate_instance)

                    if not isinstance(resp, dict):
                        resp = {"ok": True, "tool": tool_name, "result": resp}

                    resp.setdefault("peer_id", peer_id)
                    await stream.write(json.dumps(resp).encode("utf-8") + b"\n")
                    return
                except Exception as exc:
                    await stream.write(
                        json.dumps({"ok": False, "tool": tool_name, "error": str(exc), "peer_id": peer_id}).encode("utf-8") + b"\n"
                    )
                    return

            if op in {"cache.get", "cache_get", "cache"}:
                if not _cache_enabled():
                    await stream.write(json.dumps({"ok": False, "error": "cache_disabled", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await stream.write(json.dumps({"ok": False, "error": "missing_key", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                value = cache_store.get(key)
                await stream.write(
                    json.dumps({"ok": True, "key": key, "hit": value is not None, "value": _jsonable(value), "peer_id": peer_id}).encode("utf-8")
                    + b"\n"
                )
                return

            if op in {"cache.has", "cache_has"}:
                if not _cache_enabled():
                    await stream.write(json.dumps({"ok": False, "error": "cache_disabled", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await stream.write(json.dumps({"ok": False, "error": "missing_key", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                hit = bool(cache_store.has(key))
                await stream.write(json.dumps({"ok": True, "key": key, "hit": hit, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if op in {"cache.set", "cache_set"}:
                if not _cache_enabled():
                    await stream.write(json.dumps({"ok": False, "error": "cache_disabled", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await stream.write(json.dumps({"ok": False, "error": "missing_key", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                value = msg.get("value")
                ttl_s = msg.get("ttl_s")
                try:
                    ttl_value = float(ttl_s) if ttl_s is not None else None
                except Exception:
                    ttl_value = None

                cache_store.set(key, value, ttl_s=ttl_value)
                await stream.write(json.dumps({"ok": True, "key": key, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if op in {"cache.delete", "cache_del", "cache_delete"}:
                if not _cache_enabled():
                    await stream.write(json.dumps({"ok": False, "error": "cache_disabled", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await stream.write(json.dumps({"ok": False, "error": "missing_key", "peer_id": peer_id}).encode("utf-8") + b"\n")
                    return

                deleted = bool(cache_store.delete(key))
                await stream.write(json.dumps({"ok": True, "key": key, "deleted": deleted, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if op == "get":
                task_id = str(msg.get("task_id") or "")
                task = queue.get(task_id)
                await stream.write(json.dumps({"ok": True, "task": task, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            if op == "wait":
                task_id = str(msg.get("task_id") or "")
                timeout_s = float(msg.get("timeout_s") or 60.0)
                deadline = time.time() + max(0.0, timeout_s)

                task = queue.get(task_id)
                while task is not None and task.get("status") in {"queued", "running"} and time.time() < deadline:
                    await anyio.sleep(0.1)
                    task = queue.get(task_id)

                await stream.write(json.dumps({"ok": True, "task": task, "peer_id": peer_id}).encode("utf-8") + b"\n")
                return

            await stream.write(json.dumps({"ok": False, "error": "unknown_op", "peer_id": peer_id}).encode("utf-8") + b"\n")
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    host.set_stream_handler(PROTOCOL_V1, _handle)

    listen_addr = Multiaddr(f"/ip4/0.0.0.0/tcp/{cfg.listen_port}")
    print(f"ipfs_accelerate_py task queue p2p service: listening on {listen_addr}", file=sys.stderr, flush=True)

    mdns_enabled = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MDNS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MDNS", "1")
    ).strip().lower() not in {"0", "false", "no"}

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(listen_addr)

        # Bootstrap connections (best-effort)
        try:
            from libp2p.peer.peerinfo import info_from_p2p_addr

            for peer_addr in _parse_bootstrap_peers():
                try:
                    peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
                    await host.connect(peer_info)
                    print(f"ipfs_accelerate_py task queue p2p service: connected bootstrap {peer_addr}", file=sys.stderr, flush=True)
                except Exception as exc:
                    print(f"ipfs_accelerate_py task queue p2p service: bootstrap connect failed {peer_addr}: {exc}", file=sys.stderr, flush=True)
        except Exception:
            pass

        mdns = None
        if mdns_enabled:
            try:
                from libp2p.discovery.mdns.mdns import MDNSDiscovery

                mdns = MDNSDiscovery(host.get_network(), port=int(cfg.listen_port))
                mdns.start()
                print("ipfs_accelerate_py task queue p2p service: mDNS enabled", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"ipfs_accelerate_py task queue p2p service: failed to start mDNS: {exc}", file=sys.stderr, flush=True)

        public_ip = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP", "127.0.0.1")
        ).strip() or "127.0.0.1"
        announced = f"/ip4/{public_ip}/tcp/{cfg.listen_port}/p2p/{peer_id}"
        print("ipfs_accelerate_py task queue p2p service started", flush=True)
        print(f"peer_id={peer_id}", flush=True)
        print(f"multiaddr={announced}", flush=True)

        announce_file = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE", "")
        ).strip()
        if announce_file:
            try:
                os.makedirs(os.path.dirname(announce_file) or ".", exist_ok=True)
                with open(announce_file, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"peer_id": peer_id, "multiaddr": announced}, ensure_ascii=False))
                print(f"ipfs_accelerate_py task queue p2p service: wrote announce file {announce_file}", file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"ipfs_accelerate_py task queue p2p service: failed to write announce file {announce_file}: {exc}", file=sys.stderr, flush=True)

        try:
            await anyio.Event().wait()
        finally:
            try:
                if mdns is not None:
                    try:
                        mdns.listener.stop()
                    except Exception:
                        pass
                    mdns.stop()
            except Exception:
                pass


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import anyio

    parser = argparse.ArgumentParser(description="Run libp2p TaskQueue RPC service")
    parser.add_argument("--queue", required=True, help="Path to task queue DuckDB file")
    parser.add_argument("--listen-port", type=int, default=None)

    args = parser.parse_args(argv)

    async def _main() -> None:
        await serve_task_queue(queue_path=args.queue, listen_port=args.listen_port)

    anyio.run(_main, backend="trio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
