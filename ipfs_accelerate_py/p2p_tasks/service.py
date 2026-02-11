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


async def serve_task_queue(*, queue_path: str, listen_port: Optional[int] = None) -> None:
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
