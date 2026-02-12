"""libp2p client for the TaskQueue RPC service.

Supports:
- Explicit dialing by multiaddr.
- Bootstrap dialing using a configured list of peers.
- LAN discovery via mDNS (fallback when no multiaddr is provided).

Environment:
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS (comma-separated multiaddrs)
- IPFS_DATASETS_PY_TASK_P2P_DISCOVERY_TIMEOUT_S (compat, default: 5) / IPFS_ACCELERATE_PY_TASK_P2P_DISCOVERY_TIMEOUT_S
- IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT (compat, default: 9710) / IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT (used for mDNS)
- IPFS_DATASETS_PY_TASK_P2P_DHT (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_DHT
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS (compat, default: ipfs-accelerate-task-queue) / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS
- IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE (compat) / IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE
    - unset: read default XDG cache announce file(s)
    - set to a path: read announce JSON from that path
    - set to 0/false/no: disable announce-file dialing
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .protocol import PROTOCOL_V1, get_shared_token


def _truthy(text: str | None, *, default: bool = False) -> bool:
    if text is None:
        return bool(default)
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool(*, primary: str, compat: str, default: bool) -> bool:
    raw = os.environ.get(primary)
    if raw is None:
        raw = os.environ.get(compat)
    if raw is None:
        return bool(default)
    return _truthy(str(raw), default=default)


def _env_str(*, primary: str, compat: str, default: str) -> str:
    raw = os.environ.get(primary)
    if raw is None:
        raw = os.environ.get(compat)
    text = str(raw).strip() if raw is not None else ""
    return text or str(default)


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


def _default_announce_files() -> list[str]:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return [
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json"),
        os.path.join(cache_root, "ipfs_datasets_py", "task_p2p_announce.json"),
    ]


def _read_announce_multiaddr() -> str:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return ""

    candidates: list[str] = []
    if raw is not None and str(raw).strip():
        candidates.append(str(raw).strip())
    candidates.extend(_default_announce_files())

    for path in candidates:
        try:
            if not path or not os.path.exists(path):
                continue
            text = open(path, "r", encoding="utf-8").read().strip()
            if not text:
                continue
            info = json.loads(text)
            if isinstance(info, dict):
                ma = str(info.get("multiaddr") or "").strip()
                if ma and "/p2p/" in ma:
                    return ma
        except Exception:
            continue
    return ""


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


async def _dial_via_announce_file(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Optional[Dict[str, Any]]:
    ma = _read_announce_multiaddr()
    if not ma:
        return None

    # If the caller requested a specific peer, avoid dialing the announce hint
    # when it clearly doesn't match.
    if require_peer_id:
        try:
            pid = str(ma).rsplit("/p2p/", 1)[-1].strip()
            if pid and pid != require_peer_id:
                return None
        except Exception:
            pass

    try:
        resp = await _try_peer_multiaddr(host=host, peer_multiaddr=ma, message=message)
        return resp if isinstance(resp, dict) else None
    except Exception:
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


async def _dial_via_rendezvous(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Optional[Dict[str, Any]]:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS",
        default=True,
    ):
        return None

    ns = _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )

    candidates = [
        ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
        ("libp2p.discovery.rendezvous", "RendezvousClient"),
        ("libp2p.rendezvous", "RendezvousClient"),
    ]
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            cli = cls(host)
            discover = getattr(cli, "discover", None)
            if not callable(discover):
                continue

            # Newer py-libp2p returns (peers, cookie).
            cookie: bytes = b""
            try:
                peers, cookie = await discover(ns, limit=100, cookie=cookie)
            except TypeError:
                # Older implementations may have a simpler signature.
                found = discover(ns)
                peers = list(found or [])

            for peer_info in list(peers or []):
                try:
                    pid = getattr(peer_info, "peer_id", None)
                    pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                    if require_peer_id and pid_text != require_peer_id:
                        continue
                    await host.connect(peer_info)
                    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                    try:
                        resp = await _request_over_stream(stream=stream, message=message)
                        return resp if isinstance(resp, dict) else None
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            continue
    return None


async def _dial_via_dht(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Optional[Dict[str, Any]]:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_DHT",
        compat="IPFS_DATASETS_PY_TASK_P2P_DHT",
        default=True,
    ):
        return None

    ns = _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )

    candidates = [
        ("libp2p.kad_dht.kad_dht", "KadDHT"),
        ("libp2p.kad_dht", "KadDHT"),
    ]
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            # KadDHT requires a DHTMode in this libp2p build.
            try:
                from libp2p.kad_dht.kad_dht import DHTMode  # type: ignore

                dht = cls(host, DHTMode.CLIENT)
            except Exception:
                dht = cls(host)

            # Keep the DHT background loop alive while we query it.
            import anyio
            import trio
            from libp2p.tools.async_service.trio_service import background_trio_service

            async def _run_dht_service() -> None:
                async with background_trio_service(dht):
                    await trio.sleep_forever()

            async with anyio.create_task_group() as tg:
                tg.start_soon(_run_dht_service)

                if require_peer_id:
                    find_peer = getattr(dht, "find_peer", None)
                    if not callable(find_peer):
                        tg.cancel_scope.cancel()
                        continue
                    peer_info = await find_peer(require_peer_id)
                    if not peer_info:
                        tg.cancel_scope.cancel()
                        continue
                    await host.connect(peer_info)
                    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                    try:
                        resp = await _request_over_stream(stream=stream, message=message)
                        return resp if isinstance(resp, dict) else None
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass
                else:
                    # Namespace-based provider discovery: ask DHT for providers
                    # and try them in sequence.
                    find_providers = getattr(dht, "find_providers", None)
                    if not callable(find_providers):
                        tg.cancel_scope.cancel()
                        continue
                    providers = await find_providers(ns, 20)
                    for peer_info in list(providers or []):
                        try:
                            await host.connect(peer_info)
                            stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                            try:
                                resp = await _request_over_stream(stream=stream, message=message)
                                if isinstance(resp, dict):
                                    return resp
                            finally:
                                try:
                                    await stream.close()
                                except Exception:
                                    pass
                        except Exception:
                            continue

                tg.cancel_scope.cancel()
        except Exception:
            continue
    return None


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
                require_peer_id = (remote.peer_id or "").strip()

                # Zero-config: if a local service is running, it writes an
                # announce file in XDG cache; dial it first.
                ann = await _dial_via_announce_file(host=host, message=message, require_peer_id=require_peer_id)
                if isinstance(ann, dict):
                    resp = ann
                else:
                    # Then try cross-subnet mechanisms, and finally LAN mDNS.
                    boot = await _dial_via_bootstrap(host=host, message=message)
                    if isinstance(boot, dict):
                        resp = boot
                    else:
                        rv = await _dial_via_rendezvous(host=host, message=message, require_peer_id=require_peer_id)
                        if isinstance(rv, dict):
                            resp = rv
                        else:
                            dht = await _dial_via_dht(host=host, message=message, require_peer_id=require_peer_id)
                            if isinstance(dht, dict):
                                resp = dht
                            else:
                                resp = await _dial_via_mdns(host=host, message=message, require_peer_id=require_peer_id)

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


def _maybe_str_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        try:
            out[str(k)] = str(v)
        except Exception:
            continue
    return out


def _maybe_str_dict2(value: Any) -> Dict[str, str]:
    # Alias for readability in docker helpers.
    return _maybe_str_dict(value)


def _normalize_cmd(value: Any) -> Any:
    # Preserve as list[str] if provided, else accept str.
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x for x in value if x]
    if isinstance(value, str):
        return value
    return value


async def submit_docker_hub_task(
    *,
    remote: RemoteQueue,
    image: str,
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    volumes: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.execute",
    **kwargs: Any,
) -> str:
    """Submit a Docker Hub container run to the TaskQueue.

    This only submits the task; execution depends on workers enabling Docker via
    `IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1`.
    """

    payload: Dict[str, Any] = {
        "image": str(image),
    }
    if command is not None:
        payload["command"] = _normalize_cmd(command)
    if entrypoint is not None:
        payload["entrypoint"] = _normalize_cmd(entrypoint)
    if environment is not None:
        payload["environment"] = _maybe_str_dict(environment)
    if volumes is not None:
        payload["volumes"] = _maybe_str_dict2(volumes)

    # Pass through common docker execution settings (memory_limit, cpu_limit, timeout,
    # network_mode, working_dir, read_only, no_new_privileges, user, stream_output, etc.)
    for k, v in kwargs.items():
        if v is None:
            continue
        payload[str(k)] = v

    return await submit_task(remote=remote, task_type=str(task_type), model_name=str(model_name), payload=payload)


def submit_docker_hub_task_sync(
    *,
    remote: RemoteQueue,
    image: str,
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    volumes: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.execute",
    **kwargs: Any,
) -> str:
    import anyio

    async def _do() -> str:
        return await submit_docker_hub_task(
            remote=remote,
            image=image,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            volumes=volumes,
            model_name=model_name,
            task_type=task_type,
            **kwargs,
        )

    return anyio.run(_do, backend="trio")


async def submit_docker_github_task(
    *,
    remote: RemoteQueue,
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    context_path: str = ".",
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    build_args: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.github",
    **kwargs: Any,
) -> str:
    """Submit a GitHub repo build+run (Dockerfile) to the TaskQueue."""

    payload: Dict[str, Any] = {
        "repo_url": str(repo_url),
        "branch": str(branch),
        "dockerfile_path": str(dockerfile_path),
        "context_path": str(context_path),
    }
    if command is not None:
        payload["command"] = _normalize_cmd(command)
    if entrypoint is not None:
        payload["entrypoint"] = _normalize_cmd(entrypoint)
    if environment is not None:
        payload["environment"] = _maybe_str_dict(environment)
    if build_args is not None:
        payload["build_args"] = _maybe_str_dict(build_args)

    for k, v in kwargs.items():
        if v is None:
            continue
        payload[str(k)] = v

    return await submit_task(remote=remote, task_type=str(task_type), model_name=str(model_name), payload=payload)


def submit_docker_github_task_sync(
    *,
    remote: RemoteQueue,
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    context_path: str = ".",
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    build_args: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.github",
    **kwargs: Any,
) -> str:
    import anyio

    async def _do() -> str:
        return await submit_docker_github_task(
            remote=remote,
            repo_url=repo_url,
            branch=branch,
            dockerfile_path=dockerfile_path,
            context_path=context_path,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            build_args=build_args,
            model_name=model_name,
            task_type=task_type,
            **kwargs,
        )

    return anyio.run(_do, backend="trio")


def submit_task_sync(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> str:
    import anyio

    async def _do() -> str:
        return await submit_task(remote=remote, task_type=task_type, model_name=model_name, payload=payload)

    return anyio.run(_do, backend="trio")


async def submit_task_with_info(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "submit", "task_type": task_type, "model_name": model_name, "payload": payload},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit failed: {resp}")
    return {"task_id": str(resp.get("task_id")), "peer_id": str(resp.get("peer_id") or "").strip()}


def submit_task_with_info_sync(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    import anyio

    async def _do() -> Dict[str, str]:
        return await submit_task_with_info(remote=remote, task_type=task_type, model_name=model_name, payload=payload)

    return anyio.run(_do, backend="trio")


async def claim_next(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    resp = await _dial_and_request(
        remote=remote,
        message={
            "op": "claim",
            "worker_id": str(worker_id),
            "supported_task_types": list(supported_task_types or []),
            "peer_id": str(peer_id) if peer_id else "",
            "clock": clock,
        },
    )
    if not resp.get("ok"):
        raise RuntimeError(f"claim failed: {resp}")
    task = resp.get("task")
    return task if isinstance(task, dict) else None


def claim_next_sync(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    import anyio

    async def _do() -> Optional[Dict[str, Any]]:
        return await claim_next(remote=remote, worker_id=worker_id, supported_task_types=supported_task_types, peer_id=peer_id, clock=clock)

    return anyio.run(_do, backend="trio")


async def heartbeat(*, remote: RemoteQueue, peer_id: str, clock: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "peer.heartbeat", "peer_id": str(peer_id), "clock": clock},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"heartbeat failed: {resp}")
    return resp


def heartbeat_sync(*, remote: RemoteQueue, peer_id: str, clock: Dict[str, Any] | None = None) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await heartbeat(remote=remote, peer_id=peer_id, clock=clock)

    return anyio.run(_do, backend="trio")


async def list_tasks(
    *,
    remote: RemoteQueue,
    status: str | None = None,
    limit: int = 50,
    task_types: list[str] | None = None,
) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={"op": "list", "status": status, "limit": int(limit), "task_types": list(task_types or [])},
    )
    if not resp.get("ok"):
        raise RuntimeError(f"list failed: {resp}")
    return resp


def list_tasks_sync(
    *,
    remote: RemoteQueue,
    status: str | None = None,
    limit: int = 50,
    task_types: list[str] | None = None,
) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await list_tasks(remote=remote, status=status, limit=limit, task_types=task_types)

    return anyio.run(_do, backend="trio")


async def complete_task(
    *,
    remote: RemoteQueue,
    task_id: str,
    status: str = "completed",
    result: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    resp = await _dial_and_request(
        remote=remote,
        message={
            "op": "complete",
            "task_id": str(task_id),
            "status": str(status),
            "result": result,
            "error": error,
        },
    )
    if not resp.get("ok"):
        raise RuntimeError(f"complete failed: {resp}")
    return resp


def complete_task_sync(
    *,
    remote: RemoteQueue,
    task_id: str,
    status: str = "completed",
    result: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await complete_task(remote=remote, task_id=task_id, status=status, result=result, error=error)

    return anyio.run(_do, backend="trio")


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
