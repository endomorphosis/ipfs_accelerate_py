"""libp2p RPC service for the TaskQueue.

This is a thin transport wrapper around the local DuckDB-backed TaskQueue.
It enables other peers to submit tasks and wait for results.

Environment:
- IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT (compat) / IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT
- IPFS_DATASETS_PY_TASK_P2P_TOKEN (compat) / IPFS_ACCELERATE_PY_TASK_P2P_TOKEN
- IPFS_DATASETS_PY_TASK_P2P_MDNS (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_MDNS
- IPFS_DATASETS_PY_TASK_P2P_DHT (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_DHT
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS (compat, default:
    ipfs-accelerate-task-queue) / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS
- IPFS_DATASETS_PY_TASK_P2P_AUTONAT (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT
- IPFS_DATASETS_PY_TASK_P2P_RELAY (compat, default: 1)
    / IPFS_ACCELERATE_PY_TASK_P2P_RELAY
        - Enables Circuit Relay v2 protocol handlers
            (/libp2p/circuit/relay/2.0.0)
- IPFS_DATASETS_PY_TASK_P2P_RELAY_HOP (compat, default: 0) / IPFS_ACCELERATE_PY_TASK_P2P_RELAY_HOP
    - When enabled alongside *_P2P_RELAY, allow this node to act as a relay (HOP)
- IPFS_DATASETS_PY_TASK_P2P_HOLEPUNCH (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_HOLEPUNCH
        - Enables DCUtR hole punching protocol handler
            (/libp2p/dcutr)
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS (compat) / IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS
- IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP (compat) / IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP
    (for announce string; supports 'auto')
- IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE (compat) / IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE
    - unset: write announce JSON to a default XDG cache path
    - set to a path: write announce JSON to that path
    - set to 0/false/no: disable announce file writing

Protocol:
- /ipfs-datasets/task-queue/1.0.0
- Newline-delimited JSON request/response
"""

from __future__ import annotations

import functools
import ipaddress
import json
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .protocol import PROTOCOL_V1, auth_ok, get_shared_token
from .task_queue import TaskQueue
from .cache_store import DiskTTLCache, cache_enabled as _cache_enabled, default_cache_dir


_SERVICE_STATE_LOCK = threading.RLock()
_SERVICE_STATE: Dict[str, Any] = {
    "running": False,
    "peer_id": "",
    "listen_port": None,
    "started_at": 0.0,
}

_KNOWN_PEERS_LOCK = threading.RLock()
_KNOWN_PEERS: Dict[str, Dict[str, object]] = {}


def _service_mark_running(*, peer_id: str, listen_port: Optional[int]) -> None:
    with _SERVICE_STATE_LOCK:
        _SERVICE_STATE["running"] = True
        _SERVICE_STATE["peer_id"] = str(peer_id or "")
        _SERVICE_STATE["listen_port"] = int(listen_port) if listen_port is not None else None
        _SERVICE_STATE["started_at"] = float(time.time())


def _service_mark_stopped() -> None:
    with _SERVICE_STATE_LOCK:
        _SERVICE_STATE["running"] = False
        _SERVICE_STATE["peer_id"] = ""
        _SERVICE_STATE["listen_port"] = None
        _SERVICE_STATE["started_at"] = 0.0
    with _KNOWN_PEERS_LOCK:
        _KNOWN_PEERS.clear()


def get_local_service_state() -> Dict[str, Any]:
    """Return best-effort local TaskQueue p2p service state.

    Intended for in-process callers (e.g., MCP tools).
    """

    with _SERVICE_STATE_LOCK:
        return dict(_SERVICE_STATE)


def record_peer_seen(*, peer_id: str, info: Optional[Dict[str, object]] = None) -> None:
    pid = str(peer_id or "").strip()
    if not pid:
        return
    with _KNOWN_PEERS_LOCK:
        cur = _KNOWN_PEERS.get(pid) or {}
        cur["peer_id"] = pid
        cur["last_seen"] = float(time.time())
        if isinstance(info, dict):
            for k, v in info.items():
                if k in {"peer_id", "last_seen"}:
                    continue
                cur[str(k)] = v
        _KNOWN_PEERS[pid] = cur


def list_known_peers(*, alive_only: bool = True, limit: int = 200, exclude_peer_id: str = "") -> list[Dict[str, Any]]:
    """List peers recently seen by this TaskQueue p2p service."""

    now = time.time()
    peer_timeout_s = float(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_PEER_TIMEOUT_S")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_PEER_TIMEOUT_S")
        or 300.0
    )
    exclude = str(exclude_peer_id or "").strip()

    items: list[Dict[str, Any]] = []
    with _KNOWN_PEERS_LOCK:
        for pid, info in list(_KNOWN_PEERS.items()):
            if exclude and pid == exclude:
                continue
            try:
                last_seen = float(info.get("last_seen") or 0.0)
            except Exception:
                last_seen = 0.0
            age_s = max(0.0, now - last_seen)
            if alive_only and age_s > peer_timeout_s:
                continue
            row: Dict[str, Any] = {
                "peer_id": pid,
                "last_seen": last_seen,
                "age_s": age_s,
                "alive": age_s <= peer_timeout_s,
            }
            # Include any extra fields recorded.
            try:
                for k, v in info.items():
                    if k in row:
                        continue
                    row[str(k)] = v
            except Exception:
                pass
            items.append(row)

    items.sort(key=lambda x: float(x.get("last_seen") or 0.0), reverse=True)
    return items[: max(0, int(limit))]


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


def _default_announce_file() -> str:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json")


def _announce_file_path() -> str:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE")
    )
    text = str(raw).strip() if raw is not None else ""
    if text.lower() in {"0", "false", "no", "off"}:
        return ""
    if text:
        return text
    return _default_announce_file()


def _detect_outbound_ipv4() -> str:
    """Best-effort non-loopback IPv4 for dialing this host."""

    # Prefer the default route's source IP (UDP connect trick; no packets sent).
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = str(sock.getsockname()[0] or "").strip()
        finally:
            try:
                sock.close()
            except Exception:
                pass
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Fall back to hostname resolution.
    try:
        ip = str(socket.gethostbyname(socket.gethostname()) or "").strip()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass

    return "127.0.0.1"


def _dht_key_for_namespace(ns: str) -> str:
    # Newer py-libp2p KadDHT expects a string key for provide/find_providers and
    # put_value/get_value. Older builds may accept bytes; callers should
    # best-effort fall back when needed.
    return str(ns or "").strip()


def _dht_value_record_key(ns: str) -> str:
    # Deterministic DHT key for "namespace -> service multiaddr" record.
    # This supplements provider records (which can be flaky in tiny test nets).
    key = _dht_key_for_namespace(ns)
    return f"/ipfs-accelerate/task-queue/ns/{key}" if key else "/ipfs-accelerate/task-queue/ns"


async def _maybe_start_autonat(*, host) -> object | None:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_AUTONAT",
        compat="IPFS_DATASETS_PY_TASK_P2P_AUTONAT",
        default=True,
    ):
        return None

    # Best-effort: py-libp2p support varies by version.
    candidates = [
        # Older / alternate layouts.
        ("libp2p.nat.autonat", "AutoNAT"),
        ("libp2p.autonat", "AutoNAT"),
        # Current py-libp2p build in this environment.
        ("libp2p.host.autonat.autonat", "AutoNATService"),
    ]
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            svc = cls(host)
            # Some implementations are plain stream handlers rather than
            # AsyncService instances.
            if module_name == "libp2p.host.autonat.autonat":
                proto = getattr(mod, "AUTONAT_PROTOCOL_ID", None)
                handler = getattr(svc, "handle_stream", None)
                if proto is not None and callable(handler):
                    host.set_stream_handler(proto, handler)
            else:
                start = getattr(svc, "start", None)
                if callable(start):
                    maybe = start()
                    if hasattr(maybe, "__await__"):
                        await maybe

            print(
                f"ipfs_accelerate_py task queue p2p service: AutoNAT enabled ({module_name}.{symbol})",
                file=sys.stderr,
                flush=True,
            )
            return svc
        except Exception:
            continue
    print(
        "ipfs_accelerate_py task queue p2p service: AutoNAT unavailable in installed libp2p; skipping",
        file=sys.stderr,
        flush=True,
    )
    return None


def _autonat_status_dict(autonat: object | None) -> Dict[str, Any]:
    """Best-effort, JSON-friendly NAT status from whichever AutoNAT is available."""

    if autonat is None:
        return {"supported": False, "status": "unavailable", "code": None}

    code: int | None = None
    try:
        getter = getattr(autonat, "get_status", None)
        if callable(getter):
            code = int(getter())
        else:
            raw = getattr(autonat, "status", None)
            if raw is not None:
                code = int(raw)
    except Exception:
        code = None

    mapping = {0: "unknown", 1: "public", 2: "private"}
    return {"supported": True, "status": mapping.get(code, "unknown"), "code": code}


async def _maybe_make_relay_v2(*, host) -> object | None:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RELAY",
        compat="IPFS_DATASETS_PY_TASK_P2P_RELAY",
        default=True,
    ):
        return None

    allow_hop = _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RELAY_HOP",
        compat="IPFS_DATASETS_PY_TASK_P2P_RELAY_HOP",
        default=False,
    )

    try:
        from libp2p.relay.circuit_v2.protocol import CircuitV2Protocol

        svc = CircuitV2Protocol(host, allow_hop=bool(allow_hop))
        msg = f"ipfs_accelerate_py task queue p2p service: Circuit Relay v2 enabled (allow_hop={bool(allow_hop)})"
        print(msg, file=sys.stderr, flush=True)
        return svc
    except Exception as exc:
        print(
            f"ipfs_accelerate_py task queue p2p service: Circuit Relay v2 unavailable; skipping: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None


async def _maybe_make_dcutr(*, host) -> object | None:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_HOLEPUNCH",
        compat="IPFS_DATASETS_PY_TASK_P2P_HOLEPUNCH",
        default=True,
    ):
        return None

    try:
        from libp2p.relay.circuit_v2.dcutr import DCUtRProtocol

        svc = DCUtRProtocol(host)
        print(
            "ipfs_accelerate_py task queue p2p service: DCUtR (hole punching) enabled",
            file=sys.stderr,
            flush=True,
        )
        return svc
    except Exception as exc:
        msg = f"ipfs_accelerate_py task queue p2p service: DCUtR unavailable; skipping: {exc}"
        print(msg, file=sys.stderr, flush=True)
        return None


async def _maybe_start_dht(*, host, bootstrap_peers: list[str]) -> object | None:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_DHT",
        compat="IPFS_DATASETS_PY_TASK_P2P_DHT",
        default=True,
    ):
        return None

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

                dht = cls(host, DHTMode.SERVER)
            except Exception:
                dht = cls(host)

            # Newer py-libp2p uses a background `run()` coroutine (started by the caller).
            # Older versions may expose `start()` / `bootstrap()`.
            start = getattr(dht, "start", None)
            if callable(start):
                maybe = start()
                if hasattr(maybe, "__await__"):
                    await maybe
            bootstrap = getattr(dht, "bootstrap", None)
            if callable(bootstrap):
                maybe = bootstrap()
                if hasattr(maybe, "__await__"):
                    await maybe

            print(
                f"ipfs_accelerate_py task queue p2p service: DHT enabled ({module_name}.{symbol})",
                file=sys.stderr,
                flush=True,
            )
            if bootstrap_peers:
                print(
                    "ipfs_accelerate_py task queue p2p service: DHT bootstrap peers configured",
                    file=sys.stderr,
                    flush=True,
                )
            return dht
        except Exception:
            continue

    print(
        "ipfs_accelerate_py task queue p2p service: DHT unavailable in installed libp2p; skipping",
        file=sys.stderr,
        flush=True,
    )
    return None


async def _maybe_start_rendezvous(*, host, namespace: str) -> object | None:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS",
        default=True,
    ):
        return None

    ns = namespace.strip() or "ipfs-accelerate-task-queue"

    # Best-effort: rendezvous support varies by py-libp2p version.
    # Prefer client-side `register()` when a server-side service is not available.
    service_candidates = [
        ("libp2p.discovery.rendezvous.rendezvous", "RendezvousService"),
        ("libp2p.discovery.rendezvous", "RendezvousService"),
        ("libp2p.rendezvous", "RendezvousService"),
    ]
    for module_name, symbol in service_candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            svc = cls(host)
            start = getattr(svc, "start", None)
            if callable(start):
                maybe = start()
                if hasattr(maybe, "__await__"):
                    await maybe
            register = getattr(svc, "register", None)
            if callable(register):
                maybe = register(ns)
                if hasattr(maybe, "__await__"):
                    await maybe
            print(
                f"ipfs_accelerate_py task queue p2p service: rendezvous enabled (service, ns={ns})",
                file=sys.stderr,
                flush=True,
            )
            return svc
        except Exception:
            continue

    client_candidates = [
        ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
        ("libp2p.discovery.rendezvous", "RendezvousClient"),
        ("libp2p.rendezvous", "RendezvousClient"),
    ]
    for module_name, symbol in client_candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            cli = cls(host)
            register = getattr(cli, "register", None)
            if callable(register):
                await register(ns, ttl=7200)
                print(
                    f"ipfs_accelerate_py task queue p2p service: rendezvous enabled (client register, ns={ns})",
                    file=sys.stderr,
                    flush=True,
                )
                return cli
        except Exception:
            continue

    print(
        "ipfs_accelerate_py task queue p2p service: rendezvous unavailable in installed libp2p; skipping",
        file=sys.stderr,
        flush=True,
    )
    return None


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ServiceConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 9710


def _load_config() -> ServiceConfig:
    host = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_LISTEN_HOST")
        or "0.0.0.0"
    )
    port = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT") or os.environ.get(
        "IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT", "9710"
    )
    return ServiceConfig(listen_host=str(host).strip() or "0.0.0.0", listen_port=int(port))


def _parse_bootstrap_peers() -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return []

    default = [
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    ]
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return _expand_dnsaddr_peers(list(default))

    parts = [p.strip() for p in text.split(",")]
    return _expand_dnsaddr_peers([p for p in parts if p])


def _dnsaddr_resolution_enabled() -> bool:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DNSADDR_RESOLVE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DNSADDR_RESOLVE")
    )
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


@functools.lru_cache(maxsize=32)
def _resolve_dnsaddr_txt(hostname: str) -> list[str]:
    """Resolve dnsaddr TXT records for hostname.

    Returns multiaddrs (strings) extracted from TXT records of the form:
      dnsaddr=/ip4/.../tcp/.../p2p/<peerid>
    """
    if not hostname or not _dnsaddr_resolution_enabled():
        return []

    try:
        import dns.resolver  # type: ignore
    except Exception:
        return []

    qname = f"_dnsaddr.{hostname.strip().strip('.')}"
    try:
        resolver = dns.resolver.Resolver()
        # Keep startup responsive even on broken DNS.
        resolver.lifetime = 2.5
        answers = resolver.resolve(qname, "TXT")
    except Exception:
        return []

    out: list[str] = []
    for rdata in answers:
        try:
            # dnspython renders TXT with quotes, possibly split into chunks.
            txt = "".join(str(rdata).strip().strip('"').split('" "'))
        except Exception:
            continue
        txt = (txt or "").strip()
        if not txt.startswith("dnsaddr="):
            continue
        ma = txt[len("dnsaddr="):].strip()
        if ma:
            out.append(ma)
    return out


def _expand_dnsaddr_peers(peers: list[str]) -> list[str]:
    """Expand /dnsaddr/<host>/p2p/<peer> into concrete multiaddrs.

    Prefer TCP addresses since py-libp2p commonly runs over TCP.
    """
    if not peers or not _dnsaddr_resolution_enabled():
        return list(peers or [])

    def _expand_one(text: str, *, depth: int, seen_dns: set[str]) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return []
        if depth <= 0 or not text.startswith("/dnsaddr/"):
            return [text]

        if text in seen_dns:
            return [text]
        seen_dns.add(text)

        remainder = text[len("/dnsaddr/"):]
        host = remainder
        peer_id = ""
        if "/p2p/" in remainder:
            host, peer_id = remainder.split("/p2p/", 1)
            peer_id = peer_id.strip().strip("/")
        host = (host or "").strip().strip("/")

        candidates = _resolve_dnsaddr_txt(host)
        if peer_id:
            candidates = [ma for ma in candidates if f"/p2p/{peer_id}" in str(ma)]

        def _candidate_score(addr: str) -> tuple[int, int, int, int]:
            a = str(addr or "")
            has_tcp = 0 if "/tcp/" in a else 1
            is_ws = 1 if ("/ws" in a or "/wss" in a) else 0
            port_pref = 0 if "/tcp/4001" in a else 1
            if a.startswith("/ip4/"):
                net = 0
            elif a.startswith("/ip6/"):
                net = 1
            elif a.startswith("/dns4/"):
                net = 2
            elif a.startswith("/dns6/"):
                net = 3
            else:
                net = 4
            return (has_tcp, is_ws, port_pref, net)

        candidates = sorted([str(ma) for ma in candidates if str(ma).strip()], key=_candidate_score)

        out: list[str] = []
        if not candidates:
            return [text]
        for ma in candidates:
            out.extend(_expand_one(str(ma), depth=depth - 1, seen_dns=seen_dns))
        return out

    expanded: list[str] = []
    for peer_addr in peers:
        expanded.extend(_expand_one(str(peer_addr), depth=3, seen_dns=set()))

    seen: set[str] = set()
    out: list[str] = []
    for ma in expanded:
        if not ma or ma in seen:
            continue
        seen.add(ma)
        out.append(ma)
    return out


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

    cache_replicate_enabled = _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE",
        compat="IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE",
        default=False,
    )
    try:
        cache_replicate_max_peers = int(
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE_MAX_PEERS")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE_MAX_PEERS")
            or 6
        )
    except Exception:
        cache_replicate_max_peers = 6
    try:
        cache_replicate_timeout_s = float(
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE_TIMEOUT_S")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE_TIMEOUT_S")
            or 1.0
        )
    except Exception:
        cache_replicate_timeout_s = 1.0

    # Optional explicit replication targets (public-network safe).
    # When set, replication is attempted ONLY to these targets (plus any
    # peer IDs that are already known in the peerstore and match the list).
    # Supports entries like:
    #   - /ip4/203.0.113.10/tcp/9100/p2p/12D3Koo...
    #   - /dnsaddr/node.example.com/p2p/12D3Koo...
    #   - 12D3Koo... (only if already present in peerstore)
    cache_replicate_targets_raw = str(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE_TARGETS")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE_TARGETS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE_PEERS")
        or ""
    )
    cache_replicate_targets: list[str] = [
        s.strip() for s in cache_replicate_targets_raw.split(",") if str(s or "").strip()
    ]
    try:
        cache_replicate_max_message_bytes = int(
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE_MAX_MESSAGE_BYTES")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_REPLICATE_MAX_MESSAGE_BYTES")
            or 65536
        )
    except Exception:
        cache_replicate_max_message_bytes = 65536

    def _peer_id_text(p: object) -> str:
        try:
            pretty = getattr(p, "pretty", None)
            if callable(pretty):
                return str(pretty() or "").strip()
        except Exception:
            pass
        return str(p or "").strip()

    _cache_replicate_target_peerinfos: dict[str, object] | None = None

    def _get_cache_replicate_target_peerinfos() -> dict[str, object]:
        """Parse target multiaddrs into PeerInfo objects (lazy, best-effort)."""
        nonlocal _cache_replicate_target_peerinfos
        if _cache_replicate_target_peerinfos is not None:
            return _cache_replicate_target_peerinfos
        out: dict[str, object] = {}
        if not cache_replicate_targets:
            _cache_replicate_target_peerinfos = out
            return out
        try:
            from libp2p.peer.peerinfo import info_from_p2p_addr
        except Exception:
            _cache_replicate_target_peerinfos = out
            return out
        for entry in cache_replicate_targets:
            text = str(entry or "").strip()
            if not text or "/p2p/" not in text:
                continue
            try:
                pi = info_from_p2p_addr(Multiaddr(text))
                pid = getattr(pi, "peer_id", None)
                key = _peer_id_text(pid)
                if key:
                    out[key] = pi
            except Exception:
                continue
        _cache_replicate_target_peerinfos = out
        return out

    def _multiaddr_has_private_ipv4(addr: object) -> bool:
        try:
            text = str(addr or "")
        except Exception:
            return False
        if "/ip4/" not in text:
            return False
        try:
            ip_text = text.split("/ip4/", 1)[1].split("/", 1)[0].strip()
            if not ip_text:
                return False
            ip = ipaddress.ip_address(ip_text)
            return bool(getattr(ip, "is_private", False) or getattr(ip, "is_loopback", False))
        except Exception:
            return False

    def _candidate_cache_replication_peers() -> list[object]:
        if not cache_replicate_enabled:
            return []

        try:
            ps = host.get_peerstore()
        except Exception:
            return []

        # Public-network safe mode: explicit targets take priority.
        if cache_replicate_targets:
            out: list[object] = []

            # 1) Multiaddr targets (dialable)
            peerinfos = _get_cache_replicate_target_peerinfos()
            for pi in peerinfos.values():
                try:
                    pid = getattr(pi, "peer_id", None)
                    if pid is not None:
                        out.append(pid)
                except Exception:
                    continue

            # 2) Peer-id targets (only if already present in peerstore)
            target_texts = {str(t).strip() for t in cache_replicate_targets if "/p2p/" not in str(t or "")}
            if target_texts:
                try:
                    for pid in list(ps.peer_ids()):  # type: ignore[attr-defined]
                        if _peer_id_text(pid) in target_texts:
                            out.append(pid)
                except Exception:
                    pass

            # Deduplicate and apply ordering/limits.
            uniq: dict[str, object] = {}
            for pid in out:
                key = _peer_id_text(pid)
                if not key:
                    continue
                uniq.setdefault(key, pid)
            out2 = list(uniq.values())
            out2.sort(key=_peer_id_text)
            return out2[: max(0, int(cache_replicate_max_peers))]

        try:
            peer_ids = list(ps.peer_ids())  # type: ignore[attr-defined]
        except Exception:
            return []

        me = None
        try:
            me = host.get_id()
        except Exception:
            me = None

        out: list[object] = []
        for pid in peer_ids:
            try:
                if me is not None and pid == me:
                    continue
            except Exception:
                pass
            try:
                addrs = list(ps.addrs(pid))  # type: ignore[attr-defined]
            except Exception:
                addrs = []
            if not addrs:
                continue
            if not any(_multiaddr_has_private_ipv4(a) for a in addrs):
                continue
            out.append(pid)

        # Best-effort stability: deterministic order for tests/logging.
        out.sort(key=_peer_id_text)
        return out[: max(0, int(cache_replicate_max_peers))]

    async def _replicate_cache_message(message: dict) -> None:
        if not cache_replicate_enabled:
            return
        # Never re-replicate a forwarded message.
        if message.get("replicate") is False:
            return
        # Forwarded messages must not replicate further.
        fwd = dict(message)
        fwd["replicate"] = False
        token = get_shared_token()
        if token:
            fwd["token"] = token
        payload = (json.dumps(fwd, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            max_b = int(cache_replicate_max_message_bytes)
        except Exception:
            max_b = 0
        if max_b > 0 and len(payload) > max_b:
            return

        peers = _candidate_cache_replication_peers()
        if not peers:
            return

        peerinfos = _get_cache_replicate_target_peerinfos() if cache_replicate_targets else {}
        for pid in peers:
            try:
                with anyio.fail_after(float(cache_replicate_timeout_s)):
                    # If this peer is an explicit multiaddr target, make a
                    # best-effort connect first so its addrs land in peerstore.
                    try:
                        pi = peerinfos.get(_peer_id_text(pid))
                        if pi is not None:
                            await host.connect(pi)
                    except Exception:
                        pass
                    stream2 = await host.new_stream(pid, [PROTOCOL_V1])
                    try:
                        await stream2.write(payload)
                        # Drain one line best-effort so the remote handler can finish.
                        try:
                            await stream2.read(4096)
                        except Exception:
                            pass
                    finally:
                        try:
                            await stream2.close()
                        except Exception:
                            pass
            except Exception:
                continue

    # Deterministic scheduling state (best-effort, in-memory). This enables a
    # swarm to claim from a *global* queue deterministically when all peers talk
    # to the same TaskQueue service.
    from .deterministic_scheduler import MerkleClock as _MerkleClock
    from .deterministic_scheduler import select_owner_peer as _select_owner_peer
    from .deterministic_scheduler import task_hash as _task_hash

    deterministic_enabled = str(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DETERMINISTIC")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DETERMINISTIC")
        or "1"
    ).strip().lower() not in {"0", "false", "no", "off"}

    peer_timeout_s = float(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_PEER_TIMEOUT_S")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_PEER_TIMEOUT_S")
        or 300.0
    )

    sched_clock = _MerkleClock(node_id="taskqueue-service")
    known_peers: dict[str, dict[str, object]] = {}

    def _update_peer_state(peer: str, clock_dict: object | None = None, extra: object | None = None) -> None:
        pid = str(peer or "").strip()
        if not pid:
            return
        info = known_peers.get(pid) or {}
        info["peer_id"] = pid
        info["last_seen"] = float(time.time())
        if isinstance(clock_dict, dict):
            try:
                other = _MerkleClock.from_dict(clock_dict)
                info["clock"] = other.to_dict()
                sched_clock.update(other)
            except Exception:
                pass
        if isinstance(extra, dict):
            try:
                for k, v in extra.items():
                    if k in {"peer_id", "last_seen", "clock"}:
                        continue
                    info[str(k)] = v
            except Exception:
                pass
        known_peers[pid] = info
        try:
            record_peer_seen(peer_id=pid, info={k: v for k, v in info.items() if k not in {"peer_id", "last_seen"}})
        except Exception:
            pass

    def _remote_peer_id_from_stream(stream: object) -> str:
        """Best-effort extraction of remote peer id from a py-libp2p stream."""

        candidates: list[tuple[str, ...]] = [
            ("muxed_conn", "peer_id"),
            ("conn", "peer_id"),
            ("_stream", "muxed_conn", "peer_id"),
            ("_stream", "conn", "peer_id"),
        ]
        for path in candidates:
            try:
                obj: object | None = stream
                for attr in path:
                    obj = getattr(obj, attr, None)
                if obj is None:
                    continue
                pretty = getattr(obj, "pretty", None)
                if callable(pretty):
                    text = str(pretty() or "").strip()
                else:
                    text = str(obj or "").strip()
                if text:
                    return text
            except Exception:
                continue
        return ""

    def _alive_peers(requesting_peer: str) -> list[str]:
        now = time.time()
        peers: list[str] = []
        for pid, info in list(known_peers.items()):
            try:
                last_seen = float(info.get("last_seen") or 0.0)
                if (now - last_seen) <= float(peer_timeout_s):
                    peers.append(pid)
            except Exception:
                continue
        req = str(requesting_peer or "").strip()
        if req and req not in peers:
            peers.append(req)
        return peers

    def _pick_task_for_peer(
        *,
        peer_id_hint: str,
        supported_types: list[str],
        session_id: str | None = None,
    ) -> dict | None:
        try:
            candidates = queue.list(status="queued", limit=200, task_types=supported_types or None)
        except Exception:
            return None

        session = str(session_id or "").strip()

        def _required_session(payload: object) -> str:
            if not isinstance(payload, dict):
                return ""
            for k in ("session_id", "session", "p2p_session"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def _sticky_worker(payload: object) -> str:
            if not isinstance(payload, dict):
                return ""
            for k in ("sticky_worker_id", "sticky_worker"):
                v = payload.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""

        def _prio_key(t: dict) -> tuple[int, float]:
            payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
            try:
                pr = int(payload.get("priority") or 5)
            except Exception:
                pr = 5
            pr = max(1, min(10, pr))
            try:
                created = float(t.get("created_at") or 0.0)
            except Exception:
                created = 0.0
            return (10 - pr, created)

        candidates.sort(key=_prio_key)

        peers = _alive_peers(peer_id_hint)
        clock_hash = sched_clock.get_hash()
        for t in candidates:
            try:
                tid = str(t.get("task_id") or "")
                ttype = str(t.get("task_type") or "")
                mname = str(t.get("model_name") or "")

                payload = t.get("payload") if isinstance(t.get("payload"), dict) else {}
                required = _required_session(payload)
                if required:
                    if not session:
                        continue
                    if required != session:
                        continue

                # Sticky affinity: if a task is pinned to a specific worker,
                # deterministically assign it to that worker regardless of the
                # normal hash-based peer ownership.
                sticky = _sticky_worker(payload)
                if sticky:
                    if sticky != peer_id_hint:
                        continue
                    return t

                th = _task_hash(task_id=tid, task_type=ttype, model_name=mname)
                owner = _select_owner_peer(peer_ids=peers, clock_hash=clock_hash, task_hash_hex=th)
                if owner and owner == peer_id_hint:
                    return t
            except Exception:
                continue
        return None

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    print("ipfs_accelerate_py task queue p2p service: creating host...", file=sys.stderr, flush=True)

    # libp2p's ResourceManager can enter a "graceful degradation" mode when it
    # believes connection limits are exhausted. In long-running processes with
    # background discovery (DHT/rendezvous/mDNS), repeated failed dials can
    # trigger this and effectively prevent new inbound dials.
    #
    # For the TaskQueue service we prefer availability over aggressive
    # self-throttling, and we keep limits comfortably high for LAN workloads.
    try:
        from libp2p.rcmgr import ResourceLimits, new_resource_manager

        max_conns = int(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MAX_CONNECTIONS") or 100_000)
        max_streams = int(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MAX_STREAMS") or 100_000)
        max_mem_mb = int(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MAX_MEMORY_MB") or 1024)

        rm = new_resource_manager(
            limits=ResourceLimits(
                max_connections=max(64, max_conns),
                max_memory_mb=max(128, max_mem_mb),
                max_streams=max(1024, max_streams),
            ),
            enable_metrics=False,
            enable_connection_pooling=False,
            enable_memory_pooling=False,
            enable_circuit_breaker=False,
            enable_graceful_degradation=False,
        )
    except Exception:
        rm = None

    host_obj = new_host(resource_manager=rm)
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj
    peer_id = host.get_id().pretty()
    _service_mark_running(peer_id=peer_id, listen_port=cfg.listen_port)
    print("ipfs_accelerate_py task queue p2p service: host created", file=sys.stderr, flush=True)

    # Initialized early so the stream handler can safely reference these even
    # if requests arrive before auxiliary services finish starting.
    autonat: object | None = None
    dht: object | None = None
    rendezvous: object | None = None
    relay_v2: object | None = None
    dcutr: object | None = None

    async def _handle(stream) -> None:
        """Handle a single request stream.

        Important: exceptions must not escape this handler, otherwise libp2p's
        swarm service can fail and stop listening.
        """
        try:
            from libp2p.network.stream.exceptions import StreamEOF
            from libp2p.stream_muxer.exceptions import MuxedStreamEOF

            raw = bytearray()
            max_bytes = 1024 * 1024

            async def _safe_write_json(obj: dict[str, Any]) -> None:
                try:
                    await stream.write(json.dumps(obj).encode("utf-8") + b"\n")
                except Exception:
                    pass
            while len(raw) < max_bytes:
                try:
                    chunk = await stream.read(1024)
                except (StreamEOF, MuxedStreamEOF):
                    return
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
                await _safe_write_json({"ok": False, "error": "invalid_json", "peer_id": peer_id})
                return

            if not isinstance(msg, dict):
                await _safe_write_json({"ok": False, "error": "invalid_message", "peer_id": peer_id})
                return

            if not auth_ok(msg):
                await _safe_write_json({"ok": False, "error": "unauthorized", "peer_id": peer_id})
                return

            op = (msg.get("op") or "").strip().lower()

            # Prefer the actual remote peer id from the transport over any
            # caller-provided hint.
            remote_peer_id = _remote_peer_id_from_stream(stream)

            if op == "submit":
                task_type = str(msg.get("task_type") or "text-generation")
                model_name = str(msg.get("model_name") or "")
                payload = msg.get("payload")
                if not isinstance(payload, dict):
                    payload = {"payload": payload}
                task_id = queue.submit(task_type=task_type, model_name=model_name, payload=payload)
                await _safe_write_json({"ok": True, "task_id": task_id, "peer_id": peer_id})
                return

            if op in {"claim", "claim_next"}:
                worker_id = str(msg.get("worker_id") or "").strip()
                if not worker_id:
                    await _safe_write_json({"ok": False, "error": "missing_worker_id", "peer_id": peer_id})
                    return

                session_id = str(msg.get("session_id") or msg.get("session") or msg.get("p2p_session") or "").strip()

                claimed_peer_id = str(msg.get("peer") or msg.get("peer_id") or "").strip()
                # For deterministic scheduling, use the caller-provided peer id
                # when present (supports multiple logical peers behind one
                # transport in tests). Still record the transport peer id.
                peer_ident = (claimed_peer_id or remote_peer_id or worker_id).strip()
                clock_dict = msg.get("clock") if isinstance(msg.get("clock"), dict) else None

                supported = msg.get("supported_task_types")
                if supported is None:
                    supported = msg.get("task_types")
                if isinstance(supported, str):
                    supported_list = [p.strip() for p in supported.split(",") if p.strip()]
                elif isinstance(supported, (list, tuple, set)):
                    supported_list = [str(t).strip() for t in supported if str(t).strip()]
                else:
                    supported_list = []

                _update_peer_state(
                    peer_ident,
                    clock_dict,
                    extra={
                        "worker_id": worker_id,
                        "peer_hint": claimed_peer_id,
                        "transport_peer_id": str(remote_peer_id or "").strip(),
                        "supported_task_types": list(supported_list),
                        "session_id": session_id,
                    },
                )

                try:
                    if deterministic_enabled:
                        picked = _pick_task_for_peer(
                            peer_id_hint=peer_ident,
                            supported_types=supported_list,
                            session_id=session_id or None,
                        )
                        if picked is None:
                            claimed = None
                        else:
                            claimed = queue.claim(
                                task_id=str(picked.get("task_id") or ""),
                                worker_id=worker_id,
                                session_id=session_id or None,
                            )
                    else:
                        claimed = queue.claim_next(
                            worker_id=worker_id,
                            supported_task_types=supported_list,
                            session_id=session_id or None,
                        )
                except Exception as exc:
                    await _safe_write_json({"ok": False, "error": str(exc), "peer_id": peer_id})
                    return

                if claimed is None:
                    await _safe_write_json({"ok": True, "task": None, "peer_id": peer_id})
                    return

                await stream.write(
                    json.dumps(
                        {
                            "ok": True,
                            "task": {
                                "task_id": claimed.task_id,
                                "task_type": claimed.task_type,
                                "model_name": claimed.model_name,
                                "payload": claimed.payload,
                                "created_at": claimed.created_at,
                                "status": claimed.status,
                                "assigned_worker": claimed.assigned_worker,
                            },
                            "peer_id": peer_id,
                        }
                    ).encode("utf-8")
                    + b"\n"
                )
                return

            if op in {"claim_many", "claim_batch"}:
                worker_id = str(msg.get("worker_id") or "").strip()
                if not worker_id:
                    await stream.write(
                        json.dumps({"ok": False, "error": "missing_worker_id", "peer_id": peer_id}).encode("utf-8")
                        + b"\n"
                    )
                    return

                session_id = str(msg.get("session_id") or msg.get("session") or msg.get("p2p_session") or "").strip()

                claimed_peer_id = str(msg.get("peer") or msg.get("peer_id") or "").strip()
                peer_ident = (claimed_peer_id or remote_peer_id or worker_id).strip()
                clock_dict = msg.get("clock") if isinstance(msg.get("clock"), dict) else None

                supported = msg.get("supported_task_types")
                if supported is None:
                    supported = msg.get("task_types")
                if isinstance(supported, str):
                    supported_list = [p.strip() for p in supported.split(",") if p.strip()]
                elif isinstance(supported, (list, tuple, set)):
                    supported_list = [str(t).strip() for t in supported if str(t).strip()]
                else:
                    supported_list = []

                try:
                    max_tasks = int(msg.get("max_tasks") or msg.get("limit") or 1)
                except Exception:
                    max_tasks = 1
                same_type = msg.get("same_task_type")
                if same_type is None:
                    same_type = msg.get("same_type")
                same_type_bool = bool(same_type) if same_type is not None else True

                _update_peer_state(
                    peer_ident,
                    clock_dict,
                    extra={
                        "worker_id": worker_id,
                        "peer_hint": claimed_peer_id,
                        "transport_peer_id": str(remote_peer_id or "").strip(),
                        "supported_task_types": list(supported_list),
                        "session_id": session_id,
                    },
                )

                try:
                    if deterministic_enabled:
                        # Deterministic scheduling currently only selects one task.
                        picked = _pick_task_for_peer(
                            peer_id_hint=peer_ident,
                            supported_types=supported_list,
                            session_id=session_id or None,
                        )
                        if picked is None:
                            claimed_many = []
                        else:
                            one = queue.claim(
                                task_id=str(picked.get("task_id") or ""),
                                worker_id=worker_id,
                                session_id=session_id or None,
                            )
                            claimed_many = [one] if one is not None else []
                    else:
                        claimed_many = queue.claim_next_many(
                            worker_id=worker_id,
                            supported_task_types=supported_list,
                            max_tasks=int(max_tasks),
                            same_task_type=bool(same_type_bool),
                            session_id=session_id or None,
                        )
                except Exception as exc:
                    await stream.write(
                        json.dumps({"ok": False, "error": str(exc), "peer_id": peer_id}).encode("utf-8") + b"\n"
                    )
                    return

                tasks_out = []
                for claimed in list(claimed_many or []):
                    if claimed is None:
                        continue
                    tasks_out.append(
                        {
                            "task_id": claimed.task_id,
                            "task_type": claimed.task_type,
                            "model_name": claimed.model_name,
                            "payload": claimed.payload,
                            "created_at": claimed.created_at,
                            "status": claimed.status,
                            "assigned_worker": claimed.assigned_worker,
                        }
                    )

                await stream.write(
                    json.dumps({"ok": True, "tasks": tasks_out, "peer_id": peer_id}).encode("utf-8") + b"\n"
                )
                return

            if op in {"complete", "task.complete", "complete_task"}:
                task_id = str(msg.get("task_id") or "").strip()
                if not task_id:
                    await _safe_write_json({"ok": False, "error": "missing_task_id", "peer_id": peer_id})
                    return

                status = str(msg.get("status") or "completed").strip().lower()
                result = msg.get("result")
                if result is not None and not isinstance(result, dict):
                    result = {"result": result}
                error = msg.get("error")
                try:
                    ok = bool(
                        queue.complete(
                            task_id=task_id,
                            status=status,
                            result=result if isinstance(result, dict) else None,
                            error=str(error) if error else None,
                        )
                    )
                except Exception as exc:
                    await _safe_write_json({"ok": False, "error": str(exc), "peer_id": peer_id})
                    return

                await _safe_write_json({"ok": ok, "task_id": task_id, "peer_id": peer_id})
                return

            if op in {"release", "task.release", "release_task"}:
                task_id = str(msg.get("task_id") or "").strip()
                worker_id = str(msg.get("worker_id") or msg.get("assigned_worker") or "").strip()
                reason = msg.get("reason")
                if not task_id:
                    await _safe_write_json({"ok": False, "error": "missing_task_id", "peer_id": peer_id})
                    return
                if not worker_id:
                    await _safe_write_json({"ok": False, "error": "missing_worker_id", "peer_id": peer_id})
                    return
                try:
                    ok = bool(
                        queue.release(
                            task_id=task_id,
                            worker_id=worker_id,
                            reason=str(reason) if reason else None,
                        )
                    )
                except Exception as exc:
                    await _safe_write_json({"ok": False, "error": str(exc), "peer_id": peer_id})
                    return
                await _safe_write_json({"ok": ok, "task_id": task_id, "peer_id": peer_id})
                return

            if op in {"peer.heartbeat", "heartbeat", "peer"}:
                claimed_peer_id = str(msg.get("peer") or msg.get("peer_id") or "").strip()
                pid = (claimed_peer_id or remote_peer_id or "").strip()
                if not pid:
                    await _safe_write_json({"ok": False, "error": "missing_peer_id", "peer_id": peer_id})
                    return
                clock_dict = msg.get("clock") if isinstance(msg.get("clock"), dict) else None
                _update_peer_state(
                    pid,
                    clock_dict,
                    extra={
                        "peer_hint": claimed_peer_id,
                        "transport_peer_id": str(remote_peer_id or "").strip(),
                    },
                )
                await stream.write(
                    json.dumps(
                        {
                            "ok": True,
                            "peer": pid,
                            "scheduler": {
                                "clock": sched_clock.to_dict(),
                                "known_peers": [
                                    {
                                        "peer_id": str(info.get("peer_id") or ""),
                                        "last_seen": float(info.get("last_seen") or 0.0),
                                    }
                                    for info in known_peers.values()
                                ],
                            },
                            "peer_id": peer_id,
                        }
                    ).encode("utf-8")
                    + b"\n"
                )
                return

            if op in {"list", "tasks.list", "queue.list"}:
                status = msg.get("status")
                try:
                    limit = int(msg.get("limit") or 50)
                except Exception:
                    limit = 50
                task_types = msg.get("task_types")
                if isinstance(task_types, str):
                    types_list = [p.strip() for p in task_types.split(",") if p.strip()]
                elif isinstance(task_types, (list, tuple, set)):
                    types_list = [str(t).strip() for t in task_types if str(t).strip()]
                else:
                    types_list = None

                try:
                    status_filter = str(status).strip().lower() if status else None
                    tasks = queue.list(status=status_filter, limit=limit, task_types=types_list)
                except Exception as exc:
                    await _safe_write_json({"ok": False, "error": str(exc), "peer_id": peer_id})
                    return

                await _safe_write_json({"ok": True, "tasks": tasks, "peer_id": peer_id})
                return

            if op in {"cancel", "task.cancel"}:
                task_id = str(msg.get("task_id") or "").strip()
                reason = msg.get("reason")
                reason_text = str(reason).strip() if isinstance(reason, (str, int, float)) else None
                if not task_id:
                    await _safe_write_json({"ok": False, "error": "missing_task_id", "peer_id": peer_id})
                    return
                try:
                    ok = bool(queue.cancel(task_id=task_id, reason=reason_text))
                except Exception as exc:
                    await _safe_write_json({"ok": False, "error": str(exc), "peer_id": peer_id})
                    return
                await _safe_write_json({"ok": True, "cancelled": ok, "peer_id": peer_id})
                return

            if op in {"status", "capabilities", "describe"}:
                detail = bool(msg.get("detail"))
                caps = _accelerate_capabilities(accelerate_instance, detail=detail)
                nat = {
                    "autonat": _autonat_status_dict(autonat),
                }
                session = str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SESSION") or "").strip()

                resp: Dict[str, Any] = {
                    "ok": True,
                    "capabilities": caps,
                    "peer_id": peer_id,
                    "nat": nat,
                    "session": session,
                }
                if detail:
                    # Provide queue backlog counts for remote autoscalers.
                    # Keep this under detail=True because it hits DuckDB.
                    try:
                        queued_by_type = queue.counts_by_task_type(status="queued")
                    except Exception:
                        queued_by_type = {}
                    try:
                        running_by_type = queue.counts_by_task_type(status="running")
                    except Exception:
                        running_by_type = {}
                    try:
                        queued_total = int(queue.count(status="queued"))
                    except Exception:
                        queued_total = 0
                    try:
                        running_total = int(queue.count(status="running"))
                    except Exception:
                        running_total = 0

                    resp["queue"] = {
                        "queued": int(queued_total),
                        "running": int(running_total),
                        "queued_by_type": queued_by_type if isinstance(queued_by_type, dict) else {},
                        "running_by_type": running_by_type if isinstance(running_by_type, dict) else {},
                    }

                    def _truthy_env(*names: str) -> bool:
                        for n in names:
                            v = os.environ.get(n)
                            if v is None:
                                continue
                            if str(v).strip().lower() in {"1", "true", "yes", "y", "on"}:
                                return True
                            if str(v).strip().lower() in {"0", "false", "no", "off"}:
                                return False
                        return False

                    def _docker_enabled_env() -> bool:
                        # Respect explicit config when set.
                        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER")
                        raw2 = os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_ENABLE_DOCKER")
                        if raw is not None or raw2 is not None:
                            return _truthy_env(
                                "IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER",
                                "IPFS_DATASETS_PY_TASK_WORKER_ENABLE_DOCKER",
                            )

                        # Otherwise mirror the worker behavior: auto-enable when
                        # the Docker daemon appears reachable.
                        try:
                            from ipfs_accelerate_py.docker_executor import docker_daemon_available

                            return bool(docker_daemon_available())
                        except Exception:
                            return False

                    def _supported_task_types_env() -> list[str]:
                        raw = (
                            os.environ.get("IPFS_ACCELERATE_PY_TASK_WORKER_TASK_TYPES")
                            or os.environ.get("IPFS_DATASETS_PY_TASK_WORKER_TASK_TYPES")
                            or ""
                        )
                        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
                        if parts:
                            return parts
                        base = ["text-generation"]
                        if _docker_enabled_env():
                            base.extend(
                                [
                                    "docker.execute",
                                    "docker.execute_docker_container",
                                    "docker.github",
                                ]
                            )
                        return base

                    now = time.time()
                    peer_rows: list[dict[str, Any]] = []
                    worker_rows: list[dict[str, Any]] = []
                    docker_worker_ids: set[str] = set()

                    docker_task_types = {
                        "docker.execute",
                        "docker.execute_docker_container",
                        "docker.github",
                    }

                    for _pid, info in list(known_peers.items()):
                        last_seen_raw = info.get("last_seen")
                        if isinstance(last_seen_raw, (int, float)):
                            last_seen = float(last_seen_raw)
                        elif isinstance(last_seen_raw, str):
                            try:
                                last_seen = float(last_seen_raw)
                            except Exception:
                                last_seen = 0.0
                        else:
                            last_seen = 0.0
                        if (now - last_seen) > float(peer_timeout_s):
                            continue

                        wid = str(info.get("worker_id") or "").strip()
                        supported = info.get("supported_task_types")
                        if isinstance(supported, str):
                            supported_list = [p.strip() for p in supported.split(",") if p.strip()]
                        elif isinstance(supported, (list, tuple, set)):
                            supported_list = [str(t).strip() for t in supported if str(t).strip()]
                        else:
                            supported_list = []

                        row = {
                            "peer_id": str(info.get("peer_id") or "").strip(),
                            "last_seen": float(last_seen),
                            "worker_id": wid,
                            "supported_task_types": supported_list,
                            "transport_peer_id": str(info.get("transport_peer_id") or "").strip(),
                            "peer_hint": str(info.get("peer_hint") or "").strip(),
                        }
                        peer_rows.append(row)
                        if wid:
                            worker_rows.append(row)
                            if any(t in docker_task_types for t in supported_list):
                                docker_worker_ids.add(wid)

                    resp["scheduler"] = {
                        "clock": sched_clock.to_dict(),
                        "known_peers": peer_rows,
                        "known_workers": worker_rows,
                        "counts": {
                            "peers": int(len(peer_rows)),
                            "workers": int(len(worker_rows)),
                            "docker_workers": int(len(docker_worker_ids)),
                        },
                    }

                    # Also surface local (in-process) worker configuration when
                    # MCP started a task worker thread. This does not prove the
                    # worker is actively processing, but it explains why a peer
                    # may accept tasks yet never claim a given task_type.
                    local_worker_enabled = _truthy_env(
                        "IPFS_ACCELERATE_PY_MCP_P2P_TASK_WORKER",
                        "IPFS_ACCELERATE_PY_TASK_WORKER",
                        "IPFS_DATASETS_PY_TASK_WORKER",
                    )
                    local_supported = _supported_task_types_env()
                    local_docker_enabled = _docker_enabled_env()
                    resp["local_worker"] = {
                        "enabled": bool(local_worker_enabled),
                        "docker_enabled": bool(local_docker_enabled),
                        "supported_task_types": list(local_supported or []),
                    }

                    try:
                        if local_worker_enabled and any(t.startswith("docker.") for t in (local_supported or [])):
                            resp["scheduler"]["counts"]["docker_workers_configured"] = 1
                        else:
                            resp["scheduler"]["counts"]["docker_workers_configured"] = 0
                    except Exception:
                        resp["scheduler"]["counts"]["docker_workers_configured"] = 0

                await stream.write(
                    json.dumps(resp).encode("utf-8") + b"\n"
                )
                return

            if op in {"tool", "call_tool", "tool.call"}:
                allow = (
                    str(os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS", "")).lower()
                    in {"1", "true", "yes"}
                )
                if not allow:
                    await stream.write(
                        json.dumps({"ok": False, "error": "tools_disabled", "peer_id": peer_id}).encode("utf-8")
                        + b"\n"
                    )
                    return

                tool_name = str(msg.get("tool") or msg.get("tool_name") or msg.get("name") or "").strip()
                args = msg.get("args") or msg.get("arguments") or msg.get("params") or {}
                if not isinstance(args, dict):
                    args = {"value": args}

                if not tool_name:
                    await stream.write(
                        json.dumps({"ok": False, "error": "missing_tool_name", "peer_id": peer_id}).encode(
                            "utf-8"
                        )
                        + b"\n"
                    )
                    return

                timeout_s = float(msg.get("timeout_s") or 30.0)

                try:
                    # Metadata-driven routing: tools explicitly tagged as
                    # execution_context='worker' must run in thin workers.
                    # Unknown/unannotated tools default to worker for safety.
                    from ipfs_accelerate_py.tool_manifest import tool_execution_context
                    from ipfs_accelerate_py.mcp.server import get_mcp_server_instance

                    mcp_like = get_mcp_server_instance()
                    ctx = tool_execution_context(mcp_like, tool_name=tool_name) if mcp_like is not None else None
                    must_run_in_worker = (ctx != "server")

                    if must_run_in_worker:
                        # Enqueue as a task so execution happens in workers.
                        from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

                        q = TaskQueue(queue_path)
                        payload = {
                            "tool": str(tool_name),
                            "args": args if isinstance(args, dict) else {},
                            "_origin": "p2p.call_tool",
                            "_peer_id": str(peer_id),
                        }
                        tid = q.submit(task_type="tool.call", model_name="mcp.tool", payload=payload)

                        # Wait for completion and return result inline.
                        deadline = time.time() + max(0.0, float(timeout_s))
                        task = q.get(tid)
                        while (
                            task is not None
                            and task.get("status") in {"queued", "running"}
                            and time.time() < deadline
                        ):
                            await anyio.sleep(0.1)
                            task = q.get(tid)

                        if task is None:
                            resp = {
                                "ok": False,
                                "tool": tool_name,
                                "error": "task_missing",
                                "peer_id": peer_id,
                            }
                        else:
                            st = str(task.get("status") or "")
                            if st == "completed":
                                # worker returns {tool,result} shape; normalize.
                                result = task.get("result")
                                if isinstance(result, dict) and "result" in result and "tool" in result:
                                    resp = {
                                        "ok": True,
                                        "tool": str(result.get("tool") or tool_name),
                                        "result": result.get("result"),
                                        "peer_id": peer_id,
                                    }
                                else:
                                    resp = {"ok": True, "tool": tool_name, "result": result, "peer_id": peer_id}
                            elif st in {"failed", "cancelled"}:
                                resp = {
                                    "ok": False,
                                    "tool": tool_name,
                                    "error": str(task.get("error") or st),
                                    "peer_id": peer_id,
                                }
                            else:
                                resp = {"ok": False, "tool": tool_name, "error": "timeout", "peer_id": peer_id}

                        # Ephemeral: delete the internal task row now that we returned.
                        try:
                            q.delete(task_id=tid)
                        except Exception:
                            pass

                        await _safe_write_json(resp)
                        return

                    # Control-plane: invoke tool locally.
                    from ipfs_accelerate_py.tool_manifest import invoke_mcp_tool

                    if mcp_like is None:
                        resp = {"ok": False, "tool": tool_name, "error": "mcp_registry_unavailable"}
                        resp.setdefault("peer_id", peer_id)
                        await _safe_write_json(resp)
                        return
                    resp = await invoke_mcp_tool(
                        mcp_like,
                        tool_name=tool_name,
                        args=args,
                        accelerate_instance=accelerate_instance,
                    )
                    if not isinstance(resp, dict):
                        resp = {"ok": True, "tool": tool_name, "result": resp}
                    resp.setdefault("peer_id", peer_id)
                    await _safe_write_json(resp)
                    return
                except Exception as exc:
                    await _safe_write_json(
                        {"ok": False, "tool": tool_name, "error": str(exc), "peer_id": peer_id}
                    )
                    return

            if op in {"cache.get", "cache_get", "cache"}:
                if not _cache_enabled():
                    await _safe_write_json({"ok": False, "error": "cache_disabled", "peer_id": peer_id})
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await _safe_write_json({"ok": False, "error": "missing_key", "peer_id": peer_id})
                    return

                value = cache_store.get(key)
                await _safe_write_json(
                    {
                        "ok": True,
                        "key": key,
                        "hit": value is not None,
                        "value": _jsonable(value),
                        "peer_id": peer_id,
                    }
                )
                return

            if op in {"cache.has", "cache_has"}:
                if not _cache_enabled():
                    await _safe_write_json({"ok": False, "error": "cache_disabled", "peer_id": peer_id})
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await _safe_write_json({"ok": False, "error": "missing_key", "peer_id": peer_id})
                    return

                hit = bool(cache_store.has(key))
                await _safe_write_json({"ok": True, "key": key, "hit": hit, "peer_id": peer_id})
                return

            if op in {"cache.set", "cache_set"}:
                if not _cache_enabled():
                    await _safe_write_json({"ok": False, "error": "cache_disabled", "peer_id": peer_id})
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await _safe_write_json({"ok": False, "error": "missing_key", "peer_id": peer_id})
                    return

                value = msg.get("value")
                ttl_s = msg.get("ttl_s")
                try:
                    ttl_value = float(ttl_s) if ttl_s is not None else None
                except Exception:
                    ttl_value = None

                cache_store.set(key, value, ttl_s=ttl_value)
                # Best-effort write-through cache replication across LAN peers.
                # Opt-in via IPFS_ACCELERATE_PY_TASK_P2P_CACHE_REPLICATE=1.
                if cache_replicate_enabled and msg.get("replicate") is not False:
                    try:
                        await _replicate_cache_message(
                            {
                                "op": "cache.set",
                                "key": key,
                                "value": value,
                                "ttl_s": ttl_value,
                                "origin_peer_id": peer_id,
                            }
                        )
                    except Exception:
                        pass
                await _safe_write_json({"ok": True, "key": key, "peer_id": peer_id})
                return

            if op in {"cache.delete", "cache_del", "cache_delete"}:
                if not _cache_enabled():
                    await _safe_write_json({"ok": False, "error": "cache_disabled", "peer_id": peer_id})
                    return

                key = str(msg.get("key") or "").strip()
                if not key:
                    await _safe_write_json({"ok": False, "error": "missing_key", "peer_id": peer_id})
                    return

                deleted = bool(cache_store.delete(key))
                if cache_replicate_enabled and msg.get("replicate") is not False:
                    try:
                        await _replicate_cache_message(
                            {
                                "op": "cache.delete",
                                "key": key,
                                "origin_peer_id": peer_id,
                            }
                        )
                    except Exception:
                        pass
                await _safe_write_json({"ok": True, "key": key, "deleted": deleted, "peer_id": peer_id})
                return

            if op == "get":
                task_id = str(msg.get("task_id") or "")
                task = queue.get(task_id)
                await _safe_write_json({"ok": True, "task": task, "peer_id": peer_id})
                return

            if op == "wait":
                task_id = str(msg.get("task_id") or "")
                timeout_s = float(msg.get("timeout_s") or 60.0)
                deadline = time.time() + max(0.0, timeout_s)

                task = queue.get(task_id)
                while (
                    task is not None
                    and task.get("status") in {"queued", "running"}
                    and time.time() < deadline
                ):
                    await anyio.sleep(0.1)
                    task = queue.get(task_id)

                await _safe_write_json({"ok": True, "task": task, "peer_id": peer_id})
                return

            try:
                await _safe_write_json({"ok": False, "error": "unknown_op", "peer_id": peer_id})
            except Exception:
                pass
        except Exception as exc:
            # Best-effort: do not let handler exceptions take down the listener.
            try:
                await _safe_write_json(
                    {
                        "ok": False,
                        "error": "internal_error",
                        "detail": str(exc),
                        "peer_id": peer_id,
                    }
                )
            except Exception:
                pass
            try:
                print(
                    f"ipfs_accelerate_py task queue p2p service: stream handler error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                pass
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    host.set_stream_handler(PROTOCOL_V1, _handle)

    listen_addr = Multiaddr(f"/ip4/{cfg.listen_host}/tcp/{cfg.listen_port}")
    print(
        f"ipfs_accelerate_py task queue p2p service: listening on {listen_addr}",
        file=sys.stderr,
        flush=True,
    )

    mdns_enabled = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MDNS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MDNS", "1")
    ).strip().lower() not in {"0", "false", "no"}

    rendezvous_ns = _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(listen_addr)

        # Bootstrap connections (best-effort)
        try:
            from libp2p.peer.peerinfo import info_from_p2p_addr

            for peer_addr in _parse_bootstrap_peers():
                try:
                    peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
                    await host.connect(peer_info)
                    print(
                        f"ipfs_accelerate_py task queue p2p service: connected bootstrap {peer_addr}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"ipfs_accelerate_py task queue p2p service: bootstrap connect failed {peer_addr}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
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
                print(
                    f"ipfs_accelerate_py task queue p2p service: failed to start mDNS: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        try:
            autonat = await _maybe_start_autonat(host=host)
        except Exception as exc:
            print(
                f"ipfs_accelerate_py task queue p2p service: AutoNAT start failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            dht = await _maybe_start_dht(host=host, bootstrap_peers=_parse_bootstrap_peers())
        except Exception as exc:
            print(
                f"ipfs_accelerate_py task queue p2p service: DHT start failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            rendezvous = await _maybe_start_rendezvous(host=host, namespace=rendezvous_ns)
        except Exception as exc:
            print(
                f"ipfs_accelerate_py task queue p2p service: rendezvous start failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            relay_v2 = await _maybe_make_relay_v2(host=host)
        except Exception as exc:
            print(
                f"ipfs_accelerate_py task queue p2p service: relay start failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
        try:
            dcutr = await _maybe_make_dcutr(host=host)
        except Exception as exc:
            print(
                f"ipfs_accelerate_py task queue p2p service: holepunch start failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

        raw_public_ip = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_PUBLIC_IP")
        if raw_public_ip is None:
            raw_public_ip = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_PUBLIC_IP")
        public_ip = str(raw_public_ip or "").strip()
        if not public_ip or public_ip.lower() == "auto":
            # If the service is bound to a specific interface/IP, only announce
            # that address. Announcing an auto-detected LAN IP while bound to a
            # loopback-only host (common in multi-peer local tests) creates
            # undialable multiaddrs and causes confusing handshake failures.
            listen_host = str(cfg.listen_host or "").strip()
            try:
                if listen_host and listen_host not in {"0.0.0.0", "::"}:
                    ipaddress.ip_address(listen_host)
                    public_ip = listen_host
                else:
                    public_ip = _detect_outbound_ipv4()
            except Exception:
                public_ip = _detect_outbound_ipv4()
        announced = f"/ip4/{public_ip}/tcp/{cfg.listen_port}/p2p/{peer_id}"
        print("ipfs_accelerate_py task queue p2p service started", flush=True)
        print(f"peer_id={peer_id}", flush=True)
        print(f"multiaddr={announced}", flush=True)

        announce_file = _announce_file_path()
        if announce_file:
            try:
                os.makedirs(os.path.dirname(announce_file) or ".", exist_ok=True)
                with open(announce_file, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"peer_id": peer_id, "multiaddr": announced},
                            ensure_ascii=False,
                        )
                    )
                print(
                    f"ipfs_accelerate_py task queue p2p service: wrote announce file {announce_file}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"ipfs_accelerate_py task queue p2p service: failed to write announce file {announce_file}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        # Run long-lived background services (e.g., KadDHT.run) while the
        # service is alive.
        try:
            async with anyio.create_task_group() as tg:
                trio_mod = None
                try:
                    import trio as trio_mod  # type: ignore
                except Exception:
                    trio_mod = None

                # Start relay / holepunch protocol services when present.
                try:
                    from libp2p.tools.async_service.trio_service import background_trio_service

                    async def _sleep_forever() -> None:
                        if trio_mod is not None:
                            await trio_mod.sleep_forever()
                            return
                        while True:
                            await anyio.sleep(3600)

                    async def _run_trio_service(svc: object) -> None:
                        async with background_trio_service(svc):
                            await _sleep_forever()

                    if relay_v2 is not None:
                        tg.start_soon(_run_trio_service, relay_v2)
                    if dcutr is not None:
                        tg.start_soon(_run_trio_service, dcutr)
                except Exception:
                    pass

                # Start DHT background loop when present.
                try:
                    if dht is not None:
                        from libp2p.tools.async_service.trio_service import background_trio_service

                        async def _run_dht_service() -> None:
                            async with background_trio_service(dht):
                                await _sleep_forever()

                        tg.start_soon(_run_dht_service)
                        provide = getattr(dht, "provide", None)
                        put_value = getattr(dht, "put_value", None)
                        if callable(provide) or callable(put_value):
                            try:
                                ns_key = _dht_key_for_namespace(rendezvous_ns)
                                record_key = _dht_value_record_key(rendezvous_ns)

                                def _record_value_bytes() -> bytes:
                                    payload = {
                                        "peer_id": str(peer_id or ""),
                                        "multiaddr": str(announced or ""),
                                        "ts": float(time.time()),
                                    }
                                    return json.dumps(payload, ensure_ascii=False).encode("utf-8")

                                async def _publish_once() -> bool:
                                    ok_local = False

                                    if callable(provide):
                                        # Prefer string keys; fall back to bytes for older builds.
                                        try:
                                            await provide(ns_key)
                                            ok_local = True
                                        except Exception:
                                            try:
                                                await provide(ns_key.encode("utf-8"))
                                                ok_local = True
                                            except Exception:
                                                pass

                                    if callable(put_value):
                                        try:
                                            await put_value(record_key, _record_value_bytes())
                                            ok_local = True
                                        except Exception:
                                            # Some builds may accept bytes-like keys.
                                            try:
                                                await put_value(record_key.encode("utf-8"), _record_value_bytes())
                                                ok_local = True
                                            except Exception:
                                                pass

                                    return bool(ok_local)

                                # Give the DHT service a chance to start and
                                # populate routing state before advertising.
                                ok = False
                                for _ in range(20):
                                    ok = await _publish_once()
                                    if ok:
                                        break
                                    await anyio.sleep(0.2)

                                if ok:
                                    msg = (
                                        "ipfs_accelerate_py task queue p2p service: "
                                        f"DHT published namespace {rendezvous_ns}"
                                    )
                                    print(msg, file=sys.stderr, flush=True)

                                # Refresh periodically (helps public networks).
                                try:
                                    raw_interval = os.environ.get(
                                        "IPFS_ACCELERATE_PY_TASK_P2P_DHT_PROVIDE_INTERVAL_S"
                                    ) or os.environ.get(
                                        "IPFS_DATASETS_PY_TASK_P2P_DHT_PROVIDE_INTERVAL_S"
                                    )
                                    if raw_interval is not None:
                                        interval_s = float(str(raw_interval).strip())
                                    else:
                                        interval_s = 1800.0
                                except Exception:
                                    interval_s = 1800.0
                                if interval_s > 0:

                                    async def _publish_loop() -> None:
                                        while True:
                                            await anyio.sleep(float(interval_s))
                                            try:
                                                await _publish_once()
                                            except Exception:
                                                pass

                                    tg.start_soon(_publish_loop)
                            except Exception as exc:
                                print(
                                    f"ipfs_accelerate_py task queue p2p service: DHT publish failed: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                except Exception:
                    pass

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

            for svc in (rendezvous, dht, autonat, relay_v2, dcutr):
                try:
                    stop = getattr(svc, "stop", None)
                    if callable(stop):
                        maybe = stop()
                        if hasattr(maybe, "__await__"):
                            await maybe
                except Exception:
                    pass

            # Mark service stopped (best-effort) so in-process tooling can
            # detect the service lifecycle.
            try:
                _service_mark_stopped()
            except Exception:
                pass


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import anyio
    from .task_queue import default_queue_path

    parser = argparse.ArgumentParser(description="Run libp2p TaskQueue RPC service")
    parser.add_argument(
        "--queue",
        default=default_queue_path(),
        help="Path to task queue DuckDB file (default: env or shared cache path)",
    )
    parser.add_argument("--listen-port", type=int, default=None)

    args = parser.parse_args(argv)

    async def _main() -> None:
        await serve_task_queue(queue_path=args.queue, listen_port=args.listen_port)

    anyio.run(_main, backend="trio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
