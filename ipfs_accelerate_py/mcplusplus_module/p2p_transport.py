"""Profile E: libp2p+Trio P2P Transport for MCP++ protocol.

Implements the MCP++ Profile E specification using libp2p with Trio as the
async runtime. Provides peer-to-peer MCP tool invocation over the
`/mcp+p2p/1.0.0` protocol.

This module uses py-libp2p (trio-native) for:
- Peer identity (Ed25519 keypair → PeerId)
- Stream multiplexing (mplex or yamux)
- Transport (TCP + noise encryption)
- Discovery (mDNS for local, DHT for wide-area)
- Protocol handling (/mcp+p2p/1.0.0 stream handler)

Module: ipfs_accelerate_py.mcplusplus_module.p2p_transport
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ipfs_accelerate_py.mcplusplus_module.p2p.libp2p_runtime import (
    LIBP2P_INSTALL_HINT,
    PY_LIBP2P_MAIN_SPEC,
    create_libp2p_key_pair,
    ensure_libp2p_runtime,
    install_libp2p_runtime,
    install_libp2p_runtime_async,
    make_multiaddr,
    make_rendezvous_service,
    new_libp2p_host,
    peer_id_from_base58,
    peerinfo_from_multiaddr,
)
from ipfs_accelerate_py.mcplusplus_module.leanstral_topology import (
    LEANSTRAL_P2P_LISTEN_ADDR,
    LEANSTRAL_P2P_PORT,
)

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.p2p_transport")


def ensure_libp2p_installed() -> bool:
    """Auto-install libp2p from git if not already available (sync, for startup).

    Returns True if libp2p is importable after this call.
    """
    if ensure_libp2p_runtime():
        return True

    logger.info("libp2p not found — auto-installing from git (py-libp2p.git@main)...")
    if install_libp2p_runtime(quiet=True, timeout=120, upgrade=True):
        logger.info("libp2p installed successfully")
        return True
    return False


async def ensure_libp2p_installed_async() -> bool:
    """Trio-native async version of ensure_libp2p_installed (for use within Trio context)."""
    if ensure_libp2p_runtime():
        return True
    return await install_libp2p_runtime_async(quiet=True, timeout=120, upgrade=True)


# Protocol ID per MCP++ spec
MCP_P2P_PROTOCOL = "/mcp+p2p/1.0.0"

# Supported protocol versions (newest first for negotiation priority)
MCP_P2P_SUPPORTED_VERSIONS = ["/mcp+p2p/1.0.0"]

# Maximum P2P message size (16 MiB) — prevents allocation attacks via 4-byte length prefix
MAX_P2P_MESSAGE_SIZE = 16 * 1024 * 1024

# Maximum number of tracked peers (prevents unbounded memory growth)
MAX_PEERS = 500

# Default bootstrap peers for the MCP++ network
DEFAULT_BOOTSTRAP_PEERS = [
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
]

_VIRTUAL_INTERFACE_PREFIXES = (
    "br-",
    "cni",
    "docker",
    "flannel",
    "podman",
    "veth",
    "virbr",
)

_DISABLED_ENV_VALUES = frozenset({"0", "false", "no", "off"})


async def _bootstrap_dial_candidates(peer_addr: str) -> Tuple[str, ...]:
    """Resolve one configured dnsaddr peer to supported TCP descendants.

    The py-libp2p TCP transport cannot dial ``/dnsaddr`` directly.  The
    multiaddr package already implements the libp2p TXT-record algorithm, so
    preserve the configured dnsaddr as the public policy/receipt target while
    dialing its exact same-peer TCP descendants internally.
    """

    configured = str(peer_addr or "").strip()
    if not configured.startswith("/dnsaddr/"):
        return (configured,)
    try:
        from multiaddr import Multiaddr

        resolved = await Multiaddr(configured).resolve()
    except Exception:
        return (configured,)
    peer_suffix = f"/p2p/{configured.rsplit('/p2p/', 1)[-1]}" if "/p2p/" in configured else ""
    candidates = {
        str(value)
        for value in resolved
        if "/tcp/" in str(value)
        and "/ws" not in str(value)
        and "/wss" not in str(value)
        and (not peer_suffix or str(value).endswith(peer_suffix))
    }
    return tuple(
        sorted(
            candidates,
            key=lambda value: (
                0 if "/tcp/4001/" in value else 1,
                0 if value.startswith("/ip4/") else 1,
                value,
            ),
        )
    ) or (configured,)


@dataclass
class PeerInfo:
    """Information about a connected P2P peer."""

    peer_id: str
    multiaddrs: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    capabilities: List[str] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "multiaddrs": self.multiaddrs,
            "protocols": self.protocols,
            "last_seen": self.last_seen,
            "capabilities": self.capabilities,
            "latency_ms": self.latency_ms,
        }


@dataclass
class P2PMessage:
    """A message sent over the /mcp+p2p/1.0.0 protocol."""

    msg_type: str  # "request" | "response" | "notification"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = ""
    result: Any = None
    error: Optional[str] = None
    sender_peer_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def encode(self) -> bytes:
        """Encode message as length-prefixed JSON for wire transport."""
        payload = json.dumps(
            {
                "type": self.msg_type,
                "method": self.method,
                "params": self.params,
                "id": self.msg_id,
                "result": self.result,
                "error": self.error,
                "sender": self.sender_peer_id,
                "timestamp": self.timestamp,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        # 4-byte big-endian length prefix
        length = len(payload).to_bytes(4, "big")
        return length + payload

    @classmethod
    def decode(cls, data: bytes) -> "P2PMessage":
        """Decode a length-prefixed JSON message."""
        if len(data) < 4:
            raise ValueError("Message too short")
        length = int.from_bytes(data[:4], "big")
        if length > MAX_P2P_MESSAGE_SIZE:
            raise ValueError(f"Message size {length} exceeds limit {MAX_P2P_MESSAGE_SIZE}")
        if len(data) < 4 + length:
            raise ValueError(f"Incomplete message: expected {length} bytes, got {len(data) - 4}")
        payload = json.loads(data[4 : 4 + length].decode("utf-8"))
        return cls(
            msg_type=payload.get("type", "request"),
            method=payload.get("method", ""),
            params=payload.get("params", {}),
            msg_id=payload.get("id", ""),
            result=payload.get("result"),
            error=payload.get("error"),
            sender_peer_id=payload.get("sender", ""),
            timestamp=payload.get("timestamp", time.time()),
        )


class MCPp2pNode:
    """A libp2p node that speaks the /mcp+p2p/1.0.0 protocol.

    Uses Trio as the async runtime. Handles:
    - Creating a libp2p host with Ed25519 identity
    - Registering the /mcp+p2p/1.0.0 stream handler
    - Dialing peers and sending MCP requests
    - mDNS discovery for local peers
    - DHT bootstrap for wide-area discovery

    Usage with Trio::

        async with trio.open_nursery() as nursery:
            node = MCPp2pNode()
            await node.start(nursery)
            result = await node.call_tool(peer_id, "run_model", {"model": "bert"})
            await node.stop()
    """

    def __init__(
        self,
        listen_addrs: Optional[List[str]] = None,
        bootstrap_peers: Optional[List[str]] = None,
        advertise_addrs: Optional[List[str]] = None,
    ):
        env_listen = self._env_list("MCPPP_P2P_LISTEN_ADDRS")
        env_bootstrap = self._env_list("MCPPP_P2P_BOOTSTRAP_PEERS", preserve_empty=True)
        env_advertise = self._env_list("MCPPP_P2P_ADVERTISE_ADDRS")
        env_advertise_interfaces = self._env_list("MCPPP_P2P_ADVERTISE_INTERFACES")
        self._listen_addrs = (
            list(listen_addrs)
            if listen_addrs is not None
            else env_listen or [LEANSTRAL_P2P_LISTEN_ADDR]
        )
        self._bootstrap_peers = (
            list(bootstrap_peers)
            if bootstrap_peers is not None
            else env_bootstrap
            if env_bootstrap is not None
            else list(DEFAULT_BOOTSTRAP_PEERS)
        )
        self._advertise_addrs = (
            list(advertise_addrs) if advertise_addrs is not None else env_advertise or []
        )
        self._advertise_interfaces = env_advertise_interfaces or []
        self._host = None
        self._peers: Dict[str, PeerInfo] = {}
        self._tool_handler: Optional[Callable] = None
        self._started = False
        self._operational = False  # True only when libp2p is actually running
        self._concurrent_streams = 0
        self._max_concurrent_streams = 50  # Backpressure limit
        self._nursery = None
        self._host_stop_event = None
        self._host_stopped_event = None
        self._mdns_zeroconf = None
        self._mdns_service_info = None
        self._bootstrap_attempts: List[Dict[str, Any]] = []
        self._rendezvous_connectivity = None
        self._rendezvous_service = None
        self._rendezvous_attempts: List[Dict[str, Any]] = []

    @staticmethod
    def _env_list(name: str, preserve_empty: bool = False) -> Optional[List[str]]:
        """Read a comma-separated list while allowing an explicit empty value."""
        if name not in os.environ:
            return None
        values = [item.strip() for item in os.environ[name].split(",") if item.strip()]
        return values if values or preserve_empty else None

    @property
    def mdns_enabled(self) -> bool:
        """Whether local mDNS advertisement and discovery are enabled."""
        return os.environ.get("MCPPP_P2P_MDNS", "1").strip().casefold() not in _DISABLED_ENV_VALUES

    @property
    def peer_id(self) -> str:
        """Our peer ID."""
        if self._host:
            return str(self._host.get_id())
        return ""

    @property
    def multiaddrs(self) -> List[str]:
        """Our listening addresses."""
        if self._host:
            bound = [str(a) for a in self._host.get_addrs()]
            port = ""
            for address in bound:
                parts = address.split("/")
                if "tcp" in parts:
                    index = parts.index("tcp")
                    if index + 1 < len(parts):
                        port = parts[index + 1]
                        break
            addresses = self._advertise_addrs or bound
            if (
                not self._advertise_addrs
                and port
                and any(address.startswith("/ip4/0.0.0.0/") for address in bound)
            ):
                local_ips = self._local_ipv4_addresses(
                    allowed_interfaces=self._advertise_interfaces or None
                )
                if local_ips:
                    addresses = [f"/ip4/{address}/tcp/{port}" for address in local_ips]
            result = []
            for address in addresses:
                rendered = address.format(peer_id=self.peer_id, port=port)
                if "/p2p/" not in rendered:
                    rendered = f"{rendered.rstrip('/')}/p2p/{self.peer_id}"
                result.append(rendered)
            return result
        return []

    @staticmethod
    def _local_ipv4_addresses(
        allowed_interfaces: Optional[List[str]] = None,
    ) -> List[str]:
        """Return active, non-loopback IPv4 addresses allowed for advertisement.

        When ``MCPPP_P2P_ADVERTISE_INTERFACES`` is set, only those interfaces
        are considered.  Known container/overlay interface names are rejected
        even if present in the allow-list.  Operators in a container network
        namespace should use ``MCPPP_P2P_ADVERTISE_ADDRS`` to provide the
        independently dialable host addresses because the host interfaces are
        not visible from the container.
        """
        addresses = set()
        allowed = {
            str(interface).strip()
            for interface in (allowed_interfaces or [])
            if str(interface).strip()
        }
        try:
            import psutil

            stats = psutil.net_if_stats()
            for interface, entries in psutil.net_if_addrs().items():
                if interface in stats and not stats[interface].isup:
                    continue
                normalized = str(interface).casefold()
                if any(normalized.startswith(prefix) for prefix in _VIRTUAL_INTERFACE_PREFIXES):
                    continue
                if allowed and interface not in allowed:
                    continue
                for entry in entries:
                    if entry.family != socket.AF_INET:
                        continue
                    try:
                        import ipaddress

                        address = ipaddress.ip_address(entry.address)
                    except ValueError:
                        continue
                    if (
                        address.version == 4
                        and not address.is_loopback
                        and not address.is_unspecified
                        and not address.is_link_local
                        and not address.is_multicast
                    ):
                        addresses.add(str(address))
        except Exception:
            # A hostname lookup cannot honor an interface allow-list.  Fail
            # closed instead of advertising an unrelated address.
            if allowed:
                return []
            try:
                for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                    address = item[4][0]
                    if not address.startswith("127."):
                        addresses.add(address)
            except OSError:
                pass
        return sorted(addresses)

    @property
    def listen_port(self) -> Optional[int]:
        """Return the bound TCP port without confusing it with the HTTP port."""
        addresses = []
        if self._host:
            try:
                addresses = [str(address) for address in self._host.get_addrs()]
            except Exception:
                addresses = []
        addresses.extend(self._listen_addrs)
        for address in addresses:
            parts = str(address).split("/")
            try:
                index = parts.index("tcp")
                return int(parts[index + 1])
            except (ValueError, IndexError):
                continue
        return None

    @staticmethod
    def _load_or_create_key_pair():
        """Load a persistent secp256k1 identity when configured."""
        identity_file = os.environ.get("MCPPP_P2P_IDENTITY_FILE", "").strip()
        if not identity_file:
            return create_libp2p_key_pair()

        from pathlib import Path
        from libp2p.crypto.keys import KeyPair
        from libp2p.crypto.secp256k1 import Secp256k1PrivateKey

        path = Path(identity_file).expanduser()
        if path.is_file():
            private_key = Secp256k1PrivateKey.from_bytes(path.read_bytes())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            private_key = create_libp2p_key_pair().private_key
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(private_key.to_bytes())
            os.chmod(temporary, 0o600)
            temporary.replace(path)
        return KeyPair(private_key, private_key.get_public_key())

    @property
    def connected_peers(self) -> List[PeerInfo]:
        """Currently connected peers."""
        return list(self._peers.values())

    def _mount_rendezvous_service(self) -> bool:
        """Mount the rendezvous protocol on this exact service peer when requested."""

        mode = os.environ.get("MCPPP_P2P_RENDEZVOUS_SERVICE", "").strip().casefold()
        if mode != "same_as_service_peer" or self._host is None:
            self._rendezvous_service = None
            return False
        try:
            self._rendezvous_service = make_rendezvous_service(self._host)
        except Exception as exc:
            self._rendezvous_service = None
            logger.warning("Rendezvous service unavailable: %s", exc)
        return self._rendezvous_service is not None

    def set_tool_handler(self, handler: Callable) -> None:
        """Set the handler for incoming MCP tool calls.

        Handler signature: async def handler(method: str, params: dict) -> Any
        """
        self._tool_handler = handler

    async def start(self, nursery) -> None:
        """Start the libp2p node with Trio.

        Creates the host, starts listening, registers protocol handler,
        and initiates peer discovery.
        """
        if self._started:
            return
        self._nursery = nursery
        # Attempts belong to this process lifetime; never carry stale evidence
        # across a stopped/restarted node.
        self._bootstrap_attempts.clear()
        self._rendezvous_attempts.clear()

        # Auto-install libp2p if missing (run in thread to avoid blocking event loop)
        installed = await ensure_libp2p_installed_async()
        if not installed:
            logger.error(
                "libp2p could not be installed. P2P transport in stub mode. "
                "Manual install: pip install %r" % (PY_LIBP2P_MAIN_SPEC,)
            )
            self._started = True
            self._operational = False
            return

        try:
            # Generate identity through the canonical MCP++ runtime boundary.
            key_pair = self._load_or_create_key_pair()

            # Create libp2p host
            self._host = await new_libp2p_host(key_pair=key_pair)

            self._mount_rendezvous_service()

            # Register protocol handler for all supported versions
            for proto_version in MCP_P2P_SUPPORTED_VERSIONS:
                self._host.set_stream_handler(proto_version, self._handle_stream)

            # BasicHost.run starts the network background service before listen().
            # Calling network.listen() directly deadlocks while waiting for that
            # service's nursery to exist.
            listen_addrs = [make_multiaddr(addr) for addr in self._listen_addrs]
            await nursery.start(self._run_host, listen_addrs)

            self._started = True
            self._operational = True
            if self.mdns_enabled:
                await self._start_mdns_advertisement()
            logger.info("MCPp2pNode started: peer_id=%s, addrs=%s", self.peer_id, self.multiaddrs)

            # Bootstrap: connect to known peers (with timeout)
            for peer_addr in self._bootstrap_peers:
                nursery.start_soon(self._connect_bootstrap_with_timeout, peer_addr)
            rendezvous_peer = os.environ.get("IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER", "").strip()
            rendezvous_auto = os.environ.get(
                "MCPPP_P2P_RENDEZVOUS_AUTO", "1"
            ).strip().casefold() not in {"0", "false", "no", "off"}
            if rendezvous_peer and rendezvous_auto:
                nursery.start_soon(self._exercise_rendezvous_after_startup)

        except ImportError as e:
            logger.warning(
                "libp2p not available (%s). P2P transport running in stub mode. "
                "Install via MCP++ runtime: %s",
                e,
                LIBP2P_INSTALL_HINT,
            )
            self._started = True  # Mark as started in stub mode
            self._operational = False

    async def _run_host(self, listen_addrs, *, task_status) -> None:
        """Keep the libp2p host service alive for the enclosing nursery."""
        import trio

        self._host_stop_event = trio.Event()
        self._host_stopped_event = trio.Event()
        try:
            async with self._host.run(listen_addrs):
                task_status.started()
                await self._host_stop_event.wait()
        finally:
            self._operational = False
            self._host_stopped_event.set()

    async def stop(self) -> None:
        """Stop the libp2p node."""
        import trio

        await self._stop_mdns_advertisement()
        if self._host_stop_event is not None:
            self._host_stop_event.set()
        if self._host_stopped_event is not None:
            with trio.move_on_after(5.0):
                await self._host_stopped_event.wait()
        self._started = False
        self._operational = False
        self._peers.clear()
        self._rendezvous_service = None
        self._rendezvous_connectivity = None
        logger.info("MCPp2pNode stopped")

    async def _start_mdns_advertisement(self) -> None:
        """Advertise this MCP++ peer and all local addresses through Zeroconf."""
        import trio

        def register() -> None:
            from zeroconf import ServiceInfo, Zeroconf

            addresses = []
            port = 0
            for multiaddr in self.multiaddrs:
                parts = multiaddr.split("/")
                if "ip4" in parts and "tcp" in parts:
                    ip_index = parts.index("ip4")
                    tcp_index = parts.index("tcp")
                    addresses.append(socket.inet_aton(parts[ip_index + 1]))
                    port = int(parts[tcp_index + 1])
            if not addresses or not port:
                return
            service_type = "_mcp-accelerate._tcp.local."
            service_name = f"{self.peer_id}.{service_type}"
            info = ServiceInfo(
                service_type,
                service_name,
                addresses=addresses,
                port=port,
                properties={
                    b"peer_id": self.peer_id.encode(),
                    b"protocol": MCP_P2P_PROTOCOL.encode(),
                },
                server=f"{self.peer_id}.local.",
            )
            zeroconf = Zeroconf()
            zeroconf.register_service(info)
            self._mdns_zeroconf = zeroconf
            self._mdns_service_info = info

        await trio.to_thread.run_sync(register)

    async def _stop_mdns_advertisement(self) -> None:
        """Remove the Zeroconf advertisement without blocking Trio."""
        if self._mdns_zeroconf is None:
            return
        import trio

        zeroconf = self._mdns_zeroconf
        info = self._mdns_service_info
        self._mdns_zeroconf = None
        self._mdns_service_info = None

        def unregister() -> None:
            try:
                if info is not None:
                    zeroconf.unregister_service(info)
            finally:
                zeroconf.close()

        await trio.to_thread.run_sync(unregister)

    @staticmethod
    async def _read_exact(stream, length: int) -> bytes:
        """Read exactly *length* bytes from a libp2p stream."""
        chunks = []
        remaining = length
        while remaining:
            chunk = await stream.read(remaining)
            if not chunk:
                raise ConnectionError(f"Peer closed connection with {remaining} bytes remaining")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    async def _connect_bootstrap_with_timeout(self, peer_addr: str) -> None:
        """Connect to a bootstrap peer with a 10-second timeout."""
        import trio

        started = time.monotonic()
        success = False
        error = None
        with trio.move_on_after(10.0) as cancel_scope:
            success = await self._connect_bootstrap(peer_addr)
        if cancel_scope.cancelled_caught:
            error = "timeout"
            logger.debug(f"Bootstrap peer connection timed out: {peer_addr}")
        elif not success:
            error = "connect_failed"
        self._bootstrap_attempts.append(
            {
                "mechanism": "bootstrap",
                "target": peer_addr,
                "attempted": True,
                "success": bool(success and not cancel_scope.cancelled_caught),
                "timeout_s": 10.0,
                "duration_ms": (time.monotonic() - started) * 1000.0,
                "error": error,
                "observer_peer_id": self.peer_id,
                "namespace": "",
                "details": {},
            }
        )

    async def _connect_bootstrap(self, peer_addr: str) -> bool:
        """Connect to a bootstrap peer."""
        for dial_addr in await _bootstrap_dial_candidates(peer_addr):
            try:
                peer_info = peerinfo_from_multiaddr(dial_addr)
                if str(peer_info.peer_id) == self.peer_id:
                    logger.debug(
                        "Ignoring local peer bootstrap candidate: %s",
                        dial_addr,
                    )
                    continue
                await self._host.connect(peer_info)

                # Enforce max peers limit
                if len(self._peers) >= MAX_PEERS:
                    # Evict oldest peer
                    oldest = min(
                        self._peers.values(),
                        key=lambda p: p.last_seen,
                    )
                    self._peers.pop(oldest.peer_id, None)

                self._peers[str(peer_info.peer_id)] = PeerInfo(
                    peer_id=str(peer_info.peer_id),
                    multiaddrs=[dial_addr],
                    protocols=[MCP_P2P_PROTOCOL],
                )
                logger.info(
                    "Connected to bootstrap peer: %s",
                    peer_info.peer_id,
                )
                return True
            except Exception as exc:
                logger.debug(
                    "Failed to connect to bootstrap %s via %s: %s",
                    peer_addr,
                    dial_addr,
                    exc,
                )
        return False

    async def _exercise_rendezvous_after_startup(self) -> None:
        """Give bootstrap dialing a brief head start, then exercise rendezvous."""
        import trio

        await trio.sleep(0.25)
        await self.exercise_rendezvous()

    async def exercise_rendezvous(
        self,
        namespace: Optional[str] = None,
        *,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Run one bounded rendezvous registration/discovery exercise.

        The operation is opt-in through an exact
        ``IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER`` peer ID or multiaddr.  It never
        invokes an MCP tool or model inference.
        """
        import trio

        if not (0.0 < float(timeout) <= 10.0):
            raise ValueError("rendezvous timeout must be within (0, 10] seconds")

        target = os.environ.get("IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER", "").strip()
        rendezvous_namespace = (
            namespace
            or os.environ.get("IPFS_ACCELERATE_P2P_RENDEZVOUS_NS", "")
            or "leanstral-local"
        ).strip()
        receipt: Dict[str, Any] = {
            "mechanism": "rendezvous",
            "target": target,
            "namespace": rendezvous_namespace,
            "observer_peer_id": self.peer_id,
            "attempted": False,
            "success": False,
            "timeout_s": float(timeout),
            "duration_ms": 0.0,
            "error": None,
            "details": {
                "registered": False,
                "discovered_peers": [],
            },
        }
        started = time.monotonic()
        if not self._host:
            receipt["error"] = "p2p_node_not_operational"
            self._rendezvous_attempts.append(receipt)
            return dict(receipt)
        if not target:
            receipt["error"] = "rendezvous_peer_not_configured"
            self._rendezvous_attempts.append(receipt)
            return dict(receipt)
        target_peer_id = target.rsplit("/p2p/", 1)[-1] if "/p2p/" in target else target
        if target_peer_id == self.peer_id:
            receipt["error"] = "rendezvous_client_not_independent"
            self._rendezvous_attempts.append(receipt)
            return dict(receipt)

        receipt["attempted"] = True
        try:
            from ipfs_accelerate_py.mcplusplus_module.p2p.connectivity import (
                ConnectivityConfig,
                UniversalConnectivity,
            )

            connectivity = UniversalConnectivity(
                ConnectivityConfig(
                    enable_mdns=False,
                    enable_dht=False,
                    enable_relay=False,
                    enable_autonat=False,
                    enable_hole_punching=False,
                )
            )
            with trio.move_on_after(float(timeout)) as cancel_scope:
                if "/p2p/" in target:
                    await self._connect_bootstrap(target)
                await connectivity.configure_rendezvous(
                    self._host,
                    rendezvous_peer=target,
                )
                if connectivity.implemented.get("rendezvous"):
                    self._rendezvous_connectivity = connectivity
                receipt["details"]["registered"] = await connectivity.rendezvous_register(
                    rendezvous_namespace
                )
                receipt["details"]["discovered_peers"] = await connectivity.rendezvous_discover(
                    rendezvous_namespace
                )

            if cancel_scope.cancelled_caught:
                receipt["error"] = "timeout"
            elif not connectivity.implemented.get("rendezvous"):
                receipt["error"] = "rendezvous_not_implemented_by_runtime"
            elif not receipt["details"]["registered"]:
                receipt["error"] = "rendezvous_registration_failed"
            else:
                receipt["success"] = True
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["duration_ms"] = (time.monotonic() - started) * 1000.0
        self._rendezvous_attempts.append(receipt)
        return dict(receipt)

    async def _handle_stream(self, stream) -> None:
        """Handle an incoming /mcp+p2p/1.0.0 stream with backpressure."""
        # Backpressure: reject if too many concurrent handlers
        if self._concurrent_streams >= self._max_concurrent_streams:
            try:
                error_msg = P2PMessage(
                    msg_type="response",
                    error="Server overloaded (backpressure)",
                    sender_peer_id=self.peer_id or "",
                )
                await stream.write(error_msg.encode())
                await stream.close()
            except Exception:
                pass
            return

        self._concurrent_streams += 1
        try:
            import trio

            # Read length-prefixed message with read timeout
            with trio.move_on_after(10.0) as read_scope:
                length_bytes = await self._read_exact(stream, 4)
            if read_scope.cancelled_caught:
                return
            length = int.from_bytes(length_bytes, "big")
            if length > MAX_P2P_MESSAGE_SIZE:
                logger.warning("Rejecting oversized P2P message: %d bytes", length)
                return
            payload = await self._read_exact(stream, length)

            msg = P2PMessage.decode(length_bytes + payload)

            if msg.msg_type == "request" and self._tool_handler:
                # Inject sender identity so handlers can verify peer
                params = dict(msg.params) if msg.params else {}
                params["_sender_peer_id"] = msg.sender_peer_id
                try:
                    # Tool execution timeout (30s)
                    with trio.move_on_after(30.0) as exec_scope:
                        result = await self._tool_handler(msg.method, params)
                    if exec_scope.cancelled_caught:
                        response = P2PMessage(
                            msg_type="response",
                            method=msg.method,
                            msg_id=msg.msg_id,
                            error="Tool execution timeout (30s)",
                            sender_peer_id=self.peer_id,
                        )
                    else:
                        response = P2PMessage(
                            msg_type="response",
                            method=msg.method,
                            msg_id=msg.msg_id,
                            result=result,
                            sender_peer_id=self.peer_id,
                        )
                except Exception as e:
                    response = P2PMessage(
                        msg_type="response",
                        method=msg.method,
                        msg_id=msg.msg_id,
                        error=str(e),
                        sender_peer_id=self.peer_id,
                    )

                await stream.write(response.encode())

            await stream.close()
        except Exception as e:
            logger.warning(f"Stream handler error: {e}", exc_info=True)
            try:
                await stream.close()
            except Exception:
                pass
        finally:
            self._concurrent_streams -= 1

    async def call_tool(
        self,
        peer_id: str,
        method: str,
        params: Dict[str, Any],
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> Any:
        """Call a tool on a remote peer via /mcp+p2p/1.0.0 with retry and circuit breaker.

        Args:
            peer_id: Target peer's ID
            method: Tool/method name
            params: Tool parameters
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts (with exponential backoff)

        Returns:
            The tool result from the remote peer

        Raises:
            TimeoutError: If the peer doesn't respond after all retries
            ConnectionError: If we can't reach the peer after all retries
            RuntimeError: If the remote returns an error (not retried)
        """
        if not self._host:
            raise ConnectionError("P2P node not started")

        # Circuit breaker check
        cb = self._get_circuit_breaker(peer_id)
        if cb["state"] == "open":
            elapsed = time.time() - cb["opened_at"]
            if elapsed < cb["reset_timeout"]:
                raise ConnectionError(
                    f"Circuit breaker open for peer {peer_id} "
                    f"(failures: {cb['failures']}, resets in {cb['reset_timeout'] - elapsed:.0f}s)"
                )
            # Half-open: allow one attempt
            cb["state"] = "half-open"

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                result = await self._call_tool_once(peer_id, method, params, timeout)
                # Success: reset circuit breaker
                cb["failures"] = 0
                cb["state"] = "closed"
                return result
            except RuntimeError:
                # Remote application error — don't retry
                raise
            except (TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                cb["failures"] += 1
                if cb["failures"] >= cb["threshold"]:
                    cb["state"] = "open"
                    cb["opened_at"] = time.time()
                    logger.warning(
                        "Circuit breaker OPEN for peer %s after %d failures",
                        peer_id,
                        cb["failures"],
                    )
                if attempt < max_retries - 1:
                    backoff = min(2**attempt * 0.5, 10.0)
                    logger.info(
                        "P2P call to %s/%s failed (attempt %d/%d), retrying in %.1fs: %s",
                        peer_id,
                        method,
                        attempt + 1,
                        max_retries,
                        backoff,
                        e,
                    )
                    import trio

                    await trio.sleep(backoff)

        raise last_error  # type: ignore[misc]

    def _get_circuit_breaker(self, peer_id: str) -> Dict[str, Any]:
        """Get or create circuit breaker state for a peer."""
        if not hasattr(self, "_circuit_breakers"):
            self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        if peer_id not in self._circuit_breakers:
            self._circuit_breakers[peer_id] = {
                "state": "closed",  # closed | open | half-open
                "failures": 0,
                "threshold": 5,
                "opened_at": 0.0,
                "reset_timeout": 60.0,  # seconds before half-open
            }
        return self._circuit_breakers[peer_id]

    async def _call_tool_once(
        self, peer_id: str, method: str, params: Dict[str, Any], timeout: float
    ) -> Any:
        """Single attempt to call a tool on a remote peer."""

        try:
            import trio

            target = peer_id_from_base58(peer_id)

            # Open stream (try supported versions in order for negotiation)
            stream = await self._host.new_stream(target, MCP_P2P_SUPPORTED_VERSIONS)

            # Send request
            request = P2PMessage(
                msg_type="request",
                method=method,
                params=params,
                msg_id=f"{method}_{time.time()}",
                sender_peer_id=self.peer_id,
            )
            await stream.write(request.encode())

            # Read response with timeout
            with trio.move_on_after(timeout) as cancel_scope:
                length_bytes = await self._read_exact(stream, 4)
                length = int.from_bytes(length_bytes, "big")
                if length > MAX_P2P_MESSAGE_SIZE:
                    raise ConnectionError(f"Peer response exceeds {MAX_P2P_MESSAGE_SIZE} bytes")
                payload = await self._read_exact(stream, length)
                response = P2PMessage.decode(length_bytes + payload)

            if cancel_scope.cancelled_caught:
                await stream.close()
                raise TimeoutError(f"Peer {peer_id} did not respond within {timeout}s")

            await stream.close()

            if response.error:
                raise RuntimeError(f"Remote error: {response.error}")

            return response.result

        except ImportError:
            raise ConnectionError(
                f"libp2p not available. Install via MCP++ runtime: {LIBP2P_INSTALL_HINT}"
            )

    async def discover_peers(self, service_tag: str = "mcp-accelerate") -> List[PeerInfo]:
        """Discover peers via mDNS (local network).

        Uses Trio-compatible mDNS to find peers advertising the MCP++ service.
        Discovery returns dial candidates; only a successful connection may
        add a peer to the connected-peer registry.
        """
        if not self.mdns_enabled:
            logger.debug("mDNS discovery disabled by MCPPP_P2P_MDNS")
            return []

        browser = None
        zeroconf = None
        try:
            import trio
            from zeroconf import Zeroconf, ServiceBrowser

            discovered: List[PeerInfo] = []
            local_peer_id = self.peer_id
            zeroconf = Zeroconf()

            class Listener:
                def add_service(self, zc, type_, name):
                    info = zc.get_service_info(type_, name)
                    if info:
                        peer_id = name.split(".", 1)[0].strip()
                        if not peer_id or peer_id == local_peer_id:
                            return
                        peer = PeerInfo(
                            peer_id=peer_id,
                            multiaddrs=[
                                f"/ip4/{address}/tcp/{info.port}/p2p/{peer_id}"
                                for address in info.parsed_addresses()
                            ],
                            protocols=[MCP_P2P_PROTOCOL],
                        )
                        discovered.append(peer)

                def remove_service(self, zc, type_, name):
                    pass

                def update_service(self, zc, type_, name):
                    pass

            browser = ServiceBrowser(
                zeroconf,
                f"_{service_tag}._tcp.local.",
                Listener(),
            )

            # Wait briefly for discovery
            await trio.sleep(2.0)

            return discovered

        except ImportError:
            logger.debug("zeroconf not available for mDNS discovery")
            return []
        except Exception as e:
            logger.debug(f"mDNS discovery failed: {e}")
            return []
        finally:
            if browser is not None:
                try:
                    browser.cancel()
                except Exception as exc:
                    logger.debug("Failed to cancel mDNS browser: %s", exc)
            if zeroconf is not None:
                try:
                    zeroconf.close()
                except Exception as exc:
                    logger.debug("Failed to close Zeroconf: %s", exc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state for status endpoints."""
        mdns_configured = self.mdns_enabled
        rendezvous_target = os.environ.get("IPFS_ACCELERATE_P2P_RENDEZVOUS_PEER", "").strip()
        rendezvous_service_mode = (
            os.environ.get("MCPPP_P2P_RENDEZVOUS_SERVICE", "").strip().casefold()
        )
        rendezvous_service_implemented = self._rendezvous_service is not None
        rendezvous_client_implemented = bool(
            self._rendezvous_connectivity
            and self._rendezvous_connectivity.implemented.get("rendezvous")
        )
        rendezvous_client_exercised = bool(
            rendezvous_client_implemented
            and any(attempt.get("success") for attempt in self._rendezvous_attempts)
        )
        rendezvous_implemented = bool(
            rendezvous_service_implemented or rendezvous_client_implemented
        )
        rendezvous_advertised = bool(self._operational and rendezvous_service_implemented)
        return {
            "p2p_requested": True,
            "p2p_enabled": self._started,
            "peer_id": self.peer_id,
            "multiaddrs": self.multiaddrs,
            "protocol": MCP_P2P_PROTOCOL,
            "listen_addrs": list(self._listen_addrs),
            "listen_port": self.listen_port,
            "advertisement_policy": {
                "explicit_multiaddrs": list(self._advertise_addrs),
                "allowed_interfaces": list(self._advertise_interfaces),
                "reject_known_container_interfaces": True,
            },
            "started": self._started,
            "operational": self._operational,
            "connected_peers": len(self._peers),
            "peers": [p.to_dict() for p in self._peers.values()],
            "bootstrap": {
                "configured_peers": list(self._bootstrap_peers),
                "attempts": list(self._bootstrap_attempts),
            },
            "rendezvous": {
                "service": {
                    "mode": rendezvous_service_mode,
                    "configured": rendezvous_service_mode == "same_as_service_peer",
                    "implemented": rendezvous_service_implemented,
                    "peer_id": self.peer_id if rendezvous_service_implemented else "",
                },
                "client": {
                    "configured_peer": rendezvous_target,
                    "implemented": rendezvous_client_implemented,
                    "exercise_success": rendezvous_client_exercised,
                },
                "namespace": (
                    os.environ.get("IPFS_ACCELERATE_P2P_RENDEZVOUS_NS", "").strip()
                    or "leanstral-local"
                ),
                "attempts": list(self._rendezvous_attempts),
                "external_exercise_receipt_required": True,
            },
            "capabilities": {
                "mcp_stream": {
                    "configured": True,
                    "implemented": True,
                    "advertised": self._operational,
                },
                "bootstrap": {
                    "configured": bool(self._bootstrap_peers),
                    "implemented": True,
                    "advertised": bool(self._operational and self._bootstrap_peers),
                },
                "mdns": {
                    "configured": mdns_configured,
                    "implemented": self._mdns_zeroconf is not None,
                    "advertised": self._mdns_zeroconf is not None,
                },
                "rendezvous": {
                    "configured": bool(
                        rendezvous_target or rendezvous_service_mode == "same_as_service_peer"
                    ),
                    "implemented": rendezvous_implemented,
                    "advertised": rendezvous_advertised,
                },
                "pubsub": {
                    "configured": False,
                    "implemented": False,
                    "advertised": False,
                    "policy": "disabled_until_implemented",
                },
                "floodsub": {
                    "configured": False,
                    "implemented": False,
                    "advertised": False,
                    "policy": "disabled_until_implemented",
                },
            },
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_NODE: Optional[MCPp2pNode] = None
_P2P_LOCK = threading.Lock()


def get_p2p_node() -> MCPp2pNode:
    """Get or create the global P2P node singleton (thread-safe)."""
    global _NODE
    if _NODE is None:
        with _P2P_LOCK:
            if _NODE is None:
                _NODE = MCPp2pNode()
    return _NODE
