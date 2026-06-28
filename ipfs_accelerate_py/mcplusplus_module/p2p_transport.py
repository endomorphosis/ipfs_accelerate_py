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
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.p2p_transport")


def ensure_libp2p_installed() -> bool:
    """Auto-install libp2p from git if not already available (sync, for startup).

    Returns True if libp2p is importable after this call.
    """
    try:
        import libp2p  # noqa: F401
        return True
    except ImportError:
        pass

    logger.info("libp2p not found — auto-installing from git (py-libp2p@main)...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main",
             "multiaddr", "protobuf>=3.20.0"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            timeout=120,
        )
        import libp2p  # noqa: F401
        logger.info("libp2p installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ImportError) as e:
        logger.error("Failed to auto-install libp2p: %s", e)
        return False


async def ensure_libp2p_installed_async() -> bool:
    """Trio-native async version of ensure_libp2p_installed (for use within Trio context)."""
    try:
        import libp2p  # noqa: F401
        return True
    except ImportError:
        pass

    import trio
    logger.info("libp2p not found — auto-installing from git (py-libp2p@main)...")
    try:
        result = await trio.run_process(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "libp2p @ git+https://github.com/libp2p/py-libp2p.git@main",
             "multiaddr", "protobuf>=3.20.0"],
            capture_stdout=True, capture_stderr=True,
        )
        if result.returncode != 0:
            logger.error("pip install failed: %s", result.stderr.decode())
            return False
        import libp2p  # noqa: F401
        logger.info("libp2p installed successfully (async)")
        return True
    except (OSError, ImportError) as e:
        logger.error("Failed to auto-install libp2p: %s", e)
        return False

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
    "/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ",
]


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
        payload = json.dumps({
            "type": self.msg_type,
            "method": self.method,
            "params": self.params,
            "id": self.msg_id,
            "result": self.result,
            "error": self.error,
            "sender": self.sender_peer_id,
            "timestamp": self.timestamp,
        }, separators=(",", ":")).encode("utf-8")
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
        payload = json.loads(data[4:4 + length].decode("utf-8"))
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

    def __init__(self, listen_addrs: Optional[List[str]] = None,
                 bootstrap_peers: Optional[List[str]] = None):
        self._listen_addrs = listen_addrs or ["/ip4/0.0.0.0/tcp/0"]
        self._bootstrap_peers = bootstrap_peers or DEFAULT_BOOTSTRAP_PEERS
        self._host = None
        self._peers: Dict[str, PeerInfo] = {}
        self._tool_handler: Optional[Callable] = None
        self._started = False
        self._operational = False  # True only when libp2p is actually running
        self._concurrent_streams = 0
        self._max_concurrent_streams = 50  # Backpressure limit
        self._nursery = None

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
            return [str(a) for a in self._host.get_addrs()]
        return []

    @property
    def connected_peers(self) -> List[PeerInfo]:
        """Currently connected peers."""
        return list(self._peers.values())

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
        self._nursery = nursery

        # Auto-install libp2p if missing (run in thread to avoid blocking event loop)
        import trio
        installed = await trio.to_thread.run_sync(ensure_libp2p_installed)
        if not installed:
            logger.error(
                "libp2p could not be installed. P2P transport in stub mode. "
                "Manual install: pip install 'libp2p @ git+https://github.com/libp2p/py-libp2p.git@main'"
            )
            self._started = True
            self._operational = False
            return

        try:
            import trio
            from libp2p import new_host
            from libp2p.crypto.secp256k1 import create_new_key_pair
            from libp2p.peer.peerinfo import info_from_p2p_addr
            from multiaddr import Multiaddr

            # Generate identity
            key_pair = create_new_key_pair()

            # Create libp2p host
            self._host = new_host(key_pair=key_pair)

            # Register protocol handler for all supported versions
            for proto_version in MCP_P2P_SUPPORTED_VERSIONS:
                self._host.set_stream_handler(proto_version, self._handle_stream)

            # Start listening
            for addr in self._listen_addrs:
                await self._host.get_network().listen(Multiaddr(addr))

            self._started = True
            self._operational = True
            logger.info(
                "MCPp2pNode started: peer_id=%s, addrs=%s",
                self.peer_id, self.multiaddrs
            )

            # Bootstrap: connect to known peers (with timeout)
            for peer_addr in self._bootstrap_peers:
                nursery.start_soon(self._connect_bootstrap_with_timeout, peer_addr)

        except ImportError as e:
            logger.warning(
                "libp2p not available (%s). P2P transport running in stub mode. "
                "Install with: pip install libp2p", e
            )
            self._started = True  # Mark as started in stub mode
            self._operational = False

    async def stop(self) -> None:
        """Stop the libp2p node."""
        if self._host:
            await self._host.close()
        self._started = False
        self._peers.clear()
        logger.info("MCPp2pNode stopped")

    async def _connect_bootstrap_with_timeout(self, peer_addr: str) -> None:
        """Connect to a bootstrap peer with a 10-second timeout."""
        import trio
        with trio.move_on_after(10.0) as cancel_scope:
            await self._connect_bootstrap(peer_addr)
        if cancel_scope.cancelled_caught:
            logger.debug(f"Bootstrap peer connection timed out: {peer_addr}")

    async def _connect_bootstrap(self, peer_addr: str) -> None:
        """Connect to a bootstrap peer."""
        try:
            import trio
            from libp2p.peer.peerinfo import info_from_p2p_addr
            from multiaddr import Multiaddr

            peer_info = info_from_p2p_addr(Multiaddr(peer_addr))
            await self._host.connect(peer_info)

            # Enforce max peers limit
            if len(self._peers) >= MAX_PEERS:
                # Evict oldest peer
                oldest = min(self._peers.values(), key=lambda p: p.last_seen)
                self._peers.pop(oldest.peer_id, None)

            self._peers[str(peer_info.peer_id)] = PeerInfo(
                peer_id=str(peer_info.peer_id),
                multiaddrs=[peer_addr],
                protocols=[MCP_P2P_PROTOCOL],
            )
            logger.info(f"Connected to bootstrap peer: {peer_info.peer_id}")
        except Exception as e:
            logger.debug(f"Failed to connect to bootstrap {peer_addr}: {e}")

    async def _handle_stream(self, stream) -> None:
        """Handle an incoming /mcp+p2p/1.0.0 stream with backpressure."""
        # Backpressure: reject if too many concurrent handlers
        if self._concurrent_streams >= self._max_concurrent_streams:
            try:
                error_msg = P2PMessage(
                    msg_type="response", error="Server overloaded (backpressure)",
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
                length_bytes = await stream.read(4)
            if read_scope.cancelled_caught or len(length_bytes) < 4:
                return
            length = int.from_bytes(length_bytes, "big")
            if length > MAX_P2P_MESSAGE_SIZE:
                logger.warning("Rejecting oversized P2P message: %d bytes", length)
                return
            payload = await stream.read(length)

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
                            msg_type="response", method=msg.method, msg_id=msg.msg_id,
                            error="Tool execution timeout (30s)", sender_peer_id=self.peer_id,
                        )
                    else:
                        response = P2PMessage(
                            msg_type="response", method=msg.method, msg_id=msg.msg_id,
                            result=result, sender_peer_id=self.peer_id,
                        )
                except Exception as e:
                    response = P2PMessage(
                        msg_type="response", method=msg.method, msg_id=msg.msg_id,
                        error=str(e), sender_peer_id=self.peer_id,
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

    async def call_tool(self, peer_id: str, method: str,
                        params: Dict[str, Any], timeout: float = 30.0,
                        max_retries: int = 3) -> Any:
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
                        peer_id, cb["failures"],
                    )
                if attempt < max_retries - 1:
                    backoff = min(2 ** attempt * 0.5, 10.0)
                    logger.info(
                        "P2P call to %s/%s failed (attempt %d/%d), retrying in %.1fs: %s",
                        peer_id, method, attempt + 1, max_retries, backoff, e,
                    )
                    import trio
                    await trio.sleep(backoff)

        raise last_error  # type: ignore[misc]

    def _get_circuit_breaker(self, peer_id: str) -> Dict[str, Any]:
        """Get or create circuit breaker state for a peer."""
        if not hasattr(self, '_circuit_breakers'):
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

    async def _call_tool_once(self, peer_id: str, method: str,
                              params: Dict[str, Any], timeout: float) -> Any:
        """Single attempt to call a tool on a remote peer."""

        try:
            import trio
            from libp2p.peer.id import ID as PeerID

            target = PeerID.from_base58(peer_id)

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
                length_bytes = await stream.read(4)
                if len(length_bytes) < 4:
                    raise ConnectionError("Peer closed connection")
                length = int.from_bytes(length_bytes, "big")
                payload = await stream.read(length)
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
                "libp2p not available. Install with: pip install libp2p"
            )

    async def discover_peers(self, service_tag: str = "mcp-accelerate") -> List[PeerInfo]:
        """Discover peers via mDNS (local network).

        Uses Trio-compatible mDNS to find peers advertising the MCP++ service.
        """
        try:
            import trio
            from zeroconf import Zeroconf, ServiceBrowser

            discovered = []
            zc = Zeroconf()

            class Listener:
                def add_service(self, zc, type_, name):
                    info = zc.get_service_info(type_, name)
                    if info:
                        peer = PeerInfo(
                            peer_id=name.split(".")[0],
                            multiaddrs=[f"/ip4/{info.parsed_addresses()[0]}/tcp/{info.port}"],
                            protocols=[MCP_P2P_PROTOCOL],
                        )
                        discovered.append(peer)

                def remove_service(self, zc, type_, name):
                    pass

                def update_service(self, zc, type_, name):
                    pass

            browser = ServiceBrowser(zc, f"_{service_tag}._tcp.local.", Listener())

            # Wait briefly for discovery
            await trio.sleep(2.0)
            zc.close()

            # Register discovered peers (enforce MAX_PEERS limit)
            for peer in discovered:
                if len(self._peers) >= MAX_PEERS:
                    oldest = min(self._peers.values(), key=lambda p: p.last_seen)
                    self._peers.pop(oldest.peer_id, None)
                self._peers[peer.peer_id] = peer

            return discovered

        except ImportError:
            logger.debug("zeroconf not available for mDNS discovery")
            return []
        except Exception as e:
            logger.debug(f"mDNS discovery failed: {e}")
            return []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state for status endpoints."""
        return {
            "peer_id": self.peer_id,
            "multiaddrs": self.multiaddrs,
            "protocol": MCP_P2P_PROTOCOL,
            "started": self._started,
            "connected_peers": len(self._peers),
            "peers": [p.to_dict() for p in self._peers.values()],
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
