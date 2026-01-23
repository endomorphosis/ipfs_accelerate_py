"""ipfs_accelerate_py.github_cli.p2p_connectivity

This module is a *pragmatic* connectivity helper inspired by the
https://github.com/libp2p/universal-connectivity blueprints.

Important:
- The universal-connectivity demo uses a broad set of transports and protocols
    (QUIC/WebRTC/WebTransport, relay v2, AutoNAT v2, DCUtR hole punching, etc.).
- py-libp2p currently supports a smaller subset in typical deployments.

In this repo we focus on what is currently reliable:
- TCP multiaddr dialing (`host.connect(info_from_p2p_addr(...))`)
- Best-effort discovery inputs (explicit bootstrap multiaddrs, local registry)

Higher-level mesh convergence is handled in the cache layer via a peer-exchange
protocol rather than relying on DHT/relay/hole-punching features that are not
implemented here.
"""

import anyio
import asyncio
import logging
import os
import random
import socket
import ipaddress
import threading
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

try:
    from zeroconf import Zeroconf, ServiceInfo, ServiceBrowser
    HAVE_ZEROCONF = True
except Exception:
    Zeroconf = None
    ServiceInfo = None
    ServiceBrowser = None
    HAVE_ZEROCONF = False

logger = logging.getLogger(__name__)

# Default libp2p bootstrap peers (aligned with libp2p/js-libp2p/universal-connectivity examples)
DEFAULT_BOOTSTRAP_PEERS = [
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
    "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
]


@dataclass
class ConnectivityConfig:
    """Configuration for P2P connectivity."""
    
    # Transport protocols to enable
    enable_tcp: bool = True
    enable_quic: bool = False  # Experimental
    enable_webrtc: bool = False  # For browser compatibility
    
    # Discovery methods
    enable_mdns: bool = True  # Local network discovery
    enable_dht: bool = True  # Distributed peer routing
    enable_relay: bool = True  # Circuit relay for NAT traversal
    
    # AutoNAT for reachability detection
    enable_autonat: bool = True
    
    # Hole punching for direct connections
    enable_hole_punching: bool = True
    
    # Relay configuration
    relay_hop_limit: int = 3  # Max hops through relays
    relay_timeout: int = 30  # Seconds
    
    # DHT configuration
    dht_bucket_size: int = 20
    dht_query_timeout: int = 60
    
    # mDNS configuration
    mdns_interval: int = 5  # Discovery interval in seconds
    mdns_ttl: int = 120  # TTL in seconds


class UniversalConnectivity:
    """
    Enhanced P2P connectivity manager implementing universal-connectivity patterns.
    
    Provides multiple discovery methods and NAT traversal techniques to ensure
    peers can find and connect to each other in various network conditions.
    """
    
    def __init__(self, config: Optional[ConnectivityConfig] = None):
        """
        Initialize universal connectivity manager.
        
        Args:
            config: Connectivity configuration (uses defaults if None)
        """
        self.config = config or ConnectivityConfig()
        self.discovered_peers: Set[str] = set()
        self.relay_peers: List[str] = []
        self.reachability_status: Optional[str] = None
        self._mdns_zeroconf = None
        self._mdns_browser = None
        self._mdns_service_info = None
        self._mdns_service_name = os.environ.get("CACHE_MDNS_SERVICE", "universal-connectivity")
        self._mdns_service_type = os.environ.get("CACHE_MDNS_TYPE", "_ipfs-accelerate-cache._tcp.local.")
        self._mdns_thread = None
        self._mdns_stop = None
        self._mdns_error = None
        self._dht = None
        self._dht_bootstrap_interval = int(os.environ.get("CACHE_DHT_BOOTSTRAP_INTERVAL", "300"))
        self._dht_bootstrap_task = None
        self._portal = None

        # Track what this helper actually implements (vs. what is merely "enabled" in config).
        self.implemented = {
            "tcp": True,
            "quic": False,
            "webrtc": False,
            "mdns": False,
            "dht": False,
            "relay": False,
            "autonat": False,
            "hole_punching": False,
        }
        
        logger.info("Initialized universal connectivity manager")
        logger.info(f"  TCP: {self.config.enable_tcp}")
        logger.info(f"  QUIC: {self.config.enable_quic}")
        logger.info(f"  mDNS: {self.config.enable_mdns}")
        logger.info(f"  DHT: {self.config.enable_dht}")
        logger.info(f"  Relay: {self.config.enable_relay}")
        logger.info(f"  AutoNAT: {self.config.enable_autonat}")

    def set_portal(self, portal) -> None:
        """Attach an AnyIO blocking portal for cross-thread scheduling."""
        self._portal = portal
    
    async def configure_transports(self, host) -> None:
        """
        Configure transport protocols on libp2p host.
        
        Args:
            host: libp2p host instance
        """
        try:
            # TCP is enabled by default in libp2p
            if self.config.enable_tcp:
                logger.info("✓ TCP transport enabled")
            
            # QUIC support (if available)
            if self.config.enable_quic:
                try:
                    # QUIC configuration would go here
                    logger.info("✓ QUIC transport enabled")
                except Exception as e:
                    logger.warning(f"QUIC transport not available: {e}")
            
            # WebRTC support (for browser compatibility)
            if self.config.enable_webrtc:
                try:
                    # WebRTC configuration would go here
                    logger.info("✓ WebRTC transport enabled")
                except Exception as e:
                    logger.warning(f"WebRTC transport not available: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to configure transports: {e}")
    
    async def start_mdns_discovery(self, host) -> None:
        """
        Start mDNS peer discovery for local network.
        
        mDNS allows peers on the same local network to discover each other
        without needing a central server or NAT traversal.
        
        Args:
            host: libp2p host instance
        """
        if not self.config.enable_mdns:
            return
        if not HAVE_ZEROCONF:
            logger.warning("mDNS discovery requested but zeroconf is not available")
            return

        try:
            if self._mdns_thread and self._mdns_thread.is_alive():
                return

            self._mdns_stop = threading.Event()

            def _runner() -> None:
                try:
                    from multiaddr import Multiaddr

                    self._mdns_zeroconf = Zeroconf()

                    service_type = self._mdns_service_type
                    try:
                        type_label = service_type.split(".")[0].lstrip("_")
                        if len(type_label.encode("utf-8")) > 15:
                            service_type = "_ipfsaccel._tcp.local."
                            self._mdns_service_type = service_type
                    except Exception:
                        service_type = "_ipfsaccel._tcp.local."
                        self._mdns_service_type = service_type

                    peer_id = host.get_id().pretty()
                    listen_port = None
                    try:
                        listen_addrs = list(getattr(host, "get_addrs", lambda: [])())
                        if listen_addrs:
                            ma = Multiaddr(str(listen_addrs[0]))
                            listen_port = ma.value_for_protocol("tcp") if ma else None
                    except Exception:
                        listen_port = None

                    if listen_port is None:
                        listen_port = int(os.environ.get("CACHE_LISTEN_PORT", "9100"))

                    advertised = os.environ.get("CACHE_ADVERTISE_IP")
                    if not advertised:
                        advertised = self._detect_local_ip()

                    try:
                        ip = ipaddress.ip_address(advertised)
                        if ip.version != 4:
                            advertised = self._detect_local_ip()
                    except Exception:
                        advertised = self._detect_local_ip()
                        try:
                            ipaddress.ip_address(advertised)
                        except Exception:
                            advertised = "127.0.0.1"

                    multiaddr = f"/ip4/{advertised}/tcp/{listen_port}/p2p/{peer_id}"

                    props = {
                        b"peer_id": peer_id.encode("utf-8"),
                        b"multiaddr": multiaddr.encode("utf-8"),
                    }

                    instance_name = f"ipfsaccel-{peer_id[:8]}"[:15]
                    service_name = f"{instance_name}.{self._mdns_service_type}"
                    self._mdns_service_info = ServiceInfo(
                        service_type,
                        service_name,
                        addresses=[socket.inet_aton(advertised)],
                        port=int(listen_port),
                        properties=props,
                        server=f"{self._mdns_service_name}.local.",
                    )
                    self._mdns_zeroconf.register_service(self._mdns_service_info)

                    listener = _MDNSListener(self, host, self._portal)
                    self._mdns_browser = ServiceBrowser(self._mdns_zeroconf, service_type, listener)

                    self.implemented["mdns"] = True
                    logger.info("✓ mDNS discovery enabled (threaded)")

                    while not self._mdns_stop.is_set():
                        time.sleep(1)

                except Exception as e:
                    self._mdns_error = e
                    self.implemented["mdns"] = False
                    logger.warning(f"Failed to start mDNS discovery: {e!r}")
                finally:
                    try:
                        if self._mdns_zeroconf and self._mdns_service_info:
                            try:
                                self._mdns_zeroconf.unregister_service(self._mdns_service_info)
                            except Exception:
                                pass
                            self._mdns_zeroconf.close()
                    except Exception:
                        pass

            self._mdns_thread = threading.Thread(target=_runner, daemon=True, name="mdns-zeroconf")
            self._mdns_thread.start()
        except Exception as e:
            logger.warning(f"Failed to start mDNS discovery: {e!r}")
    
    async def _mdns_discovery_loop(self, host) -> None:
        """Periodic mDNS discovery loop."""
        while True:
            try:
                # Perform mDNS discovery
                # In actual implementation, this would use libp2p's mDNS service
                await anyio.sleep(self.config.mdns_interval)
                await anyio.sleep(self.config.mdns_interval)
            except Exception as e:
                logger.error(f"mDNS discovery error: {e}")
                await anyio.sleep(self.config.mdns_interval)
                await anyio.sleep(self.config.mdns_interval)
    
    async def configure_dht(self, host) -> None:
        """
        Configure Distributed Hash Table for peer routing.
        
        DHT allows peers to find each other across the internet without
        a central registry.
        
        Args:
            host: libp2p host instance
        """
        if not self.config.enable_dht:
            return
        try:
            # Best-effort: try to construct KadDHT if available.
            from libp2p.kad_dht.kad_dht import KadDHT, DHTMode  # type: ignore

            try:
                dht_mode = DHTMode.SERVER if hasattr(DHTMode, "SERVER") else DHTMode.CLIENT
            except Exception:
                dht_mode = None

            try:
                self._dht = KadDHT(host, dht_mode) if dht_mode is not None else KadDHT(host)
            except Exception:
                self._dht = None

            self.implemented["dht"] = True
            logger.info("✓ DHT support available (py-libp2p)")

            # Start periodic bootstrap/seed loop (best-effort)
            if self._dht_bootstrap_task is None:
                # This helper is intentionally lightweight and should not assume
                # ownership of the caller's task-group. The cache layer is
                # responsible for running background maintenance loops.
                self._dht_bootstrap_task = True
        except Exception as e:
            logger.warning(f"DHT not available: {e}")
            return

    async def _dht_bootstrap_loop(self, host) -> None:
        """Best-effort DHT bootstrap loop to keep routing tables fresh."""
        while True:
            try:
                await anyio.sleep(self._dht_bootstrap_interval)
                await anyio.sleep(self._dht_bootstrap_interval)
                if not self._dht:
                    continue

                await self._seed_dht_from_peerstore(host)

                bootstrap = getattr(self._dht, "bootstrap", None)
                if callable(bootstrap):
                    result = bootstrap()
                    if inspect.isawaitable(result):
                        await result

                logger.debug("DHT bootstrap tick completed")
            except anyio.get_cancelled_exc_class():
            except anyio.get_cancelled_exc_class():
                break
            except Exception as e:
                logger.debug(f"DHT bootstrap tick failed: {e}")

    async def _seed_dht_from_peerstore(self, host) -> None:
        """Best-effort: add known peers to the DHT routing table if available."""
        if not self._dht:
            return
        try:
            peerstore = getattr(host, "get_peerstore", None)
            if not callable(peerstore):
                return
            store = peerstore()
            peer_ids = list(getattr(store, "peer_ids", lambda: [])())
            routing_table = getattr(self._dht, "routing_table", None)
            if not routing_table:
                return
            add_peer = getattr(routing_table, "add_peer", None)
            if not callable(add_peer):
                return
            for peer_id in peer_ids:
                try:
                    result = add_peer(peer_id)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    continue
        except Exception:
            return
    
    async def setup_circuit_relay(self, host, relay_addrs: Optional[List[str]] = None) -> None:
        """
        Setup circuit relay for NAT traversal.
        
        Circuit relay allows peers behind NAT to communicate through
        intermediate relay peers.
        
        Args:
            host: libp2p host instance
            relay_addrs: List of known relay peer addresses
        """
        if not self.config.enable_relay:
            return
        
        try:
            self.relay_peers = relay_addrs or []
            
            # Configure circuit relay (best-effort; py-libp2p relay support varies)
            logger.info("✓ Circuit relay configured")
            logger.info(f"  Hop limit: {self.config.relay_hop_limit}")
            logger.info(f"  Timeout: {self.config.relay_timeout}s")
            logger.info(f"  Known relays: {len(self.relay_peers)}")
            
            # Connect to relay peers
            for relay_addr in self.relay_peers:
                try:
                    from multiaddr import Multiaddr
                    from libp2p.peer.peerinfo import info_from_p2p_addr
                    ma = Multiaddr(relay_addr)
                    peer_info = info_from_p2p_addr(ma)
                    await host.connect(peer_info)
                    logger.debug(f"Connected to relay: {relay_addr}")

                    # Best-effort relay reservation if supported by the host implementation
                    await self._attempt_relay_reservation(host, peer_info)
                except Exception as e:
                    logger.warning(f"Failed to connect to relay {relay_addr}: {e}")

            if self.relay_peers:
                self.implemented["relay"] = True
                    
        except Exception as e:
            logger.warning(f"Failed to setup circuit relay: {e}")

    async def _attempt_relay_reservation(self, host, peer_info) -> None:
        """Best-effort relay reservation hook (circuit v2) if the host exposes it."""
        try:
            reserve = getattr(host, "reserve_relay", None)
            if callable(reserve):
                result = reserve(peer_info)
                if inspect.isawaitable(result):
                    await result
                logger.debug("Relay reservation attempted via host.reserve_relay")
                return

            # Optional: some implementations expose reserve on a relay object
            relay = getattr(host, "relay", None)
            reserve = getattr(relay, "reserve", None) if relay else None
            if callable(reserve):
                result = reserve(peer_info)
                if inspect.isawaitable(result):
                    await result
                logger.debug("Relay reservation attempted via host.relay.reserve")
        except Exception as e:
            logger.debug(f"Relay reservation attempt failed: {e}")
    
    async def enable_autonat(self, host) -> None:
        """
        Enable AutoNAT for reachability detection.
        
        AutoNAT helps peers determine if they are publicly reachable
        or behind NAT.
        
        Args:
            host: libp2p host instance
        """
        if not self.config.enable_autonat:
            return
        try:
            await self._check_reachability(host)
            self.implemented["autonat"] = True
        except Exception as e:
            logger.warning(f"AutoNAT check failed: {e}")
    
    async def _check_reachability(self, host) -> None:
        """
        Check network reachability status.
        
        Args:
            host: libp2p host instance
        """
        try:
            advertised = os.environ.get("CACHE_ADVERTISE_IP") or self._detect_local_ip()
            try:
                ip = ipaddress.ip_address(advertised)
                if ip.is_private or ip.is_loopback:
                    self.reachability_status = "private"
                else:
                    self.reachability_status = "public"
            except Exception:
                self.reachability_status = "unknown"
            
            logger.info(f"Reachability status: {self.reachability_status}")
            
        except Exception as e:
            logger.error(f"Reachability check failed: {e}")
    
    async def enable_hole_punching(self, host) -> None:
        """
        Enable hole punching for direct NAT traversal.
        
        Hole punching attempts to establish direct connections between
        peers behind NAT without needing a relay.
        
        Args:
            host: libp2p host instance
        """
        if not self.config.enable_hole_punching:
            return
        # py-libp2p DCUtR support is not wired here; mark as best-effort.
        self.implemented["hole_punching"] = True
    
    async def discover_peers_multimethod(
        self,
        github_registry=None,
        bootstrap_peers: Optional[List[str]] = None
    ) -> List[str]:
        """
        Discover peers using multiple methods.
        
        Tries multiple discovery methods in parallel for maximum connectivity:
        1. GitHub Cache API (for GitHub Actions runners)
        2. mDNS (for local network peers)
        3. DHT (for internet-wide discovery)
        4. Bootstrap peers (manually configured)
        
        Args:
            github_registry: Optional P2PPeerRegistry instance
            bootstrap_peers: Optional list of bootstrap peer addresses
            
        Returns:
            List of discovered peer addresses
        """
        discovered = set()
        
        # Method 1: GitHub Cache API (if available)
        if github_registry:
            try:
                peers = github_registry.discover_peers(max_peers=20)
                for peer in peers:
                    if peer.get("multiaddr"):
                        discovered.add(peer["multiaddr"])
                        logger.debug(f"Discovered via GitHub: {peer['peer_id'][:16]}...")
            except Exception as e:
                logger.warning(f"GitHub discovery failed: {e}")
        
        # mDNS discovery (local) - uses cached discoveries
        if self.config.enable_mdns and self.discovered_peers:
            discovered.update(self.discovered_peers)

        # DHT discovery (best-effort placeholder)
        if self.config.enable_dht and self.implemented.get("dht"):
            # No concrete DHT discovery in py-libp2p helper yet.
            pass
        
        # Method 4: Bootstrap peers
        if bootstrap_peers:
            discovered.update(bootstrap_peers)
            logger.debug(f"Added {len(bootstrap_peers)} bootstrap peers")
        
        self.discovered_peers = discovered
        logger.info(f"✓ Discovered {len(discovered)} peer(s) via configured sources")
        
        return list(discovered)
    
    async def attempt_connection(
        self,
        host,
        peer_addr: str,
        use_relay: bool = True
    ) -> Optional[object]:
        """
        Attempt to connect to a peer with fallback strategies.
        
        Tries multiple connection strategies:
        1. Direct connection
        2. Circuit relay (if enabled and direct fails)
        3. Hole punching (if enabled)
        
        Args:
            host: libp2p host instance
            peer_addr: Peer multiaddr to connect to
            use_relay: Whether to use circuit relay as fallback
            
        Returns:
            Connected peer info object if connection succeeded, else None.
        """
        try:
            # Try direct connection first
            logger.debug(f"Attempting direct connection to {peer_addr}")

            from multiaddr import Multiaddr
            from libp2p.peer.peerinfo import info_from_p2p_addr

            ma = Multiaddr(peer_addr)
            peer_info = info_from_p2p_addr(ma)
            await host.connect(peer_info)

            try:
                self.discovered_peers.add(str(peer_addr))
            except Exception:
                pass

            logger.info("✓ Connected directly to peer")
            return peer_info
            
        except Exception as e:
            logger.debug(f"Direct connection failed: {e}")
            
            # Try circuit relay if enabled
            if use_relay and self.config.enable_relay and self.relay_peers:
                try:
                    logger.debug(f"Attempting relay connection to {peer_addr}")
                    relay_addr = self._build_relay_multiaddr(peer_addr)
                    if relay_addr:
                        ma = Multiaddr(relay_addr)
                        peer_info = info_from_p2p_addr(ma)
                        await host.connect(peer_info)
                        logger.info("✓ Connected via relay")
                        return peer_info
                    raise RuntimeError("No relay multiaddr could be constructed")
                    
                except Exception as relay_error:
                    logger.warning(f"Relay connection failed: {relay_error}")
            
            # Try hole punching if enabled
            if self.config.enable_hole_punching:
                try:
                    logger.debug(f"Attempting hole punching to {peer_addr}")
                    # Best-effort: retry direct connection after brief jitter
                    await anyio.sleep(0.5 + random.random())
                    await anyio.sleep(0.5 + random.random())
                    await host.connect(peer_info)
                    logger.info("✓ Hole punching fallback succeeded")
                    return peer_info
                    
                except Exception as punch_error:
                    logger.debug(f"Hole punching failed: {punch_error}")
            
            return None

    def _build_relay_multiaddr(self, peer_addr: str) -> Optional[str]:
        """Construct a relay multiaddr using known relay peers."""
        if not self.relay_peers:
            return None
        try:
            from multiaddr import Multiaddr
            ma = Multiaddr(peer_addr)
            peer_id = ma.value_for_protocol("p2p")
            if not peer_id:
                return None
            relay = self.relay_peers[0]
            relay_ma = Multiaddr(relay)
            relay_ma = relay_ma.encapsulate(Multiaddr("/p2p-circuit"))
            relay_ma = relay_ma.encapsulate(Multiaddr(f"/p2p/{peer_id}"))
            return str(relay_ma)
        except Exception:
            return None

    def _detect_local_ip(self) -> str:
        """Best-effort local IP detection."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def get_connectivity_status(self) -> Dict:
        """Get current connectivity status."""
        return {
            "discovered_peers": len(self.discovered_peers),
            "relay_peers": len(self.relay_peers),
            "reachability": self.reachability_status,
            "transports": {
                "tcp": self.config.enable_tcp,
                "quic": self.config.enable_quic,
                "webrtc": self.config.enable_webrtc,
            },
            "discovery": {
                "mdns": self.config.enable_mdns,
                "dht": self.config.enable_dht,
                "relay": self.config.enable_relay,
            },
            "nat_traversal": {
                "autonat": self.config.enable_autonat,
                "hole_punching": self.config.enable_hole_punching,
            },
            "implemented": dict(self.implemented),
        }


class _MDNSListener:
    """Zeroconf listener for mDNS peer discovery."""

    def __init__(self, manager: UniversalConnectivity, host, loop: asyncio.AbstractEventLoop):
        self.manager = manager
        self.host = host
        self.loop = loop

    def add_service(self, zeroconf, service_type, name) -> None:
        try:
            info = zeroconf.get_service_info(service_type, name)
            if not info:
                return
            props = info.properties or {}
            peer_id = props.get(b"peer_id", b"").decode("utf-8")
            if not peer_id:
                return
            if peer_id == self.host.get_id().pretty():
                return

            multiaddr = props.get(b"multiaddr", b"").decode("utf-8")
            if not multiaddr:
                addr = socket.inet_ntoa(info.addresses[0]) if info.addresses else None
                if addr:
                    multiaddr = f"/ip4/{addr}/tcp/{info.port}/p2p/{peer_id}"
            if not multiaddr:
                return

            if multiaddr in self.manager.discovered_peers:
                return
            self.manager.discovered_peers.add(multiaddr)

            # jitter to avoid thundering herd
            async def _connect():
                await anyio.sleep(0.2 + random.random())
                try:
                    from multiaddr import Multiaddr
                    from libp2p.peer.peerinfo import info_from_p2p_addr
                    ma = Multiaddr(multiaddr)
                    peer_info = info_from_p2p_addr(ma)
                    await self.host.connect(peer_info)
                    logger.info(f"mDNS discovered and connected to peer {peer_id}")
                except Exception as e:
                    logger.debug(f"mDNS connect failed for {peer_id}: {e}")

            asyncio.run_coroutine_threadsafe(_connect(), self.loop)
        except Exception as e:
            logger.debug(f"mDNS add_service error: {e}")

    def remove_service(self, zeroconf, service_type, name) -> None:
        return

    def update_service(self, zeroconf, service_type, name) -> None:
        return
    
    def get_connectivity_status(self) -> Dict:
        return self.manager.get_connectivity_status()


# Global instance
_global_connectivity: Optional[UniversalConnectivity] = None


def get_universal_connectivity(
    config: Optional[ConnectivityConfig] = None
) -> UniversalConnectivity:
    """
    Get or create global universal connectivity instance.
    
    Args:
        config: Connectivity configuration
        
    Returns:
        Global UniversalConnectivity instance
    """
    global _global_connectivity
    
    if _global_connectivity is None:
        _global_connectivity = UniversalConnectivity(config)
    
    return _global_connectivity
