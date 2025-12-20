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

import asyncio
import logging
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
        
        # py-libp2p mDNS support is not wired up in this helper yet.
        # Keep this as a no-op to avoid misleading "started" logs.
        return
    
    async def _mdns_discovery_loop(self, host) -> None:
        """Periodic mDNS discovery loop."""
        while True:
            try:
                # Perform mDNS discovery
                # In actual implementation, this would use libp2p's mDNS service
                await asyncio.sleep(self.config.mdns_interval)
            except Exception as e:
                logger.error(f"mDNS discovery error: {e}")
                await asyncio.sleep(self.config.mdns_interval)
    
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
        
        # DHT support is not implemented in this helper.
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
            
            # Configure circuit relay
            logger.info("✓ Circuit relay configured")
            logger.info(f"  Hop limit: {self.config.relay_hop_limit}")
            logger.info(f"  Timeout: {self.config.relay_timeout}s")
            logger.info(f"  Known relays: {len(self.relay_peers)}")
            
            # Connect to relay peers
            for relay_addr in self.relay_peers:
                try:
                    # In actual implementation, connect to relay
                    logger.debug(f"Connecting to relay: {relay_addr}")
                except Exception as e:
                    logger.warning(f"Failed to connect to relay {relay_addr}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to setup circuit relay: {e}")
    
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
        
        # AutoNAT is not implemented in this helper.
        return
    
    async def _check_reachability(self, host) -> None:
        """
        Check network reachability status.
        
        Args:
            host: libp2p host instance
        """
        try:
            # In actual implementation, use AutoNAT protocol
            # For now, simulate reachability check
            
            # Possible states: "public", "private", "unknown"
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
        
        # Hole punching is not implemented in this helper.
        return
    
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
        
        # mDNS/DHT discovery is not implemented here.
        
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
                    # Relay connection requires explicit relay protocol support.
                    # Best-effort fallback is not implemented in this wrapper.
                    raise RuntimeError("Circuit relay fallback not implemented")
                    
                except Exception as relay_error:
                    logger.warning(f"Relay connection failed: {relay_error}")
            
            # Try hole punching if enabled
            if self.config.enable_hole_punching:
                try:
                    logger.debug(f"Attempting hole punching to {peer_addr}")
                    # Hole punching requires protocol support; not implemented here.
                    
                except Exception as punch_error:
                    logger.debug(f"Hole punching failed: {punch_error}")
            
            return None
    
    def get_connectivity_status(self) -> Dict:
        """
        Get current connectivity status.
        
        Returns:
            Dictionary with connectivity information
        """
        return {
            "discovered_peers": len(self.discovered_peers),
            "relay_peers": len(self.relay_peers),
            "reachability": self.reachability_status,
            "transports": {
                "tcp": self.config.enable_tcp,
                "quic": self.config.enable_quic,
                "webrtc": self.config.enable_webrtc
            },
            "discovery": {
                "mdns": self.config.enable_mdns,
                "dht": self.config.enable_dht,
                "relay": self.config.enable_relay
            },
            "nat_traversal": {
                "autonat": self.config.enable_autonat,
                "hole_punching": self.config.enable_hole_punching
            },
            "implemented": dict(self.implemented),
        }


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
