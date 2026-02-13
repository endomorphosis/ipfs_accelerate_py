"""
P2P Bootstrap Helper for MCP++ (refactored from original github_cli module).

This module provides a simplified peer discovery mechanism that works
in GitHub Actions without requiring gh cache upload/download commands,
refactored for the MCP++ Trio-native architecture.

Module: ipfs_accelerate_py.mcplusplus_module.p2p.bootstrap
Refactored from: ipfs_accelerate_py/github_cli/p2p_bootstrap_helper.py

Uses:
1. Environment variables for static bootstrap peers
2. GitHub repository dispatch events for dynamic peer discovery
3. Simple file-based peer registry for local testing
"""

import json
import logging
import os
import socket
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ...common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
except ImportError:
    try:
        from ..common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
    except ImportError:
        try:
            from test.common.storage_wrapper import get_storage_wrapper, HAVE_STORAGE_WRAPPER
        except ImportError:
            HAVE_STORAGE_WRAPPER = False

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

storage_wrapper = get_storage_wrapper if HAVE_STORAGE_WRAPPER else None

logger = logging.getLogger(__name__)


class SimplePeerBootstrap:
    """
    Simplified peer bootstrap for P2P cache sharing.
    
    Works in GitHub Actions without gh cache commands by using:
    - Environment variables for peer configuration
    - File-based registry for local development
    - Fallback to static bootstrap peers
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        peer_ttl_minutes: int = 30
    ):
        """
        Initialize peer bootstrap helper.
        
        Args:
            cache_dir: Directory for peer registry (defaults to ~/.cache/p2p_peers)
            peer_ttl_minutes: How long peer entries are valid
        """
        # Initialize storage wrapper
        self.storage = None
        if storage_wrapper is not None:
            try:
                self.storage = storage_wrapper(auto_detect_ci=True)
            except Exception:
                self.storage = None
        
        if cache_dir is None:
            # Allow overriding for hardened environments (e.g., systemd ProtectHome)
            env_dir = os.environ.get("IPFS_ACCELERATE_P2P_CACHE_DIR")
            cache_dir = Path(env_dir) if env_dir else (Path.home() / ".cache" / "p2p_peers")

        self.cache_dir = Path(cache_dir)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Common under systemd with ProtectHome=read-only (Errno 30)
            fallback = Path("/tmp") / "ipfs_accelerate_p2p_peers"
            logger.warning(f"⚠ P2P peer registry dir not writable ({self.cache_dir}): {e} - falling back to {fallback}")
            self.cache_dir = fallback
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.peer_ttl = timedelta(minutes=peer_ttl_minutes)
        
        # Detect runner information
        self.runner_name = self._detect_runner_name()
        self.public_ip = self._detect_public_ip()
        
        logger.info(f"Peer bootstrap initialized: runner={self.runner_name}, ip={self.public_ip}")
    
    def _detect_runner_name(self) -> str:
        """Detect the GitHub Actions runner name."""
        # Try environment variables
        runner_name = os.environ.get("RUNNER_NAME")
        if runner_name:
            return runner_name
        
        # Try hostname
        try:
            return socket.gethostname()
        except Exception:
            return "unknown-runner"
    
    def _detect_public_ip(self) -> Optional[str]:
        """
        Detect the public IP address of this runner.
        
        This is needed for NAT traversal and peer connectivity.
        """
        try:
            # Try multiple services for redundancy
            services = [
                "https://api.ipify.org",
                "https://ifconfig.me/ip",
                "https://icanhazip.com"
            ]
            
            import urllib.request
            for service in services:
                try:
                    with urllib.request.urlopen(service, timeout=5) as response:
                        return response.read().decode('utf-8').strip()
                except Exception:
                    continue
            
            return None
        except Exception as e:
            logger.warning(f"Failed to detect public IP: {e}")
            return None
    
    def register_peer(
        self,
        peer_id: str,
        listen_port: int,
        multiaddr: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Register this runner as an active peer in local file registry.
        
        Args:
            peer_id: libp2p peer ID
            listen_port: Port the peer is listening on
            multiaddr: Full libp2p multiaddr
            metadata: Optional additional metadata
            
        Returns:
            True if registration succeeded
        """
        try:
            peer_info = {
                "peer_id": peer_id,
                "runner_name": self.runner_name,
                "public_ip": self.public_ip,
                "listen_port": listen_port,
                "multiaddr": multiaddr,
                "last_seen": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store peer info in local file
            peer_file = self.cache_dir / f"peer_{self.runner_name}.json"
            peer_data = json.dumps(peer_info, indent=2)

            # Best-effort: write to distributed storage for debugging, but rely on
            # the local file registry for actual peer discovery.
            if self.storage:
                try:
                    self.storage.write_file(peer_data, filename=peer_file.name, pin=False)
                except Exception:
                    pass

            with open(peer_file, "w") as f:
                f.write(peer_data)
            
            logger.info(f"✓ Registered peer: {peer_id[:16]}... on {self.public_ip}:{listen_port}")
            logger.info(f"  Registry file: {peer_file}")
            return True
                
        except Exception as e:
            logger.error(f"Error registering peer: {e}")
            return False
    
    def discover_peers(self, max_peers: int = 10) -> List[Dict]:
        """
        Discover active peers from file registry.
        
        Args:
            max_peers: Maximum number of peers to return
            
        Returns:
            List of peer info dictionaries
        """
        try:
            peers = []
            
            # Read all peer files
            for peer_file in self.cache_dir.glob("peer_*.json"):
                # Skip our own file
                if peer_file.name == f"peer_{self.runner_name}.json":
                    continue
                
                try:
                    with open(peer_file, "r") as f:
                        peer_info = json.load(f)
                    
                    # Check if peer is still active (within TTL)
                    last_seen = datetime.fromisoformat(peer_info["last_seen"])
                    if datetime.utcnow() - last_seen < self.peer_ttl:
                        peers.append(peer_info)
                    else:
                        logger.debug(f"Peer {peer_info.get('peer_id', 'unknown')[:16]}... expired")
                        # Clean up stale peer file
                        peer_file.unlink()
                        
                except Exception as e:
                    logger.debug(f"Failed to read peer file {peer_file}: {e}")
                    continue
                
                if len(peers) >= max_peers:
                    break
            
            logger.info(f"✓ Discovered {len(peers)} active peer(s)")
            return peers
            
        except Exception as e:
            logger.error(f"Error discovering peers: {e}")
            return []
    
    def get_bootstrap_addrs(self, max_peers: int = 5) -> List[str]:
        """
        Get bootstrap multiaddrs for discovered peers.
        
        Also includes environment variable bootstrap peers and fallback to
        standard libp2p bootstrap nodes if no other peers are available.
        
        Args:
            max_peers: Maximum number of bootstrap peers
            
        Returns:
            List of libp2p multiaddrs
        """
        bootstrap_addrs = []
        
        # First, check environment variable for static bootstrap peers
        env_peers = os.environ.get("CACHE_BOOTSTRAP_PEERS", "")
        if env_peers:
            for peer_addr in env_peers.split(","):
                peer_addr = peer_addr.strip()
                if peer_addr and peer_addr not in bootstrap_addrs:
                    bootstrap_addrs.append(peer_addr)
                    logger.info(f"  ✓ Static bootstrap peer: {peer_addr}")
        
        # Then, discover dynamic peers from file registry
        peers = self.discover_peers(max_peers)
        for peer in peers:
            multiaddr = peer.get("multiaddr")
            if multiaddr and multiaddr not in bootstrap_addrs:
                bootstrap_addrs.append(multiaddr)
                logger.info(f"  ✓ Discovered peer: {multiaddr}")
        
        # If no peers found, optionally add public libp2p bootstrap nodes.
        # This is OFF by default because:
        # - it can slow startup with repeated failed connect attempts
        # - many environments cannot reach these nodes due to transport/NAT/firewall
        # - it can be misleading ("discovered" peers that are not actually connectable)
        enable_public_bootstrap = os.environ.get("IPFS_ACCELERATE_ENABLE_PUBLIC_BOOTSTRAP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not bootstrap_addrs and enable_public_bootstrap:
            libp2p_bootstrap_nodes = [
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
                "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
            ]
            bootstrap_addrs.extend(libp2p_bootstrap_nodes)
            logger.info(
                f"  ℹ️  Using {len(libp2p_bootstrap_nodes)} public libp2p bootstrap node(s) (IPFS_ACCELERATE_ENABLE_PUBLIC_BOOTSTRAP=1)"
            )
        
        return bootstrap_addrs[:max_peers]
    
    def cleanup_stale_peers(self) -> int:
        """
        Remove stale peer entries from the registry.
        
        Returns:
            Number of peers cleaned up
        """
        try:
            cleaned = 0
            
            for peer_file in self.cache_dir.glob("peer_*.json"):
                try:
                    with open(peer_file, "r") as f:
                        peer_info = json.load(f)
                    
                    last_seen = datetime.fromisoformat(peer_info["last_seen"])
                    if datetime.utcnow() - last_seen > self.peer_ttl:
                        peer_file.unlink()
                        cleaned += 1
                        logger.debug(f"Cleaned up stale peer: {peer_file.name}")
                        
                except Exception as e:
                    logger.debug(f"Error checking peer file {peer_file}: {e}")
                    continue
            
            if cleaned > 0:
                logger.info(f"✓ Cleaned up {cleaned} stale peer(s)")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning up peers: {e}")
            return 0
    
    def heartbeat(self, peer_id: str, listen_port: int, multiaddr: str) -> None:
        """
        Send periodic heartbeat to keep peer entry fresh.
        
        Should be called every ~5-10 minutes.
        """
        self.register_peer(peer_id, listen_port, multiaddr)


def get_bootstrap_peers_for_cache() -> List[str]:
    """
    Convenience function to get bootstrap peers for cache initialization.
    
    Returns:
        List of bootstrap peer multiaddrs
    """
    helper = SimplePeerBootstrap()
    return helper.get_bootstrap_addrs()
