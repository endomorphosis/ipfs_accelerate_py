"""
Network Kit Module

This module provides a unified interface for IPFS network and peer operations
by wrapping the external ipfs_kit_py package. It follows the kit pattern
established for other modules and provides core functionality that is exposed
through both CLI and MCP tools.

Architecture:
    External ipfs_kit_py package (git submodule)
        ↓
    network_kit.py (this module - wraps external package)
        ↓
    ├─ unified_cli.py (CLI wrapper)
    └─ mcp/unified_tools.py (MCP wrapper)

Key Features:
- Peer management (connect, disconnect, list)
- DHT operations (put, get)
- Swarm information and statistics
- Bandwidth monitoring
- libp2p network operations
- Graceful fallback when ipfs_kit_py unavailable

Usage:
    from ipfs_accelerate_py.kit.network_kit import NetworkKit, NetworkConfig
    
    # Initialize
    config = NetworkConfig()
    kit = NetworkKit(config)
    
    # List peers
    result = kit.list_peers()
    print(f"Connected peers: {len(result.data['peers'])}")
    
    # Connect to peer
    result = kit.connect_peer("/ip4/1.2.3.4/tcp/4001/p2p/QmPeerID")
    
    # DHT operations
    result = kit.dht_put("key", "value")
    result = kit.dht_get("key")
"""

import os
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union

# Configure logging
logger = logging.getLogger("ipfs_accelerate.kit.network")


@dataclass
class NetworkConfig:
    """Configuration for network operations."""
    
    # Whether to enable ipfs_kit_py integration
    enable_ipfs_kit: bool = True
    
    # Timeout for operations (seconds)
    timeout: int = 30
    
    # Whether to use local IPFS node if available
    use_local_node: bool = True
    
    # Default bootstrap peers
    bootstrap_peers: List[str] = field(default_factory=list)


@dataclass
class PeerInfo:
    """Information about a peer."""
    
    peer_id: str
    addresses: List[str] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    latency: Optional[int] = None  # milliseconds
    connected: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BandwidthStats:
    """Bandwidth statistics."""
    
    total_in: int = 0  # bytes
    total_out: int = 0  # bytes
    rate_in: float = 0.0  # bytes/sec
    rate_out: float = 0.0  # bytes/sec
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NetworkResult:
    """Result from a network operation."""
    
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class NetworkKit:
    """
    Network operations kit.
    
    Provides unified interface for IPFS network and peer operations by wrapping
    the external ipfs_kit_py package. Includes graceful fallback when the package
    is unavailable.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize network kit.
        
        Args:
            config: Configuration for network operations
        """
        self.config = config or NetworkConfig()
        self.ipfs_client = None
        self.ipfs_available = False
        
        # Try to initialize ipfs_kit_py client
        if self.config.enable_ipfs_kit:
            self._init_ipfs_client()
    
    def _init_ipfs_client(self):
        """Initialize IPFS client from ipfs_kit_py package."""
        try:
            # Try importing the external ipfs_kit_py package
            import ipfs_kit_py
            
            # Try to get the network client
            try:
                from ipfs_accelerate_py.ipfs_kit_integration import IPFSKitStorage
                self.ipfs_client = IPFSKitStorage(enable_ipfs_kit=True)
                self.ipfs_available = True
                logger.info("Network client initialized using ipfs_kit_integration")
            except ImportError:
                # Try direct import
                try:
                    from ipfs_kit_py import IPFSApi
                    self.ipfs_client = IPFSApi()
                    self.ipfs_available = True
                    logger.info("Network client initialized using IPFSApi")
                except (ImportError, AttributeError):
                    logger.warning("Could not initialize ipfs_kit_py client")
                    
        except ImportError as e:
            logger.warning(f"ipfs_kit_py not available: {e}. Using fallback mode.")
    
    def list_peers(self) -> NetworkResult:
        """
        List connected peers.
        
        Returns:
            NetworkResult with list of peers
        """
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    if hasattr(self.ipfs_client, 'swarm_peers'):
                        peers_data = self.ipfs_client.swarm_peers()
                        peers = [
                            PeerInfo(
                                peer_id=p.get('Peer', p.get('peer_id', 'unknown')),
                                addresses=p.get('Addrs', []),
                                connected=True
                            )
                            for p in (peers_data if isinstance(peers_data, list) else peers_data.get('Peers', []))
                        ]
                        
                        return NetworkResult(
                            success=True,
                            message=f"Found {len(peers)} connected peers",
                            data={'peers': [p.to_dict() for p in peers], 'count': len(peers)}
                        )
                        
                except Exception as e:
                    logger.warning(f"ipfs_kit_py list_peers failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'swarm', 'peers'],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    peers = []
                    for line in result.stdout.strip().split('\n'):
                        if line:
                            # Extract peer ID from multiaddr
                            parts = line.split('/p2p/')
                            if len(parts) == 2:
                                peer_id = parts[1]
                                peers.append(PeerInfo(
                                    peer_id=peer_id,
                                    addresses=[line],
                                    connected=True
                                ))
                    
                    return NetworkResult(
                        success=True,
                        message=f"Found {len(peers)} connected peers (CLI)",
                        data={'peers': [p.to_dict() for p in peers], 'count': len(peers)}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'swarm', 'peers'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message="IPFS not available (no ipfs_kit_py or CLI)",
                    error=str(e),
                    data={'peers': [], 'count': 0}
                )
                
        except Exception as e:
            logger.error(f"Error listing peers: {e}")
            return NetworkResult(
                success=False,
                message=f"Error listing peers: {str(e)}",
                error=type(e).__name__,
                data={'peers': [], 'count': 0}
            )
    
    def connect_peer(self, peer_address: str) -> NetworkResult:
        """
        Connect to a peer.
        
        Args:
            peer_address: Multiaddr of peer to connect to
            
        Returns:
            NetworkResult with connection status
        """
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    if hasattr(self.ipfs_client, 'swarm_connect'):
                        result = self.ipfs_client.swarm_connect(peer_address)
                        return NetworkResult(
                            success=True,
                            message=f"Connected to peer: {peer_address}",
                            data={'peer_address': peer_address, 'connected': True}
                        )
                        
                except Exception as e:
                    logger.warning(f"ipfs_kit_py connect_peer failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'swarm', 'connect', peer_address],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return NetworkResult(
                        success=True,
                        message=f"Connected to peer (CLI): {peer_address}",
                        data={'peer_address': peer_address, 'connected': True}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'swarm', 'connect'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"Failed to connect to peer: {str(e)}",
                    error=str(e),
                    data={'peer_address': peer_address, 'connected': False}
                )
                
        except Exception as e:
            logger.error(f"Error connecting to peer: {e}")
            return NetworkResult(
                success=False,
                message=f"Error connecting to peer: {str(e)}",
                error=type(e).__name__,
                data={'peer_address': peer_address, 'connected': False}
            )
    
    def disconnect_peer(self, peer_id: str) -> NetworkResult:
        """
        Disconnect from a peer.
        
        Args:
            peer_id: Peer ID to disconnect from
            
        Returns:
            NetworkResult with disconnection status
        """
        try:
            # Try using ipfs_kit_py if available
            if self.ipfs_available and self.ipfs_client:
                try:
                    if hasattr(self.ipfs_client, 'swarm_disconnect'):
                        result = self.ipfs_client.swarm_disconnect(peer_id)
                        return NetworkResult(
                            success=True,
                            message=f"Disconnected from peer: {peer_id}",
                            data={'peer_id': peer_id, 'connected': False}
                        )
                        
                except Exception as e:
                    logger.warning(f"ipfs_kit_py disconnect_peer failed, trying fallback: {e}")
            
            # Fallback: try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'swarm', 'disconnect', peer_id],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return NetworkResult(
                        success=True,
                        message=f"Disconnected from peer (CLI): {peer_id}",
                        data={'peer_id': peer_id, 'connected': False}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'swarm', 'disconnect'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"Failed to disconnect from peer: {str(e)}",
                    error=str(e),
                    data={'peer_id': peer_id, 'connected': True}
                )
                
        except Exception as e:
            logger.error(f"Error disconnecting from peer: {e}")
            return NetworkResult(
                success=False,
                message=f"Error disconnecting from peer: {str(e)}",
                error=type(e).__name__,
                data={'peer_id': peer_id, 'connected': True}
            )
    
    def dht_put(self, key: str, value: str) -> NetworkResult:
        """
        Put a value in the DHT.
        
        Args:
            key: DHT key
            value: Value to store
            
        Returns:
            NetworkResult with put status
        """
        try:
            # Try using IPFS CLI for DHT operations
            try:
                result = subprocess.run(
                    ['ipfs', 'dht', 'put', key, value],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    return NetworkResult(
                        success=True,
                        message=f"DHT put successful: {key}",
                        data={'key': key, 'value': value, 'stored': True}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'dht', 'put'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"DHT put failed: {str(e)}",
                    error=str(e),
                    data={'key': key, 'value': value, 'stored': False}
                )
                
        except Exception as e:
            logger.error(f"Error in DHT put: {e}")
            return NetworkResult(
                success=False,
                message=f"Error in DHT put: {str(e)}",
                error=type(e).__name__,
                data={'key': key, 'value': value, 'stored': False}
            )
    
    def dht_get(self, key: str) -> NetworkResult:
        """
        Get a value from the DHT.
        
        Args:
            key: DHT key to retrieve
            
        Returns:
            NetworkResult with retrieved value
        """
        try:
            # Try using IPFS CLI for DHT operations
            try:
                result = subprocess.run(
                    ['ipfs', 'dht', 'get', key],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    value = result.stdout.strip()
                    return NetworkResult(
                        success=True,
                        message=f"DHT get successful: {key}",
                        data={'key': key, 'value': value, 'found': True}
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'dht', 'get'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"DHT get failed: {str(e)}",
                    error=str(e),
                    data={'key': key, 'value': None, 'found': False}
                )
                
        except Exception as e:
            logger.error(f"Error in DHT get: {e}")
            return NetworkResult(
                success=False,
                message=f"Error in DHT get: {str(e)}",
                error=type(e).__name__,
                data={'key': key, 'value': None, 'found': False}
            )
    
    def get_swarm_info(self) -> NetworkResult:
        """
        Get swarm information and statistics.
        
        Returns:
            NetworkResult with swarm info
        """
        try:
            # Get peer list first
            peers_result = self.list_peers()
            peer_count = peers_result.data.get('count', 0)
            
            # Try using IPFS CLI for additional info
            try:
                # Get swarm addresses
                result = subprocess.run(
                    ['ipfs', 'swarm', 'addrs', 'local'],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                local_addrs = []
                if result.returncode == 0:
                    local_addrs = [line.strip() for line in result.stdout.strip().split('\n') if line]
                
                swarm_info = {
                    'peer_count': peer_count,
                    'local_addresses': local_addrs,
                    'connected': peer_count > 0
                }
                
                return NetworkResult(
                    success=True,
                    message=f"Swarm info retrieved: {peer_count} peers",
                    data=swarm_info
                )
                
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"Could not get swarm info: {str(e)}",
                    error=str(e),
                    data={'peer_count': peer_count, 'local_addresses': [], 'connected': False}
                )
                
        except Exception as e:
            logger.error(f"Error getting swarm info: {e}")
            return NetworkResult(
                success=False,
                message=f"Error getting swarm info: {str(e)}",
                error=type(e).__name__,
                data={'peer_count': 0, 'local_addresses': [], 'connected': False}
            )
    
    def get_bandwidth_stats(self) -> NetworkResult:
        """
        Get bandwidth statistics.
        
        Returns:
            NetworkResult with bandwidth stats
        """
        try:
            # Try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'stats', 'bw'],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    # Parse bandwidth stats
                    stats = BandwidthStats()
                    for line in result.stdout.strip().split('\n'):
                        if 'TotalIn' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                stats.total_in = int(parts[1])
                        elif 'TotalOut' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                stats.total_out = int(parts[1])
                        elif 'RateIn' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                stats.rate_in = float(parts[1])
                        elif 'RateOut' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                stats.rate_out = float(parts[1])
                    
                    return NetworkResult(
                        success=True,
                        message="Bandwidth stats retrieved",
                        data=stats.to_dict()
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'stats', 'bw'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"Could not get bandwidth stats: {str(e)}",
                    error=str(e),
                    data=BandwidthStats().to_dict()
                )
                
        except Exception as e:
            logger.error(f"Error getting bandwidth stats: {e}")
            return NetworkResult(
                success=False,
                message=f"Error getting bandwidth stats: {str(e)}",
                error=type(e).__name__,
                data=BandwidthStats().to_dict()
            )
    
    def ping_peer(self, peer_id: str, count: int = 3) -> NetworkResult:
        """
        Ping a peer to test connectivity.
        
        Args:
            peer_id: Peer ID to ping
            count: Number of pings (default: 3)
            
        Returns:
            NetworkResult with ping results
        """
        try:
            # Try using IPFS CLI
            try:
                result = subprocess.run(
                    ['ipfs', 'ping', '-n', str(count), peer_id],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout
                )
                
                if result.returncode == 0:
                    # Parse ping results
                    latencies = []
                    for line in result.stdout.strip().split('\n'):
                        if 'time=' in line:
                            try:
                                time_part = line.split('time=')[1].split()[0]
                                latency = float(time_part.replace('ms', ''))
                                latencies.append(latency)
                            except (IndexError, ValueError):
                                pass
                    
                    avg_latency = sum(latencies) / len(latencies) if latencies else 0
                    
                    return NetworkResult(
                        success=True,
                        message=f"Ping successful: {peer_id}",
                        data={
                            'peer_id': peer_id,
                            'count': count,
                            'success_count': len(latencies),
                            'avg_latency_ms': avg_latency,
                            'latencies': latencies
                        }
                    )
                else:
                    raise subprocess.CalledProcessError(
                        result.returncode, ['ipfs', 'ping'], result.stderr
                    )
                    
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                return NetworkResult(
                    success=False,
                    message=f"Ping failed: {str(e)}",
                    error=str(e),
                    data={'peer_id': peer_id, 'count': 0, 'success_count': 0, 'avg_latency_ms': 0}
                )
                
        except Exception as e:
            logger.error(f"Error pinging peer: {e}")
            return NetworkResult(
                success=False,
                message=f"Error pinging peer: {str(e)}",
                error=type(e).__name__,
                data={'peer_id': peer_id, 'count': 0, 'success_count': 0, 'avg_latency_ms': 0}
            )


# Singleton instance getter
_network_kit_instance = None

def get_network_kit(config: Optional[NetworkConfig] = None) -> NetworkKit:
    """
    Get the singleton NetworkKit instance.
    
    Args:
        config: Optional configuration (used only for first call)
        
    Returns:
        NetworkKit instance
    """
    global _network_kit_instance
    if _network_kit_instance is None:
        _network_kit_instance = NetworkKit(config)
    return _network_kit_instance
