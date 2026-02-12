"""
Dashboard Data Operations for IPFS Accelerate MCP Server

This module provides reusable functions for fetching dashboard data including
user information, cache statistics, peer system status, and system metrics.

These functions can be used as:
1. Package imports: `from ipfs_accelerate_py.mcp.tools.dashboard_data import get_user_info`
2. CLI tools: via ipfs-accelerate CLI
3. MCP server tools: via MCP server JavaScript SDK
"""

import os
import time
import logging
import json
import importlib.metadata
from typing import Dict, Any, Optional

# Try to import storage wrapper with comprehensive fallback
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
            def get_storage_wrapper(*args, **kwargs):
                return None

logger = logging.getLogger("ipfs_accelerate_mcp.tools.dashboard_data")

# Initialize storage wrapper at module level
_storage = get_storage_wrapper() if HAVE_STORAGE_WRAPPER else None

# Try to import psutil for system metrics
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    psutil = None

# Track start time for uptime calculation
_START_TIME = time.time()


def get_user_info() -> Dict[str, Any]:
    """
    Get current GitHub user information.
    
    Returns authenticated user data from GitHub including username,
    authentication status, and token type.
    
    Returns:
        Dictionary containing:
        - authenticated: bool - Whether user is authenticated
        - username: str - GitHub username (if authenticated)
        - name: str - User's full name (if available)
        - email: str - User's email (if available)
        - token_type: str - Type of authentication token
        - error: str - Error message (if authentication failed)
    
    Example:
        >>> user_info = get_user_info()
        >>> if user_info['authenticated']:
        ...     print(f"Logged in as: {user_info['username']}")
    """
    try:
        import subprocess
        import json
        
        # Quick check for GITHUB_TOKEN environment variable first
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            # Try to get user info directly with token
            try:
                result = subprocess.run(
                    ['gh', 'api', '/user'],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    env={**os.environ, 'GH_TOKEN': github_token}
                )
                if result.returncode == 0:
                    user_data = json.loads(result.stdout)
                    return {
                        'authenticated': True,
                        'username': user_data.get('login', 'Unknown'),
                        'name': user_data.get('name', ''),
                        'email': user_data.get('email', ''),
                        'avatar_url': user_data.get('avatar_url', ''),
                        'public_repos': user_data.get('public_repos', 0),
                        'followers': user_data.get('followers', 0),
                        'following': user_data.get('following', 0),
                        'token_type': 'environment'
                    }
            except subprocess.TimeoutExpired:
                logger.debug("Timeout fetching user info via GITHUB_TOKEN")
            except Exception as e:
                logger.debug(f"Could not fetch user info via GITHUB_TOKEN: {e}")
        
        # Try using GitHubCLI wrapper with short timeout
        try:
            from ipfs_accelerate_py.github_cli import GitHubCLI
            # Disable auto_refresh to avoid interactive prompts
            gh = GitHubCLI(auto_refresh_token=False)
            auth_status = gh.get_auth_status()
            
            if auth_status.get('authenticated'):
                # Try to get detailed user info with very short timeout
                try:
                    result = subprocess.run(
                        ['gh', 'api', '/user'],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        user_data = json.loads(result.stdout)
                        return {
                            'authenticated': True,
                            'username': user_data.get('login', auth_status.get('username', 'Unknown')),
                            'name': user_data.get('name', ''),
                            'email': user_data.get('email', ''),
                            'avatar_url': user_data.get('avatar_url', ''),
                            'public_repos': user_data.get('public_repos', 0),
                            'followers': user_data.get('followers', 0),
                            'following': user_data.get('following', 0),
                            'token_type': auth_status.get('token_type', 'unknown')
                        }
                except (subprocess.TimeoutExpired, Exception) as e:
                    logger.debug(f"Could not fetch detailed user info: {e}")
                
                # Fallback to basic auth status
                return {
                    'authenticated': True,
                    'username': auth_status.get('username', 'Unknown'),
                    'token_type': auth_status.get('token_type', 'unknown')
                }
            else:
                return {
                    'authenticated': False,
                    'error': 'Not authenticated with GitHub'
                }
        except Exception as e:
            logger.debug(f"GitHubCLI not available: {e}")
            return {
                'authenticated': False,
                'error': 'GitHub CLI not configured'
            }
            
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return {
            'authenticated': False,
            'error': str(e)
        }


def get_cache_stats() -> Dict[str, Any]:
    """
    Get GitHub API cache statistics.
    
    Returns cache performance metrics including total entries, size,
    hit rate, and P2P status.
    
    Returns:
        Dictionary containing:
        - available: bool - Whether cache is available
        - total_entries: int - Number of cached entries
        - total_size_mb: float - Cache size in megabytes
        - hit_rate: float - Cache hit rate (0.0-1.0)
        - total_hits: int - Total cache hits
        - total_misses: int - Total cache misses
        - p2p_enabled: bool - Whether P2P cache sharing is enabled
        - p2p_peers: int - Number of P2P peers
        - error: str - Error message (if cache unavailable)
    
    Example:
        >>> cache_stats = get_cache_stats()
        >>> if cache_stats['available']:
        ...     print(f"Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")
    """
    try:
        from ipfs_accelerate_py.github_cli.cache import get_global_cache
        cache = get_global_cache()
        stats = cache.get_stats()

        total_size_bytes = float(stats.get('total_size_bytes', 0) or 0)
        total_size_mb = total_size_bytes / (1024 * 1024)

        # hit_rate is 0..1 in the cache layer
        hit_rate_float = float(stats.get('hit_rate', 0) or 0)
        hit_rate_percent = max(0.0, min(1.0, hit_rate_float)) * 100.0

        connected_peers = int(
            (stats.get('connected_peers') if stats.get('connected_peers') is not None else stats.get('p2p_peers'))
            or 0
        )

        return {
            'available': True,
            'total_entries': stats.get('total_entries', 0),
            'total_size_mb': total_size_mb,
            # Back-compat numeric hit rate (0..1)
            'hit_rate': hit_rate_float,
            # Dashboard-friendly formatted strings
            'cache_size': f"{total_size_mb:.2f} MB",
            'hit_rate_percent': hit_rate_percent,
            'hit_rate_display': f"{hit_rate_percent:.1f}%",
            'total_hits': stats.get('hits', 0),
            'total_misses': stats.get('misses', 0),
            'total_requests': stats.get('hits', 0) + stats.get('misses', 0),
            'cache_dir': str(stats.get('cache_dir', '')),
            'p2p_enabled': stats.get('p2p_enabled', False),
            'p2p_peers': connected_peers,
            'connected_peers': connected_peers,
            'content_addressing_available': stats.get('content_addressing_available', False),

            # P2P cache sharing metrics
            'local_hits': stats.get('local_hits', stats.get('hits', 0)),
            'peer_hits': stats.get('peer_hits', 0),
            'peer_id': stats.get('peer_id'),
            'known_peer_multiaddrs': stats.get('known_peer_multiaddrs'),
            'peer_exchange_last_iso': stats.get('peer_exchange_last_iso'),
            'peer_exchange_last': stats.get('peer_exchange_last'),
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {
            'available': False,
            'error': str(e)
        }


def get_peer_status() -> Dict[str, Any]:
    """
    Get P2P peer system status.
    
    Returns information about the peer-to-peer cache sharing system
    including whether it's enabled, active, and connected peer count.
    
    Returns:
        Dictionary containing:
        - enabled: bool - Whether P2P is enabled
        - active: bool - Whether P2P is actively running
        - peer_count: int - Number of connected peers
        - peers: list - List of peer information (if available)
        - error: str - Error message (if P2P unavailable)
    
    Example:
        >>> peer_status = get_peer_status()
        >>> if peer_status['enabled']:
        ...     print(f"Connected to {peer_status['peer_count']} peers")
    """
    def _get_libp2p_info() -> Dict[str, Any]:
        """Return installed libp2p version and (if available) VCS ref info."""
        try:
            dist = importlib.metadata.distribution("libp2p")
        except importlib.metadata.PackageNotFoundError:
            return {
                "available": False,
                "error": "libp2p not installed"
            }

        info: Dict[str, Any] = {
            "available": True,
            "version": dist.version,
        }

        # If installed from a VCS URL, pip writes direct_url.json into dist-info
        try:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct_url = json.loads(direct_url_text)
                info["direct_url"] = direct_url
                vcs_info = direct_url.get("vcs_info") or {}
                if isinstance(vcs_info, dict):
                    info["vcs_ref"] = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
                    info["vcs"] = vcs_info.get("vcs")
        except Exception:
            # Best-effort only
            pass

        return info

    try:
        from ipfs_accelerate_py.github_cli.cache import get_global_cache
        cache = get_global_cache()
        
        # Get cache stats which include P2P info
        stats = cache.get_stats()
        
        # Try to get more detailed P2P info if available
        enabled = bool(stats.get('p2p_enabled', False))
        connected_peer_count = int(
            (stats.get('connected_peers') if stats.get('connected_peers') is not None else stats.get('p2p_peers'))
            or 0
        )

        peer_info = {
            # Back-compat fields
            'enabled': enabled,
            'active': enabled,
            # peer_count is the libp2p network connected-peer count
            'peer_count': connected_peer_count,
            'peers': [],

            # Frontend-friendly aliases
            'p2p_enabled': enabled,
            'status': 'Active' if enabled else 'Disabled',

            # Explicit counters (useful for debugging/telemetry)
            'connected_peer_count': connected_peer_count,
            'registered_peer_count': 0,

            # Diagnostics
            'libp2p': _get_libp2p_info(),
        }

        # Pass through peer-exchange debug fields if available
        for k in (
            'peer_exchange_protocol',
            'peer_exchange_interval_s',
            'peer_exchange_last',
            'peer_exchange_last_iso',
            'peer_exchange_last_by_peer',
            'known_peer_multiaddrs',
        ):
            if k in stats:
                peer_info[k] = stats.get(k)

        # Pass through universal-connectivity diagnostics if available
        if isinstance(stats.get('connectivity'), dict):
            peer_info['connectivity'] = stats.get('connectivity')
        
        # Try to get peer registry info if available
        try:
            repo = os.environ.get('IPFS_ACCELERATE_GITHUB_REPO') or os.environ.get('GITHUB_REPOSITORY', '')
            if repo and peer_info['enabled']:
                from ipfs_accelerate_py.github_cli.p2p_peer_registry import P2PPeerRegistry
                registry = P2PPeerRegistry(repo=repo)
                
                # Get registered peers (back-compat with older registry APIs)
                peers = []
                list_peers = getattr(registry, 'list_peers', None)
                if callable(list_peers):
                    peers = list_peers(max_peers=50)
                else:
                    discover_peers = getattr(registry, 'discover_peers', None)
                    if callable(discover_peers):
                        peers = discover_peers(max_peers=50)

                if isinstance(peers, list):
                    peer_info['peers'] = [
                        {
                            'peer_id': p.get('peer_id', 'unknown'),
                            'runner_name': p.get('runner_name') or p.get('metadata', {}).get('runner_name', 'unknown'),
                            'listen_port': p.get('listen_port', 0),
                            'last_seen': p.get('last_seen', '')
                        }
                        for p in peers
                        if isinstance(p, dict)
                    ]
                    peer_info['registered_peer_count'] = len(peer_info['peers'])
                peer_info['status'] = 'Active' if enabled else peer_info['status']
        except Exception as e:
            logger.debug(f"Could not get detailed peer info: {e}")
        
        return peer_info
    except Exception as e:
        logger.error(f"Error getting peer status: {e}")
        return {
            'enabled': False,
            'active': False,
            'peer_count': 0,
            'peers': [],
            'p2p_enabled': False,
            'status': 'Error',
            'libp2p': _get_libp2p_info(),
            'error': str(e)
        }


def get_system_metrics(start_time: Optional[float] = None) -> Dict[str, Any]:
    """
    Get real system metrics.
    
    Returns current system performance metrics including CPU usage,
    memory usage, and uptime.
    
    Args:
        start_time: Optional start time for uptime calculation.
                   If None, uses module-level start time.
    
    Returns:
        Dictionary containing:
        - cpu_percent: float - CPU usage percentage
        - memory_percent: float - Memory usage percentage
        - memory_used_gb: float - Used memory in GB
        - memory_total_gb: float - Total memory in GB
        - uptime: str - Human-readable uptime string
        - uptime_seconds: int - Uptime in seconds
        - active_connections: int - Number of active connections
        - pid: int - Process ID
        - error: str - Error message (if metrics unavailable)
    
    Example:
        >>> metrics = get_system_metrics()
        >>> print(f"CPU: {metrics['cpu_percent']}%, Memory: {metrics['memory_percent']}%")
    """
    if start_time is None:
        start_time = _START_TIME
    
    try:
        # Try to import psutil for detailed metrics
        if HAVE_PSUTIL:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory info
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            memory_total_gb = memory.total / (1024 ** 3)
            
            # Get process info for uptime
            process = psutil.Process(os.getpid())
            uptime_seconds = time.time() - process.create_time()
            
            # Get connection count (approximate from open files/sockets)
            connections = len(process.connections())
            
        else:
            # Fallback to basic metrics if psutil not available
            logger.debug("psutil not available, using basic metrics")
            cpu_percent = 0
            memory_percent = 0
            memory_used_gb = 0
            memory_total_gb = 0
            uptime_seconds = time.time() - start_time
            connections = 1  # At least the dashboard connection
        
        # Format uptime
        if uptime_seconds < 60:
            uptime = f"{int(uptime_seconds)}s"
        elif uptime_seconds < 3600:
            uptime = f"{int(uptime_seconds / 60)}m"
        elif uptime_seconds < 86400:
            uptime = f"{int(uptime_seconds / 3600)}h"
        else:
            uptime = f"{int(uptime_seconds / 86400)}d"
        
        return {
            'cpu_percent': round(cpu_percent, 1),
            'memory_percent': round(memory_percent, 1),
            'memory_used_gb': round(memory_used_gb, 2),
            'memory_total_gb': round(memory_total_gb, 2),
            'uptime': uptime,
            'uptime_seconds': int(uptime_seconds),
            'active_connections': connections,
            'pid': os.getpid()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        # Return fallback data
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'uptime': 'unknown',
            'uptime_seconds': 0,
            'active_connections': 0,
            'error': str(e)
        }


def register_tools(mcp):
    """
    Register dashboard data tools with the MCP server.
    
    This makes the dashboard data functions available as MCP tools
    that can be called by MCP clients/SDKs.
    
    Args:
        mcp: MCP server instance
    """
    
    @mcp.tool()
    def get_dashboard_user_info() -> Dict[str, Any]:
        """
        Get GitHub user information for dashboard
        
        Returns the current GitHub user's authentication status and profile
        information including username, name, email, and token type.
        
        Returns:
            Dictionary with user information
        """
        return get_user_info()
    
    @mcp.tool()
    def get_dashboard_cache_stats() -> Dict[str, Any]:
        """
        Get cache statistics for dashboard
        
        Returns GitHub API cache performance metrics including total entries,
        size, hit rate, and P2P status.
        
        Returns:
            Dictionary with cache statistics
        """
        return get_cache_stats()
    
    @mcp.tool()
    def get_dashboard_peer_status() -> Dict[str, Any]:
        """
        Get P2P peer system status for dashboard
        
        Returns information about the peer-to-peer cache sharing system
        including enabled status, active peers, and peer count.
        
        Returns:
            Dictionary with peer system status
        """
        return get_peer_status()
    
    @mcp.tool()
    def get_dashboard_system_metrics() -> Dict[str, Any]:
        """
        Get system metrics for dashboard
        
        Returns real-time system performance metrics including CPU usage,
        memory usage, uptime, and active connections.
        
        Returns:
            Dictionary with system metrics
        """
        return get_system_metrics()
    
    logger.info("Registered dashboard data tools with MCP server")
