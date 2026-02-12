"""
GitHub API Response Cache

This module provides caching capabilities for GitHub API responses to reduce
the number of API calls and avoid rate limiting.

Uses content-addressed hashing (multiformats) to intelligently detect stale cache.
Supports P2P cache sharing via libp2p for distributed runners with encryption
using GitHub token as shared secret (only runners with same GitHub access can decrypt).
"""

import json
import logging
import os
import time
import hashlib
import anyio
import anyio
import base64
import threading
from datetime import datetime, timezone
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from threading import Lock

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

storage_wrapper = get_storage_wrapper if HAVE_STORAGE_WRAPPER else None

if HAVE_STORAGE_WRAPPER:
    try:
        _storage = get_storage_wrapper(auto_detect_ci=True)
    except Exception:
        _storage = None
else:
    _storage = None

# Try to import cryptography for message encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    HAVE_CRYPTO = True
except ImportError:
    HAVE_CRYPTO = False
    Fernet = None
    PBKDF2HMAC = None
    hashes = None
    default_backend = None

# Try to import multiformats for content-addressed caching
# Note: multiformats provides its own multihash submodule, separate from pymultihash
try:
    from multiformats import CID
    from multiformats import multihash as multiformats_multihash
    HAVE_MULTIFORMATS = True
except ImportError:
    HAVE_MULTIFORMATS = False
    CID = None
    multiformats_multihash = None

# Try to import libp2p for P2P cache sharing
try:
    # Apply libp2p compatibility patches first
    from .libp2p_compat import ensure_libp2p_compatible
    if ensure_libp2p_compatible():
        from libp2p import new_host
        from libp2p.peer.peerinfo import info_from_p2p_addr
        HAVE_LIBP2P = True
        # INetStream is only used for type hints, so it's optional
        try:
            from libp2p.network.stream.net_stream_interface import INetStream
        except ImportError:
            INetStream = None  # Type hint only, not critical
    else:
        raise ImportError("libp2p compatibility check failed")
except ImportError:
    HAVE_LIBP2P = False
    new_host = None
    info_from_p2p_addr = None
    INetStream = None

logger = logging.getLogger(__name__)



@dataclass
class CacheEntry:
    """Represents a cached API response with content-based validation."""
    data: Any
    timestamp: float
    ttl: int  # Time to live in seconds
    content_hash: Optional[str] = None  # Multihash of validation fields
    validation_fields: Optional[Dict[str, Any]] = None  # Fields used for hash
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.timestamp > self.ttl
    
    def is_stale(self, current_validation_fields: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if cache is stale by comparing validation fields.
        
        Args:
            current_validation_fields: Current values of validation fields
            
        Returns:
            True if cache is stale (hash mismatch), False if still valid
        """
        # If no validation fields, fall back to TTL-based expiration
        if not self.content_hash or not current_validation_fields:
            return self.is_expired()
        
        # Compute hash of current validation fields
        current_hash = GitHubAPICache._compute_validation_hash(current_validation_fields)
        
        # Cache is stale if hash changed
        return current_hash != self.content_hash


class GitHubAPICache:
    """
    Cache for GitHub API responses with TTL and persistence.
    
    Features:
    - In-memory caching with TTL
    - Optional disk persistence
    - Thread-safe operations
    - Automatic expiration
    - Cache statistics
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        default_ttl: int = 300,  # 5 minutes
        max_cache_size: int = 1000,
        enable_persistence: bool = True,
        enable_p2p: bool = True,
        enable_task_p2p_cache: bool | None = None,
        p2p_listen_port: int = 9100,  # Default P2P port (avoiding 9000 for MCP server)
        p2p_bootstrap_peers: Optional[List[str]] = None,
        github_repo: Optional[str] = None,
        enable_peer_discovery: bool = True,
        enable_universal_connectivity: bool = True
    ):
        """
        Initialize the GitHub API cache.
        
        Args:
            cache_dir: Directory for persistent cache (default: ~/.cache/github_cli)
            default_ttl: Default time-to-live for cache entries in seconds
            max_cache_size: Maximum number of entries to keep in memory
            enable_persistence: Whether to persist cache to disk
            enable_p2p: Whether to enable P2P cache sharing via libp2p
            p2p_listen_port: Port for libp2p to listen on (default: 9000)
            p2p_bootstrap_peers: List of bootstrap peer multiaddrs
            github_repo: GitHub repository for peer discovery (e.g., 'owner/repo')
            enable_peer_discovery: Whether to use GitHub cache API for peer discovery
            enable_universal_connectivity: Whether to enable universal connectivity patterns
        """
        # Initialize storage wrapper
        if storage_wrapper:
            try:
                self.storage = storage_wrapper()
            except:
                self.storage = None
        else:
            self.storage = None
        
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.enable_persistence = enable_persistence
        self.enable_p2p = enable_p2p and HAVE_LIBP2P
        self.enable_universal_connectivity = enable_universal_connectivity

        # Optional: reuse the accelerate libp2p task-service cache (cache.get/set)
        # to avoid hammering the GitHub API across machines.
        if enable_task_p2p_cache is None:
            raw_disable = os.environ.get("IPFS_ACCELERATE_PY_GITHUB_CACHE_DISABLE_TASK_P2P") or os.environ.get(
                "IPFS_DATASETS_PY_GITHUB_CACHE_DISABLE_TASK_P2P"
            )
            disabled = str(raw_disable or "").strip().lower() in {"1", "true", "yes", "on"}
            self.enable_task_p2p_cache = not disabled
        else:
            self.enable_task_p2p_cache = bool(enable_task_p2p_cache)

        # Encryption for task-p2p cache values (must be derived from GitHub auth token).
        self._task_p2p_cipher = None
        if self.enable_task_p2p_cache:
            if not HAVE_CRYPTO:
                logger.warning("⚠ Task P2P cache disabled: cryptography not available for encryption")
                self.enable_task_p2p_cache = False
            else:
                try:
                    self._init_task_p2p_encryption()
                except Exception as e:
                    logger.warning(f"⚠ Task P2P cache disabled: encryption init failed: {e}")
                    self.enable_task_p2p_cache = False
        
        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            env_dir = os.environ.get("IPFS_ACCELERATE_CACHE_DIR")
            self.cache_dir = Path(env_dir) if env_dir else (Path.home() / ".cache" / "github_cli")

        if self.enable_persistence:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

                # mkdir(exist_ok=True) can succeed even when the directory is not writable.
                # Under systemd ProtectHome=read-only, ~/.cache often exists but cannot be written.
                write_probe = self.cache_dir / ".write_test"
                try:
                    with open(write_probe, "w") as f:
                        f.write("ok")
                    write_probe.unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception as e:
                    raise OSError(f"Cache directory not writable: {e}")
            except OSError as e:
                # Common under systemd with ProtectHome=read-only (Errno 30)
                fallback = Path("/tmp") / "ipfs_accelerate_github_cli_cache"
                logger.warning(f"⚠ Cache dir not writable ({self.cache_dir}): {e} - falling back to {fallback}")
                self.cache_dir = fallback
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = Lock()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "peer_hits": 0,
            "expirations": 0,
            "evictions": 0,
            "api_calls_saved": 0,
            "api_calls_made": 0,  # Track actual REST API calls made
            "graphql_api_calls_made": 0,  # Track GraphQL API calls separately
            "graphql_cache_hits": 0  # Track GraphQL cache hits separately
        }
        
        # Aggregate stats from all peers
        self._aggregate_stats = {
            "total_api_calls": 0,
            "total_cache_hits": 0,
            "peer_stats": {},  # Map of peer_id -> stats
            "last_sync": 0
        }
        
        # P2P networking
        self._p2p_host = None
        self._p2p_protocol = "/github-cache/1.0.0"
        self._p2p_peer_exchange_protocol = "/github-cache/peer-exchange/1.0.0"
        self._p2p_listen_port = p2p_listen_port
        if p2p_bootstrap_peers:
            self._p2p_bootstrap_peers = p2p_bootstrap_peers
        else:
            try:
                from .p2p_connectivity import DEFAULT_BOOTSTRAP_PEERS as _DEFAULT_BOOTSTRAP_PEERS
                self._p2p_bootstrap_peers = list(_DEFAULT_BOOTSTRAP_PEERS)
            except Exception:
                self._p2p_bootstrap_peers = []
        self._p2p_connected_peers: Dict[str, Any] = {}
        self._p2p_known_peer_addrs: Set[str] = set(self._p2p_bootstrap_peers)
        self._peer_exchange_last: Dict[str, float] = {}
        self._peer_exchange_interval = 300  # seconds
        self._p2p_portal = None
        self._p2p_thread_running = False
        self._max_bootstrap_peers = 10  # Limit bootstrap peers to prevent connection overload
        self._p2p_init_lock = Lock()  # Lock to prevent concurrent P2P initialization
        self._p2p_initialized = False  # Flag to track if P2P is already initialized
        self._universal_connectivity = None  # Universal connectivity manager
        self._last_bootstrap_refresh = 0  # Timestamp of last bootstrap refresh
        self._bootstrap_refresh_interval = 300  # Re-bootstrap every 5 minutes (like rust-peer)
        self._last_discovery_refresh = 0
        self._discovery_refresh_interval = int(os.environ.get("CACHE_DISCOVERY_REFRESH_INTERVAL", "120"))
        self._min_connected_peers = int(os.environ.get("CACHE_MIN_CONNECTED_PEERS", "1"))
        self._max_discovery_connect_attempts = int(os.environ.get("CACHE_DISCOVERY_CONNECT_ATTEMPTS", "3"))

        # P2P runtime (py-libp2p is Trio-based). We run it in a dedicated thread
        # so the rest of the cache can remain sync-friendly.
        self._p2p_thread: Optional[threading.Thread] = None
        self._p2p_thread_running = False
        self._p2p_trio_token = None
        self._p2p_cancel_scope = None
        self._p2p_ready = threading.Event()

        # Background work queue (e.g., broadcast cache entries) executed on the
        # libp2p Trio thread.
        self._p2p_broadcast_send = None
        
        # Peer discovery - use simplified bootstrap helper
        self.github_repo = github_repo or os.environ.get("IPFS_ACCELERATE_GITHUB_REPO") or os.environ.get("GITHUB_REPOSITORY")
        self.enable_peer_discovery = enable_peer_discovery
        self._peer_registry = None
        self._bootstrap_helper = None
        
        if self.enable_peer_discovery and self.enable_p2p:
            try:
                # Prefer the GitHub issue-backed registry for cross-host rendezvous.
                if self.github_repo:
                    from .p2p_peer_registry import P2PPeerRegistry

                    self._bootstrap_helper = P2PPeerRegistry(repo=self.github_repo)
                else:
                    raise RuntimeError("No github_repo available")
                
                # Get bootstrap peers from environment and discovered peers
                discovered_peers = self._bootstrap_helper.get_bootstrap_addrs(max_peers=10)
                if discovered_peers:
                    # Merge with explicitly provided bootstrap peers
                    self._p2p_bootstrap_peers.extend(discovered_peers)
                    # Remove duplicates
                    self._p2p_bootstrap_peers = list(set(self._p2p_bootstrap_peers))
                    logger.info(f"✓ P2P peer discovery enabled, found {len(discovered_peers)} peer(s)")
                else:
                    logger.info("✓ P2P peer discovery enabled (no peers discovered yet)")
            except Exception as e:
                # Fallback to the local file-based bootstrap helper.
                try:
                    from .p2p_bootstrap_helper import SimplePeerBootstrap

                    self._bootstrap_helper = SimplePeerBootstrap()
                    discovered_peers = self._bootstrap_helper.get_bootstrap_addrs(max_peers=10)
                    if discovered_peers:
                        self._p2p_bootstrap_peers.extend(discovered_peers)
                        self._p2p_bootstrap_peers = list(set(self._p2p_bootstrap_peers))
                        logger.info(f"✓ P2P peer discovery enabled (local), found {len(discovered_peers)} peer(s)")
                    else:
                        logger.info("✓ P2P peer discovery enabled (local, no peers discovered yet)")
                except Exception as e2:
                    logger.warning(f"⚠ P2P peer discovery unavailable: {e} / {e2}")
                    self.enable_peer_discovery = False
        
        # Encryption for P2P messages (using GitHub token as shared secret)
        self._encryption_key = None
        self._cipher = None
        if self.enable_p2p and HAVE_CRYPTO:
            try:
                self._init_encryption()
                logger.info("✓ P2P message encryption enabled")
            except Exception as e:
                logger.warning(f"⚠ P2P encryption unavailable: {e}")
        
        # Load persistent cache if enabled
        if self.enable_persistence:
            self._load_from_disk()
        
        # Initialize P2P if enabled
        logger.info(f"P2P Configuration: enable_p2p={self.enable_p2p}, HAVE_LIBP2P={HAVE_LIBP2P}, port={p2p_listen_port}")
        if self.enable_p2p:
            try:
                logger.info(f"Starting P2P initialization on port {p2p_listen_port}...")
                self._init_p2p()
                logger.info(f"✓ P2P cache sharing enabled on port {p2p_listen_port}")
            except Exception as e:
                logger.warning(f"⚠ Failed to initialize P2P: {e}")
                self.enable_p2p = False
        else:
            logger.info(f"P2P cache sharing disabled (enable_p2p={enable_p2p}, HAVE_LIBP2P={HAVE_LIBP2P})")
    
    def __del__(self):
        """Cleanup when cache is destroyed."""
        self.shutdown()
    
    def shutdown(self) -> None:
        """Gracefully shutdown P2P and save cache."""
        # Save all cache entries to disk
        if self.enable_persistence:
            try:
                with self._lock:
                    for cache_key, entry in self._cache.items():
                        self._save_to_disk(cache_key, entry)
                logger.info(f"✓ Saved {len(self._cache)} cache entries to disk")
            except Exception as e:
                logger.warning(f"Failed to save cache during shutdown: {e}")
        
        # Close P2P host and cleanup connections
        if self.enable_p2p:
            try:
                # Close all peer connections
                if self._p2p_connected_peers:
                    logger.info(f"Closing {len(self._p2p_connected_peers)} peer connections")
                    self._p2p_connected_peers.clear()
                
                # Stop P2P Trio thread (cancels the host.run() context and closes host)
                try:
                    import trio

                    if self._p2p_trio_token is not None and self._p2p_cancel_scope is not None:
                        def _cancel() -> None:
                            try:
                                self._p2p_cancel_scope.cancel()
                            except Exception:
                                pass

                        trio.from_thread.run_sync(_cancel, trio_token=self._p2p_trio_token)
                except Exception:
                    pass

                # Best-effort join
                try:
                    if self._p2p_thread and self._p2p_thread.is_alive():
                        self._p2p_thread.join(timeout=2.0)
                except Exception:
                    pass

                self._p2p_thread_running = False
                self._p2p_host = None
                logger.info("✓ P2P runtime stopped")
                        
            except Exception as e:
                logger.warning(f"Failed to shutdown P2P cleanly: {e}")
    
    def _make_cache_key(self, operation: str, *args, **kwargs) -> str:
        """
        Create a cache key from operation and parameters.
        
        Args:
            operation: Operation name (e.g., 'list_repos', 'workflow_runs')
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        key_parts = [operation] + list(map(str, args)) + [f"{k}={v}" for k, v in sorted_kwargs]
        return ":".join(key_parts)
    
    @staticmethod
    def _compute_validation_hash(validation_fields: Dict[str, Any]) -> str:
        """
        Compute content-addressed hash of validation fields using multiformats.
        
        Args:
            validation_fields: Fields to hash (e.g., {'updatedAt': '2025-11-06T10:00:00Z'})
            
        Returns:
            CID string if multiformats available, otherwise SHA256 hex
        """
        # Sort fields for deterministic hashing
        sorted_fields = json.dumps(validation_fields, sort_keys=True)
        
        if HAVE_MULTIFORMATS:
            # Use multiformats for content-addressed hashing
            content_bytes = sorted_fields.encode('utf-8')
            hasher = hashlib.sha256()
            hasher.update(content_bytes)
            digest = hasher.digest()
            
            # Wrap in multihash (using multiformats' multihash, not pymultihash)
            mh = multiformats_multihash.wrap(digest, 'sha2-256')
            # Create CID
            cid = CID('base32', 1, 'raw', mh)
            return str(cid)
        else:
            # Fallback to simple SHA256 hex
            hasher = hashlib.sha256()
            hasher.update(sorted_fields.encode('utf-8'))
            return hasher.hexdigest()
    
    @staticmethod
    def _extract_validation_fields(operation: str, data: Any) -> Optional[Dict[str, Any]]:
        """
        Extract validation fields from API response based on operation type.
        
        Args:
            operation: Operation name
            data: API response data
            
        Returns:
            Dictionary of fields to use for validation hashing
        """
        if not data:
            return None
        
        validation = {}
        
        # Repository operations - use updatedAt/pushedAt
        if operation.startswith('list_repos') or operation == 'get_repo_info':
            if isinstance(data, list):
                # For list operations, hash all repo update times
                for repo in data:
                    if isinstance(repo, dict):
                        repo_id = repo.get('name') or repo.get('url', '')
                        validation[repo_id] = {
                            'updatedAt': repo.get('updatedAt'),
                            'pushedAt': repo.get('pushedAt')
                        }
            elif isinstance(data, dict):
                # For single repo
                validation['updatedAt'] = data.get('updatedAt')
                validation['pushedAt'] = data.get('pushedAt')
        
        # Issue operations - use updatedAt/state/comments
        elif 'issue' in operation:
            if isinstance(data, list):
                for issue in data:
                    if isinstance(issue, dict):
                        issue_id = str(issue.get('number', issue.get('id', '')))
                        validation[issue_id] = {
                            'state': issue.get('state'),
                            'updatedAt': issue.get('updatedAt') or issue.get('updated_at'),
                            'comments': issue.get('comments', 0)
                        }
            elif isinstance(data, dict):
                validation['state'] = data.get('state')
                validation['updatedAt'] = data.get('updatedAt') or data.get('updated_at')
                validation['comments'] = data.get('comments', 0)
        
        # Pull request operations - use updatedAt/state/mergeable/reviews
        elif 'pull' in operation or 'pr' in operation:
            if isinstance(data, list):
                for pr in data:
                    if isinstance(pr, dict):
                        pr_id = str(pr.get('number', pr.get('id', '')))
                        validation[pr_id] = {
                            'state': pr.get('state'),
                            'updatedAt': pr.get('updatedAt') or pr.get('updated_at'),
                            'mergeable': pr.get('mergeable') or pr.get('mergeableState'),
                            'reviews': pr.get('reviews', {}).get('totalCount', 0) if isinstance(pr.get('reviews'), dict) else 0
                        }
            elif isinstance(data, dict):
                validation['state'] = data.get('state')
                validation['updatedAt'] = data.get('updatedAt') or data.get('updated_at')
                validation['mergeable'] = data.get('mergeable') or data.get('mergeableState')
                validation['reviews'] = data.get('reviews', {}).get('totalCount', 0) if isinstance(data.get('reviews'), dict) else 0
        
        # Comment operations - use updatedAt/body hash
        elif 'comment' in operation:
            if isinstance(data, list):
                for comment in data:
                    if isinstance(comment, dict):
                        comment_id = str(comment.get('id', ''))
                        validation[comment_id] = {
                            'updatedAt': comment.get('updatedAt') or comment.get('updated_at'),
                            'bodyLength': len(comment.get('body', ''))
                        }
            elif isinstance(data, dict):
                validation['updatedAt'] = data.get('updatedAt') or data.get('updated_at')
                validation['bodyLength'] = len(data.get('body', ''))
        
        # Commit operations - use sha
        elif 'commit' in operation:
            if isinstance(data, list):
                for commit in data:
                    if isinstance(commit, dict):
                        commit_sha = commit.get('sha') or commit.get('oid', '')
                        validation[commit_sha] = {
                            'sha': commit_sha,
                            'date': commit.get('committedDate') or commit.get('commit', {}).get('committer', {}).get('date')
                        }
            elif isinstance(data, dict):
                validation['sha'] = data.get('sha') or data.get('oid', '')
                validation['date'] = data.get('committedDate') or data.get('commit', {}).get('committer', {}).get('date')
        
        # Release operations - use tag/publishedAt
        elif 'release' in operation:
            if isinstance(data, list):
                for release in data:
                    if isinstance(release, dict):
                        release_id = str(release.get('id', release.get('tagName', '')))
                        validation[release_id] = {
                            'tagName': release.get('tagName') or release.get('tag_name'),
                            'publishedAt': release.get('publishedAt') or release.get('published_at')
                        }
            elif isinstance(data, dict):
                validation['tagName'] = data.get('tagName') or data.get('tag_name')
                validation['publishedAt'] = data.get('publishedAt') or data.get('published_at')
        
        # Workflow operations - use updatedAt/status/conclusion
        elif 'workflow' in operation:
            if isinstance(data, list):
                for workflow in data:
                    if isinstance(workflow, dict):
                        wf_id = str(workflow.get('databaseId', workflow.get('id', '')))
                        validation[wf_id] = {
                            'status': workflow.get('status'),
                            'conclusion': workflow.get('conclusion'),
                            'updatedAt': workflow.get('updatedAt')
                        }
            elif isinstance(data, dict):
                validation['status'] = data.get('status')
                validation['conclusion'] = data.get('conclusion')
                validation['updatedAt'] = data.get('updatedAt')
        
        # Runner operations - use status/busy
        elif 'runner' in operation:
            if isinstance(data, list):
                for runner in data:
                    if isinstance(runner, dict):
                        runner_id = str(runner.get('id', runner.get('name', '')))
                        validation[runner_id] = {
                            'status': runner.get('status'),
                            'busy': runner.get('busy')
                        }
            elif isinstance(data, dict):
                validation['status'] = data.get('status')
                validation['busy'] = data.get('busy')
        
        # Branch operations - use sha/protection
        elif 'branch' in operation:
            if isinstance(data, list):
                for branch in data:
                    if isinstance(branch, dict):
                        branch_name = branch.get('name', '')
                        validation[branch_name] = {
                            'name': branch_name,
                            'protected': branch.get('protected', False),
                            'sha': branch.get('commit', {}).get('sha', '') if isinstance(branch.get('commit'), dict) else ''
                        }
            elif isinstance(data, dict):
                validation['name'] = data.get('name', '')
                validation['protected'] = data.get('protected', False)
                validation['sha'] = data.get('commit', {}).get('sha', '') if isinstance(data.get('commit'), dict) else ''
        
        # Tag operations - use name/sha
        elif 'tag' in operation:
            if isinstance(data, list):
                for tag in data:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                        validation[tag_name] = {
                            'name': tag_name,
                            'sha': tag.get('commit', {}).get('sha', '') if isinstance(tag.get('commit'), dict) else ''
                        }
            elif isinstance(data, dict):
                validation['name'] = data.get('name', '')
                validation['sha'] = data.get('commit', {}).get('sha', '') if isinstance(data.get('commit'), dict) else ''
        
        # Deployment operations - use updatedAt/state
        elif 'deployment' in operation:
            if isinstance(data, list):
                for deployment in data:
                    if isinstance(deployment, dict):
                        deployment_id = str(deployment.get('id', ''))
                        validation[deployment_id] = {
                            'id': deployment_id,
                            'state': deployment.get('state'),
                            'updatedAt': deployment.get('updatedAt') or deployment.get('updated_at')
                        }
            elif isinstance(data, dict):
                validation['id'] = str(data.get('id', ''))
                validation['state'] = data.get('state')
                validation['updatedAt'] = data.get('updatedAt') or data.get('updated_at')
        
        # Check/status operations - use status/conclusion
        elif 'check' in operation or 'status' in operation:
            if isinstance(data, list):
                for check in data:
                    if isinstance(check, dict):
                        check_id = str(check.get('id', check.get('name', '')))
                        validation[check_id] = {
                            'status': check.get('status'),
                            'conclusion': check.get('conclusion'),
                            'completedAt': check.get('completedAt') or check.get('completed_at')
                        }
            elif isinstance(data, dict):
                validation['status'] = data.get('status')
                validation['conclusion'] = data.get('conclusion')
                validation['completedAt'] = data.get('completedAt') or data.get('completed_at')
        
        # Copilot operations - hash the prompt for deterministic results
        elif operation.startswith('copilot_'):
            # Copilot responses should be stable for same prompts
            # No validation needed - rely on TTL
            return None
        
        return validation if validation else None
    
    def get(
        self,
        operation: str,
        *args,
        validation_fields: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached response with optional validation field checking.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            validation_fields: Current validation fields to check staleness
            **kwargs: Keyword arguments
            
        Returns:
            Cached data or None if not found/expired/stale
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)

        # 1) Local cache first (fast path)
        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is not None:
                if entry.is_expired():
                    logger.debug(f"Cache entry expired for {cache_key}")
                    del self._cache[cache_key]
                    self._stats["expirations"] += 1
                elif validation_fields and entry.is_stale(validation_fields):
                    logger.debug(f"Cache entry stale (hash mismatch) for {cache_key}")
                    del self._cache[cache_key]
                    self._stats["expirations"] += 1
                else:
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit for {cache_key}")
                    return entry.data

        # 2) Optional remote/libp2p task-service cache on miss.
        remote_entry = None
        if self.enable_task_p2p_cache:
            try:
                remote_entry = self._task_p2p_cache_get(cache_key)
            except Exception:
                remote_entry = None

        if isinstance(remote_entry, dict):
            try:
                data = remote_entry.get("data")
                ts = float(remote_entry.get("timestamp") or 0)
                ttl = int(remote_entry.get("ttl") or self.default_ttl)
                content_hash = remote_entry.get("content_hash")
                vfields = remote_entry.get("validation_fields")
                recovered = CacheEntry(
                    data=data,
                    timestamp=ts,
                    ttl=ttl,
                    content_hash=content_hash,
                    validation_fields=vfields if isinstance(vfields, dict) else None,
                )
            except Exception:
                recovered = None

            if recovered is not None and not recovered.is_expired():
                # Validate staleness if caller provided validation fields.
                if validation_fields and recovered.is_stale(validation_fields):
                    recovered = None

            if recovered is not None:
                with self._lock:
                    self._cache[cache_key] = recovered
                    self._stats["peer_hits"] += 1
                    self._stats["api_calls_saved"] += 1
                    if self.enable_persistence:
                        self._save_to_disk(cache_key, recovered)
                return recovered.data

        # Miss
        with self._lock:
            self._stats["misses"] += 1
        return None
    
    def get_stale(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get a cached response even if expired - useful as fallback when API rate limit is hit.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached data or None if not found (ignores expiration)
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            entry = self._cache.get(cache_key)
            
            if entry is None:
                return None
            
            # Return data even if expired
            logger.info(f"Using stale cache data for {cache_key} (API rate limit fallback)")
            return entry.data
    
    def put(
        self,
        operation: str,
        data: Any,
        ttl: Optional[int] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Store a response in the cache with content-based validation.
        
        Args:
            operation: Operation name
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        ttl = ttl if ttl is not None else self.default_ttl
        
        # Extract validation fields and compute hash
        validation_fields = self._extract_validation_fields(operation, data)
        content_hash = None
        if validation_fields:
            content_hash = self._compute_validation_hash(validation_fields)
            logger.debug(f"Computed validation hash for {cache_key}: {content_hash[:16]}...")
        
        entry = None
        with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self.max_cache_size:
                self._evict_oldest()
            
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=ttl,
                content_hash=content_hash,
                validation_fields=validation_fields
            )
            
            self._cache[cache_key] = entry
            logger.debug(f"Cached {cache_key} with TTL {ttl}s")
            
            # Persist to disk if enabled
            if self.enable_persistence:
                self._save_to_disk(cache_key, entry)

        # Write-through to the libp2p task-service cache (best-effort).
        if self.enable_task_p2p_cache and entry is not None:
            try:
                remote_payload = {
                    "cache_key": cache_key,
                    "data": entry.data,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                    "content_hash": entry.content_hash,
                    "validation_fields": entry.validation_fields,
                }
                self._task_p2p_cache_set(cache_key, remote_payload, ttl_s=float(entry.ttl))
            except Exception:
                pass

        # Broadcast to P2P peers if enabled (outside the lock)
        if self.enable_p2p and entry is not None:
            self._broadcast_in_background(cache_key, entry)

    def _get_github_auth_secret(self) -> str:
        """Return GitHub auth token to use as shared secret.

        Uses `GITHUB_TOKEN` if available, otherwise tries `gh auth token`.
        """

        github_token = (os.environ.get("GITHUB_TOKEN") or "").strip()
        if github_token:
            return github_token

        # Try to get token from gh CLI
        try:
            import subprocess

            result = subprocess.run(
                ["gh", "auth", "token"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                token = (result.stdout or "").strip()
                if token:
                    return token
        except Exception:
            pass

        raise RuntimeError("GitHub auth token unavailable (set GITHUB_TOKEN or run 'gh auth login')")

    def _init_task_p2p_encryption(self) -> None:
        """Initialize task-p2p encryption cipher derived from GitHub auth token."""

        if not HAVE_CRYPTO:
            raise RuntimeError("cryptography not available")

        shared_secret = self._get_github_auth_secret()

        # Deterministic derivation so nodes with the same token decrypt each other.
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"github-task-p2p-cache",  # fixed salt
            iterations=100000,
            backend=default_backend(),
        )
        key = base64.urlsafe_b64encode(kdf.derive(shared_secret.encode("utf-8")))
        self._task_p2p_cipher = Fernet(key)

    def _task_p2p_encrypt_value(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt a JSON-serializable dict for storage in task-p2p cache."""

        if not self._task_p2p_cipher:
            raise RuntimeError("Task P2P cipher not initialized")
        plaintext = json.dumps(value, sort_keys=True).encode("utf-8")
        ct = self._task_p2p_cipher.encrypt(plaintext)
        return {"enc": "fernet-v1", "ct": ct.decode("ascii")}

    def _task_p2p_decrypt_value(self, wrapped: Any) -> Any:
        """Decrypt a task-p2p cache value if it is encrypted; otherwise return as-is."""

        if not isinstance(wrapped, dict) or "enc" not in wrapped:
            return wrapped

        if wrapped.get("enc") != "fernet-v1":
            return None
        ct = wrapped.get("ct")
        if not isinstance(ct, str) or not ct:
            return None
        if not self._task_p2p_cipher:
            return None
        try:
            pt = self._task_p2p_cipher.decrypt(ct.encode("ascii"))
            return json.loads(pt.decode("utf-8"))
        except Exception:
            return None

    def _task_p2p_remote(self):
        """Build a p2p_tasks RemoteQueue from environment (best-effort).

        Returns None when remote is not configured.
        """

        remote_peer_id = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_PEER_ID")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_PEER_ID")
            or ""
        ).strip()
        remote_multiaddr = (
            os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_MULTIADDR")
            or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_MULTIADDR")
            or ""
        ).strip()

        if not remote_multiaddr and not remote_peer_id:
            return None

        from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue

        return RemoteQueue(peer_id=remote_peer_id, multiaddr=remote_multiaddr)

    def _task_p2p_cache_get(self, cache_key: str) -> Any | None:
        """Best-effort get from accelerate's libp2p task-service cache."""

        if not cache_key:
            return None
        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_get_sync

            remote = self._task_p2p_remote()
            if remote is None:
                return None

            resp = cache_get_sync(remote=remote, key=str(cache_key), timeout_s=10.0)
            if not isinstance(resp, dict) or not resp.get("ok"):
                return None
            if not resp.get("hit"):
                return None
            value = resp.get("value")
            return self._task_p2p_decrypt_value(value)
        except Exception:
            return None

    def _task_p2p_cache_set(self, cache_key: str, value: Any, *, ttl_s: float | None = None) -> None:
        """Best-effort set to accelerate's libp2p task-service cache."""

        if not cache_key:
            return
        try:
            from ipfs_accelerate_py.p2p_tasks.client import cache_set_sync

            remote = self._task_p2p_remote()
            if remote is None:
                return

            # Require encryption for remote storage.
            if isinstance(value, dict):
                wrapped = self._task_p2p_encrypt_value(value)
            else:
                # Only dict payloads are supported (keeps remote format stable)
                wrapped = self._task_p2p_encrypt_value({"value": value})

            cache_set_sync(remote=remote, key=str(cache_key), value=wrapped, ttl_s=ttl_s, timeout_s=10.0)
        except Exception:
            return
    
    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry."""
        if not self._cache:
            return
        
        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1
        logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def invalidate(self, operation: str, *args, **kwargs) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            operation: Operation name
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        cache_key = self._make_cache_key(operation, *args, **kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                logger.debug(f"Invalidated cache entry: {cache_key}")
                
                # Remove from disk if persistence enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
                    if cache_file.exists():
                        cache_file.unlink()
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'list_repos' will invalidate all list_repos calls)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(pattern)]
            
            for key in keys_to_delete:
                del self._cache[key]
                
                # Remove from disk if persistence enabled
                if self.enable_persistence:
                    cache_file = self.cache_dir / f"{self._sanitize_filename(key)}.json"
                    if cache_file.exists():
                        cache_file.unlink()
            
            logger.info(f"Invalidated {len(keys_to_delete)} cache entries matching '{pattern}'")
            return len(keys_to_delete)
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "peer_hits": 0,
                "expirations": 0,
                "evictions": 0,
                "api_calls_saved": 0
            }
            
            # Clear disk cache if persistence enabled
            if self.enable_persistence and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            logger.info(f"Cleared {count} cache entries")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            local_hits = self._stats["hits"]
            peer_hits = self._stats["peer_hits"]
            total_hits = local_hits + peer_hits
            total_requests = total_hits + self._stats["misses"]
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            # API calls saved = hits (local + peer) that would have been API calls
            api_calls_saved = total_hits
            
            stats = {
                **self._stats,
                "local_hits": local_hits,
                "total_hits": total_hits,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "total_entries": len(self._cache),
                "max_cache_size": self.max_cache_size,
                "api_calls_saved": api_calls_saved,
                "p2p_enabled": self.enable_p2p,
                "content_addressing_available": HAVE_MULTIFORMATS,
                "cache_dir": str(self.cache_dir) if self.enable_persistence else "",
            }

            # Best-effort disk stats for dashboard/debugging.
            if self.enable_persistence:
                total_size_bytes = 0
                disk_entries = 0
                try:
                    if self.cache_dir.exists():
                        for cache_file in self.cache_dir.glob("*.json"):
                            disk_entries += 1
                            try:
                                total_size_bytes += cache_file.stat().st_size
                            except OSError:
                                pass
                except Exception:
                    pass
                stats["disk_entries"] = disk_entries
                stats["total_size_bytes"] = total_size_bytes
            
            # Add P2P-specific stats if enabled
            if self.enable_p2p:
                connected_peers = len(self._p2p_connected_peers)
                # Stable keys used by dashboard/API
                stats["connected_peers"] = connected_peers
                stats["p2p_peers"] = connected_peers
                stats["bootstrap_peers_configured"] = len(self._p2p_bootstrap_peers)

                # Peer-exchange diagnostics (helps validate mesh convergence without DHT/relay)
                stats["peer_exchange_protocol"] = getattr(self, "_p2p_peer_exchange_protocol", None)
                stats["peer_exchange_interval_s"] = getattr(self, "_peer_exchange_interval", None)
                peer_exchange_last_by_peer = getattr(self, "_peer_exchange_last", None)
                latest_peer_exchange_ts = None
                if isinstance(peer_exchange_last_by_peer, dict):
                    try:
                        ts_values = [
                            v
                            for v in peer_exchange_last_by_peer.values()
                            if isinstance(v, (int, float))
                        ]
                        latest_peer_exchange_ts = max(ts_values) if ts_values else None
                    except Exception:
                        latest_peer_exchange_ts = None
                    stats["peer_exchange_last_by_peer"] = peer_exchange_last_by_peer
                elif isinstance(peer_exchange_last_by_peer, (int, float)):
                    latest_peer_exchange_ts = peer_exchange_last_by_peer
                    stats["peer_exchange_last_by_peer"] = None
                else:
                    stats["peer_exchange_last_by_peer"] = None

                stats["peer_exchange_last"] = latest_peer_exchange_ts
                if isinstance(latest_peer_exchange_ts, (int, float)):
                    stats["peer_exchange_last_iso"] = datetime.fromtimestamp(
                        latest_peer_exchange_ts, tz=timezone.utc
                    ).isoformat()
                else:
                    stats["peer_exchange_last_iso"] = None

                stats["known_peer_multiaddrs"] = len(
                    getattr(self, "_p2p_known_peer_addrs", set()) or set()
                )
                if self._p2p_host:
                    stats["peer_id"] = self._p2p_host.get_id().pretty()
                
                # Include aggregate stats from all peers
                stats["aggregate"] = self._get_aggregate_stats()
                
                # Add universal connectivity stats if available
                if self._universal_connectivity:
                    stats["connectivity"] = self._universal_connectivity.get_connectivity_status()
            
            return stats
    
    def _get_aggregate_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all P2P peers.
        
        NOTE: This method expects self._lock to already be held by the caller!
        
        Returns:
            Dictionary with aggregate statistics
        """
        # Lock is already held by caller (get_stats), don't acquire again
        # Sync with peers if needed (every 60 seconds)
        current_time = time.time()
        if current_time - self._aggregate_stats["last_sync"] > 60:
            self._sync_stats_with_peers()
        
        return {
            "total_api_calls": self._aggregate_stats["total_api_calls"],
            "total_cache_hits": self._aggregate_stats["total_cache_hits"],
            "total_peers": len(self._aggregate_stats["peer_stats"]) + 1,  # +1 for self
            "peer_breakdown": self._aggregate_stats["peer_stats"],
            "last_synced": self._aggregate_stats["last_sync"]
        }
    
    def _sync_stats_with_peers(self) -> None:
        """
        Sync statistics with connected P2P peers.
        
        Broadcasts local stats and receives stats from peers to build aggregate view.
        """
        if not self.enable_p2p or not self._p2p_connected_peers:
            return
        
        try:
            # Prepare local stats for sharing
            local_stats = {
                "api_calls_made": self._stats["api_calls_made"],
                "cache_hits": self._stats["hits"] + self._stats["peer_hits"],
                "timestamp": time.time()
            }
            
            # Broadcast stats to peers
            self._broadcast_stats(local_stats)
            
            # Update aggregate with local stats
            peer_id = self._p2p_host.get_id().pretty() if self._p2p_host else "local"
            self._aggregate_stats["peer_stats"][peer_id] = local_stats
            
            # Calculate totals
            total_calls = self._stats["api_calls_made"]
            total_hits = self._stats["hits"] + self._stats["peer_hits"]
            
            for peer_stats in self._aggregate_stats["peer_stats"].values():
                total_calls += peer_stats.get("api_calls_made", 0)
                total_hits += peer_stats.get("cache_hits", 0)
            
            self._aggregate_stats["total_api_calls"] = total_calls
            self._aggregate_stats["total_cache_hits"] = total_hits
            self._aggregate_stats["last_sync"] = time.time()
            
            logger.debug(f"Synced stats: {total_calls} total API calls across {len(self._aggregate_stats['peer_stats'])} peers")
            
        except Exception as e:
            logger.warning(f"Failed to sync stats with peers: {e}")
    
    def _broadcast_stats(self, stats: Dict[str, Any]) -> None:
        """
        Broadcast statistics to all connected P2P peers.
        
        Args:
            stats: Statistics dictionary to broadcast
        """
        if not self.enable_p2p or not self._p2p_connected_peers:
            return
        
        try:
            # Encrypt stats if encryption is enabled
            if self._cipher:
                stats_json = json.dumps(stats)
                encrypted_stats = self._cipher.encrypt(stats_json.encode())
                message = base64.b64encode(encrypted_stats).decode()
            else:
                message = json.dumps(stats)
            
            # Send stats to each connected peer
            # Note: This would need to be implemented with actual P2P message sending
            # For now, this is a placeholder for the P2P protocol
            logger.debug(f"Broadcasting stats to {len(self._p2p_connected_peers)} peers")
            
        except Exception as e:
            logger.warning(f"Failed to broadcast stats: {e}")
    
    def _handle_peer_stats(self, peer_id: str, stats: Dict[str, Any]) -> None:
        """
        Handle incoming statistics from a peer.
        
        Args:
            peer_id: Peer ID
            stats: Statistics from peer
        """
        with self._lock:
            self._aggregate_stats["peer_stats"][peer_id] = stats
            logger.debug(f"Received stats from peer {peer_id[:16]}...")
    
    def increment_api_call_count(self, api_type: str = "rest", operation: str = "unknown") -> None:
        """
        Increment the count of API calls made and log the operation.
        
        Args:
            api_type: Type of API call ("rest", "graphql", or "code_scanning")
            operation: Name of the operation being performed
        
        Should be called whenever an actual API call is made (not cached).
        """
        with self._lock:
            # Track by API type
            if api_type == "graphql":
                self._stats["graphql_api_calls_made"] += 1
                logger.info(f"📡 GraphQL API call #{self._stats['graphql_api_calls_made']}: {operation}")
            elif api_type == "code_scanning":
                if "code_scanning_api_calls" not in self._stats:
                    self._stats["code_scanning_api_calls"] = 0
                self._stats["code_scanning_api_calls"] += 1
                logger.info(f"🔍 CodeQL API call #{self._stats['code_scanning_api_calls']}: {operation}")
            else:
                self._stats["api_calls_made"] += 1
                logger.info(f"📡 REST API call #{self._stats['api_calls_made']}: {operation}")
            
            # Add to API call log (keep last 100 calls)
            if "api_call_log" not in self._stats:
                self._stats["api_call_log"] = []
            
            import time
            self._stats["api_call_log"].append({
                "timestamp": time.time(),
                "api_type": api_type,
                "operation": operation,
                "count": self._stats.get(f"{api_type}_api_calls_made" if api_type == "graphql" else "code_scanning_api_calls" if api_type == "code_scanning" else "api_calls_made", 0)
            })
            
            # Keep only last 100 calls
            if len(self._stats["api_call_log"]) > 100:
                self._stats["api_call_log"] = self._stats["api_call_log"][-100:]
    
    def increment_graphql_cache_hit(self) -> None:
        """
        Increment the count of GraphQL cache hits.
        
        Should be called when a GraphQL query is served from cache.
        """
        with self._lock:
            self._stats["graphql_cache_hits"] += 1
            logger.debug(f"GraphQL cache hits: {self._stats['graphql_cache_hits']}")
    
    def _sanitize_filename(self, key: str) -> str:
        """Sanitize a cache key for use as a filename."""
        # Replace invalid filename characters with underscores
        return key.replace("/", "_").replace(":", "_").replace("*", "_")
    
    def _save_to_disk(self, cache_key: str, entry: CacheEntry) -> None:
        """Save a cache entry to disk."""
        try:
            cache_file = self.cache_dir / f"{self._sanitize_filename(cache_key)}.json"
            cache_data = {
                "cache_key": cache_key,
                "data": entry.data,
                "timestamp": entry.timestamp,
                "ttl": entry.ttl,
                "content_hash": entry.content_hash,
                "validation_fields": entry.validation_fields
            }
            
            cache_json = json.dumps(cache_data)
            with open(cache_file, 'w') as f:
                f.write(cache_json)
            
            # Update distributed storage
            if self.storage:
                try:
                    self.storage.store_file(str(cache_file), cache_json, pin=False)
                except:
                    pass  # Silently fail distributed storage updates
            
            logger.debug(f"Saved cache entry to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache entries from disk."""
        if not self.cache_dir.exists():
            return
        
        loaded_count = 0
        expired_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    # Try distributed storage first
                    if self.storage:
                        try:
                            cached_data = self.storage.get_file(str(cache_file))
                            if cached_data:
                                cache_data = json.loads(cached_data)
                            else:
                                with open(cache_file, 'r') as f:
                                    cache_data = json.load(f)
                                # Cache for future use
                                self.storage.store_file(str(cache_file), json.dumps(cache_data), pin=False)
                        except:
                            with open(cache_file, 'r') as f:
                                cache_data = json.load(f)
                    else:
                        with open(cache_file, 'r') as f:
                            cache_data = json.load(f)
                    
                    entry = CacheEntry(
                        data=cache_data["data"],
                        timestamp=cache_data["timestamp"],
                        ttl=cache_data["ttl"],
                        content_hash=cache_data.get("content_hash"),
                        validation_fields=cache_data.get("validation_fields")
                    )
                    
                    # Only load non-expired entries
                    if not entry.is_expired():
                        # Use original cache key if available (v2 format), else fall back to stem (legacy)
                        cache_key = cache_data.get("cache_key") or cache_file.stem
                        self._cache[cache_key] = entry
                        loaded_count += 1
                    else:
                        # Remove expired cache file
                        try:
                            cache_file.unlink()
                        except OSError:
                            # If running with a read-only cache dir, just leave the file in place.
                            pass
                        expired_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to load cache file {cache_file}: {e}")
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} cache entries from disk ({expired_count} expired)")
        
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    def _init_encryption(self) -> None:
        """
        Initialize encryption for P2P messages using a shared secret.
        
        By default, this derives the secret from the GitHub token so that GitHub Actions
        runners within the same job can share encrypted cache entries.

        For cross-machine / non-Actions usage, set `CACHE_P2P_SHARED_SECRET` to a
        pre-shared value on all nodes to ensure they can decrypt each other.
        """
        if not HAVE_CRYPTO:
            raise RuntimeError("cryptography not available, install with: pip install cryptography")
        
        # Prefer an explicit shared secret (useful for cross-machine testing).
        shared_secret = os.environ.get("CACHE_P2P_SHARED_SECRET")
        secret_source = "CACHE_P2P_SHARED_SECRET"

        # Fallback to GitHub token (legacy/default behavior)
        if not shared_secret:
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                # Try to get token from gh CLI
                try:
                    import subprocess
                    result = subprocess.run(
                        ["gh", "auth", "token"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        github_token = result.stdout.strip()
                except Exception as e:
                    logger.debug(f"Failed to get GitHub token from gh CLI: {e}")

            if not github_token:
                raise RuntimeError(
                    "Shared secret not available (set CACHE_P2P_SHARED_SECRET, or set GITHUB_TOKEN / run 'gh auth login')"
                )

            shared_secret = github_token
            secret_source = "GITHUB_TOKEN/gh"
        
        # Derive encryption key from GitHub token using PBKDF2
        # This ensures all runners with same token get same encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"github-cache-p2p",  # Fixed salt for deterministic key derivation
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(shared_secret.encode('utf-8')))
        self._encryption_key = key
        self._cipher = Fernet(key)
        
        logger.debug(f"Encryption key derived from {secret_source}")
    
    def _encrypt_message(self, data: Dict[str, Any]) -> bytes:
        """
        Encrypt a message for P2P transmission.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Encrypted bytes
        """
        if not self._cipher:
            # No encryption available, send plaintext (with warning)
            logger.warning("P2P message sent unencrypted (cryptography not available)")
            return json.dumps(data).encode('utf-8')
        
        try:
            plaintext = json.dumps(data).encode('utf-8')
            encrypted = self._cipher.encrypt(plaintext)
            return encrypted
        except Exception as e:
            logger.error(f"Failed to encrypt message: {e}")
            raise
    
    def _decrypt_message(self, encrypted_data: bytes) -> Optional[Dict[str, Any]]:
        """
        Decrypt a P2P message.
        
        Args:
            encrypted_data: Encrypted bytes
            
        Returns:
            Decrypted dictionary or None if decryption fails
        """
        if not self._cipher:
            # No encryption available, try parsing as plaintext
            try:
                return json.loads(encrypted_data.decode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to parse unencrypted message: {e}")
                return None
        
        try:
            decrypted = self._cipher.decrypt(encrypted_data)
            return json.loads(decrypted.decode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to decrypt message (wrong key or corrupted): {e}")
            return None
    
    def _init_p2p(self) -> None:
        """Initialize P2P networking for cache sharing in a Trio background thread (thread-safe)."""
        if not HAVE_LIBP2P:
            raise RuntimeError(
                "libp2p not available, install with: "
                "pip install 'libp2p @ git+https://github.com/libp2p/py-libp2p@main'"
            )

        with self._p2p_init_lock:
            if self._p2p_initialized or self._p2p_thread_running:
                logger.debug("P2P already initialized, skipping")
                return

            self._p2p_ready.clear()

            def _runner() -> None:
                try:
                    import trio

                    async def _main() -> None:
                        self._p2p_trio_token = trio.lowlevel.current_trio_token()
                        with trio.CancelScope() as scope:
                            self._p2p_cancel_scope = scope
                            await self._run_p2p_node()

                    trio.run(_main)
                except Exception as e:
                    logger.error(f"P2P runtime error: {e}")
                finally:
                    self._p2p_thread_running = False

            self._p2p_thread = threading.Thread(target=_runner, daemon=True, name="p2p-trio")
            self._p2p_thread.start()

            # Wait briefly so startup errors surface quickly, but don't block forever.
            if not self._p2p_ready.wait(timeout=3.0):
                logger.warning("⚠ P2P is still starting in background")

            self._p2p_thread_running = True
            self._p2p_initialized = True
            logger.info("✓ P2P initialized (Trio)")

    async def _enqueue_broadcast(self, cache_key: str, entry: CacheEntry) -> None:
        """Enqueue a broadcast job onto the P2P Trio thread."""
        if self._p2p_broadcast_send is None:
            return
        try:
            send_nowait = getattr(self._p2p_broadcast_send, "send_nowait", None)
            if callable(send_nowait):
                send_nowait((cache_key, entry))
                return
        except Exception:
            pass

        await self._p2p_broadcast_send.send((cache_key, entry))

    async def _broadcast_worker(self, recv) -> None:
        """Process broadcast jobs from the queue."""
        async for cache_key, entry in recv:
            try:
                await self._broadcast_cache_entry(cache_key, entry)
            except Exception:
                pass

    async def _run_p2p_node(self) -> None:
        """Run libp2p host + background tasks forever (until cancelled)."""
        import trio
        from multiaddr import Multiaddr

        listen_ma = Multiaddr(f"/ip4/0.0.0.0/tcp/{self._p2p_listen_port}")

        # Create host (new_host is synchronous in this libp2p version)
        self._p2p_host = new_host()

        # Set stream handlers
        self._p2p_host.set_stream_handler(self._p2p_protocol, self._handle_cache_stream)
        self._p2p_host.set_stream_handler(
            self._p2p_peer_exchange_protocol,
            self._handle_peer_exchange_stream,
        )

        send, recv = trio.open_memory_channel(200)
        self._p2p_broadcast_send = send

        async with self._p2p_host.run([listen_ma]):
            try:
                logger.info(f"P2P host listening on {self._p2p_listen_port}")
                logger.info(f"Peer ID: {self._p2p_host.get_id().pretty()}")
            except Exception:
                pass

            # Mark ready once we're actually listening.
            self._p2p_ready.set()

            async with trio.open_nursery() as nursery:
                nursery.start_soon(self._broadcast_worker, recv)
                nursery.start_soon(self._bootstrap_and_maintain)
                await trio.sleep_forever()

    async def _bootstrap_and_maintain(self) -> None:
        """Connect to bootstrap peers and then periodically maintain connectivity."""
        import trio

        # Initial bootstrap (best-effort)
        await self._initial_bootstrap_connect()

        # Maintenance loop
        while True:
            await trio.sleep(60)
            try:
                await self._periodic_connection_maintenance()
            except Exception:
                pass
    
    def _validate_multiaddr(self, addr: Optional[str]) -> bool:
        """
        Validate a libp2p multiaddr format.
        
        Args:
            addr: Multiaddr string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not addr or not isinstance(addr, str):
            return False
        
        # Basic format validation
        # Must start with /ip4 or /ip6
        # Must contain /tcp/ and /p2p/
        if not (addr.startswith('/ip4') or addr.startswith('/ip6')):
            return False
        
        if '/tcp/' not in addr or '/p2p/' not in addr:
            return False
        
        return True
    
    def _get_public_ip(self) -> str:
        """Get public IP address for P2P multiaddr."""
        import urllib.request
        
        # Try multiple services
        services = [
            'https://api.ipify.org',
            'https://ifconfig.me/ip',
            'https://icanhazip.com'
        ]
        
        for service in services:
            try:
                with urllib.request.urlopen(service, timeout=5) as response:
                    ip = response.read().decode('utf-8').strip()
                    if ip:
                        return ip
            except Exception:
                continue
        
        # Fallback to localhost if all services fail
        return "127.0.0.1"
    
    async def _start_p2p_host(self) -> None:
        """Legacy entrypoint (kept for compatibility)."""
        raise RuntimeError("Use _init_p2p() to start the Trio P2P runtime")

    async def _initial_bootstrap_connect(self) -> None:
        """Initial peer registry registration + bootstrap connect (best-effort)."""
        import trio

        if not self._p2p_host:
            return

        # Best-effort register this peer with discovery backend.
        try:
            if self._bootstrap_helper:
                peer_id = self._p2p_host.get_id().pretty()
                advertised = self._get_advertised_multiaddrs()

                # When using the local file-based registry (SimplePeerBootstrap),
                # peers are typically on the same host. Advertising a public IP
                # can fail due to hairpin NAT or firewall policies, so prefer a
                # loopback multiaddr for local discovery.
                force_localhost_env = os.environ.get("IPFS_ACCELERATE_P2P_FORCE_LOCALHOST")
                force_localhost = force_localhost_env == "1"
                disable_force_localhost = force_localhost_env == "0"
                is_local_registry = type(self._bootstrap_helper).__name__ == "SimplePeerBootstrap"
                if (force_localhost or is_local_registry) and not disable_force_localhost:
                    multiaddr = f"/ip4/127.0.0.1/tcp/{self._p2p_listen_port}/p2p/{peer_id}"
                else:
                    multiaddr = advertised[0] if advertised else f"/ip4/127.0.0.1/tcp/{self._p2p_listen_port}/p2p/{peer_id}"
                self._bootstrap_helper.register_peer(peer_id, self._p2p_listen_port, multiaddr)
        except Exception:
            pass

        # Deduplicate bootstrap peers before connecting
        self._p2p_bootstrap_peers = list(dict.fromkeys(self._p2p_bootstrap_peers))
        self._p2p_known_peer_addrs.update(self._p2p_bootstrap_peers)

        for peer_addr in self._p2p_bootstrap_peers:
            try:
                with trio.fail_after(15.0):
                    await self._connect_to_peer(peer_addr)
            except Exception as e:
                logger.debug(f"Bootstrap connect failed: {peer_addr} ({e})")

        self._last_bootstrap_refresh = time.time()
    
    async def _connect_to_peer(self, peer_addr: str) -> None:
        """Connect to a peer by multiaddr."""
        if not self._p2p_host:
            return
        
        # Convert string to Multiaddr if needed
        from multiaddr import Multiaddr
        if isinstance(peer_addr, str):
            peer_addr = Multiaddr(peer_addr)
        
        peer_info = info_from_p2p_addr(peer_addr)
        await self._p2p_host.connect(peer_info)
        
        peer_id = peer_info.peer_id.pretty()
        self._p2p_connected_peers[peer_id] = peer_info
        logger.info(f"Connected to peer: {peer_id}")

        # Opportunistically perform peer exchange to learn additional multiaddrs.
        try:
            import trio
            with trio.fail_after(10.0):
                await self._peer_exchange_with_peer(peer_info)
        except Exception:
            pass
    
    async def _periodic_connection_maintenance(self) -> None:
        """Periodically refresh bootstrap connections and prune dead peers."""
        import trio

        if not self._p2p_host:
            return

        current_time = time.time()

        # Periodic bootstrap refresh
        if current_time - self._last_bootstrap_refresh > self._bootstrap_refresh_interval:
            reconnected = 0
            for peer_addr in self._p2p_bootstrap_peers[:5]:
                try:
                    with trio.fail_after(10.0):
                        await self._connect_to_peer(peer_addr)
                    reconnected += 1
                except Exception:
                    pass
            if reconnected > 0:
                logger.info(f"✓ Refreshed {reconnected} bootstrap connection(s)")
            self._last_bootstrap_refresh = current_time

        # Periodic peer exchange with already-connected peers
        for peer_id, peer_info in list(self._p2p_connected_peers.items()):
            last = self._peer_exchange_last.get(peer_id, 0)
            if current_time - last < self._peer_exchange_interval:
                continue
            try:
                with trio.fail_after(10.0):
                    await self._peer_exchange_with_peer(peer_info)
            except Exception:
                pass

        # Refresh peers from registry (GitHub issue comments or local registry)
        try:
            helper = getattr(self, "_bootstrap_helper", None)
            if helper is not None and hasattr(helper, "get_bootstrap_addrs"):
                addrs = helper.get_bootstrap_addrs(max_peers=10)
                if isinstance(addrs, list):
                    for addr in addrs:
                        if isinstance(addr, str) and self._validate_multiaddr(addr):
                            self._p2p_known_peer_addrs.add(addr)
                            if addr not in self._p2p_bootstrap_peers and len(self._p2p_bootstrap_peers) < self._max_bootstrap_peers:
                                self._p2p_bootstrap_peers.append(addr)
        except Exception:
            pass

        # Heartbeat our own registry entry so other machines can find us.
        try:
            if self._p2p_host and self._bootstrap_helper is not None:
                peer_id = self._p2p_host.get_id().pretty()
                advertised = self._get_advertised_multiaddrs()
                multiaddr = advertised[0] if advertised else f"/ip4/127.0.0.1/tcp/{self._p2p_listen_port}/p2p/{peer_id}"
                heartbeat = getattr(self._bootstrap_helper, "heartbeat", None)
                if callable(heartbeat):
                    heartbeat(peer_id, self._p2p_listen_port, multiaddr)
                else:
                    register = getattr(self._bootstrap_helper, "register_peer", None)
                    if callable(register):
                        register(peer_id, self._p2p_listen_port, multiaddr)
        except Exception:
            pass

        # Best-effort: if we still have few/no connections, try dialing some known peers.
        try:
            if self._p2p_host and len(self._p2p_connected_peers) < max(1, self._min_connected_peers):
                import trio

                attempts = 0
                for addr in list(self._p2p_known_peer_addrs):
                    if attempts >= self._max_discovery_connect_attempts:
                        break
                    try:
                        with trio.fail_after(8.0):
                            await self._connect_to_peer(addr)
                        attempts += 1
                    except Exception:
                        continue
        except Exception:
            pass

    async def _refresh_discovery_and_connect(self) -> None:
        """Refresh discovery sources and attempt connections to newly learned peers."""
        if not self._universal_connectivity or not self._p2p_host:
            return

        # Build a simple registry wrapper for the bootstrap helper
        registry = None
        if self._bootstrap_helper:
            class SimpleRegistry:
                def __init__(self, helper):
                    self.helper = helper
                def discover_peers(self, max_peers=10):
                    return self.helper.discover_peers(max_peers=max_peers)

            registry = SimpleRegistry(self._bootstrap_helper)

        discovered = await self._universal_connectivity.discover_peers_multimethod(
            github_registry=registry,
            bootstrap_peers=self._p2p_bootstrap_peers
        )

        for addr in discovered:
            if self._validate_multiaddr(addr):
                self._p2p_known_peer_addrs.add(addr)
                if addr not in self._p2p_bootstrap_peers and len(self._p2p_bootstrap_peers) < self._max_bootstrap_peers:
                    self._p2p_bootstrap_peers.append(addr)

        # Try connecting to a few newly learned peers
        attempts = 0
        for addr in list(self._p2p_known_peer_addrs):
            if attempts >= self._max_discovery_connect_attempts:
                break
            try:
                if self._universal_connectivity:
                    peer_info = await self._universal_connectivity.attempt_connection(
                        self._p2p_host,
                        addr,
                        use_relay=True
                    )
                    if peer_info is not None:
                        try:
                            peer_id = peer_info.peer_id.pretty()
                            self._p2p_connected_peers[peer_id] = peer_info
                        except Exception:
                            pass
                else:
                    await self._connect_to_peer(addr)
                attempts += 1
            except Exception:
                continue

    async def _ensure_min_connections(self) -> None:
        """Best-effort to keep at least a minimum number of connections."""
        if not self._p2p_host:
            return

        connected = len(self._p2p_connected_peers)
        if connected >= self._min_connected_peers:
            return

        needed = self._min_connected_peers - connected
        candidates = list(self._p2p_known_peer_addrs)

        attempts = 0
        for addr in candidates:
            if attempts >= needed:
                break
            try:
                if self._universal_connectivity:
                    peer_info = await self._universal_connectivity.attempt_connection(
                        self._p2p_host,
                        addr,
                        use_relay=True
                    )
                    if peer_info is not None:
                        try:
                            peer_id = peer_info.peer_id.pretty()
                            self._p2p_connected_peers[peer_id] = peer_info
                        except Exception:
                            pass
                else:
                    await self._connect_to_peer(addr)
                attempts += 1
            except Exception:
                continue

    def _get_advertised_multiaddrs(self) -> List[str]:
        """Return best-effort dialable multiaddrs for this host."""
        if not self._p2p_host:
            return []

        peer_id = self._p2p_host.get_id().pretty()
        addrs: List[str] = []

        # Prefer explicit operator-provided public IP.
        env_ip = os.environ.get("IPFS_ACCELERATE_PUBLIC_IP")
        if env_ip:
            addrs.append(f"/ip4/{env_ip}/tcp/{self._p2p_listen_port}/p2p/{peer_id}")

        # Fall back to detected public IP (best-effort, may fail under restricted egress).
        try:
            public_ip = self._get_public_ip()
            if public_ip and public_ip != "127.0.0.1":
                addrs.append(f"/ip4/{public_ip}/tcp/{self._p2p_listen_port}/p2p/{peer_id}")
        except Exception:
            pass

        # If host exposes listen addrs, include them too.
        try:
            host_addrs = getattr(self._p2p_host, "get_addrs", None)
            if callable(host_addrs):
                for ma in host_addrs():
                    try:
                        s = str(ma)
                        if s and "/p2p/" not in s:
                            s = f"{s}/p2p/{peer_id}"
                        if self._validate_multiaddr(s):
                            addrs.append(s)
                    except Exception:
                        continue
        except Exception:
            pass

        # Deduplicate preserving order.
        seen = set()
        out = []
        for a in addrs:
            if a not in seen:
                seen.add(a)
                out.append(a)
        return out

    async def _handle_peer_exchange_stream(self, stream: 'INetStream') -> None:
        """Handle inbound peer exchange request (encrypted if encryption enabled)."""
        try:
            payload = await stream.read()
            message = self._decrypt_message(payload) if self._cipher else json.loads(payload.decode("utf-8"))

            if not isinstance(message, dict):
                await stream.write(b"ERROR")
                return

            # Learn sender's advertised addrs (best signal) and their known peers (gossip).
            sender_addrs = message.get("addrs") or []
            if isinstance(sender_addrs, list):
                for addr in sender_addrs:
                    if not isinstance(addr, str):
                        continue
                    if self._validate_multiaddr(addr):
                        self._p2p_known_peer_addrs.add(addr)

            peer_addrs = message.get("peers") or []
            if isinstance(peer_addrs, list):
                for addr in peer_addrs:
                    if not isinstance(addr, str):
                        continue
                    if self._validate_multiaddr(addr):
                        self._p2p_known_peer_addrs.add(addr)

            # Keep bootstrap list fed with a bounded number of known addrs.
            for addr in list(sorted(self._p2p_known_peer_addrs))[: self._max_bootstrap_peers]:
                if addr not in self._p2p_bootstrap_peers and len(self._p2p_bootstrap_peers) < self._max_bootstrap_peers:
                    self._p2p_bootstrap_peers.append(addr)

            # Respond with our current known peers + our advertised multiaddrs
            response = {
                "addrs": self._get_advertised_multiaddrs(),
                "peers": list(sorted(self._p2p_known_peer_addrs))[:50],
                "ts": time.time(),
            }

            resp_bytes = self._encrypt_message(response) if self._cipher else json.dumps(response).encode("utf-8")
            await stream.write(resp_bytes)
        except Exception as e:
            logger.debug(f"Peer exchange handler error: {e}")
            try:
                await stream.write(b"ERROR")
            except Exception:
                pass
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    async def _peer_exchange_with_peer(self, peer_info: Any) -> None:
        """Exchange known peer multiaddrs with a connected peer."""
        if not self._p2p_host:
            return

        peer_id = getattr(peer_info, "peer_id", None)
        if peer_id is None:
            return

        peer_id_str = peer_id.pretty()
        self._peer_exchange_last[peer_id_str] = time.time()

        message = {
            "addrs": self._get_advertised_multiaddrs(),
            "peers": list(sorted(self._p2p_known_peer_addrs))[:50],
            "ts": time.time(),
        }
        req_bytes = self._encrypt_message(message) if self._cipher else json.dumps(message).encode("utf-8")

        stream = await self._p2p_host.new_stream(peer_id, [self._p2p_peer_exchange_protocol])
        await stream.write(req_bytes)
        resp = await stream.read()
        await stream.close()

        try:
            data = self._decrypt_message(resp) if self._cipher else json.loads(resp.decode("utf-8"))
        except Exception:
            return

        if not isinstance(data, dict):
            return

        # Merge peer's self-advertised addrs and known peers.
        for addr in (data.get("addrs") or []):
            if isinstance(addr, str) and self._validate_multiaddr(addr):
                self._p2p_known_peer_addrs.add(addr)
        for addr in (data.get("peers") or []):
            if isinstance(addr, str) and self._validate_multiaddr(addr):
                self._p2p_known_peer_addrs.add(addr)

        # Attempt to connect to a small number of newly learned peers.
        candidates = deque([a for a in self._p2p_known_peer_addrs if a not in self._p2p_bootstrap_peers])
        attempts = 0
        while candidates and attempts < 3:
            addr = candidates.popleft()
            if len(self._p2p_bootstrap_peers) < self._max_bootstrap_peers:
                self._p2p_bootstrap_peers.append(addr)
            try:
                import trio
                with trio.fail_after(5.0):
                    await self._connect_to_peer(addr)
            except Exception:
                pass
            attempts += 1
    
    async def _handle_cache_stream(self, stream: 'INetStream') -> None:
        """Handle incoming encrypted cache entry from peer."""
        try:
            # Read encrypted cache entry data
            encrypted_data = await stream.read()

            # A peer can disconnect before sending a payload. Treat this as a
            # normal stream close (not an error).
            if not encrypted_data:
                logger.debug("Received empty cache stream; peer likely disconnected")
                return
            
            # Decrypt message
            message = self._decrypt_message(encrypted_data)
            if message is None:
                logger.warning("Failed to decrypt message from peer (unauthorized or corrupted)")
                await stream.write(b"ERROR: Decryption failed")
                return
            
            # Extract cache entry
            cache_key = message.get("key")
            entry_data = message.get("entry")
            
            if not cache_key or not entry_data:
                logger.warning("Received invalid cache entry from peer")
                await stream.write(b"ERROR: Invalid format")
                return
            
            # Reconstruct cache entry
            entry = CacheEntry(
                data=entry_data["data"],
                timestamp=entry_data["timestamp"],
                ttl=entry_data["ttl"],
                content_hash=entry_data.get("content_hash"),
                validation_fields=entry_data.get("validation_fields")
            )
            
            # Verify content hash if available
            if entry.content_hash and entry.validation_fields:
                expected_hash = self._compute_validation_hash(entry.validation_fields)
                if expected_hash != entry.content_hash:
                    logger.warning(f"Content hash mismatch for {cache_key}, rejecting")
                    return
            
            # Store in cache if not expired
            if not entry.is_expired():
                with self._lock:
                    # Only store if we don't have it or our version is older
                    existing = self._cache.get(cache_key)
                    if not existing or existing.timestamp < entry.timestamp:
                        self._cache[cache_key] = entry
                        self._stats["peer_hits"] += 1
                        logger.debug(f"Received cache entry from peer: {cache_key}")
                        
                        # Persist if enabled
                        if self.enable_persistence:
                            self._save_to_disk(cache_key, entry)
            
            # Send acknowledgment
            await stream.write(b"OK")
        
        except Exception as e:
            msg = repr(e) if e is not None else ""
            benign_disconnect_markers = (
                "Yamux connection closed",
                "Connection closed during read operation",
                "expected 2 bytes but received 0 bytes",
                "StreamReset",
            )
            if any(marker in msg for marker in benign_disconnect_markers):
                logger.debug("Cache stream closed by peer: %r", e)
            else:
                logger.error("Error handling cache stream: %r", e)
        finally:
            await stream.close()
    
    async def _broadcast_cache_entry(self, cache_key: str, entry: CacheEntry) -> None:
        """Broadcast encrypted cache entry to all connected peers."""
        if not self._p2p_host or not self._p2p_connected_peers:
            return
        
        try:
            # Prepare message
            message = {
                "key": cache_key,
                "entry": {
                    "data": entry.data,
                    "timestamp": entry.timestamp,
                    "ttl": entry.ttl,
                    "content_hash": entry.content_hash,
                    "validation_fields": entry.validation_fields
                }
            }
            
            # Encrypt message (only peers with same GitHub token can decrypt)
            encrypted_bytes = self._encrypt_message(message)
            
            # Send to all connected peers
            for peer_id, peer_info in list(self._p2p_connected_peers.items()):
                try:
                    stream = await self._p2p_host.new_stream(peer_info.peer_id, [self._p2p_protocol])
                    await stream.write(encrypted_bytes)
                    
                    # Wait for ack
                    ack = await stream.read()
                    if ack == b"OK":
                        logger.debug(f"Broadcast cache entry to {peer_id}: {cache_key}")
                    
                    await stream.close()
                
                except Exception as e:
                    logger.warning(f"Failed to broadcast to peer {peer_id}: {e}")
        
        except Exception as e:
            logger.error(f"Error broadcasting cache entry: {e}")
    
    def _broadcast_in_background(self, cache_key: str, entry: CacheEntry) -> None:
        """Broadcast cache entry in background (non-blocking)."""
        if not self.enable_p2p or self._p2p_trio_token is None:
            return

        try:
            import trio

            trio.from_thread.run(
                self._enqueue_broadcast,
                cache_key,
                entry,
                trio_token=self._p2p_trio_token,
            )
        except Exception:
            pass


# Global cache instance (can be configured at module level)
_global_cache: Optional[GitHubAPICache] = None
_global_cache_lock = Lock()


def get_global_cache(**kwargs) -> GitHubAPICache:
    """
    Get or create the global GitHub API cache instance with thread-safe initialization.
    
    Uses double-checked locking pattern to ensure only one instance is created
    even when called concurrently from multiple threads.
    
    Automatically reads P2P configuration from environment variables:
    - CACHE_ENABLE_P2P: Enable P2P cache sharing (default: true)
    - CACHE_LISTEN_PORT: P2P listen port (default: 9000)
    - CACHE_BOOTSTRAP_PEERS: Comma-separated list of peer multiaddrs
    - CACHE_DEFAULT_TTL: Default cache TTL in seconds (default: 300)
    - CACHE_DIR: Cache directory (default: ~/.cache/github_cli)
    
    Args:
        **kwargs: Arguments to pass to GitHubAPICache constructor (overrides env vars)
        
    Returns:
        Global GitHubAPICache instance
    """
    global _global_cache
    
    # First check without lock for performance (double-checked locking)
    if _global_cache is None:
        with _global_cache_lock:
            # Second check with lock to ensure only one thread creates the instance
            if _global_cache is None:
                # Read from environment if not provided
                if 'enable_p2p' not in kwargs:
                    kwargs['enable_p2p'] = os.environ.get('CACHE_ENABLE_P2P', 'true').lower() == 'true'
                
                if 'p2p_listen_port' not in kwargs:
                    kwargs['p2p_listen_port'] = int(os.environ.get('CACHE_LISTEN_PORT', '9100'))
                
                if 'p2p_bootstrap_peers' not in kwargs:
                    peers_str = os.environ.get('CACHE_BOOTSTRAP_PEERS', '')
                    if peers_str:
                        kwargs['p2p_bootstrap_peers'] = [p.strip() for p in peers_str.split(',') if p.strip()]
                
                if 'default_ttl' not in kwargs:
                    kwargs['default_ttl'] = int(os.environ.get('CACHE_DEFAULT_TTL', '300'))
                
                if 'cache_dir' not in kwargs and 'CACHE_DIR' in os.environ:
                    kwargs['cache_dir'] = os.environ['CACHE_DIR']
                
                _global_cache = GitHubAPICache(**kwargs)
    
    return _global_cache


def configure_cache(**kwargs) -> GitHubAPICache:
    """
    Configure the global cache with custom settings in a thread-safe manner.
    
    Args:
        **kwargs: Arguments to pass to GitHubAPICache constructor
        
    Returns:
        Configured GitHubAPICache instance
    """
    global _global_cache
    with _global_cache_lock:
        _global_cache = GitHubAPICache(**kwargs)
    return _global_cache
