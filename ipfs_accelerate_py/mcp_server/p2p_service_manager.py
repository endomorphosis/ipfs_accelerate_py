"""Canonical P2P service manager facade for unified MCP runtime.

This module adapts the source `p2p_service_manager` surface to canonical
`ipfs_accelerate_py.p2p_tasks` runtime and service primitives.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_DEFAULT_QUEUE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "ipfs_datasets_py", "task_queue.duckdb")


@dataclass
class P2PServiceState:
    running: bool
    peer_id: str
    listen_port: Optional[int]
    started_at: float
    last_error: str = ""
    workflow_scheduler_available: bool = False
    peer_registry_available: bool = False
    bootstrap_available: bool = False
    connected_peers: int = 0
    active_workflows: int = 0


class P2PServiceManager:
    """Compatibility manager delegating to canonical p2p_tasks runtime."""

    def __init__(
        self,
        *,
        enabled: bool,
        queue_path: str = _DEFAULT_QUEUE_PATH,
        listen_port: Optional[int] = None,
        enable_tools: bool = True,
        enable_cache: bool = True,
        auth_mode: str = "mcp_token",
        startup_timeout_s: float = 2.0,
        enable_workflow_scheduler: bool = True,
        enable_peer_registry: bool = True,
        enable_bootstrap: bool = True,
        bootstrap_nodes: Optional[List[str]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.queue_path = str(queue_path or _DEFAULT_QUEUE_PATH)
        self.listen_port = int(listen_port) if listen_port is not None else None
        self.enable_tools = bool(enable_tools)
        self.enable_cache = bool(enable_cache)
        self.auth_mode = str(auth_mode or "mcp_token")
        self.startup_timeout_s = float(startup_timeout_s)

        self.enable_workflow_scheduler = bool(enable_workflow_scheduler)
        self.enable_peer_registry = bool(enable_peer_registry)
        self.enable_bootstrap = bool(enable_bootstrap)
        self.bootstrap_nodes = list(bootstrap_nodes or [])

        self._runtime = None
        self._handle = None
        self._env_restore: Dict[str, Optional[str]] = {}

        self._workflow_scheduler = None
        self._peer_registry = None
        self._mcplusplus_available = False

        self._connection_pool: Dict[str, Any] = {}
        self._pool_max_size = 10
        self._pool_lock = threading.Lock()
        self._pool_hits = 0
        self._pool_misses = 0

    def _setdefault_env(self, key: str, value: str) -> None:
        if key not in os.environ:
            self._env_restore.setdefault(key, None)
            os.environ[key] = value

    def _apply_env(self) -> None:
        self._setdefault_env("IPFS_ACCELERATE_PY_TASK_QUEUE_PATH", self.queue_path)
        self._setdefault_env("IPFS_DATASETS_PY_TASK_QUEUE_PATH", self.queue_path)
        self._setdefault_env("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_TOOLS", "1" if self.enable_tools else "0")
        self._setdefault_env("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_CACHE", "1" if self.enable_cache else "0")
        self._setdefault_env("IPFS_DATASETS_PY_TASK_P2P_ENABLE_CACHE", "1" if self.enable_cache else "0")
        self._setdefault_env("IPFS_ACCELERATE_PY_MCP_P2P_SERVICE", "1")
        self._setdefault_env("IPFS_DATASETS_PY_TASK_P2P_AUTH_MODE", self.auth_mode)
        self._setdefault_env("IPFS_ACCELERATE_PY_TASK_P2P_AUTH_MODE", self.auth_mode)

    def _restore_env(self) -> None:
        for key, prior in list(self._env_restore.items()):
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior
        self._env_restore.clear()

    def start(self, *, accelerate_instance: Any | None = None) -> bool:
        if not self.enabled:
            return False

        self._apply_env()
        from ipfs_accelerate_py.p2p_tasks.runtime import TaskQueueP2PServiceRuntime

        if self._runtime is None:
            self._runtime = TaskQueueP2PServiceRuntime()

        self._handle = self._runtime.start(
            queue_path=self.queue_path,
            listen_port=self.listen_port,
            accelerate_instance=accelerate_instance,
        )

        try:
            self._handle.started.wait(timeout=max(0.0, self.startup_timeout_s))
        except Exception:
            pass

        return bool(getattr(self._runtime, "running", False))

    def stop(self) -> bool:
        if self._runtime is None:
            return True
        try:
            ok = bool(self._runtime.stop(timeout_s=2.0))
            self._restore_env()
            return ok
        except Exception:
            self._restore_env()
            return False

    def state(self) -> P2PServiceState:
        running = False
        peer_id = ""
        listen_port: Optional[int] = None
        started_at = 0.0

        try:
            from ipfs_accelerate_py.p2p_tasks.service import get_local_service_state

            st = get_local_service_state() or {}
            running = bool(st.get("running"))
            peer_id = str(st.get("peer_id") or "")
            listen_port = st.get("listen_port") if isinstance(st.get("listen_port"), int) else None
            started_at = float(st.get("started_at") or 0.0)
        except Exception:
            running = bool(getattr(self._runtime, "running", False)) if self._runtime is not None else False
            started_at = time.time() if running else 0.0

        connected_peers = 0
        try:
            from ipfs_accelerate_py.p2p_tasks.service import list_known_peers

            peers = list_known_peers(alive_only=True, limit=1000)
            connected_peers = len(peers) if isinstance(peers, list) else 0
        except Exception:
            connected_peers = 0

        return P2PServiceState(
            running=running,
            peer_id=peer_id,
            listen_port=listen_port,
            started_at=started_at,
            last_error=str(getattr(self._runtime, "last_error", "") or "") if self._runtime is not None else "",
            workflow_scheduler_available=self._workflow_scheduler is not None,
            peer_registry_available=self._peer_registry is not None,
            bootstrap_available=bool(self._mcplusplus_available and self.enable_bootstrap),
            connected_peers=connected_peers,
            active_workflows=0,
        )

    def get_workflow_scheduler(self) -> Optional[Any]:
        return self._workflow_scheduler

    def get_peer_registry(self) -> Optional[Any]:
        return self._peer_registry

    def has_advanced_features(self) -> bool:
        return self._mcplusplus_available

    def acquire_connection(self, peer_id: str) -> Optional[Any]:
        with self._pool_lock:
            conn = self._connection_pool.pop(str(peer_id), None)
            if conn is not None:
                self._pool_hits += 1
            else:
                self._pool_misses += 1
            return conn

    def release_connection(self, peer_id: str, conn: Any) -> bool:
        if conn is None:
            return False
        with self._pool_lock:
            if len(self._connection_pool) >= self._pool_max_size:
                return False
            self._connection_pool[str(peer_id)] = conn
            return True

    def clear_connection_pool(self) -> int:
        with self._pool_lock:
            count = len(self._connection_pool)
            self._connection_pool.clear()
            self._pool_hits = 0
            self._pool_misses = 0
            return count

    def get_pool_stats(self) -> Dict[str, Any]:
        with self._pool_lock:
            total = self._pool_hits + self._pool_misses
            return {
                "size": len(self._connection_pool),
                "max_size": self._pool_max_size,
                "hits": self._pool_hits,
                "misses": self._pool_misses,
                "hit_rate": (self._pool_hits / total) if total > 0 else None,
            }

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "p2p_enabled": self.enabled,
            "mcplusplus_available": self._mcplusplus_available,
            "workflow_scheduler": self._workflow_scheduler is not None,
            "peer_registry": self._peer_registry is not None,
            "bootstrap": bool(self.enable_bootstrap and self._mcplusplus_available),
            "tools_enabled": self.enable_tools,
            "cache_enabled": self.enable_cache,
            "connection_pool_max_size": self._pool_max_size,
        }


__all__ = ["P2PServiceState", "P2PServiceManager"]
