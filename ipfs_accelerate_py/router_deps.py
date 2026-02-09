"""Router dependency injection helpers.

Why this exists
---------------
Python's import system caches modules in-process (via ``sys.modules``), so the
same package isn't *re-imported* repeatedly.

However, higher-level integrations (e.g., ipfs_accelerate_py / ipfs_kit_py)
may still be *re-initialized* repeatedly if every call site constructs new
clients/managers.

This module provides a tiny dependency container that routers can use to:
- reuse already-created Accelerate managers/clients
- allow upstream applications to inject pre-configured instances

It is intentionally lightweight and safe to import in CI/minimal contexts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol


class RemoteCacheProtocol(Protocol):
    """Optional remote cache interface.

    This is a Protocol defining the interface for remote caches. Implementations
    only need to provide `get(key)` and `set(key, value)` methods.

    The default RouterDeps does not provide a remote cache; callers can inject
    one (e.g., backed by libp2p, IPFS Kit, etc.).
    """

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any) -> Any: ...


@dataclass
class RouterDeps:
    """Mutable dependency container used by routers.

    Users can create one instance and pass it into routers to ensure all
    router calls share the same underlying clients/managers.
    """

    accelerate_managers: dict[str, Any] = field(default_factory=dict)
    backend_manager: Any | None = None
    # Generic cache for router-resolved instances (providers, clients, etc.).
    # Keys should be stable strings; values are arbitrary objects.
    router_cache: dict[str, Any] = field(default_factory=dict)

    # Optional remote/distributed cache. If provided, routers may consult it on
    # cache miss, and may write-through to it on cache set.
    remote_cache: Any | None = None

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_cached(self, key: str) -> Any | None:
        if not key:
            return None
        with self._lock:
            return self.router_cache.get(key)

    def get_cached_or_remote(self, key: str) -> Any | None:
        """Return cached value, optionally consulting a remote cache on miss."""

        cached = self.get_cached(key)
        if cached is not None:
            return cached

        remote = self.remote_cache
        getter = getattr(remote, "get", None)
        if callable(getter):
            try:
                value = getter(key)
            except Exception:
                value = None
            if value is not None:
                self.set_cached(key, value)
            return value
        return None

    def set_cached(self, key: str, value: Any) -> Any:
        if not key:
            return value
        with self._lock:
            self.router_cache[key] = value
            return value

    def set_cached_and_remote(self, key: str, value: Any) -> Any:
        """Set local cache and best-effort write-through to remote cache."""

        self.set_cached(key, value)
        remote = self.remote_cache
        setter = getattr(remote, "set", None)
        if callable(setter):
            try:
                setter(key, value)
            except Exception:
                pass
        return value

    def get_or_create(self, key: str, factory: callable) -> Any:
        if not key:
            return factory()
        with self._lock:
            if key in self.router_cache:
                return self.router_cache[key]
        value = factory()
        with self._lock:
            self.router_cache[key] = value
        return value

    def get_backend_manager(
        self,
        *,
        purpose: str = "router",
        enable_health_checks: bool = True,
        load_balancing_strategy: str = "round_robin",
    ) -> Any | None:
        """Return a cached InferenceBackendManager for ``purpose`` if available.

        Creates the manager lazily on first access.
        Returns ``None`` if backend manager is unavailable.
        
        Note: The enable_health_checks and load_balancing_strategy parameters are
        currently not used as get_backend_manager() returns a singleton. Future
        versions may support per-purpose configuration.
        """

        if not purpose or not str(purpose).strip():
            purpose = "router"

        with self._lock:
            if self.backend_manager is not None:
                return self.backend_manager

            # Lazy import to avoid import-time side effects.
            try:
                from ipfs_accelerate_py.inference_backend_manager import (
                    get_backend_manager,
                )
            except Exception:
                return None

            try:
                # Note: get_backend_manager() returns a singleton and doesn't accept
                # configuration parameters. The enable_health_checks and
                # load_balancing_strategy parameters are for future use.
                manager = get_backend_manager()
                if manager is not None:
                    self.backend_manager = manager
                    return manager
            except Exception:
                return None

            return None


_DEFAULT_DEPS: RouterDeps | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_router_deps() -> RouterDeps:
    """Return the process-global default dependency container."""

    global _DEFAULT_DEPS
    if _DEFAULT_DEPS is not None:
        return _DEFAULT_DEPS
    with _DEFAULT_LOCK:
        if _DEFAULT_DEPS is None:
            _DEFAULT_DEPS = RouterDeps()
        return _DEFAULT_DEPS


def set_default_router_deps(deps: RouterDeps | None) -> None:
    """Override the process-global default dependency container."""

    global _DEFAULT_DEPS
    with _DEFAULT_LOCK:
        _DEFAULT_DEPS = deps
