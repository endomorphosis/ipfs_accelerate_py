"""IPFS backend router for ipfs_accelerate_py.

This module provides a stable entry point for basic IPFS operations with a
pluggable backend strategy:
- Preferred: ipfs_kit_py backend when explicitly enabled and available
- Fallback 1: HuggingFace model cache for model storage
- Fallback 2: Local Kubo via the `ipfs` CLI

Design goals:
- Avoid importing ipfs_kit_py at module import time
- Prefer ipfs_kit_py for distributed storage
- Fall back gracefully to HF cache and Kubo
- Keep behavior predictable in benchmarks/CI

Environment variables:
- `IPFS_BACKEND`: force backend name (registered provider)
- `ENABLE_IPFS_KIT`: enable ipfs_kit_py backend (preferred, default: true)
- `ENABLE_HF_CACHE`: enable HuggingFace cache backend (default: true) 
- `IPFS_KIT_DISABLE`: disable ipfs_kit_py backend completely
- `KUBO_CMD`: override ipfs CLI command (default: "ipfs")
"""

from __future__ import annotations

import os
import importlib
import subprocess
import tempfile
import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Optional, Protocol, runtime_checkable
from pathlib import Path

try:
    from .router_deps import RouterDeps, get_default_router_deps
except ImportError:
    # Fallback for when imported standalone
    RouterDeps = None
    get_default_router_deps = lambda: None


def _truthy(value: Optional[str]) -> bool:
    """Check if an environment variable value is truthy."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_enabled() -> bool:
    """Check if backend caching is enabled."""
    return os.environ.get("IPFS_ROUTER_CACHE", "1").strip() != "0"


_DEFAULT_BACKEND_OVERRIDE: IPFSBackend | None = None


def set_default_ipfs_backend(backend: IPFSBackend | None) -> None:
    """Inject a process-global backend instance.

    If set, all router calls will use this backend unless an explicit backend
    is passed at call time.
    """
    global _DEFAULT_BACKEND_OVERRIDE
    _DEFAULT_BACKEND_OVERRIDE = backend


def _backend_cache_key() -> tuple:
    """Generate cache key from environment variables."""
    return (
        os.getenv("IPFS_BACKEND", "").strip(),
        os.getenv("ENABLE_IPFS_KIT", "").strip(),
        os.getenv("IPFS_KIT_DISABLE", "").strip(),
        os.getenv("ENABLE_HF_CACHE", "").strip(),
        os.getenv("KUBO_CMD", "").strip(),
        os.getenv("HF_HOME", "").strip(),
    )


@runtime_checkable
class IPFSBackend(Protocol):
    """Protocol for IPFS backend implementations."""
    
    def add_bytes(self, data: bytes, *, pin: bool = True) -> str: ...

    def cat(self, cid: str) -> bytes: ...

    def pin(self, cid: str) -> None: ...

    def unpin(self, cid: str) -> None: ...

    def block_put(self, data: bytes, *, codec: str = "raw") -> str: ...

    def block_get(self, cid: str) -> bytes: ...

    def add_path(
        self,
        path: str,
        *,
        recursive: bool = True,
        pin: bool = True,
        chunker: Optional[str] = None,
    ) -> str: ...

    def get_to_path(self, cid: str, *, output_path: str) -> None: ...

    def ls(self, cid: str) -> list[str]: ...

    def dag_export(self, cid: str) -> bytes: ...


ProviderFactory = Callable[[], IPFSBackend]


@dataclass(frozen=True)
class ProviderInfo:
    """Information about a registered backend provider."""
    name: str
    factory: ProviderFactory


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}


def register_ipfs_backend(name: str, factory: ProviderFactory) -> None:
    """Register a new IPFS backend provider."""
    if not name or not name.strip():
        raise ValueError("Backend name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


class IPFSKitBackend:
    """IPFS backend using ipfs_kit_py (preferred)."""
    
    def __init__(self, cache_dir: Optional[str] = None, deps: object = None) -> None:
        """Initialize ipfs_kit_py backend.
        
        Args:
            cache_dir: Directory for local caching
            deps: Optional dependency injection container
        """
        self._cache_dir = cache_dir or os.getenv("IPFS_KIT_CACHE_DIR") or \
                         os.path.join(os.path.expanduser("~"), ".cache", "ipfs_kit")
        self._deps = deps
        self._storage = None
        self._init_storage()
    
    def _init_storage(self):
        """Initialize ipfs_kit storage."""
        try:
            # Use existing IPFSKitStorage from ipfs_kit_integration
            from .ipfs_kit_integration import get_storage
            self._storage = get_storage(
                enable_ipfs_kit=True,
                cache_dir=self._cache_dir,
                deps=self._deps,
                force_fallback=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ipfs_kit_py backend: {e}")
    
    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        """Add bytes to IPFS and return CID."""
        return self._storage.store(data, pin=pin)
    
    def cat(self, cid: str) -> bytes:
        """Retrieve data by CID."""
        result = self._storage.retrieve(cid)
        if result is None:
            raise RuntimeError(f"CID not found: {cid}")
        return result
    
    def pin(self, cid: str) -> None:
        """Pin content by CID."""
        if not self._storage.pin(cid):
            raise RuntimeError(f"Failed to pin CID: {cid}")
    
    def unpin(self, cid: str) -> None:
        """Unpin content by CID."""
        if not self._storage.unpin(cid):
            raise RuntimeError(f"Failed to unpin CID: {cid}")
    
    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        """Store a raw block and return its CID."""
        # For raw blocks, just use add_bytes
        return self.add_bytes(data, pin=True)
    
    def block_get(self, cid: str) -> bytes:
        """Get a raw block by CID."""
        return self.cat(cid)
    
    def add_path(
        self,
        path: str,
        *,
        recursive: bool = True,
        pin: bool = True,
        chunker: Optional[str] = None,
    ) -> str:
        """Add a file or directory to IPFS."""
        return self._storage.store(Path(path), pin=pin)
    
    def get_to_path(self, cid: str, *, output_path: str) -> None:
        """Retrieve content and save to path."""
        data = self.cat(cid)
        Path(output_path).write_bytes(data)
    
    def ls(self, cid: str) -> list[str]:
        """List directory contents."""
        # This would require more complex IPFS directory handling
        # For now, return empty list as not all backends support this
        return []
    
    def dag_export(self, cid: str) -> bytes:
        """Export DAG as CAR file."""
        # Not implemented in basic storage layer
        raise RuntimeError("dag_export not available in ipfs_kit backend")


class HuggingFaceCacheBackend:
    """IPFS backend using HuggingFace model cache."""
    
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize HuggingFace cache backend.
        
        Args:
            cache_dir: Directory for cache (defaults to HF_HOME)
        """
        self._cache_dir = Path(cache_dir or os.getenv("HF_HOME") or 
                               os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        self._ipfs_cache = self._cache_dir / "ipfs_blocks"
        self._ipfs_cache.mkdir(parents=True, exist_ok=True)
    
    def _generate_cid(self, data: bytes) -> str:
        """Generate a CID-like identifier for data."""
        hash_value = hashlib.sha256(data).hexdigest()
        return f"bafy{hash_value[:56]}"
    
    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        """Store bytes in HF cache and return CID."""
        cid = self._generate_cid(data)
        block_path = self._ipfs_cache / cid
        block_path.write_bytes(data)
        
        # Store metadata about pinning
        if pin:
            meta_path = self._ipfs_cache / f"{cid}.meta"
            meta_path.write_text(json.dumps({"pinned": True}))
        
        return cid
    
    def cat(self, cid: str) -> bytes:
        """Retrieve data by CID from HF cache."""
        block_path = self._ipfs_cache / cid
        if not block_path.exists():
            raise RuntimeError(f"CID not found in HF cache: {cid}")
        return block_path.read_bytes()
    
    def pin(self, cid: str) -> None:
        """Mark content as pinned in HF cache."""
        meta_path = self._ipfs_cache / f"{cid}.meta"
        meta_path.write_text(json.dumps({"pinned": True}))
    
    def unpin(self, cid: str) -> None:
        """Unmark content as pinned in HF cache."""
        meta_path = self._ipfs_cache / f"{cid}.meta"
        if meta_path.exists():
            meta_path.unlink()
    
    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        """Store a raw block in HF cache."""
        return self.add_bytes(data, pin=True)
    
    def block_get(self, cid: str) -> bytes:
        """Get a raw block by CID from HF cache."""
        return self.cat(cid)
    
    def add_path(
        self,
        path: str,
        *,
        recursive: bool = True,
        pin: bool = True,
        chunker: Optional[str] = None,
    ) -> str:
        """Add file to HF cache."""
        data = Path(path).read_bytes()
        return self.add_bytes(data, pin=pin)
    
    def get_to_path(self, cid: str, *, output_path: str) -> None:
        """Retrieve content and save to path."""
        data = self.cat(cid)
        Path(output_path).write_bytes(data)
    
    def ls(self, cid: str) -> list[str]:
        """List directory contents (not supported in HF cache)."""
        return []
    
    def dag_export(self, cid: str) -> bytes:
        """Export DAG (not supported in HF cache)."""
        raise RuntimeError("dag_export not available in HF cache backend")


class KuboCLIBackend:
    """IPFS backend using local Kubo CLI."""
    
    def __init__(self, cmd: Optional[str] = None) -> None:
        """Initialize Kubo CLI backend.
        
        Args:
            cmd: IPFS CLI command (defaults to 'ipfs')
        """
        self._cmd = cmd or os.getenv("KUBO_CMD", "ipfs")

    def _run(self, args: list[str], *, input_bytes: Optional[bytes] = None) -> bytes:
        """Run an IPFS CLI command."""
        proc = subprocess.run(
            [self._cmd, *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", errors="replace").strip() or "ipfs command failed"
            raise RuntimeError(msg)
        return proc.stdout

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        """Add bytes to IPFS via CLI."""
        pin_flag = "true" if pin else "false"
        out = self._run(["add", "-Q", f"--pin={pin_flag}", "--stdin-name", "data.bin"], input_bytes=data)
        return out.decode("utf-8", errors="replace").strip()

    def cat(self, cid: str) -> bytes:
        """Retrieve data by CID via CLI."""
        return self._run(["cat", cid])

    def pin(self, cid: str) -> None:
        """Pin content by CID via CLI."""
        self._run(["pin", "add", cid])

    def unpin(self, cid: str) -> None:
        """Unpin content by CID via CLI."""
        self._run(["pin", "rm", cid])

    def block_put(self, data: bytes, *, codec: str = "raw") -> str:
        """Store a raw block via CLI."""
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
            handle.flush()
            try:
                out = self._run(["block", "put", "--cid-version", "1", "--format", str(codec), handle.name])
            except RuntimeError as e:
                # Some IPFS CLIs don't support these flags
                msg = str(e)
                if "unknown option" in msg or "flag provided but not defined" in msg:
                    out = self._run(["block", "put", "--format", str(codec), handle.name])
                else:
                    raise
            finally:
                try:
                    os.unlink(handle.name)
                except:
                    pass
        return out.decode("utf-8", errors="replace").strip()

    def block_get(self, cid: str) -> bytes:
        """Get a raw block by CID via CLI."""
        return self._run(["block", "get", cid])

    def add_path(
        self,
        path: str,
        *,
        recursive: bool = True,
        pin: bool = True,
        chunker: Optional[str] = None,
    ) -> str:
        """Add file or directory to IPFS via CLI."""
        pin_flag = "true" if pin else "false"
        args: list[str] = ["add", "-Q", f"--pin={pin_flag}"]
        if recursive:
            args.append("-r")
        if chunker:
            args.extend(["--chunker", str(chunker)])
        args.append(path)
        out = self._run(args)
        return out.decode("utf-8", errors="replace").strip()

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        """Retrieve content and save to path via CLI."""
        self._run(["get", cid, "-o", output_path])

    def ls(self, cid: str) -> list[str]:
        """List directory contents via CLI."""
        out = self._run(["ls", cid]).decode("utf-8", errors="replace")
        names: list[str] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Expected: <hash> <size> <name>
            parts = line.split()
            if len(parts) >= 3:
                names.append(" ".join(parts[2:]))
        return names

    def dag_export(self, cid: str) -> bytes:
        """Export DAG as CAR file via CLI."""
        return self._run(["dag", "export", cid])


def _get_ipfs_kit_backend(deps: object = None) -> Optional[IPFSBackend]:
    """Get ipfs_kit_py backend if available and enabled."""
    # Check if disabled
    if _truthy(os.getenv("IPFS_KIT_DISABLE")):
        return None
    
    # Check if enabled (default: true)
    if not _truthy(os.getenv("ENABLE_IPFS_KIT", "true")):
        return None

    try:
        backend = IPFSKitBackend(deps=deps)
        return backend
    except Exception:
        return None


def _get_hf_cache_backend() -> Optional[IPFSBackend]:
    """Get HuggingFace cache backend if enabled."""
    if not _truthy(os.getenv("ENABLE_HF_CACHE", "true")):
        return None

    try:
        return HuggingFaceCacheBackend()
    except Exception:
        return None


def _get_kubo_backend() -> Optional[IPFSBackend]:
    """Get Kubo CLI backend (always available)."""
    try:
        return KuboCLIBackend()
    except Exception:
        return None


@lru_cache(maxsize=1)
def _get_default_backend_cached(cache_key: tuple, deps: object = None) -> IPFSBackend:
    """Get the default backend with caching.
    
    This tries backends in order of preference:
    1. ipfs_kit_py (preferred for distributed storage)
    2. HuggingFace cache (good for model storage)
    3. Kubo CLI (fallback)
    """
    # Check for explicit backend override
    backend_name = os.getenv("IPFS_BACKEND", "").strip()
    if backend_name and backend_name in _PROVIDER_REGISTRY:
        provider = _PROVIDER_REGISTRY[backend_name]
        return provider.factory()
    
    # Try backends in order of preference
    backends = [
        ("ipfs_kit", lambda: _get_ipfs_kit_backend(deps)),
        ("hf_cache", _get_hf_cache_backend),
        ("kubo", _get_kubo_backend),
    ]
    
    for name, factory in backends:
        try:
            backend = factory()
            if backend is not None:
                return backend
        except Exception:
            continue
    
    # If all fail, use Kubo as absolute fallback
    return KuboCLIBackend()


def get_backend(*, deps: object = None, backend: Optional[IPFSBackend] = None) -> IPFSBackend:
    """Get the IPFS backend to use.
    
    Args:
        deps: Optional dependency injection container
        backend: Optional explicit backend instance
    
    Returns:
        IPFSBackend instance
    """
    # Use explicit backend if provided
    if backend is not None:
        return backend
    
    # Check for global override
    if _DEFAULT_BACKEND_OVERRIDE is not None:
        return _DEFAULT_BACKEND_OVERRIDE
    
    # Get cached backend
    if _cache_enabled():
        cache_key = _backend_cache_key()
        return _get_default_backend_cached(cache_key, deps)
    
    # No caching - create new backend
    return _get_default_backend_cached.__wrapped__(_backend_cache_key(), deps)


# Convenience functions that use the default backend

def add_bytes(data: bytes, *, pin: bool = True, backend: Optional[IPFSBackend] = None, deps: object = None) -> str:
    """Add bytes to IPFS and return CID."""
    return get_backend(deps=deps, backend=backend).add_bytes(data, pin=pin)


def cat(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> bytes:
    """Retrieve data by CID."""
    return get_backend(deps=deps, backend=backend).cat(cid)


def pin(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> None:
    """Pin content by CID."""
    get_backend(deps=deps, backend=backend).pin(cid)


def unpin(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> None:
    """Unpin content by CID."""
    get_backend(deps=deps, backend=backend).unpin(cid)


def block_put(data: bytes, *, codec: str = "raw", backend: Optional[IPFSBackend] = None, deps: object = None) -> str:
    """Store a raw block and return its CID."""
    return get_backend(deps=deps, backend=backend).block_put(data, codec=codec)


def block_get(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> bytes:
    """Get a raw block by CID."""
    return get_backend(deps=deps, backend=backend).block_get(cid)


def add_path(
    path: str,
    *,
    recursive: bool = True,
    pin: bool = True,
    chunker: Optional[str] = None,
    backend: Optional[IPFSBackend] = None,
    deps: object = None
) -> str:
    """Add a file or directory to IPFS."""
    return get_backend(deps=deps, backend=backend).add_path(path, recursive=recursive, pin=pin, chunker=chunker)


def get_to_path(cid: str, *, output_path: str, backend: Optional[IPFSBackend] = None, deps: object = None) -> None:
    """Retrieve content and save to path."""
    get_backend(deps=deps, backend=backend).get_to_path(cid, output_path=output_path)


def ls(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> list[str]:
    """List directory contents."""
    return get_backend(deps=deps, backend=backend).ls(cid)


def dag_export(cid: str, *, backend: Optional[IPFSBackend] = None, deps: object = None) -> bytes:
    """Export DAG as CAR file."""
    return get_backend(deps=deps, backend=backend).dag_export(cid)
