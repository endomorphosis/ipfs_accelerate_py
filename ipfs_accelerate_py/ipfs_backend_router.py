"""IPFS backend router for ipfs_accelerate_py.

This module provides a stable entry point for basic IPFS operations with a
pluggable backend strategy:
- Preferred: ipfs_kit_py backend when explicitly enabled and available
- Fallback 1: HuggingFace model cache for model storage (cache-only role)
- Fallback 2: Local Kubo via the `ipfs` CLI

Design goals:
- Avoid importing ipfs_kit_py at module import time
- Prefer ipfs_kit_py for distributed storage
- Fall back gracefully to HF cache and Kubo with explicit degradation
- Accurately report backend roles (ipfs_kit_py / Kubo / cache)
- Never claim that a cache synthetic identifier is a strict CIDv1
- Keep behavior predictable in benchmarks/CI

Environment variables:
- `IPFS_BACKEND`: force backend name (registered provider)
- `ENABLE_IPFS_KIT`: enable ipfs_kit_py backend (preferred, default: true)
- `ENABLE_HF_CACHE`: enable HuggingFace cache backend (default: true)
- `IPFS_KIT_DISABLE`: disable ipfs_kit_py backend completely
- `KUBO_CMD`: override ipfs CLI command (default: "ipfs")
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

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


class BackendRole(str, Enum):
    """Closed vocabulary of IPFS transport roles for coordination.

    Roles are reporting/classification labels.  A successful ``add`` is never
    enough to claim strict CIDv1 / codec / CAR conformance — that is decided by
    capability probes and the verified IPLD adapter.
    """

    IPFS_KIT = "ipfs_kit_py"
    KUBO = "kubo"
    CACHE = "cache"
    UNKNOWN = "unknown"


# Preferred selection order for automatic routing (not capability authority).
PREFERRED_BACKEND_ORDER: Tuple[str, ...] = ("ipfs_kit", "hf_cache", "kubo")


@dataclass(frozen=True)
class BackendCapabilityDescriptor:
    """Static + probe-time capability matrix entry for one backend instance.

    ``synthetic_identifiers`` means the transport may return non-canonical
    identifiers (for example Hugging Face cache ``bafy…`` prefixes).  Those
    handles must never be admitted into coordination manifests as CIDs.
    ``preserves_requested_codec`` is False when an add/put path cannot be
    trusted to honor the codec the caller requested.
    """

    name: str
    role: BackendRole
    synthetic_identifiers: bool
    admits_strict_cid: bool
    supports_raw: bool
    supports_dag_json: bool
    supports_car: bool
    preserves_requested_codec: bool
    pin_supported: bool
    degraded: bool = False
    degradation_reasons: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["role"] = self.role.value
        payload["degradation_reasons"] = list(self.degradation_reasons)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class BackendSelectionReceipt:
    """Explicit report of which backend was selected and why others were skipped.

    Degradation is always explicit: falling back from ipfs_kit_py to cache or
    Kubo records reasons rather than silently presenting cache as IPFS.
    """

    selected_name: str
    selected_role: BackendRole
    preferred_order: Tuple[str, ...]
    attempted: Tuple[Tuple[str, str], ...]  # (name, outcome)
    degraded: bool
    degradation_reasons: Tuple[str, ...]
    capability: BackendCapabilityDescriptor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_name": self.selected_name,
            "selected_role": self.selected_role.value,
            "preferred_order": list(self.preferred_order),
            "attempted": [{"name": n, "outcome": o} for n, o in self.attempted],
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "capability": self.capability.to_dict(),
        }


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
    """IPFS backend using ipfs_kit_py (preferred distributed transport).

    Role: ``ipfs_kit_py``.  Add paths are not assumed to preserve a requested
    codec, and CAR export is not available through the basic storage layer.
    Coordination code must verify returned identifiers through the verified
    IPLD adapter rather than trusting an add receipt alone.
    """

    BACKEND_NAME = "ipfs_kit"
    BACKEND_ROLE = BackendRole.IPFS_KIT

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

    def capability_descriptor(self) -> BackendCapabilityDescriptor:
        """Return the closed capability matrix for this backend instance."""
        return BackendCapabilityDescriptor(
            name=self.BACKEND_NAME,
            role=self.BACKEND_ROLE,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=False,
            supports_car=False,
            preserves_requested_codec=False,
            pin_supported=True,
            degraded=False,
            degradation_reasons=(),
            notes=(
                "ipfs_kit_py add is not assumed to preserve a requested codec",
                "dag_export/CAR is not available in the basic storage layer",
            ),
        )

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
        # Codec is not guaranteed to be preserved by the kit storage layer.
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
    """Local/cache transport using the HuggingFace model cache layout.

    Role: ``cache``.  Identifiers are synthetic content keys shaped like
    ``bafy…`` prefixes and are **not** strict CIDv1 objects.  They must never
    be admitted into coordination manifests.  CAR export is unsupported.
    """

    BACKEND_NAME = "hf_cache"
    BACKEND_ROLE = BackendRole.CACHE

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """Initialize HuggingFace cache backend.

        Args:
            cache_dir: Directory for cache (defaults to HF_HOME)
        """
        self._cache_dir = Path(cache_dir or os.getenv("HF_HOME") or
                               os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        self._ipfs_cache = self._cache_dir / "ipfs_blocks"
        self._ipfs_cache.mkdir(parents=True, exist_ok=True)

    def capability_descriptor(self) -> BackendCapabilityDescriptor:
        """Return the closed capability matrix for this cache transport."""
        return BackendCapabilityDescriptor(
            name=self.BACKEND_NAME,
            role=self.BACKEND_ROLE,
            synthetic_identifiers=True,
            admits_strict_cid=False,
            supports_raw=True,
            supports_dag_json=False,
            supports_car=False,
            preserves_requested_codec=False,
            pin_supported=True,
            degraded=True,
            degradation_reasons=(
                "huggingface_cache_is_local_transport_not_strict_cid",
            ),
            notes=(
                "synthetic bafy… identifiers are cache keys, not IPLD CIDs",
                "classification remains cache-only until strict CID conformance",
            ),
        )

    def _generate_cid(self, data: bytes) -> str:
        """Generate a synthetic cache key (not a strict CIDv1).

        Retained for transport compatibility.  Callers that need coordination
        authority must route through the verified IPLD adapter, which never
        admits this identifier as a CID.
        """
        hash_value = hashlib.sha256(data).hexdigest()
        return f"bafy{hash_value[:56]}"

    def add_bytes(self, data: bytes, *, pin: bool = True) -> str:
        """Store bytes in HF cache and return a synthetic cache key."""
        cid = self._generate_cid(data)
        block_path = self._ipfs_cache / cid
        block_path.write_bytes(data)

        # Store metadata about pinning
        if pin:
            meta_path = self._ipfs_cache / f"{cid}.meta"
            meta_path.write_text(json.dumps({"pinned": True}))

        return cid

    def cat(self, cid: str) -> bytes:
        """Retrieve data by cache key from HF cache."""
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
        """Store a raw block in HF cache (codec is not preserved as CID)."""
        return self.add_bytes(data, pin=True)

    def block_get(self, cid: str) -> bytes:
        """Get a raw block by cache key from HF cache."""
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
    """IPFS backend using local Kubo CLI.

    Role: ``kubo``.  Supports CAR via ``ipfs dag export`` when the daemon and
    CLI are healthy.  Returned identifiers still require verified rehash before
    coordination-manifest admission.
    """

    BACKEND_NAME = "kubo"
    BACKEND_ROLE = BackendRole.KUBO

    def __init__(self, cmd: Optional[str] = None) -> None:
        """Initialize Kubo CLI backend.

        Args:
            cmd: IPFS CLI command (defaults to 'ipfs')
        """
        self._cmd = cmd or os.getenv("KUBO_CMD", "ipfs")

    def capability_descriptor(self) -> BackendCapabilityDescriptor:
        """Return the closed capability matrix for the Kubo CLI transport."""
        return BackendCapabilityDescriptor(
            name=self.BACKEND_NAME,
            role=self.BACKEND_ROLE,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=True,
            supports_car=True,
            preserves_requested_codec=True,
            pin_supported=True,
            degraded=False,
            degradation_reasons=(),
            notes=(
                "CAR export is capability-gated on a working ipfs dag export",
                "block put requests CIDv1 with the declared format/codec",
            ),
        )

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
                except OSError:
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
    """Get Kubo CLI backend (always available as last-resort transport)."""
    try:
        return KuboCLIBackend()
    except Exception:
        return None


def _backend_role_of(backend: IPFSBackend) -> BackendRole:
    """Classify a backend instance into a closed role vocabulary."""
    role = getattr(backend, "BACKEND_ROLE", None)
    if isinstance(role, BackendRole):
        return role
    if isinstance(backend, IPFSKitBackend):
        return BackendRole.IPFS_KIT
    if isinstance(backend, HuggingFaceCacheBackend):
        return BackendRole.CACHE
    if isinstance(backend, KuboCLIBackend):
        return BackendRole.KUBO
    name = getattr(backend, "BACKEND_NAME", "") or type(backend).__name__
    lowered = str(name).lower()
    if "kit" in lowered:
        return BackendRole.IPFS_KIT
    if "kubo" in lowered or "cli" in lowered:
        return BackendRole.KUBO
    if "cache" in lowered or "hf" in lowered or "huggingface" in lowered:
        return BackendRole.CACHE
    return BackendRole.UNKNOWN


def describe_backend(backend: IPFSBackend) -> BackendCapabilityDescriptor:
    """Return an accurate capability/role descriptor for ``backend``.

    Prefers a backend-provided ``capability_descriptor()`` so roles stay
    authoritative.  Unknown adapters are reported as degraded/unknown rather
    than silently elevated to conformant IPFS.
    """
    descriptor_fn = getattr(backend, "capability_descriptor", None)
    if callable(descriptor_fn):
        descriptor = descriptor_fn()
        if isinstance(descriptor, BackendCapabilityDescriptor):
            return descriptor

    role = _backend_role_of(backend)
    name = str(getattr(backend, "BACKEND_NAME", None) or type(backend).__name__)
    if role is BackendRole.CACHE:
        return BackendCapabilityDescriptor(
            name=name,
            role=role,
            synthetic_identifiers=True,
            admits_strict_cid=False,
            supports_raw=True,
            supports_dag_json=False,
            supports_car=False,
            preserves_requested_codec=False,
            pin_supported=True,
            degraded=True,
            degradation_reasons=("cache_role_without_explicit_descriptor",),
            notes=("cache transports are non-authoritative for coordination CIDs",),
        )
    if role is BackendRole.IPFS_KIT:
        return BackendCapabilityDescriptor(
            name=name,
            role=role,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=False,
            supports_car=False,
            preserves_requested_codec=False,
            pin_supported=True,
            degraded=False,
            notes=("codec preservation is not assumed for ipfs_kit_py",),
        )
    if role is BackendRole.KUBO:
        return BackendCapabilityDescriptor(
            name=name,
            role=role,
            synthetic_identifiers=False,
            admits_strict_cid=True,
            supports_raw=True,
            supports_dag_json=True,
            supports_car=True,
            preserves_requested_codec=True,
            pin_supported=True,
            degraded=False,
        )
    return BackendCapabilityDescriptor(
        name=name,
        role=BackendRole.UNKNOWN,
        synthetic_identifiers=True,
        admits_strict_cid=False,
        supports_raw=True,
        supports_dag_json=False,
        supports_car=False,
        preserves_requested_codec=False,
        pin_supported=False,
        degraded=True,
        degradation_reasons=("unknown_backend_role",),
        notes=("unknown transports fail closed for coordination admission",),
    )


def _probe_named_backend(
    name: str,
    *,
    deps: object = None,
) -> Tuple[Optional[IPFSBackend], str]:
    """Attempt to construct a named backend; return (instance, outcome)."""
    if name == "ipfs_kit":
        if _truthy(os.getenv("IPFS_KIT_DISABLE")):
            return None, "disabled:IPFS_KIT_DISABLE"
        if not _truthy(os.getenv("ENABLE_IPFS_KIT", "true")):
            return None, "disabled:ENABLE_IPFS_KIT"
        try:
            return IPFSKitBackend(deps=deps), "selected"
        except Exception as exc:
            return None, f"unavailable:{type(exc).__name__}"
    if name == "hf_cache":
        if not _truthy(os.getenv("ENABLE_HF_CACHE", "true")):
            return None, "disabled:ENABLE_HF_CACHE"
        try:
            return HuggingFaceCacheBackend(), "selected"
        except Exception as exc:
            return None, f"unavailable:{type(exc).__name__}"
    if name == "kubo":
        try:
            return KuboCLIBackend(), "selected"
        except Exception as exc:
            return None, f"unavailable:{type(exc).__name__}"
    if name in _PROVIDER_REGISTRY:
        try:
            return _PROVIDER_REGISTRY[name].factory(), "selected"
        except Exception as exc:
            return None, f"unavailable:{type(exc).__name__}"
    return None, "unknown_provider"


def select_backend_with_receipt(
    *,
    deps: object = None,
    backend: Optional[IPFSBackend] = None,
) -> Tuple[IPFSBackend, BackendSelectionReceipt]:
    """Select a backend and emit an explicit degradation/role receipt.

    Prefer ``ipfs_kit_py``, then Hugging Face cache, then Kubo.  Cache selection
    is always reported as degraded relative to strict coordination IPLD.
    """
    if backend is not None:
        capability = describe_backend(backend)
        receipt = BackendSelectionReceipt(
            selected_name=capability.name,
            selected_role=capability.role,
            preferred_order=PREFERRED_BACKEND_ORDER,
            attempted=((capability.name, "explicit"),),
            degraded=capability.degraded or capability.role is BackendRole.CACHE,
            degradation_reasons=capability.degradation_reasons
            + (
                ("explicit_backend_override",)
                if capability.role is BackendRole.CACHE
                else ()
            ),
            capability=capability,
        )
        return backend, receipt

    if _DEFAULT_BACKEND_OVERRIDE is not None:
        capability = describe_backend(_DEFAULT_BACKEND_OVERRIDE)
        receipt = BackendSelectionReceipt(
            selected_name=capability.name,
            selected_role=capability.role,
            preferred_order=PREFERRED_BACKEND_ORDER,
            attempted=((capability.name, "global_override"),),
            degraded=True,
            degradation_reasons=capability.degradation_reasons
            + ("global_backend_override",),
            capability=capability,
        )
        return _DEFAULT_BACKEND_OVERRIDE, receipt

    forced = os.getenv("IPFS_BACKEND", "").strip()
    attempted: List[Tuple[str, str]] = []
    degradation: List[str] = []

    if forced:
        chosen, outcome = _probe_named_backend(forced, deps=deps)
        attempted.append((forced, outcome))
        if chosen is not None:
            capability = describe_backend(chosen)
            degraded = capability.degraded or capability.role is BackendRole.CACHE
            if degraded and not capability.degradation_reasons:
                degradation.append(f"forced_backend:{forced}")
            receipt = BackendSelectionReceipt(
                selected_name=capability.name,
                selected_role=capability.role,
                preferred_order=PREFERRED_BACKEND_ORDER,
                attempted=tuple(attempted),
                degraded=degraded,
                degradation_reasons=capability.degradation_reasons
                + tuple(degradation)
                + ((f"forced_backend:{forced}",) if forced != capability.name else ()),
                capability=capability,
            )
            return chosen, receipt
        degradation.append(f"forced_backend_unavailable:{forced}:{outcome}")

    preferred = PREFERRED_BACKEND_ORDER
    for name in preferred:
        chosen, outcome = _probe_named_backend(name, deps=deps)
        attempted.append((name, outcome))
        if chosen is None:
            if outcome.startswith("disabled"):
                degradation.append(f"{name}:{outcome}")
            elif outcome.startswith("unavailable"):
                degradation.append(f"{name}:{outcome}")
            continue
        capability = describe_backend(chosen)
        degraded = bool(degradation) or capability.degraded or capability.role is BackendRole.CACHE
        reasons = list(degradation)
        reasons.extend(capability.degradation_reasons)
        if name != preferred[0] and not any(r.startswith(preferred[0]) for r in reasons):
            reasons.append(f"fallback_from:{preferred[0]}")
        if capability.role is BackendRole.CACHE:
            reasons.append("selected_cache_role_not_strict_cid")
        receipt = BackendSelectionReceipt(
            selected_name=capability.name,
            selected_role=capability.role,
            preferred_order=preferred,
            attempted=tuple(attempted),
            degraded=degraded,
            degradation_reasons=tuple(dict.fromkeys(reasons)),
            capability=capability,
        )
        return chosen, receipt

    # Absolute fallback: construct Kubo even if earlier probe failed mildly.
    fallback = KuboCLIBackend()
    capability = describe_backend(fallback)
    attempted.append(("kubo", "absolute_fallback"))
    receipt = BackendSelectionReceipt(
        selected_name=capability.name,
        selected_role=capability.role,
        preferred_order=preferred,
        attempted=tuple(attempted),
        degraded=True,
        degradation_reasons=tuple(
            dict.fromkeys(
                list(degradation)
                + list(capability.degradation_reasons)
                + ["absolute_kubo_fallback"]
            )
        ),
        capability=capability,
    )
    return fallback, receipt


@lru_cache(maxsize=1)
def _get_default_backend_cached(cache_key: tuple, deps: object = None) -> IPFSBackend:
    """Get the default backend with caching.

    This tries backends in order of preference:
    1. ipfs_kit_py (preferred for distributed storage)
    2. HuggingFace cache (local/cache transport)
    3. Kubo CLI (fallback)
    """
    backend, _receipt = select_backend_with_receipt(deps=deps)
    return backend


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


def get_backend_selection_receipt(
    *,
    deps: object = None,
    backend: Optional[IPFSBackend] = None,
) -> BackendSelectionReceipt:
    """Return the selection/degradation receipt for the effective backend."""
    _chosen, receipt = select_backend_with_receipt(deps=deps, backend=backend)
    return receipt


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
