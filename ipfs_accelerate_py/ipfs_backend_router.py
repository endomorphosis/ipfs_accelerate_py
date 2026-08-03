"""IPFS backend router for ipfs_accelerate_py.

This module provides a stable entry point for basic IPFS operations with a
pluggable backend strategy:
- Preferred: ipfs_kit_py backend when explicitly enabled and available
- Fallback 1: HuggingFace model cache for model storage (cache-only role)
- Fallback 2: Local Kubo via the `ipfs` CLI

Design goals:
- Avoid importing ipfs_kit_py at module import time
- Prefer ipfs_kit_py for distributed storage
- Fall back gracefully to HF cache and Kubo with *explicit* degradation
- Accurately report backend roles (ipfs_kit_py / kubo / cache)
- Never treat synthetic HF ``bafy…`` cache keys as multiformats CIDs
- Do not assume codec preservation or CAR export unless the role claims it
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
import subprocess
import tempfile
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable
from pathlib import Path

try:
    from .router_deps import RouterDeps, get_default_router_deps
except ImportError:
    # Fallback for when imported standalone
    RouterDeps = None
    get_default_router_deps = lambda: None  # noqa: E731


# ---------------------------------------------------------------------------
# Backend role vocabulary (replication-facing)
# ---------------------------------------------------------------------------


class BackendRole(str, Enum):
    """Closed role set for IPFS/IPLD replication adapters.

    Roles are structural classifications, not a ranking:

    * ``IPFS_KIT`` — preferred distributed kit; codec preservation is not
      assumed and must be verified by callers.
    * ``KUBO`` — local Kubo CLI; can advertise CAR when the CLI supports it.
    * ``CACHE`` — local/cache transport only (e.g. HuggingFace).  Identifiers
      emitted by this role must never be admitted as IPLD CIDs for
      coordination manifests until the adapter becomes conformant.
    * ``MEMORY`` — in-process conformant store (tests / local verification).
    * ``UNKNOWN`` — unregistered / custom provider; treat as non-conformant
      until inspected and verified.
    """

    IPFS_KIT = "ipfs_kit_py"
    KUBO = "kubo"
    CACHE = "cache"
    MEMORY = "memory"
    UNKNOWN = "unknown"


# Stable string aliases used in receipts and tests.
ROLE_IPFS_KIT = BackendRole.IPFS_KIT.value
ROLE_KUBO = BackendRole.KUBO.value
ROLE_CACHE = BackendRole.CACHE.value
ROLE_MEMORY = BackendRole.MEMORY.value
ROLE_UNKNOWN = BackendRole.UNKNOWN.value


def _truthy(value: Optional[str]) -> bool:
    """Check if an environment variable value is truthy."""
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_enabled() -> bool:
    """Check if backend caching is enabled."""
    return os.environ.get("IPFS_ROUTER_CACHE", "1").strip() != "0"


_DEFAULT_BACKEND_OVERRIDE: "IPFSBackend | None" = None


def set_default_ipfs_backend(backend: "IPFSBackend | None") -> None:
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


@dataclass(frozen=True)
class BackendCapabilityInfo:
    """Static capability matrix for a concrete backend instance or name.

    ``conformant_cid`` is true only when the adapter is known to emit real
    multiformats CIDv1 strings (not synthetic cache keys).  CAR support is
    capability-gated and must never be assumed from role alone for unknown
    providers.
    """

    name: str
    role: str
    conformant_cid: bool
    supports_raw: bool
    supports_dag_json: bool
    supports_car: bool
    supports_pin: bool
    codec_preservation_guaranteed: bool
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "conformant_cid": self.conformant_cid,
            "supports_raw": self.supports_raw,
            "supports_dag_json": self.supports_dag_json,
            "supports_car": self.supports_car,
            "supports_pin": self.supports_pin,
            "codec_preservation_guaranteed": self.codec_preservation_guaranteed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BackendSelectionReceipt:
    """Explicit selection + degradation record for the active backend.

    Degradation is never silent: when the preferred ``ipfs_kit_py`` path is
    unavailable, ``degraded`` is true and ``degradation_reasons`` lists why.
    """

    selected_name: str
    selected_role: str
    preferred_name: str
    preferred_available: bool
    degraded: bool
    degradation_reasons: Tuple[str, ...]
    capabilities: BackendCapabilityInfo
    candidate_order: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_name": self.selected_name,
            "selected_role": self.selected_role,
            "preferred_name": self.preferred_name,
            "preferred_available": self.preferred_available,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
            "capabilities": self.capabilities.to_dict(),
            "candidate_order": list(self.candidate_order),
        }


_PROVIDER_REGISTRY: Dict[str, ProviderInfo] = {}

# Last selection metadata (process-local; cleared with cache).
_LAST_SELECTION_RECEIPT: Optional[BackendSelectionReceipt] = None


def register_ipfs_backend(name: str, factory: ProviderFactory) -> None:
    """Register a new IPFS backend provider."""
    if not name or not name.strip():
        raise ValueError("Backend name must be non-empty")
    _PROVIDER_REGISTRY[name] = ProviderInfo(name=name, factory=factory)


def _role_for_name(name: str) -> str:
    normalized = (name or "").strip().lower()
    if normalized in {"ipfs_kit", "ipfs_kit_py", "kit"}:
        return ROLE_IPFS_KIT
    if normalized in {"kubo", "ipfs", "kubo_cli"}:
        return ROLE_KUBO
    if normalized in {"hf_cache", "huggingface", "hf", "cache"}:
        return ROLE_CACHE
    if normalized in {"memory", "in_memory", "in-memory", "mem"}:
        return ROLE_MEMORY
    return ROLE_UNKNOWN


def classify_backend_role(backend: object) -> str:
    """Return the structural role for a backend instance."""

    if isinstance(backend, IPFSKitBackend):
        return ROLE_IPFS_KIT
    if isinstance(backend, KuboCLIBackend):
        return ROLE_KUBO
    if isinstance(backend, HuggingFaceCacheBackend):
        return ROLE_CACHE
    # Honor optional explicit attributes on custom adapters.
    explicit = getattr(backend, "backend_role", None)
    if isinstance(explicit, BackendRole):
        return explicit.value
    if isinstance(explicit, str) and explicit.strip():
        # Prefer exact enum values over name heuristics.
        for role in BackendRole:
            if explicit.strip() == role.value:
                return role.value
        return _role_for_name(explicit)
    name = getattr(backend, "backend_name", None)
    if isinstance(name, str) and name.strip():
        return _role_for_name(name)
    return ROLE_UNKNOWN


def describe_backend_capabilities(backend: object) -> BackendCapabilityInfo:
    """Describe known capabilities for a backend without probing the network.

    CAR / codec claims are fail-closed for cache and unknown roles: callers
    must not assume export or codec preservation.
    """

    role = classify_backend_role(backend)
    if role == ROLE_IPFS_KIT:
        return BackendCapabilityInfo(
            name="ipfs_kit",
            role=role,
            conformant_cid=True,  # expected to emit real CIDs; still re-verify
            supports_raw=True,
            supports_dag_json=False,  # add must not be assumed codec-preserving
            supports_car=False,  # dag_export not available in kit adapter
            supports_pin=True,
            codec_preservation_guaranteed=False,
            notes=(
                "ipfs_kit_py is preferred for distributed storage",
                "codec preservation is not assumed on add; re-verify CIDs",
                "CAR/dag_export is not available through this adapter",
            ),
        )
    if role == ROLE_KUBO:
        return BackendCapabilityInfo(
            name="kubo",
            role=role,
            conformant_cid=True,
            supports_raw=True,
            supports_dag_json=True,  # block put --format when CLI supports it
            supports_car=True,  # dag export via CLI when available
            supports_pin=True,
            codec_preservation_guaranteed=False,  # still re-verify
            notes=(
                "local Kubo CLI backend",
                "CAR export is capability-gated on the installed CLI",
                "add may return CIDv0 unless block put is used",
            ),
        )
    if role == ROLE_CACHE:
        return BackendCapabilityInfo(
            name="hf_cache",
            role=role,
            conformant_cid=False,  # synthetic bafy… keys are NOT multiformats CIDs
            supports_raw=True,  # stores bytes locally
            supports_dag_json=False,
            supports_car=False,
            supports_pin=True,  # local pin metadata only
            codec_preservation_guaranteed=False,
            notes=(
                "HuggingFace cache is local/cache transport only",
                "synthetic bafy… identifiers must never enter coordination manifests",
                "CAR export is unsupported",
            ),
        )
    if role == ROLE_MEMORY:
        return BackendCapabilityInfo(
            name="memory",
            role=role,
            conformant_cid=True,
            supports_raw=True,
            supports_dag_json=True,
            supports_car=True,
            supports_pin=True,
            codec_preservation_guaranteed=True,
            notes=(
                "in-process conformant store for verification and tests",
                "emits real CIDv1 raw/dag-json identifiers",
            ),
        )
    # Unknown / custom
    name = getattr(backend, "backend_name", None)
    label = str(name).strip() if isinstance(name, str) and name.strip() else "unknown"
    return BackendCapabilityInfo(
        name=label,
        role=ROLE_UNKNOWN,
        conformant_cid=False,
        supports_raw=True,
        supports_dag_json=False,
        supports_car=False,
        supports_pin=hasattr(backend, "pin"),
        codec_preservation_guaranteed=False,
        notes=(
            "unregistered backend; treat as non-conformant until verified",
            "CAR and codec support are not assumed",
        ),
    )


def get_last_backend_selection() -> Optional[BackendSelectionReceipt]:
    """Return the most recent selection receipt, if any."""

    return _LAST_SELECTION_RECEIPT


def _record_selection(receipt: BackendSelectionReceipt) -> BackendSelectionReceipt:
    global _LAST_SELECTION_RECEIPT
    _LAST_SELECTION_RECEIPT = receipt
    return receipt


class IPFSKitBackend:
    """IPFS backend using ipfs_kit_py (preferred distributed role)."""

    backend_name = "ipfs_kit"
    backend_role = BackendRole.IPFS_KIT

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
        """Store a block and return its CID.

        Codec preservation is **not** guaranteed by ipfs_kit_py; callers that
        need a specific codec must recompute and verify the CID locally.
        """
        # Kit storage path does not accept a codec argument today.
        _ = codec
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
        _ = recursive, chunker
        return self._storage.store(Path(path), pin=pin)

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        """Retrieve content and save to path."""
        data = self.cat(cid)
        Path(output_path).write_bytes(data)

    def ls(self, cid: str) -> list[str]:
        """List directory contents."""
        # This would require more complex IPFS directory handling
        # For now, return empty list as not all backends support this
        _ = cid
        return []

    def dag_export(self, cid: str) -> bytes:
        """Export DAG as CAR file (unsupported — fail closed)."""
        _ = cid
        raise RuntimeError(
            "dag_export/CAR not available in ipfs_kit backend "
            f"(role={ROLE_IPFS_KIT})"
        )


class HuggingFaceCacheBackend:
    """Local/cache transport using HuggingFace model cache directories.

    Role: ``cache``.  Identifiers produced by this adapter are synthetic
    ``bafy…`` cache keys derived from a truncated hex digest.  They are **not**
    multiformats CIDv1 objects and must never be admitted into coordination
    manifests as IPLD CIDs.  CAR export is unsupported.
    """

    backend_name = "hf_cache"
    backend_role = BackendRole.CACHE

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
        """Generate a synthetic cache key (NOT a multiformats CID).

        Compatibility note: historical callers and tests expect a ``bafy``
        prefix.  The verified IPLD adapter must reject these strings via
        ``validate_cid`` / rehash admission rather than treating them as CIDs.
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
        """Store bytes in HF cache (codec is ignored; cache-only)."""
        _ = codec
        return self.add_bytes(data, pin=True)

    def block_get(self, cid: str) -> bytes:
        """Get a block by cache key from HF cache."""
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
        _ = recursive, chunker
        data = Path(path).read_bytes()
        return self.add_bytes(data, pin=pin)

    def get_to_path(self, cid: str, *, output_path: str) -> None:
        """Retrieve content and save to path."""
        data = self.cat(cid)
        Path(output_path).write_bytes(data)

    def ls(self, cid: str) -> list[str]:
        """List directory contents (not supported in HF cache)."""
        _ = cid
        return []

    def dag_export(self, cid: str) -> bytes:
        """Export DAG (unsupported for cache role — fail closed)."""
        _ = cid
        raise RuntimeError(
            "dag_export/CAR not available in HF cache backend "
            f"(role={ROLE_CACHE}; cache-only until conformant)"
        )


class KuboCLIBackend:
    """IPFS backend using local Kubo CLI (role: kubo)."""

    backend_name = "kubo"
    backend_role = BackendRole.KUBO

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
    """Get Kubo CLI backend (always available as last resort)."""
    try:
        return KuboCLIBackend()
    except Exception:
        return None


def _probe_ipfs_kit_availability(deps: object = None) -> Tuple[bool, Optional[str]]:
    """Return (available, reason_if_not)."""
    if _truthy(os.getenv("IPFS_KIT_DISABLE")):
        return False, "IPFS_KIT_DISABLE is set"
    if not _truthy(os.getenv("ENABLE_IPFS_KIT", "true")):
        return False, "ENABLE_IPFS_KIT is false"
    try:
        backend = IPFSKitBackend(deps=deps)
        _ = backend
        return True, None
    except Exception as exc:
        return False, f"ipfs_kit_py initialization failed: {exc}"


def _build_selection_receipt(
    *,
    selected: IPFSBackend,
    selected_name: str,
    preferred_available: bool,
    degradation_reasons: Sequence[str],
    candidate_order: Sequence[str],
) -> BackendSelectionReceipt:
    caps = describe_backend_capabilities(selected)
    role = classify_backend_role(selected)
    reasons = tuple(str(r) for r in degradation_reasons if r)
    # Preferring non-kit, or selecting cache, is always an explicit degradation.
    degraded = bool(reasons) or role == ROLE_CACHE or selected_name != "ipfs_kit"
    if role == ROLE_CACHE and "selected cache-only HuggingFace backend" not in reasons:
        reasons = reasons + (
            "selected cache-only HuggingFace backend; synthetic identifiers "
            "are not coordination CIDs",
        )
    if not preferred_available and not any("ipfs_kit" in r for r in reasons):
        reasons = ("preferred ipfs_kit_py backend unavailable",) + reasons
    return BackendSelectionReceipt(
        selected_name=selected_name,
        selected_role=role,
        preferred_name="ipfs_kit",
        preferred_available=preferred_available,
        degraded=degraded,
        degradation_reasons=reasons,
        capabilities=caps,
        candidate_order=tuple(candidate_order),
    )


def select_backend(
    *,
    deps: object = None,
    backend: Optional[IPFSBackend] = None,
) -> Tuple[IPFSBackend, BackendSelectionReceipt]:
    """Select a backend and return it with an explicit selection receipt.

    Preference order (when ``IPFS_BACKEND`` is unset):

    1. ``ipfs_kit_py`` (preferred distributed role)
    2. HuggingFace cache (cache-only; degraded for coordination)
    3. Kubo CLI (fallback)

    Degradation is always recorded on the receipt; it is never silent.
    """
    if backend is not None:
        name = getattr(backend, "backend_name", None)
        label = str(name) if isinstance(name, str) and name.strip() else "explicit"
        receipt = _build_selection_receipt(
            selected=backend,
            selected_name=label,
            preferred_available=classify_backend_role(backend) == ROLE_IPFS_KIT,
            degradation_reasons=(
                ()
                if classify_backend_role(backend) == ROLE_IPFS_KIT
                else ("explicit backend override is not preferred ipfs_kit_py",)
            ),
            candidate_order=(label,),
        )
        return backend, _record_selection(receipt)

    if _DEFAULT_BACKEND_OVERRIDE is not None:
        override = _DEFAULT_BACKEND_OVERRIDE
        name = getattr(override, "backend_name", None)
        label = str(name) if isinstance(name, str) and name.strip() else "override"
        receipt = _build_selection_receipt(
            selected=override,
            selected_name=label,
            preferred_available=classify_backend_role(override) == ROLE_IPFS_KIT,
            degradation_reasons=(
                ()
                if classify_backend_role(override) == ROLE_IPFS_KIT
                else ("process-global backend override is not preferred ipfs_kit_py",)
            ),
            candidate_order=(label,),
        )
        return override, _record_selection(receipt)

    preferred_ok, preferred_reason = _probe_ipfs_kit_availability(deps)
    candidate_order = ("ipfs_kit", "hf_cache", "kubo")
    degradation: List[str] = []
    if not preferred_ok and preferred_reason:
        degradation.append(preferred_reason)

    # Explicit registered provider
    backend_name = os.getenv("IPFS_BACKEND", "").strip()
    if backend_name and backend_name in _PROVIDER_REGISTRY:
        provider = _PROVIDER_REGISTRY[backend_name]
        selected = provider.factory()
        reasons = list(degradation)
        if _role_for_name(backend_name) != ROLE_IPFS_KIT:
            reasons.append(f"IPFS_BACKEND={backend_name!r} forces non-preferred role")
        receipt = _build_selection_receipt(
            selected=selected,
            selected_name=backend_name,
            preferred_available=preferred_ok,
            degradation_reasons=reasons,
            candidate_order=(backend_name,),
        )
        return selected, _record_selection(receipt)

    factories: List[Tuple[str, Callable[[], Optional[IPFSBackend]]]] = [
        ("ipfs_kit", lambda: _get_ipfs_kit_backend(deps)),
        ("hf_cache", _get_hf_cache_backend),
        ("kubo", _get_kubo_backend),
    ]

    for name, factory in factories:
        try:
            candidate = factory()
        except Exception as exc:
            degradation.append(f"{name} factory raised: {exc}")
            continue
        if candidate is None:
            if name == "hf_cache" and not _truthy(os.getenv("ENABLE_HF_CACHE", "true")):
                degradation.append("ENABLE_HF_CACHE is false")
            elif name != "ipfs_kit":
                degradation.append(f"{name} unavailable")
            continue
        reasons = list(degradation)
        if name != "ipfs_kit":
            reasons.append(f"fell back to {name}")
        receipt = _build_selection_receipt(
            selected=candidate,
            selected_name=name,
            preferred_available=preferred_ok and name == "ipfs_kit",
            degradation_reasons=reasons,
            candidate_order=candidate_order,
        )
        return candidate, _record_selection(receipt)

    # Absolute fallback — Kubo CLI object even if prior factory returned None.
    fallback = KuboCLIBackend()
    reasons = list(degradation) + ["absolute fallback to Kubo CLI"]
    receipt = _build_selection_receipt(
        selected=fallback,
        selected_name="kubo",
        preferred_available=False,
        degradation_reasons=reasons,
        candidate_order=candidate_order,
    )
    return fallback, _record_selection(receipt)


@lru_cache(maxsize=1)
def _get_default_backend_cached(cache_key: tuple, deps: object = None) -> IPFSBackend:
    """Get the default backend with caching.

    Selection side effects (receipt recording) run on each cache miss.
    """
    _ = cache_key
    backend, _receipt = select_backend(deps=deps)
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
        _, _ = select_backend(backend=backend)
        return backend

    # Check for global override
    if _DEFAULT_BACKEND_OVERRIDE is not None:
        selected, _ = select_backend()
        return selected

    # Get cached backend (receipt recorded on cache miss via select_backend)
    if _cache_enabled():
        cache_key = _backend_cache_key()
        # Ensure a receipt exists even on cache hit by reusing last receipt
        # or refreshing when missing.
        result = _get_default_backend_cached(cache_key, deps)
        if _LAST_SELECTION_RECEIPT is None:
            _, _ = select_backend(deps=deps)
        return result

    # No caching - create new backend
    selected, _ = select_backend(deps=deps)
    return selected


def get_backend_with_receipt(
    *,
    deps: object = None,
    backend: Optional[IPFSBackend] = None,
) -> Tuple[IPFSBackend, BackendSelectionReceipt]:
    """Like :func:`get_backend` but always returns an explicit selection receipt."""

    if backend is not None:
        return select_backend(backend=backend)
    if _DEFAULT_BACKEND_OVERRIDE is not None:
        return select_backend()
    # Bypass instance cache so the receipt matches the returned instance.
    return select_backend(deps=deps)


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
