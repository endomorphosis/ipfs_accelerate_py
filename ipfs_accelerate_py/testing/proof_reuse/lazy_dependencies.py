"""Bounded first-use lazy installation for proof-reuse capabilities.

This module is the accelerator zero-config dependency surface
(``ProofReuseLazyDependencyInstaller@1``).  Importing it is cold-safe: no
package is installed, no network call is made, and the pytest plugin is not
imported.  Installation runs only when a requested proof-reuse capability is
missing *and* the package auto-install policy permits it.

Failure always degrades to a typed capability reason so callers continue with
normal test execution.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import urlsplit

from .services import (
    DATASETS_GROTH16_ARTIFACTS_ROOT_ENV,
    DATASETS_GROTH16_BINARY_ENV,
    DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES,
    DATASETS_GROTH16_BUNDLED_BINARIES_SHA256,
    DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256,
    DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
    DATASETS_GROTH16_RELEASE_MANIFESTS_SHA256,
    DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256,
    DATASETS_GROTH16_REVIEWED_FILES_SHA256,
    DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT,
    DATASETS_VERIFIER_DEPENDENCY,
    DATASETS_VERIFIER_REVISION,
    DEFAULT_NLTK_DATA_RESOURCES,
    JSONSCHEMA_DEPENDENCY,
    MULTIFORMATS_DEPENDENCY,
    NLTK_DATA_RESOURCE_ALLOWLIST,
    NLTK_DEPENDENCY,
    PACKAGE_AUTO_INSTALL_ENV,
    PROOF_REUSE_CACHE_DIR_ENV,
    PROOF_REUSE_DEPENDENCY_ALLOWLIST,
    PROOF_REUSE_GROTH16_BUILD_ENV,
    PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV,
    PROOF_REUSE_GROTH16_ENDPOINT_ENV,
    PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV,
    PROOF_REUSE_NLTK_DATA_DIR_ENV,
    PROOF_REUSE_NLTK_DOWNLOAD_ENV,
    PROOF_REUSE_PROVISION_DIR_ENV,
    AllowlistedPipInstaller,
    DefaultProofReuseServices,
    ProofReuseDependency,
    TEST_PASS_GROTH16_CIRCUIT_CID,
    TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256,
    TEST_PASS_GROTH16_CIRCUIT_VERSION,
    TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256,
    TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS,
    automatic_install_enabled,
    compose_default_proof_reuse_services,
    groth16_build_enabled,
    isolated_pip_environment,
    nltk_data_download_enabled,
    package_auto_install_policy_permits,
    proof_reuse_install_permitted,
    proof_reuse_dependency_plan,
    GROTH16_NATIVE_BUILD_RECEIPT_INTERFACE,
    validate_groth16_capability_payload,
    validate_groth16_release_manifest_payload,
)

PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE: Final = (
    "ProofReuseLazyDependencyInstaller@1"
)
ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE: Final = "AcceleratorProofReuseBootstrap@1"

PLUGIN_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PROOF_REUSE_CONFIG_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.config"

# Internal, non-authoritative provenance continuity for a reviewed artifacts
# directory published to a child process.  Possessing or forging this value can
# only enable the stricter reviewed-digest checks; it never makes an artifact
# valid by itself.
_GROTH16_REVIEWED_ARTIFACTS_MARKER_ENV: Final = (
    "IPFS_ACCEL_PROOF_REUSE_REVIEWED_GROTH16_ARTIFACTS"
)

# Typed capability reason codes (closed vocabulary).
REASON_AVAILABLE: Final = "available"
REASON_AUTO_INSTALL_DISABLED: Final = "auto_install_disabled"
REASON_OFFLINE_INDEX: Final = "offline_index"
REASON_RESOLVER_FAILURE: Final = "resolver_failure"
REASON_INCOMPATIBLE_VERSION: Final = "incompatible_version"
REASON_READ_ONLY_ENVIRONMENT: Final = "read_only_environment"
REASON_DEPENDENCY_MISSING: Final = "dependency_missing"
REASON_NOT_ALLOWLISTED: Final = "not_allowlisted"
REASON_INSTALL_FAILED: Final = "install_failed"
REASON_NLTK_DATA_MISSING: Final = "nltk_data_missing"
REASON_NLTK_DOWNLOAD_DISABLED: Final = "nltk_download_disabled"
REASON_GROTH16_BUILD_DISABLED: Final = "groth16_build_disabled"
REASON_GROTH16_SOURCE_INVALID: Final = "groth16_source_invalid"
REASON_GROTH16_TOOLCHAIN_MISSING: Final = "groth16_toolchain_missing"
REASON_GROTH16_BINARY_MISSING: Final = "groth16_binary_missing"
REASON_GROTH16_CAPABILITY_MISMATCH: Final = "groth16_capability_mismatch"
REASON_GROTH16_ENDPOINT_UNCONFIGURED: Final = "groth16_endpoint_unconfigured"
REASON_GROTH16_KEYS_MISSING: Final = "groth16_keys_missing"
REASON_GROTH16_CIRCUIT_UNCONFIGURED: Final = "groth16_circuit_unconfigured"
REASON_LOCK_TIMEOUT: Final = "provision_lock_timeout"
REASON_PROVISION_TIMEOUT: Final = "provision_timeout"

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"", "0", "false", "no", "off", "disabled"})

_CAPABILITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "content_addressing": MULTIFORMATS_DEPENDENCY.module_name,
        "cid": MULTIFORMATS_DEPENDENCY.module_name,
        "multiformats": MULTIFORMATS_DEPENDENCY.module_name,
        "jsonschema": JSONSCHEMA_DEPENDENCY.module_name,
        "certificate_schema": JSONSCHEMA_DEPENDENCY.module_name,
        "nltk": NLTK_DEPENDENCY.module_name,
        "nltk_python": NLTK_DEPENDENCY.module_name,
        "datasets_zk": DATASETS_VERIFIER_DEPENDENCY.module_name,
        "datasets_verifier": DATASETS_VERIFIER_DEPENDENCY.module_name,
        "zk_verifier": DATASETS_VERIFIER_DEPENDENCY.module_name,
    }
)

_NLTK_DATA_CAPABILITY_ALIASES: Final = frozenset(
    {"nltk_data", "nltk_corpora", "nltk_resources"}
)
_GROTH16_NATIVE_CAPABILITY_ALIASES: Final = frozenset(
    {"groth16", "groth16_native", "groth16_binary", "datasets_groth16"}
)
_GROTH16_ENDPOINT_CAPABILITY_ALIASES: Final = frozenset({"groth16_endpoint"})
_GROTH16_KEYS_CAPABILITY_ALIASES: Final = frozenset(
    {"groth16_keys", "groth16_artifacts", "groth16_verifying_key"}
)
_GROTH16_CIRCUIT_CAPABILITY_ALIASES: Final = frozenset(
    {"groth16_circuit", "groth16_circuit_binding"}
)

_VERSION_CONSTRAINT_RE: Final = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+)\s*(?P<spec>.*)$"
)
_SPEC_TOKEN_RE: Final = re.compile(
    r"(==|!=|<=|>=|<|>|~=)\s*([0-9]+(?:\.[0-9A-Za-z.*+!~_-]*)?)"
)

_INSTALL_THREAD_LOCK = threading.Lock()
_NLTK_DATA_THREAD_LOCK = threading.Lock()
_GROTH16_BUILD_THREAD_LOCK = threading.Lock()

_GROTH16_BUILD_RECEIPT_INTERFACE: Final = GROTH16_NATIVE_BUILD_RECEIPT_INTERFACE
_GROTH16_BUILD_RECEIPT_NAME: Final = "groth16-native-build.json"
_GROTH16_BUILD_RECEIPT_MAX_BYTES: Final = 64 * 1024
_GROTH16_BUILD_BINARY_MAX_BYTES: Final = 128 * 1024 * 1024


def _groth16_build_binary_relative() -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return f"target/release/groth16{suffix}"


def _groth16_cached_binary_relative() -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return f"bin/{_platform_binary_name()}/groth16{suffix}"


@dataclass(frozen=True, slots=True)
class ProofReuseCapabilityResolution:
    """Outcome of one capability probe or first-use installation attempt."""

    available: bool
    reason_code: str
    capability: str
    module_name: str = ""
    distribution: str = ""
    installed: bool = False
    module: Any = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    capability_kind: str = "python_distribution"
    fallback_action: str = "RUN"

    def __post_init__(self) -> None:
        if self.fallback_action not in {"RUN", "DEFERRED"}:
            raise ValueError("fallback_action must be RUN or DEFERRED")

    @property
    def action(self) -> str:
        """Tests always continue; reasons are diagnostic only."""

        return self.fallback_action

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason_code": self.reason_code,
            "capability": self.capability,
            "module_name": self.module_name,
            "distribution": self.distribution,
            "installed": self.installed,
            "capability_kind": self.capability_kind,
            "action": self.action,
            "diagnostics": dict(self.diagnostics),
        }


def resolve_capability_module_name(capability: str) -> str:
    """Map a capability alias or module name onto the allowlist key."""

    text = str(capability or "").strip()
    if not text:
        return ""
    aliased = _CAPABILITY_ALIASES.get(text) or _CAPABILITY_ALIASES.get(text.lower())
    if aliased:
        return aliased
    return text


def _parse_version_tuple(version: str) -> tuple[int | str, ...] | None:
    text = str(version or "").strip()
    if not text:
        return None
    parts: list[int | str] = []
    for token in text.replace("-", ".").split("."):
        if not token:
            continue
        if token.isdigit():
            parts.append(int(token))
            continue
        numeric = re.match(r"^(\d+)", token)
        if numeric:
            parts.append(int(numeric.group(1)))
        else:
            parts.append(token)
    return tuple(parts) or None


def _compare_versions(left: str, right: str) -> int | None:
    left_parts = _parse_version_tuple(left)
    right_parts = _parse_version_tuple(right)
    if left_parts is None or right_parts is None:
        return None
    width = max(len(left_parts), len(right_parts))
    padded_left = left_parts + (0,) * (width - len(left_parts))
    padded_right = right_parts + (0,) * (width - len(right_parts))
    for left_item, right_item in zip(padded_left, padded_right):  # noqa: B905
        if type(left_item) is not type(right_item):
            left_item = str(left_item)
            right_item = str(right_item)
        if left_item < right_item:
            return -1
        if left_item > right_item:
            return 1
    return 0


def _satisfies_spec(installed_version: str, distribution: str) -> bool | None:
    """Return whether *installed_version* satisfies *distribution* constraints.

    ``None`` means the constraint could not be evaluated (treat as compatible).
    """

    match = _VERSION_CONSTRAINT_RE.match(str(distribution or "").strip())
    if match is None:
        return None
    spec = (match.group("spec") or "").strip()
    if not spec:
        return True
    tokens = _SPEC_TOKEN_RE.findall(spec)
    if not tokens:
        return None
    for operator, expected in tokens:
        compared = _compare_versions(installed_version, expected.rstrip(".*"))
        if compared is None:
            return None
        if operator == "==" and compared != 0:
            return False
        if operator == "!=" and compared == 0:
            return False
        if operator == ">=" and compared < 0:
            return False
        if operator == "<=" and compared > 0:
            return False
        if operator == ">" and compared <= 0:
            return False
        if operator == "<" and compared >= 0:
            return False
        if operator == "~=":
            # Compatible release: >= expected and < next major of the
            # penultimate segment (PEP 440 simplified).
            if compared < 0:
                return False
            expected_parts = _parse_version_tuple(expected.rstrip(".*"))
            if expected_parts is None or len(expected_parts) < 2:
                continue
            upper = list(expected_parts[:-1])
            last = upper[-1]
            if isinstance(last, int):
                upper[-1] = last + 1
                upper_version = ".".join(str(part) for part in upper)
                upper_compared = _compare_versions(installed_version, upper_version)
                if upper_compared is not None and upper_compared >= 0:
                    return False
    return True


def _installed_distribution_version(distribution: str) -> str | None:
    match = _VERSION_CONSTRAINT_RE.match(str(distribution or "").strip())
    name = match.group("name") if match else str(distribution or "").strip()
    if not name:
        return None
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


def _classify_install_failure(
    *,
    returncode: int | None,
    output: str,
    error: BaseException | None = None,
) -> str:
    text = (output or "").lower()
    error_text = str(error or "").lower()
    combined = f"{text}\n{error_text}"

    if error is not None and isinstance(error, OSError):
        errno_value = getattr(error, "errno", None)
        if errno_value in {13, 30}:  # EACCES, EROFS
            return REASON_READ_ONLY_ENVIRONMENT
        if "read-only" in error_text or "read only" in error_text:
            return REASON_READ_ONLY_ENVIRONMENT
        if "permission denied" in error_text:
            return REASON_READ_ONLY_ENVIRONMENT

    readonly_markers = (
        "read-only file system",
        "readonly file system",
        "errno 30",
        "permission denied",
        "errno 13",
        "operation not permitted",
        "access is denied",
        "cannot write",
        "read only file system",
    )
    if any(marker in combined for marker in readonly_markers):
        return REASON_READ_ONLY_ENVIRONMENT

    offline_markers = (
        "network is unreachable",
        "name or service not known",
        "temporary failure in name resolution",
        "failed to establish a new connection",
        "connection refused",
        "connection timed out",
        "no matching distribution found",
        "could not find a version that satisfies",
        "offline mode",
        "pip is configured with locations that require tls",
        "failed to fetch",
        "max retries exceeded",
        "nodename nor servname provided",
        "getaddrinfo failed",
        "living off the land",
        "there was a problem confirming the ssl certificate",
        "proxy error",
        "http error 503",
        "http error 502",
        "http error 504",
    )
    if any(marker in combined for marker in offline_markers):
        return REASON_OFFLINE_INDEX

    incompatible_markers = (
        "incompatible",
        "requires-python",
        "does not provide the required",
        "conflicting dependencies",
        "resolutionimpossible",
        "has requirement",
        "version conflict",
        "is not a supported wheel",
    )
    if any(marker in combined for marker in incompatible_markers):
        return REASON_INCOMPATIBLE_VERSION

    if returncode not in (0, None) or error is not None:
        return REASON_RESOLVER_FAILURE
    return REASON_INSTALL_FAILED


def _load_provision_lock_backend() -> tuple[str, Any]:
    """Return the supported native non-blocking file-lock implementation."""

    try:
        import fcntl

        return "fcntl", fcntl
    except ImportError:  # pragma: no cover - Windows.
        if os.name != "nt":
            return "", None
        try:
            import msvcrt

            return "msvcrt", msvcrt
        except ImportError:  # pragma: no cover - broken Windows runtime.
            return "", None


@contextmanager
def _bounded_interprocess_provision_fence(
    lock_root: str | os.PathLike[str] | None,
    *,
    capability_key: str,
    timeout_seconds: float,
) -> Iterator[bool]:
    """Acquire a strict, bounded cross-process provisioning lock.

    Package/resource installation must not proceed unlocked when the lock path
    or a supported native lock primitive is unavailable. POSIX uses ``fcntl``;
    Windows uses a one-byte non-blocking ``msvcrt`` lock.
    """

    if lock_root:
        root = Path(lock_root)
    else:
        getuid = getattr(os, "getuid", None)
        identity = (
            str(getuid())
            if callable(getuid)
            else hashlib.sha256(str(Path.home()).encode()).hexdigest()[:16]
        )
        root = (
            Path(tempfile.gettempdir())
            / f"ipfs-accelerate-proof-reuse-provision-{identity}"
        )
    try:
        if root.is_symlink():
            yield False
            return
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        root_stat = os.lstat(root)
    except OSError:
        yield False
        return
    if not stat.S_ISDIR(root_stat.st_mode):
        yield False
        return
    if os.name != "nt":
        getuid = getattr(os, "getuid", None)
        if callable(getuid) and root_stat.st_uid != getuid():
            yield False
            return
        if root_stat.st_mode & 0o077:
            yield False
            return
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", capability_key)[:120] or "global"
    lock_path = root / f"{safe_key}.lock"
    descriptor = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise OSError("provision lock is not a regular file")
        if os.name != "nt":
            getuid = getattr(os, "getuid", None)
            if callable(getuid) and lock_stat.st_uid != getuid():
                raise OSError("provision lock has a different owner")
            if lock_stat.st_mode & 0o077:
                raise OSError("provision lock is not private")
        handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        descriptor = -1
    except OSError:
        if descriptor >= 0:
            os.close(descriptor)
        yield False
        return

    acquired = False
    lock_backend, lock_module = _load_provision_lock_backend()
    try:
        if lock_backend == "msvcrt":
            try:
                handle.seek(0, os.SEEK_END)
                if handle.tell() < 1:
                    handle.write("\0")
                    handle.flush()
                handle.seek(0)
            except OSError:
                lock_backend = ""
        if lock_backend:
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    if lock_backend == "fcntl":
                        lock_module.flock(
                            handle.fileno(),
                            lock_module.LOCK_EX | lock_module.LOCK_NB,
                        )
                    else:
                        handle.seek(0)
                        lock_module.locking(handle.fileno(), lock_module.LK_NBLCK, 1)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield acquired
    finally:
        if acquired:
            try:
                if lock_backend == "fcntl":
                    lock_module.flock(handle.fileno(), lock_module.LOCK_UN)
                elif lock_backend == "msvcrt":
                    handle.seek(0)
                    lock_module.locking(handle.fileno(), lock_module.LK_UNLCK, 1)
            except OSError:
                pass
        try:
            handle.close()
        except OSError:
            pass


def _platform_binary_name() -> str:
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
    return f"{platform.system().lower()}-{machine}"


def _reviewed_groth16_artifacts_marker(root: Path) -> str | None:
    """Bind a resolved directory to the immutable reviewed artifact manifest."""

    try:
        if root.is_symlink():
            return None
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_dir():
        return None
    payload = {
        "interface": PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE,
        "datasets_revision": DATASETS_VERIFIER_REVISION,
        "artifacts_root": str(resolved),
        "reviewed_artifacts_sha256": dict(
            sorted(DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256.items())
        ),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_regular_file(
    path: Path,
    *,
    max_bytes: int,
) -> str | None:
    loaded = _read_regular_file_bytes(path, max_bytes=max_bytes)
    return hashlib.sha256(loaded[0]).hexdigest() if loaded is not None else None


def _read_regular_file_bytes(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, os.stat_result] | None:
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            return None
        if metadata.st_size <= 0 or metadata.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            return None
        return b"".join(chunks), metadata
    except OSError:
        return None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _configured_regular_file(value: str) -> Path | None:
    try:
        requested = Path(value).expanduser()
        if requested.is_symlink():
            return None
        path = requested.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    try:
        if not path.is_file() or path.is_symlink():
            return None
    except OSError:
        return None
    return path


class ProofReuseLazyDependencyInstaller:
    """Bounded, allowlisted, fenced first-use installer for proof-reuse deps.

    Implements ``ProofReuseLazyDependencyInstaller@1``.  Compatible with the
    plugin's installer injection point (``install(dependency) -> bool``).
    """

    interface: str = PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
        native_timeout_seconds: float = 900.0,
        lock_timeout_seconds: float = 30.0,
        environ: Mapping[str, str] | None = None,
        importer: Callable[[str], Any] | None = None,
        lock_root: str | os.PathLike[str] | None = None,
        pip_installer: AllowlistedPipInstaller | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1 <= float(timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must be between 1 and 600")
        if (
            isinstance(native_timeout_seconds, bool)
            or not isinstance(native_timeout_seconds, (int, float))
            or not 30 <= float(native_timeout_seconds) <= 1800
        ):
            raise ValueError("native_timeout_seconds must be between 30 and 1800")
        if (
            isinstance(lock_timeout_seconds, bool)
            or not isinstance(lock_timeout_seconds, (int, float))
            or not 0.1 <= float(lock_timeout_seconds) <= 120
        ):
            raise ValueError("lock_timeout_seconds must be between 0.1 and 120")
        self._runner = runner
        self._process_runner = runner or subprocess.run
        self._timeout_seconds = float(timeout_seconds)
        self._native_timeout_seconds = float(native_timeout_seconds)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._uses_process_environment = environ is None
        self._environ = dict(os.environ if environ is None else environ)
        self._importer = importer or importlib.import_module
        configured_cache_root = self._environ.get(PROOF_REUSE_CACHE_DIR_ENV, "").strip()
        self._lock_root = (
            str(lock_root)
            if lock_root is not None
            else (
                str(Path(configured_cache_root).expanduser() / "provision-locks")
                if configured_cache_root
                else None
            )
        )
        configured_provision_root = self._environ.get(
            PROOF_REUSE_PROVISION_DIR_ENV, ""
        ).strip()
        if configured_provision_root:
            self._provision_root = Path(configured_provision_root).expanduser()
        elif lock_root is not None:
            self._provision_root = Path(lock_root).expanduser() / "receipts"
        elif configured_cache_root:
            self._provision_root = (
                Path(configured_cache_root).expanduser() / "provisioning"
            )
        else:
            home = self._environ.get("HOME", "").strip()
            home_path = Path(home).expanduser() if home else Path.home()
            self._provision_root = (
                home_path
                / ".cache"
                / "ipfs_accelerate_py"
                / "proof_reuse"
                / "provisioning"
            )
        self._pip = pip_installer or AllowlistedPipInstaller(
            runner=runner,
            timeout_seconds=self._timeout_seconds,
            environ=self._environ,
            provision_root=self._provision_root,
        )
        self._outcomes: dict[str, ProofReuseCapabilityResolution] = {}
        self._provision_outcomes: dict[str, ProofReuseCapabilityResolution] = {}
        self._reviewed_groth16_artifacts_root: Path | None = None
        self._groth16_backend_source_kind = "unavailable"
        self._validated_native_binary_identities: dict[
            Path, tuple[str, int]
        ] = {}
        self._lock = threading.Lock()

    def allowlist(self) -> Mapping[str, ProofReuseDependency]:
        return PROOF_REUSE_DEPENDENCY_ALLOWLIST

    def dependency_plan(self) -> dict[str, Any]:
        return proof_reuse_dependency_plan(self._environ)

    def install_permitted(self) -> bool:
        return proof_reuse_install_permitted(self._environ)

    def prepare_dependency(self, dependency: ProofReuseDependency) -> bool:
        """Prepare exact verifier provenance before its first Python import."""

        if dependency.module_name != DATASETS_VERIFIER_DEPENDENCY.module_name:
            return True
        if sys.version_info < (3, 12):
            return False
        prepare = getattr(self._pip, "prepare_dependency", None)
        if not callable(prepare):
            return False
        activate_cached = getattr(self._pip, "activate_cached_datasets_verifier", None)
        if callable(activate_cached) and activate_cached():
            return True
        if not self.install_permitted():
            return False
        with _INSTALL_THREAD_LOCK, _bounded_interprocess_provision_fence(
            self._lock_root,
            capability_key=dependency.module_name,
            timeout_seconds=self._lock_timeout_seconds,
        ) as acquired:
            if not acquired:
                return False
            return bool(prepare(dependency, allow_install=True))

    def validate_module_provenance(
        self, dependency: ProofReuseDependency, module: Any
    ) -> bool:
        validator = getattr(self._pip, "validate_module_provenance", None)
        return bool(callable(validator) and validator(dependency, module))

    def validate_authority_module_provenance(self, module: Any) -> bool:
        """Validate one datasets ZKP module against the exact snapshot root."""

        validator = getattr(
            self._pip, "validate_datasets_authority_module", None
        )
        return bool(callable(validator) and validator(module))

    def _lookup_dependency(
        self,
        module_name: str,
        dependency: ProofReuseDependency | None = None,
    ) -> ProofReuseDependency | None:
        if dependency is not None:
            allowed = PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(
                getattr(dependency, "module_name", "")
            )
            if allowed == dependency:
                return dependency
            return None
        return PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(module_name)

    def _import_module(self, module_name: str) -> Any | None:
        try:
            return self._importer(module_name)
        except ModuleNotFoundError:
            return None
        except Exception:
            return None

    def _check_present(
        self,
        dependency: ProofReuseDependency,
        *,
        capability: str,
    ) -> ProofReuseCapabilityResolution | None:
        module = self._import_module(dependency.module_name)
        if module is None:
            return None
        missing_symbols = [
            symbol
            for symbol in dependency.required_symbols
            if getattr(module, symbol, None) is None
        ]
        if missing_symbols:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_INCOMPATIBLE_VERSION,
                capability=capability,
                module_name=dependency.module_name,
                distribution=dependency.distribution,
                diagnostics={"missing_symbols": missing_symbols},
            )
        if not self.validate_module_provenance(dependency, module):
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_INCOMPATIBLE_VERSION,
                capability=capability,
                module_name=dependency.module_name,
                distribution=dependency.distribution,
                diagnostics={
                    "module_provenance": "unreviewed",
                    "fallback_action": "RUN",
                },
            )
        version = _installed_distribution_version(dependency.distribution)
        if version is not None:
            compatible = _satisfies_spec(version, dependency.distribution)
            if compatible is False:
                return ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_INCOMPATIBLE_VERSION,
                    capability=capability,
                    module_name=dependency.module_name,
                    distribution=dependency.distribution,
                    diagnostics={"installed_version": version},
                )
        return ProofReuseCapabilityResolution(
            available=True,
            reason_code=REASON_AVAILABLE,
            capability=capability,
            module_name=dependency.module_name,
            distribution=dependency.distribution,
            module=module,
            diagnostics={"installed_version": version or ""},
        )

    def ensure_capability(
        self,
        capability: str,
        *,
        dependency: ProofReuseDependency | None = None,
        force_install: bool = False,
        consent: bool | None = None,
    ) -> ProofReuseCapabilityResolution:
        """Ensure one allowlisted proof-reuse capability is importable.

        Never raises for optional-boundary faults.  Returns a typed reason so
        callers always continue with RUN semantics.
        """

        requested_capability = str(capability or "").strip().lower()
        if requested_capability in _NLTK_DATA_CAPABILITY_ALIASES:
            return self.ensure_nltk_data(
                consent=consent,
                force=force_install,
            )
        if requested_capability in _GROTH16_NATIVE_CAPABILITY_ALIASES:
            return self.ensure_groth16_native_backend(
                consent=consent,
                force_build=force_install,
            )
        if requested_capability in _GROTH16_ENDPOINT_CAPABILITY_ALIASES:
            return self.inspect_groth16_endpoint()
        if requested_capability in _GROTH16_KEYS_CAPABILITY_ALIASES:
            return self.inspect_groth16_keys()
        if requested_capability in _GROTH16_CIRCUIT_CAPABILITY_ALIASES:
            return self.inspect_groth16_circuit()

        module_name = resolve_capability_module_name(capability)
        capability_key = module_name or str(capability or "unknown")
        with self._lock:
            cached = self._outcomes.get(capability_key)
            if cached is not None and not force_install:
                return cached

        selected = self._lookup_dependency(module_name, dependency)
        if selected is None:
            resolution = ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_NOT_ALLOWLISTED,
                capability=str(capability or ""),
                module_name=module_name,
                diagnostics={"action": "run"},
            )
            with self._lock:
                self._outcomes[capability_key] = resolution
            return resolution

        is_datasets_verifier = (
            selected.module_name == DATASETS_VERIFIER_DEPENDENCY.module_name
        )
        if is_datasets_verifier and sys.version_info < (3, 12):
            resolution = ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_INCOMPATIBLE_VERSION,
                capability=capability_key,
                module_name=selected.module_name,
                distribution=selected.distribution,
                diagnostics={
                    "requires_python": ">=3.12",
                    "install_process_started": False,
                    "fallback_action": "RUN",
                },
            )
            with self._lock:
                self._outcomes[capability_key] = resolution
            return resolution

        if is_datasets_verifier and selected.module_name in sys.modules:
            present = self._check_present(selected, capability=capability_key)
            if present is not None:
                # A preloaded exact source is usable. An arbitrary preloaded
                # canonical namespace cannot be safely replaced in-process.
                with self._lock:
                    self._outcomes[capability_key] = present
                return present

        if is_datasets_verifier and self.install_permitted():
            activate_cached = getattr(
                self._pip, "activate_cached_datasets_verifier", None
            )
            prepared = bool(callable(activate_cached) and activate_cached())
            # Do not import an arbitrary same-version package before the exact
            # private overlay is ready; importing it would populate canonical
            # parent modules and make safe replacement ambiguous.
            present = (
                self._check_present(selected, capability=capability_key)
                if prepared
                else None
            )
        else:
            activate_cached = getattr(self._pip, "activate_cached_dependency", None)
            if callable(activate_cached):
                try:
                    activate_cached(selected)
                except Exception:
                    pass
            present = self._check_present(selected, capability=capability_key)
        if present is not None and present.available and not force_install:
            with self._lock:
                self._outcomes[capability_key] = present
            return present
        if present is not None and not present.available:
            # Present but incompatible — do not auto-upgrade outside policy.
            if not self.install_permitted():
                with self._lock:
                    self._outcomes[capability_key] = present
                return present

        if not self.install_permitted():
            resolution = ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_AUTO_INSTALL_DISABLED,
                capability=capability_key,
                module_name=selected.module_name,
                distribution=selected.distribution,
                diagnostics={
                    "proof_reuse_auto_install": automatic_install_enabled(
                        self._environ
                    ),
                    "package_auto_install": package_auto_install_policy_permits(
                        self._environ
                    ),
                    "unavailable_reason": selected.unavailable_reason,
                },
            )
            with self._lock:
                self._outcomes[capability_key] = resolution
            return resolution

        with _INSTALL_THREAD_LOCK, _bounded_interprocess_provision_fence(
            self._lock_root,
            capability_key=selected.module_name,
            timeout_seconds=self._lock_timeout_seconds,
        ) as acquired:
            if not acquired:
                resolution = ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_LOCK_TIMEOUT,
                    capability=capability_key,
                    module_name=selected.module_name,
                    distribution=selected.distribution,
                    diagnostics={"install_process_started": False},
                )
                with self._lock:
                    self._outcomes[capability_key] = resolution
                return resolution
            # Re-check under the fence — another process may have installed.
            if is_datasets_verifier:
                activate_cached = getattr(
                    self._pip, "activate_cached_datasets_verifier", None
                )
                prepared = bool(callable(activate_cached) and activate_cached())
                present = (
                    self._check_present(selected, capability=capability_key)
                    if prepared
                    else None
                )
            else:
                activate_cached = getattr(
                    self._pip, "activate_cached_dependency", None
                )
                if callable(activate_cached):
                    try:
                        activate_cached(selected)
                    except Exception:
                        pass
                present = self._check_present(selected, capability=capability_key)
            if present is not None and present.available and not force_install:
                with self._lock:
                    self._outcomes[capability_key] = present
                return present

            install_error: BaseException | None = None
            install_output = ""
            returncode: int | None = None
            succeeded = False
            try:
                if self._runner is not None and not is_datasets_verifier:
                    # Instrumented path: capture output for typed classification.
                    succeeded, returncode, install_output, install_error = (
                        self._run_instrumented_install(selected)
                    )
                else:
                    succeeded = self._pip.install(selected) is True
            except Exception as exc:  # noqa: BLE001 - optional boundary.
                install_error = exc
                succeeded = False

            if succeeded:
                importlib.invalidate_caches()
                present = self._check_present(selected, capability=capability_key)
                if present is not None and present.available:
                    resolution = ProofReuseCapabilityResolution(
                        available=True,
                        reason_code=REASON_AVAILABLE,
                        capability=capability_key,
                        module_name=selected.module_name,
                        distribution=selected.distribution,
                        installed=True,
                        module=present.module,
                        diagnostics=dict(present.diagnostics),
                    )
                    with self._lock:
                        self._outcomes[capability_key] = resolution
                    return resolution
                reason = REASON_INCOMPATIBLE_VERSION
            else:
                reason = _classify_install_failure(
                    returncode=returncode,
                    output=install_output,
                    error=install_error,
                )
                if reason == REASON_INSTALL_FAILED:
                    reason = REASON_DEPENDENCY_MISSING

            resolution = ProofReuseCapabilityResolution(
                available=False,
                reason_code=reason,
                capability=capability_key,
                module_name=selected.module_name,
                distribution=selected.distribution,
                diagnostics={
                    "unavailable_reason": selected.unavailable_reason,
                    "returncode": returncode,
                    "output": (install_output or "")[:500],
                    "error": str(install_error or "")[:300],
                },
            )
            with self._lock:
                self._outcomes[capability_key] = resolution
            return resolution

    def _run_instrumented_install(
        self,
        dependency: ProofReuseDependency,
    ) -> tuple[bool, int | None, str, BaseException | None]:
        """Run pip via the injected runner and capture classification inputs."""

        allowed = PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(dependency.module_name)
        if allowed != dependency:
            return False, None, "", None
        private_install = getattr(self._pip, "install_with_diagnostics", None)
        if not callable(private_install):
            # A custom installer without an explicit private-target contract
            # must fail closed rather than mutate the interpreter environment.
            return False, None, "private target installer unavailable", None
        try:
            result = private_install(dependency)
        except Exception as exc:  # noqa: BLE001
            return False, None, "", exc
        if (
            not isinstance(result, tuple)
            or len(result) != 4
            or not isinstance(result[0], bool)
            or (
                result[1] is not None
                and (
                    isinstance(result[1], bool)
                    or not isinstance(result[1], int)
                )
            )
            or not isinstance(result[2], str)
            or (
                result[3] is not None
                and not isinstance(result[3], BaseException)
            )
        ):
            return False, None, "invalid private target installer result", None
        return result

    def _remember_provision(
        self,
        key: str,
        resolution: ProofReuseCapabilityResolution,
    ) -> ProofReuseCapabilityResolution:
        with self._lock:
            self._provision_outcomes[key] = resolution
        return resolution

    def _remember_provision_attempt(
        self,
        key: str,
        resolution: ProofReuseCapabilityResolution,
    ) -> ProofReuseCapabilityResolution:
        diagnostics = dict(resolution.diagnostics)
        diagnostics["provision_attempted"] = True
        return self._remember_provision(
            key, replace(resolution, diagnostics=diagnostics)
        )

    def _cached_provision_attempt(
        self, key: str
    ) -> ProofReuseCapabilityResolution | None:
        with self._lock:
            resolution = self._provision_outcomes.get(key)
        if (
            resolution is None
            or resolution.available
            or not resolution.diagnostics.get("provision_attempted")
        ):
            return None
        return resolution

    def _nltk_download_root(self) -> Path | None:
        configured = str(self._environ.get(PROOF_REUSE_NLTK_DATA_DIR_ENV, "")).strip()
        if not configured:
            nltk_data = str(self._environ.get("NLTK_DATA", "")).strip()
            if nltk_data:
                configured = nltk_data.split(os.pathsep)[0].strip()
        try:
            requested = (
                Path(configured).expanduser()
                if configured
                else Path(self._environ.get("HOME", str(Path.home()))).expanduser()
                / "nltk_data"
            )
            if requested.is_symlink():
                return None
            root = requested.resolve(strict=False)
        except (OSError, RuntimeError):
            return None
        # A downloader target must be a dedicated directory, never a broad
        # filesystem root or an existing non-directory/symlink.
        if root == Path(root.anchor):
            return None
        try:
            if root.exists() and not root.is_dir():
                return None
        except OSError:
            return None
        return root

    @staticmethod
    def _missing_nltk_resources(
        nltk_module: Any,
        resources: tuple[str, ...],
        download_root: Path,
    ) -> list[str]:
        data = getattr(nltk_module, "data", None)
        find = getattr(data, "find", None)
        if not callable(find):
            return list(resources)
        data_paths = getattr(data, "path", None)
        if isinstance(data_paths, list) and str(download_root) not in data_paths:
            data_paths.insert(0, str(download_root))
        missing: list[str] = []
        for package_id in resources:
            resource = NLTK_DATA_RESOURCE_ALLOWLIST[package_id]
            found = False
            for find_path in resource.find_paths:
                try:
                    try:
                        find(find_path, paths=[str(download_root)])
                    except TypeError:
                        find(find_path)
                    found = True
                    break
                except Exception:
                    continue
            if not found:
                missing.append(package_id)
        return missing

    def ensure_nltk_data(
        self,
        resources: tuple[str, ...] = DEFAULT_NLTK_DATA_RESOURCES,
        *,
        consent: bool | None = None,
        force: bool = False,
    ) -> ProofReuseCapabilityResolution:
        """Provision only allowlisted NLTK data on an explicit first-use call.

        The Python distribution is handled by the pip allowlist.  Corpus/model
        data is a separate, subprocess-bounded network operation and requires
        the NLTK-specific opt-in (or ``consent=True``) in addition to the two
        package installation policies.
        """

        if isinstance(resources, (str, bytes)):
            selected: tuple[str, ...] = ()
        else:
            try:
                selected = tuple(str(item).strip() for item in resources)
            except TypeError:
                selected = ()
        if (
            not selected
            or any(
                not item or item not in NLTK_DATA_RESOURCE_ALLOWLIST
                for item in selected
            )
            or len(set(selected)) != len(selected)
        ):
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_NOT_ALLOWLISTED,
                capability="nltk_data",
                module_name=NLTK_DEPENDENCY.module_name,
                distribution=NLTK_DEPENDENCY.distribution,
                capability_kind="network_data_download",
                diagnostics={"requested_resource_count": len(selected)},
            )

        python_resolution = self.ensure_capability(NLTK_DEPENDENCY.module_name)
        if not python_resolution.available:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=python_resolution.reason_code,
                capability="nltk_data",
                module_name=NLTK_DEPENDENCY.module_name,
                distribution=NLTK_DEPENDENCY.distribution,
                capability_kind="network_data_download",
                diagnostics={"python_dependency": python_resolution.to_dict()},
            )
        nltk_module = python_resolution.module or self._import_module(
            NLTK_DEPENDENCY.module_name
        )
        download_root = self._nltk_download_root()
        if nltk_module is None or download_root is None:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_READ_ONLY_ENVIRONMENT,
                capability="nltk_data",
                module_name=NLTK_DEPENDENCY.module_name,
                distribution=NLTK_DEPENDENCY.distribution,
                capability_kind="network_data_download",
            )

        missing = self._missing_nltk_resources(nltk_module, selected, download_root)
        if force:
            missing = list(selected)
        if not missing and not force:
            return self._remember_provision(
                "nltk_data",
                ProofReuseCapabilityResolution(
                    available=True,
                    reason_code=REASON_AVAILABLE,
                    capability="nltk_data",
                    module_name=NLTK_DEPENDENCY.module_name,
                    distribution=NLTK_DEPENDENCY.distribution,
                    capability_kind="network_data_download",
                    diagnostics={
                        "resources": list(selected),
                        "downloaded": [],
                    },
                ),
            )

        root_fingerprint = hashlib.sha256(
            str(download_root).encode("utf-8")
        ).hexdigest()[:20]
        attempt_key = f"nltk_data:{root_fingerprint}:" + ",".join(sorted(selected))
        cached_attempt = self._cached_provision_attempt(attempt_key)
        if cached_attempt is not None and not force:
            return cached_attempt

        explicit_consent = (
            nltk_data_download_enabled(self._environ)
            if consent is None
            else consent is True
        )
        if not explicit_consent:
            return self._remember_provision(
                "nltk_data",
                ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_NLTK_DOWNLOAD_DISABLED,
                    capability="nltk_data",
                    module_name=NLTK_DEPENDENCY.module_name,
                    distribution=NLTK_DEPENDENCY.distribution,
                    capability_kind="network_data_download",
                    diagnostics={
                        "missing_resources": missing,
                        "consent_environment_variable": (PROOF_REUSE_NLTK_DOWNLOAD_ENV),
                    },
                ),
            )
        if not self.install_permitted():
            return self._remember_provision(
                "nltk_data",
                ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_AUTO_INSTALL_DISABLED,
                    capability="nltk_data",
                    module_name=NLTK_DEPENDENCY.module_name,
                    distribution=NLTK_DEPENDENCY.distribution,
                    capability_kind="network_data_download",
                    diagnostics={"missing_resources": missing},
                ),
            )

        with _NLTK_DATA_THREAD_LOCK, _bounded_interprocess_provision_fence(
            self._lock_root,
            capability_key=f"nltk-data-root-{root_fingerprint}",
            timeout_seconds=self._lock_timeout_seconds,
        ) as acquired:
            if not acquired:
                return self._remember_provision(
                    "nltk_data",
                    ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=REASON_LOCK_TIMEOUT,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                    ),
                )
            missing = self._missing_nltk_resources(nltk_module, selected, download_root)
            if force:
                missing = list(selected)
            if not missing and not force:
                return self._remember_provision(
                    "nltk_data",
                    ProofReuseCapabilityResolution(
                        available=True,
                        reason_code=REASON_AVAILABLE,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                        diagnostics={"resources": list(selected)},
                    ),
                )
            try:
                download_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                return self._remember_provision(
                    "nltk_data",
                    ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=REASON_READ_ONLY_ENVIRONMENT,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                    ),
                )
            command = (
                sys.executable,
                "-I",
                "-m",
                "nltk.downloader",
                "--quiet",
                "--exit-on-error",
                "--dir",
                str(download_root),
                *(("--force",) if force else ()),
                *missing,
            )
            run_environment = isolated_pip_environment(self._environ)
            existing_nltk_data = str(run_environment.get("NLTK_DATA", "")).strip()
            run_environment["NLTK_DATA"] = os.pathsep.join(
                part for part in (str(download_root), existing_nltk_data) if part
            )
            try:
                completed = self._process_runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    env=run_environment,
                )
            except subprocess.TimeoutExpired as exc:
                return self._remember_provision_attempt(
                    attempt_key,
                    ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=REASON_PROVISION_TIMEOUT,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                        diagnostics={"error": str(exc)[:300]},
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - optional boundary.
                reason = _classify_install_failure(
                    returncode=None, output="", error=exc
                )
                return self._remember_provision_attempt(
                    attempt_key,
                    ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=reason,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                        diagnostics={"error": str(exc)[:300]},
                    ),
                )
            stdout = str(getattr(completed, "stdout", "") or "")
            stderr = str(getattr(completed, "stderr", "") or "")
            returncode = int(getattr(completed, "returncode", 1))
            if returncode != 0:
                reason = _classify_install_failure(
                    returncode=returncode,
                    output=f"{stdout}\n{stderr}",
                )
                return self._remember_provision_attempt(
                    attempt_key,
                    ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=reason,
                        capability="nltk_data",
                        module_name=NLTK_DEPENDENCY.module_name,
                        distribution=NLTK_DEPENDENCY.distribution,
                        capability_kind="network_data_download",
                        diagnostics={
                            "missing_resources": missing,
                            "output": f"{stdout}\n{stderr}"[:500],
                        },
                    ),
                )
            still_missing = self._missing_nltk_resources(
                nltk_module, selected, download_root
            )
            resolution = ProofReuseCapabilityResolution(
                available=not still_missing,
                reason_code=(
                    REASON_AVAILABLE if not still_missing else REASON_NLTK_DATA_MISSING
                ),
                capability="nltk_data",
                module_name=NLTK_DEPENDENCY.module_name,
                distribution=NLTK_DEPENDENCY.distribution,
                installed=not still_missing,
                capability_kind="network_data_download",
                diagnostics={
                    "resources": list(selected),
                    "downloaded": missing,
                    "missing_resources": still_missing,
                },
            )
            return self._remember_provision_attempt(attempt_key, resolution)

    @staticmethod
    def _validated_groth16_backend_tree(backend: Path) -> Path | None:
        """Validate every executable Cargo input by exact reviewed bytes."""

        try:
            if backend.is_symlink():
                return None
            resolved = backend.resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not resolved.is_dir():
            return None
        backend = resolved
        expected_rust = {
            relative
            for relative in DATASETS_GROTH16_REVIEWED_FILES_SHA256
            if relative.startswith("src/")
        }
        try:
            actual_rust = {
                path.relative_to(backend).as_posix()
                for path in (backend / "src").rglob("*.rs")
                if path.is_file() and not path.is_symlink()
            }
        except (OSError, RuntimeError):
            return None
        if actual_rust != expected_rust:
            return None
        for relative, expected_digest in DATASETS_GROTH16_REVIEWED_FILES_SHA256.items():
            actual = _sha256_regular_file(
                backend / relative,
                max_bytes=16 * 1024 * 1024,
            )
            if actual != expected_digest:
                return None
        return backend

    @classmethod
    def _installed_datasets_groth16_backend(cls) -> Path | None:
        """Find exact wheel package data without importing ``ipfs_datasets_py``."""

        try:
            distribution = importlib.metadata.distribution("ipfs-datasets-py")
            candidate = Path(
                distribution.locate_file(
                    "ipfs_datasets_py/processors/groth16_backend"
                )
            )
        except Exception:
            return None
        return cls._validated_groth16_backend_tree(candidate)

    def _validated_groth16_backend_dir(self) -> Path | None:
        source_resolver = getattr(self._pip, "validated_local_datasets_source", None)
        if callable(source_resolver):
            try:
                source = source_resolver()
            except Exception:
                source = None
            if isinstance(source, Path):
                validated = self._validated_groth16_backend_tree(
                    source
                    / "ipfs_datasets_py"
                    / "processors"
                    / "groth16_backend"
                )
                if validated is not None:
                    self._groth16_backend_source_kind = "reviewed_local_git"
                    return validated
        installed = self._installed_datasets_groth16_backend()
        if installed is not None:
            self._groth16_backend_source_kind = "installed_distribution_package_data"
            return installed
        self._groth16_backend_source_kind = "unavailable"
        return None

    def _configured_groth16_binary(self) -> Path | None:
        for name in (DATASETS_GROTH16_BINARY_ENV, "GROTH16_BINARY"):
            value = str(self._environ.get(name, "")).strip()
            if not value:
                continue
            path = _configured_regular_file(value)
            if path is None:
                return None
            try:
                cached_path = (
                    self._provision_root / _groth16_cached_binary_relative()
                ).resolve(strict=False)
            except (OSError, RuntimeError):
                cached_path = None
            if cached_path is not None and path == cached_path:
                # Internally published cache paths must always traverse receipt
                # and digest validation; they are not operator trust anchors.
                continue
            if os.name != "nt" and not os.access(path, os.X_OK):
                return None
            return path
        return None

    @staticmethod
    def _reviewed_bundled_groth16_platforms(backend: Path) -> tuple[str, ...]:
        available: list[str] = []
        for platform_name, expected in DATASETS_GROTH16_BUNDLED_BINARIES_SHA256.items():
            candidate = backend / "bin" / platform_name / "groth16"
            digest = _sha256_regular_file(candidate, max_bytes=128 * 1024 * 1024)
            release_manifest = (
                backend / "bin" / platform_name / "release-manifest.json"
            )
            try:
                release_payload = (
                    release_manifest.read_bytes()
                    if not release_manifest.is_symlink()
                    and release_manifest.is_file()
                    and release_manifest.stat().st_size <= 64 * 1024
                    else b""
                )
            except OSError:
                release_payload = b""
            if (
                digest == expected
                and validate_groth16_release_manifest_payload(
                    release_payload,
                    platform_name=platform_name,
                    binary_sha256=expected,
                )
            ):
                available.append(platform_name)
        return tuple(sorted(available))

    @classmethod
    def _reviewed_bundled_groth16_binary(
        cls,
        backend: Path,
        *,
        required_circuit_version: int | None = None,
    ) -> Path | None:
        platform_name = _platform_binary_name()
        if platform_name not in cls._reviewed_bundled_groth16_platforms(backend):
            return None
        capabilities = DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES.get(
            platform_name, ()
        )
        if (
            required_circuit_version is not None
            and required_circuit_version not in capabilities
        ):
            return None
        candidate = backend / "bin" / platform_name / "groth16"
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            return None
        return candidate

    @classmethod
    def _groth16_platform_diagnostics(cls, backend: Path) -> dict[str, Any]:
        native_platform = _platform_binary_name()
        bundled_platforms = cls._reviewed_bundled_groth16_platforms(backend)
        foreign_platforms = tuple(
            item for item in bundled_platforms if item != native_platform
        )
        return {
            "native_platform": native_platform,
            "reviewed_bundled_platforms": list(bundled_platforms),
            "foreign_bundled_platforms": list(foreign_platforms),
            "foreign_binary_execution_attempted": False,
            "native_build_required": native_platform not in bundled_platforms,
            "reviewed_bundled_supported_circuit_versions": list(
                DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES.get(
                    native_platform, ()
                )
            ),
        }

    def _probe_groth16_binary_capabilities(
        self,
        binary: Path,
        *,
        required_circuit_version: int,
    ) -> tuple[bool, str]:
        """Run only the bounded artifact-free PTR-151 capability command."""

        environment = {
            "PATH": os.defpath,
            DATASETS_GROTH16_ARTIFACTS_ROOT_ENV: str(
                self._provision_root / ".capability-probe-no-artifacts"
            ),
        }
        for name in ("SYSTEMROOT", "WINDIR"):
            value = str(self._environ.get(name, "") or "").strip()
            if value:
                environment[name] = value
        try:
            resolved = binary.resolve(strict=True)
        except (OSError, RuntimeError):
            return False, "capability_probe_failed"
        identity = self._validated_native_binary_identities.get(resolved)
        if identity is None:
            return False, "capability_binary_identity_unvalidated"
        loaded = _read_regular_file_bytes(
            resolved, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
        )
        if loaded is None:
            return False, "capability_binary_identity_unvalidated"
        binary_bytes, metadata = loaded
        if (
            hashlib.sha256(binary_bytes).hexdigest() != identity[0]
            or metadata.st_size != identity[1]
        ):
            return False, "capability_binary_identity_mismatch"
        executable_fd = -1
        try:
            command_path = str(resolved)
            runner_kwargs: dict[str, Any] = {}
            if os.name != "nt":
                memfd_create = getattr(os, "memfd_create", None)
                if not callable(memfd_create) or not Path("/proc/self/fd").is_dir():
                    return False, "capability_fd_execution_unavailable"
                executable_fd = memfd_create(
                    "proof-reuse-reviewed-groth16",
                    getattr(os, "MFD_CLOEXEC", 0),
                )
                view = memoryview(binary_bytes)
                while view:
                    written = os.write(executable_fd, view)
                    if written <= 0:
                        return False, "capability_fd_copy_failed"
                    view = view[written:]
                os.fchmod(executable_fd, 0o500)
                command_path = f"/proc/self/fd/{executable_fd}"
                runner_kwargs["pass_fds"] = (executable_fd,)
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                completed = self._process_runner(
                    (command_path, "capabilities", "--json"),
                    cwd=str(Path(binary.anchor)),
                    check=False,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=min(5.0, self._native_timeout_seconds),
                    env=environment,
                    **runner_kwargs,
                )
                stdout_file.seek(0)
                stderr_file.seek(0)
                captured_stdout = stdout_file.read(65_537)
                captured_stderr = stderr_file.read(65_537)
        except Exception:
            return False, "capability_probe_failed"
        finally:
            if executable_fd >= 0:
                os.close(executable_fd)
        after = _read_regular_file_bytes(
            resolved, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
        )
        if (
            after is None
            or hashlib.sha256(after[0]).hexdigest() != identity[0]
            or after[1].st_size != identity[1]
        ):
            return False, "capability_binary_changed_during_probe"
        stdout_raw = getattr(completed, "stdout", b"") or captured_stdout
        stderr_raw = getattr(completed, "stderr", b"") or captured_stderr
        stdout = (
            stdout_raw.encode("utf-8")
            if isinstance(stdout_raw, str)
            else bytes(stdout_raw)
        )
        stderr = (
            stderr_raw.encode("utf-8")
            if isinstance(stderr_raw, str)
            else bytes(stderr_raw)
        )
        if (
            getattr(completed, "returncode", 1) != 0
            or len(stdout) > 65_536
            or len(stderr) > 65_536
            or stderr
            or not validate_groth16_capability_payload(
                stdout,
                required_circuit_version=required_circuit_version,
            )
        ):
            return False, "capability_payload_mismatch"
        return True, "available"

    @staticmethod
    def _reviewed_groth16_source_fingerprint() -> str:
        return DATASETS_GROTH16_REVIEWED_SOURCE_FINGERPRINT

    @contextmanager
    def _reviewed_groth16_build_snapshot(self) -> Iterator[Path | None]:
        """Materialize only reviewed commit blobs into a private build tree."""

        blob_reader = getattr(self._pip, "reviewed_datasets_blobs", None)
        backend_prefix = "ipfs_datasets_py/processors/groth16_backend/"
        requested = {
            f"{backend_prefix}{relative}": digest
            for relative, digest in DATASETS_GROTH16_REVIEWED_FILES_SHA256.items()
        }
        if callable(blob_reader):
            try:
                blobs = blob_reader(requested)
            except Exception:
                blobs = None
        else:
            blobs = None
        if not isinstance(blobs, Mapping) or set(blobs) != set(requested):
            # Standalone wheels package the exact locked Cargo source.  Read it
            # only after the complete source closure has passed byte checks;
            # no import or distribution-version label is trusted.
            installed_backend = self._installed_datasets_groth16_backend()
            if installed_backend is not None:
                installed_blobs: dict[str, bytes] = {}
                for repository_relative, expected_digest in requested.items():
                    relative = repository_relative[len(backend_prefix) :]
                    try:
                        payload = (installed_backend / relative).read_bytes()
                    except OSError:
                        installed_blobs = {}
                        break
                    if hashlib.sha256(payload).hexdigest() != expected_digest:
                        installed_blobs = {}
                        break
                    installed_blobs[repository_relative] = payload
                blobs = installed_blobs
        if not isinstance(blobs, Mapping) or set(blobs) != set(requested):
            yield None
            return
        receipt_path, root_status = self._groth16_receipt_path(create=True)
        if receipt_path is None or root_status != "ready":
            yield None
            return
        try:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix="groth16-reviewed-build-",
                dir=receipt_path.parent,
            )
        except OSError:
            yield None
            return
        with temporary_directory as temporary_root:
            try:
                root = Path(temporary_root)
                if os.name != "nt":
                    root.chmod(0o700)
                backend = root / "backend"
                for repository_relative, expected_digest in requested.items():
                    if not repository_relative.startswith(backend_prefix):
                        yield None
                        return
                    relative = repository_relative[len(backend_prefix) :]
                    payload = blobs.get(repository_relative)
                    if not isinstance(payload, bytes):
                        yield None
                        return
                    destination = backend / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                    if (
                        _sha256_regular_file(destination, max_bytes=16 * 1024 * 1024)
                        != expected_digest
                    ):
                        yield None
                        return
                actual_files = {
                    path.relative_to(backend).as_posix()
                    for path in backend.rglob("*")
                    if path.is_file()
                }
                if actual_files != set(DATASETS_GROTH16_REVIEWED_FILES_SHA256):
                    yield None
                    return
            except OSError:
                yield None
                return
            yield backend

    def _isolated_cargo_environment(
        self, build_root: Path, target_root: Path
    ) -> dict[str, str] | None:
        """Create a minimal Cargo environment with no inherited wrappers/config."""

        cargo_home = build_root / "cargo-home"
        isolated_home = build_root / "home"
        try:
            cargo_home.mkdir(mode=0o700)
            isolated_home.mkdir(mode=0o700)
            target_root.mkdir(mode=0o700)
        except OSError:
            return None
        allowed_environment = (
            "PATH",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "ALL_PROXY",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "CARGO_HTTP_CAINFO",
            "RUSTUP_HOME",
            "RUSTUP_TOOLCHAIN",
            "SYSTEMROOT",
            "WINDIR",
            "TMPDIR",
            "TEMP",
            "TMP",
        )
        environment = {
            name: str(self._environ[name])
            for name in allowed_environment
            if str(self._environ.get(name, "")).strip()
        }
        if "RUSTUP_HOME" not in environment:
            original_home = str(self._environ.get("HOME", "")).strip()
            if original_home:
                try:
                    requested_rustup_home = Path(original_home).expanduser() / ".rustup"
                    rustup_home = requested_rustup_home.resolve(strict=True)
                    rustup_stat = rustup_home.stat()
                    rustup_home_ok = stat.S_ISDIR(rustup_stat.st_mode)
                    if os.name != "nt":
                        getuid = getattr(os, "getuid", None)
                        getgid = getattr(os, "getgid", None)
                        rustup_home_ok = rustup_home_ok and not (
                            rustup_stat.st_mode & 0o002
                        )
                        if rustup_stat.st_mode & 0o020:
                            rustup_home_ok = (
                                rustup_home_ok
                                and callable(getgid)
                                and rustup_stat.st_gid == getgid()
                            )
                        if callable(getuid):
                            rustup_home_ok = (
                                rustup_home_ok and rustup_stat.st_uid == getuid()
                            )
                    if rustup_home_ok:
                        environment["RUSTUP_HOME"] = str(rustup_home)
                except (OSError, RuntimeError):
                    pass
        environment.update(
            {
                "HOME": str(isolated_home),
                "CARGO_HOME": str(cargo_home),
                "CARGO_TARGET_DIR": str(target_root),
                "CARGO_TERM_COLOR": "never",
                "CARGO_NET_GIT_FETCH_WITH_CLI": "false",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        return environment

    def _groth16_receipt_path(self, *, create: bool) -> tuple[Path | None, str]:
        """Return an owner-private receipt path without following a leaf link."""

        root = self._provision_root
        try:
            if root.is_symlink():
                return None, "insecure_receipt_directory"
            if create:
                root.mkdir(mode=0o700, parents=True, exist_ok=True)
            elif not root.exists():
                return None, "absent"
            root_stat = root.stat()
        except OSError:
            return None, "receipt_directory_unavailable"
        if not stat.S_ISDIR(root_stat.st_mode):
            return None, "insecure_receipt_directory"
        if os.name != "nt":
            getuid = getattr(os, "getuid", None)
            if callable(getuid) and root_stat.st_uid != getuid():
                return None, "insecure_receipt_directory"
            if root_stat.st_mode & 0o077:
                return None, "insecure_receipt_directory"
        return root / _GROTH16_BUILD_RECEIPT_NAME, "ready"

    @staticmethod
    def _read_private_receipt(path: Path) -> tuple[dict[str, Any] | None, str]:
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return None, "absent"
        except OSError:
            return None, "invalid_receipt"
        try:
            receipt_stat = os.fstat(descriptor)
            if not stat.S_ISREG(receipt_stat.st_mode):
                return None, "invalid_receipt"
            if receipt_stat.st_size > _GROTH16_BUILD_RECEIPT_MAX_BYTES:
                return None, "invalid_receipt"
            if os.name != "nt":
                getuid = getattr(os, "getuid", None)
                if callable(getuid) and receipt_stat.st_uid != getuid():
                    return None, "invalid_receipt"
                if receipt_stat.st_mode & 0o077:
                    return None, "invalid_receipt"
            payload_bytes = os.read(descriptor, _GROTH16_BUILD_RECEIPT_MAX_BYTES + 1)
        except OSError:
            return None, "invalid_receipt"
        finally:
            os.close(descriptor)
        if len(payload_bytes) > _GROTH16_BUILD_RECEIPT_MAX_BYTES:
            return None, "invalid_receipt"
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, "invalid_receipt"
        if not isinstance(payload, dict):
            return None, "invalid_receipt"
        return payload, "loaded"

    def _validated_receipted_groth16_binary(
        self,
        backend: Path,
        *,
        required_circuit_version: int | None = None,
    ) -> tuple[Path | None, str]:
        receipt_path, root_status = self._groth16_receipt_path(create=False)
        if receipt_path is None:
            return None, root_status
        payload, receipt_status = self._read_private_receipt(receipt_path)
        if payload is None:
            return None, receipt_status
        required_keys = {
            "interface",
            "reviewed_datasets_revision",
            "reviewed_source_fingerprint",
            "native_platform",
            "binary_relative_path",
            "binary_sha256",
            "binary_size",
            "cargo_locked",
            "trusted_setup",
            "supported_circuit_versions",
            "test_pass_circuit_version",
            "test_pass_circuit_identity_sha256",
            "test_pass_circuit_cid",
            "test_pass_provider_source_sha256",
            "locked_source_identity",
            "capability_payload_sha256",
        }
        if set(payload) != required_keys:
            return None, "invalid_receipt"
        if payload.get("interface") != _GROTH16_BUILD_RECEIPT_INTERFACE:
            return None, "invalid_receipt"
        if payload.get("reviewed_datasets_revision") != DATASETS_VERIFIER_REVISION:
            return None, "source_mismatch"
        if (
            payload.get("reviewed_source_fingerprint")
            != self._reviewed_groth16_source_fingerprint()
        ):
            return None, "source_mismatch"
        if payload.get("native_platform") != _platform_binary_name():
            return None, "platform_mismatch"
        expected_relative = _groth16_cached_binary_relative()
        if payload.get("binary_relative_path") != expected_relative:
            return None, "invalid_receipt"
        if payload.get("cargo_locked") is not True:
            return None, "invalid_receipt"
        if payload.get("trusted_setup") is not False:
            return None, "invalid_receipt"
        versions = payload.get("supported_circuit_versions")
        if versions != list(TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS):
            return None, "capability_mismatch"
        if (
            required_circuit_version is not None
            and required_circuit_version not in versions
        ):
            return None, "capability_mismatch"
        if payload.get("test_pass_circuit_version") != TEST_PASS_GROTH16_CIRCUIT_VERSION:
            return None, "circuit_identity_mismatch"
        if (
            payload.get("test_pass_circuit_identity_sha256")
            != TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256
        ):
            return None, "circuit_identity_mismatch"
        if payload.get("test_pass_circuit_cid") != TEST_PASS_GROTH16_CIRCUIT_CID:
            return None, "circuit_identity_mismatch"
        if (
            payload.get("test_pass_provider_source_sha256")
            != TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
        ):
            return None, "source_mismatch"
        if payload.get("locked_source_identity") != DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY:
            return None, "source_mismatch"
        if payload.get("capability_payload_sha256") not in set(
            DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256.values()
        ):
            return None, "capability_mismatch"
        expected_digest = payload.get("binary_sha256")
        if (
            not isinstance(expected_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
        ):
            return None, "invalid_receipt"
        binary = receipt_path.parent / expected_relative
        actual_digest = _sha256_regular_file(
            binary, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
        )
        if actual_digest is None:
            return None, "binary_missing"
        if actual_digest != expected_digest:
            return None, "binary_digest_mismatch"
        expected_size = payload.get("binary_size")
        try:
            actual_size = binary.stat().st_size
        except OSError:
            return None, "binary_missing"
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or actual_size != expected_size
        ):
            return None, "binary_size_mismatch"
        if os.name != "nt" and not os.access(binary, os.X_OK):
            return None, "binary_not_executable"
        try:
            self._validated_native_binary_identities[binary.resolve(strict=True)] = (
                expected_digest,
                expected_size,
            )
        except (OSError, RuntimeError):
            return None, "binary_missing"
        return binary, "available"

    def _write_groth16_build_receipt(
        self, backend: Path, binary: Path
    ) -> tuple[str, Path | None]:
        binary_digest = _sha256_regular_file(
            binary, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
        )
        if binary_digest is None:
            return "binary_missing", None
        receipt_path, root_status = self._groth16_receipt_path(create=True)
        if receipt_path is None:
            return root_status, None
        cached_binary = receipt_path.parent / _groth16_cached_binary_relative()
        binary_temporary_name = ""
        binary_descriptor = -1
        try:
            cached_binary.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.name != "nt":
                cached_binary.parent.chmod(0o700)
            binary_descriptor, binary_temporary_name = tempfile.mkstemp(
                prefix=".groth16-",
                suffix=".tmp",
                dir=cached_binary.parent,
            )
            with binary.open("rb") as source_handle, os.fdopen(
                binary_descriptor, "wb"
            ) as target_handle:
                binary_descriptor = -1
                shutil.copyfileobj(source_handle, target_handle, 1024 * 1024)
                target_handle.flush()
                os.fsync(target_handle.fileno())
            if os.name != "nt":
                Path(binary_temporary_name).chmod(0o700)
            copied_digest = _sha256_regular_file(
                Path(binary_temporary_name),
                max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES,
            )
            if copied_digest != binary_digest:
                return "binary_copy_mismatch", None
            os.replace(binary_temporary_name, cached_binary)
            binary_temporary_name = ""
        except OSError:
            return "binary_copy_failed", None
        finally:
            if binary_descriptor >= 0:
                os.close(binary_descriptor)
            if binary_temporary_name:
                try:
                    Path(binary_temporary_name).unlink()
                except OSError:
                    pass
        payload = {
            "interface": _GROTH16_BUILD_RECEIPT_INTERFACE,
            "reviewed_datasets_revision": DATASETS_VERIFIER_REVISION,
            "reviewed_source_fingerprint": (
                self._reviewed_groth16_source_fingerprint()
            ),
            "native_platform": _platform_binary_name(),
            "binary_relative_path": _groth16_cached_binary_relative(),
            "binary_sha256": binary_digest,
            "binary_size": binary.stat().st_size,
            "cargo_locked": True,
            "trusted_setup": False,
            "supported_circuit_versions": list(
                TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
            ),
            "test_pass_circuit_version": TEST_PASS_GROTH16_CIRCUIT_VERSION,
            "test_pass_circuit_identity_sha256": (
                TEST_PASS_GROTH16_CIRCUIT_IDENTITY_SHA256
            ),
            "test_pass_circuit_cid": TEST_PASS_GROTH16_CIRCUIT_CID,
            "test_pass_provider_source_sha256": (
                TEST_PASS_GROTH16_PROVIDER_SOURCE_SHA256
            ),
            "locked_source_identity": DATASETS_GROTH16_LOCKED_SOURCE_IDENTITY,
            "capability_payload_sha256": next(
                iter(DATASETS_GROTH16_CAPABILITY_PAYLOADS_SHA256.values())
            ),
        }
        temporary_name = ""
        descriptor = -1
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".groth16-native-build-",
                suffix=".tmp",
                dir=receipt_path.parent,
            )
            if os.name != "nt":
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, receipt_path)
            temporary_name = ""
            if os.name != "nt":
                receipt_path.chmod(0o600)
        except OSError:
            return "write_failed", None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_name:
                try:
                    Path(temporary_name).unlink()
                except OSError:
                    pass
        validated, status = self._validated_receipted_groth16_binary(backend)
        if validated == cached_binary and status == "available":
            return "persisted", cached_binary
        return status, None

    def _publish_groth16_binary(
        self,
        binary: Path,
        *,
        backend: Path,
        receipt_path: Path | None = None,
        supported_circuit_versions: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        """Activate a validated binary only after an explicit capability call."""

        value = str(binary)
        artifacts_value = str(backend / "artifacts")
        try:
            self._reviewed_groth16_artifacts_root = (backend / "artifacts").resolve(
                strict=False
            )
        except (OSError, RuntimeError):
            self._reviewed_groth16_artifacts_root = None
        self._environ[DATASETS_GROTH16_BINARY_ENV] = value
        self._environ.setdefault(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, artifacts_value)
        if receipt_path is not None:
            self._environ[PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV] = str(receipt_path)
        reviewed_marker = _reviewed_groth16_artifacts_marker(backend / "artifacts")
        installer_artifacts_marker_published = False
        configured_artifacts = str(
            self._environ.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "")
        ).strip()
        if (
            reviewed_marker is not None
            and configured_artifacts
            and _reviewed_groth16_artifacts_marker(Path(configured_artifacts))
            == reviewed_marker
        ):
            self._environ[_GROTH16_REVIEWED_ARTIFACTS_MARKER_ENV] = reviewed_marker
            installer_artifacts_marker_published = True
        process_environment_published = False
        process_environment_preserved = False
        artifacts_environment_published = False
        process_artifacts_marker_published = False
        process_receipt_published = False
        if self._uses_process_environment:
            existing = str(os.environ.get(DATASETS_GROTH16_BINARY_ENV, "")).strip()
            if not existing:
                os.environ[DATASETS_GROTH16_BINARY_ENV] = value
                process_environment_published = True
            elif existing != value:
                process_environment_preserved = True
            if not str(os.environ.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "")).strip():
                os.environ[DATASETS_GROTH16_ARTIFACTS_ROOT_ENV] = artifacts_value
                artifacts_environment_published = True
            process_artifacts = str(
                os.environ.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "")
            ).strip()
            if (
                reviewed_marker is not None
                and process_artifacts
                and _reviewed_groth16_artifacts_marker(Path(process_artifacts))
                == reviewed_marker
            ):
                os.environ[_GROTH16_REVIEWED_ARTIFACTS_MARKER_ENV] = reviewed_marker
                process_artifacts_marker_published = True
            if receipt_path is not None and not str(
                os.environ.get(PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV, "")
            ).strip():
                os.environ[PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV] = str(receipt_path)
                process_receipt_published = True
        return {
            "activation_environment_variable": DATASETS_GROTH16_BINARY_ENV,
            "installer_environment_activated": True,
            "process_environment_published": process_environment_published,
            "existing_process_environment_preserved": (process_environment_preserved),
            "artifacts_environment_variable": (DATASETS_GROTH16_ARTIFACTS_ROOT_ENV),
            "artifacts_environment_published": artifacts_environment_published,
            "installer_reviewed_artifacts_marker_published": (
                installer_artifacts_marker_published
            ),
            "process_reviewed_artifacts_marker_published": (
                process_artifacts_marker_published
            ),
            "native_receipt_environment_variable": (
                PROOF_REUSE_GROTH16_NATIVE_RECEIPT_ENV
            ),
            "native_receipt_published": receipt_path is not None,
            "process_native_receipt_published": process_receipt_published,
            "supported_circuit_versions": list(supported_circuit_versions),
        }

    def ensure_groth16_native_backend(
        self,
        *,
        consent: bool | None = None,
        force_build: bool = False,
        required_circuit_version: int | None = TEST_PASS_GROTH16_CIRCUIT_VERSION,
    ) -> ProofReuseCapabilityResolution:
        """Ensure the reviewed datasets Rust binary, never trusted setup.

        A configured external binary is operator-supplied.  Otherwise only the
        exact reviewed datasets source may reach ``cargo build --locked``.
        Circuit selection and proving/verifying keys are inspected separately.
        """

        if (
            required_circuit_version is not None
            and (
                isinstance(required_circuit_version, bool)
                or not isinstance(required_circuit_version, int)
                or required_circuit_version <= 0
            )
        ):
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_CIRCUIT_UNCONFIGURED,
                capability="groth16_native",
                capability_kind="native_executable",
                fallback_action="DEFERRED",
                diagnostics={"required_circuit_version": None},
            )

        configured_binary = self._configured_groth16_binary()
        configured_diagnostics: dict[str, Any] = {}
        if configured_binary is not None and not force_build:
            capability_ready = required_circuit_version is None
            capability_reason = (
                "generic_operator_binary"
                if capability_ready
                else "operator_manifest_unavailable"
            )
            manifest_diagnostics: Mapping[str, Any] = {}
            if required_circuit_version is not None:
                try:
                    from .publication import validate_groth16_native_manifest_identity

                    (
                        capability_ready,
                        capability_reason,
                        manifest_diagnostics,
                    ) = validate_groth16_native_manifest_identity(
                        binary_path=configured_binary,
                        environ=self._environ,
                        required_circuit_version=required_circuit_version,
                    )
                except Exception:
                    capability_ready = False
                    capability_reason = "operator_manifest_validation_failed"
            if capability_ready:
                return ProofReuseCapabilityResolution(
                    available=True,
                    reason_code=REASON_AVAILABLE,
                    capability="groth16_native",
                    installed=False,
                    capability_kind="native_executable",
                    fallback_action="DEFERRED",
                    diagnostics={
                        "binary_source": "operator_configured",
                        "binary_path": str(configured_binary),
                        "native_platform": _platform_binary_name(),
                        "required_circuit_version": required_circuit_version,
                        "required_capability_validated": (
                            required_circuit_version is not None
                        ),
                        "keys_and_circuit_checked": False,
                        **dict(manifest_diagnostics),
                    },
                )
            configured_diagnostics = {
                "configured_binary_rejected": True,
                "configured_binary_reason": capability_reason,
                "required_circuit_version": required_circuit_version,
                **dict(manifest_diagnostics),
            }

        backend = self._validated_groth16_backend_dir()
        if backend is None:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_SOURCE_INVALID,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics=configured_diagnostics,
            )
        platform_diagnostics = self._groth16_platform_diagnostics(backend)
        platform_diagnostics["backend_source_kind"] = (
            self._groth16_backend_source_kind
        )
        platform_diagnostics.update(configured_diagnostics)
        platform_diagnostics["required_circuit_version"] = required_circuit_version
        try:
            bundled = self._reviewed_bundled_groth16_binary(
                backend,
                required_circuit_version=required_circuit_version,
            )
        except TypeError:
            # Preserve narrow test/injected installer compatibility while the
            # returned bundle still undergoes the explicit capability check.
            candidate = self._reviewed_bundled_groth16_binary(backend)
            platform_name = _platform_binary_name()
            capabilities = DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES.get(
                platform_name, ()
            )
            bundled = (
                candidate
                if required_circuit_version is None
                or required_circuit_version in capabilities
                else None
            )
        if bundled is not None and not force_build:
            bundled_loaded = _read_regular_file_bytes(
                bundled, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
            )
            if bundled_loaded is not None:
                bundled_digest = DATASETS_GROTH16_BUNDLED_BINARIES_SHA256.get(
                    _platform_binary_name(), ""
                )
                if hashlib.sha256(bundled_loaded[0]).hexdigest() == bundled_digest:
                    self._validated_native_binary_identities[
                        bundled.resolve(strict=True)
                    ] = (bundled_digest, bundled_loaded[1].st_size)
            bundled_versions = DATASETS_GROTH16_BUNDLED_BINARY_CAPABILITIES.get(
                _platform_binary_name(), ()
            )
            capability_ready, capability_status = (
                self._probe_groth16_binary_capabilities(
                    bundled,
                    required_circuit_version=(
                        required_circuit_version
                        or TEST_PASS_GROTH16_CIRCUIT_VERSION
                    ),
                )
            )
            if not capability_ready:
                return ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_GROTH16_CAPABILITY_MISMATCH,
                    capability="groth16_native",
                    capability_kind="native_executable",
                    fallback_action="DEFERRED",
                    diagnostics={
                        **platform_diagnostics,
                        "binary_source": "reviewed_bundled_binary",
                        "capability_probe_status": capability_status,
                        "process_started": True,
                    },
                )
            activation = self._publish_groth16_binary(
                bundled,
                backend=backend,
                supported_circuit_versions=bundled_versions,
            )
            return ProofReuseCapabilityResolution(
                available=True,
                reason_code=REASON_AVAILABLE,
                capability="groth16_native",
                capability_kind="native_executable",
                fallback_action="DEFERRED",
                diagnostics={
                    "binary_source": "reviewed_bundled_binary",
                    "binary_path": str(bundled),
                    "required_circuit_version": required_circuit_version,
                    "required_capability_validated": (
                        required_circuit_version is None
                        or required_circuit_version in bundled_versions
                    ),
                    "capability_probe_status": capability_status,
                    "process_started": True,
                    "keys_and_circuit_checked": False,
                    **platform_diagnostics,
                    **activation,
                },
            )

        receipted_binary, receipt_status = self._validated_receipted_groth16_binary(
            backend,
            required_circuit_version=required_circuit_version,
        )
        platform_diagnostics["previous_native_build_receipt_status"] = receipt_status
        if receipted_binary is not None and not force_build:
            capability_ready, capability_status = (
                self._probe_groth16_binary_capabilities(
                    receipted_binary,
                    required_circuit_version=(
                        required_circuit_version
                        or TEST_PASS_GROTH16_CIRCUIT_VERSION
                    ),
                )
            )
            if not capability_ready:
                return ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_GROTH16_CAPABILITY_MISMATCH,
                    capability="groth16_native",
                    capability_kind="native_executable",
                    fallback_action="DEFERRED",
                    diagnostics={
                        **platform_diagnostics,
                        "binary_source": "validated_build_receipt",
                        "capability_probe_status": capability_status,
                        "process_started": True,
                    },
                )
            receipt_path, _ = self._groth16_receipt_path(create=False)
            activation = self._publish_groth16_binary(
                receipted_binary,
                backend=backend,
                receipt_path=receipt_path,
                supported_circuit_versions=(
                    TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
                ),
            )
            return ProofReuseCapabilityResolution(
                available=True,
                reason_code=REASON_AVAILABLE,
                capability="groth16_native",
                installed=False,
                capability_kind="native_executable",
                fallback_action="DEFERRED",
                diagnostics={
                    "binary_source": "validated_build_receipt",
                    "binary_path": str(receipted_binary),
                    "required_circuit_version": required_circuit_version,
                    "required_capability_validated": True,
                    "capability_probe_status": capability_status,
                    "process_started": True,
                    "keys_and_circuit_checked": False,
                    **platform_diagnostics,
                    **activation,
                },
            )

        if (
            required_circuit_version is not None
            and required_circuit_version
            not in TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
        ):
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_SOURCE_INVALID,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics={
                    **platform_diagnostics,
                    "reviewed_source_supported_circuit_versions": list(
                        TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
                    ),
                    "required_capability_validated": False,
                },
            )

        provision_key = (
            "groth16_native:"
            f"{_platform_binary_name()}:"
            f"{self._reviewed_groth16_source_fingerprint()}:"
            f"v{required_circuit_version or 'generic'}"
        )
        cached_attempt = self._cached_provision_attempt(provision_key)
        if cached_attempt is not None and not force_build:
            return cached_attempt

        explicit_consent = (
            groth16_build_enabled(self._environ) if consent is None else consent is True
        )
        if not explicit_consent:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_BUILD_DISABLED,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics={
                    "consent_environment_variable": (PROOF_REUSE_GROTH16_BUILD_ENV),
                    "trusted_setup_attempted": False,
                    **platform_diagnostics,
                },
            )
        if not self.install_permitted():
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_AUTO_INSTALL_DISABLED,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics=platform_diagnostics,
            )
        cargo = shutil.which("cargo")
        if not cargo:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_TOOLCHAIN_MISSING,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics=platform_diagnostics,
            )
        cargo_executable = Path(os.path.abspath(cargo))
        try:
            cargo_executable_ok = cargo_executable.is_file() and (
                os.name == "nt" or os.access(cargo_executable, os.X_OK)
            )
        except OSError:
            cargo_executable_ok = False
        if not cargo_executable_ok:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_TOOLCHAIN_MISSING,
                capability="groth16_native",
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics=platform_diagnostics,
            )

        with _GROTH16_BUILD_THREAD_LOCK, _bounded_interprocess_provision_fence(
            self._lock_root,
            capability_key="datasets-groth16-cargo-build",
            timeout_seconds=self._lock_timeout_seconds,
        ) as acquired:
            if not acquired:
                return ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_LOCK_TIMEOUT,
                    capability="groth16_native",
                    capability_kind="cargo_native_build",
                    fallback_action="DEFERRED",
                    diagnostics=platform_diagnostics,
                )
            # Validate again after waiting.  Never compile a checkout that
            # changed while another process held the fence.
            backend = self._validated_groth16_backend_dir()
            if backend is None:
                return ProofReuseCapabilityResolution(
                    available=False,
                    reason_code=REASON_GROTH16_SOURCE_INVALID,
                    capability="groth16_native",
                    capability_kind="cargo_native_build",
                    fallback_action="DEFERRED",
                    diagnostics=platform_diagnostics,
                )
            receipted_binary, receipt_status = self._validated_receipted_groth16_binary(
                backend,
                required_circuit_version=required_circuit_version,
            )
            platform_diagnostics["previous_native_build_receipt_status"] = (
                receipt_status
            )
            if receipted_binary is not None and not force_build:
                capability_ready, capability_status = (
                    self._probe_groth16_binary_capabilities(
                        receipted_binary,
                        required_circuit_version=(
                            required_circuit_version
                            or TEST_PASS_GROTH16_CIRCUIT_VERSION
                        ),
                    )
                )
                if not capability_ready:
                    return ProofReuseCapabilityResolution(
                        available=False,
                        reason_code=REASON_GROTH16_CAPABILITY_MISMATCH,
                        capability="groth16_native",
                        capability_kind="native_executable",
                        fallback_action="DEFERRED",
                        diagnostics={
                            **platform_diagnostics,
                            "binary_source": "validated_build_receipt",
                            "capability_probe_status": capability_status,
                            "process_started": True,
                        },
                    )
                receipt_path, _ = self._groth16_receipt_path(create=False)
                activation = self._publish_groth16_binary(
                    receipted_binary,
                    backend=backend,
                    receipt_path=receipt_path,
                    supported_circuit_versions=(
                        TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
                    ),
                )
                return ProofReuseCapabilityResolution(
                    available=True,
                    reason_code=REASON_AVAILABLE,
                    capability="groth16_native",
                    installed=False,
                    capability_kind="native_executable",
                    fallback_action="DEFERRED",
                    diagnostics={
                        "binary_source": "validated_build_receipt",
                        "binary_path": str(receipted_binary),
                        "required_circuit_version": required_circuit_version,
                        "required_capability_validated": True,
                        "capability_probe_status": capability_status,
                        "process_started": True,
                        "keys_and_circuit_checked": False,
                        **platform_diagnostics,
                        **activation,
                    },
                )
            with self._reviewed_groth16_build_snapshot() as build_backend:
                if build_backend is None:
                    return self._remember_provision_attempt(
                        provision_key,
                        ProofReuseCapabilityResolution(
                            available=False,
                            reason_code=REASON_GROTH16_SOURCE_INVALID,
                            capability="groth16_native",
                            capability_kind="cargo_native_build",
                            fallback_action="DEFERRED",
                            diagnostics={
                                "immutable_source_snapshot": False,
                                **platform_diagnostics,
                            },
                        ),
                    )
                target_root = build_backend.parent / "target"
                run_environment = self._isolated_cargo_environment(
                    build_backend.parent, target_root
                )
                invocation_root = Path(build_backend.anchor)
                root_cargo_config = any(
                    (invocation_root / ".cargo" / name).exists()
                    for name in ("config", "config.toml")
                )
                if run_environment is None or root_cargo_config:
                    return self._remember_provision_attempt(
                        provision_key,
                        ProofReuseCapabilityResolution(
                            available=False,
                            reason_code=REASON_GROTH16_SOURCE_INVALID,
                            capability="groth16_native",
                            capability_kind="cargo_native_build",
                            fallback_action="DEFERRED",
                            diagnostics={
                                "isolated_cargo_environment": False,
                                **platform_diagnostics,
                            },
                        ),
                    )
                manifest = build_backend / "Cargo.toml"
                command = (
                    str(cargo_executable),
                    "build",
                    "--locked",
                    "--release",
                    "--manifest-path",
                    str(manifest),
                    "--target-dir",
                    str(target_root),
                )
                try:
                    completed = self._process_runner(
                        command,
                        cwd=str(invocation_root),
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=self._native_timeout_seconds,
                        env=run_environment,
                    )
                except subprocess.TimeoutExpired as exc:
                    return self._remember_provision_attempt(
                        provision_key,
                        ProofReuseCapabilityResolution(
                            available=False,
                            reason_code=REASON_PROVISION_TIMEOUT,
                            capability="groth16_native",
                            capability_kind="cargo_native_build",
                            fallback_action="DEFERRED",
                            diagnostics={"error": str(exc)[:300]},
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 - optional boundary.
                    reason = _classify_install_failure(
                        returncode=None, output="", error=exc
                    )
                    return self._remember_provision_attempt(
                        provision_key,
                        ProofReuseCapabilityResolution(
                            available=False,
                            reason_code=reason,
                            capability="groth16_native",
                            capability_kind="cargo_native_build",
                            fallback_action="DEFERRED",
                            diagnostics={"error": str(exc)[:300]},
                        ),
                    )
                stdout = str(getattr(completed, "stdout", "") or "")
                stderr = str(getattr(completed, "stderr", "") or "")
                returncode = int(getattr(completed, "returncode", 1))
                if returncode != 0:
                    return self._remember_provision_attempt(
                        provision_key,
                        ProofReuseCapabilityResolution(
                            available=False,
                            reason_code=_classify_install_failure(
                                returncode=returncode,
                                output=f"{stdout}\n{stderr}",
                            ),
                            capability="groth16_native",
                            capability_kind="cargo_native_build",
                            fallback_action="DEFERRED",
                            diagnostics={"output": f"{stdout}\n{stderr}"[:500]},
                        ),
                    )
                binary = build_backend.parent / _groth16_build_binary_relative()
                try:
                    binary_ok = (
                        binary.is_file()
                        and not binary.is_symlink()
                        and binary.stat().st_size > 0
                        and (os.name == "nt" or os.access(binary, os.X_OK))
                    )
                except OSError:
                    binary_ok = False
                capability_status = "binary_missing"
                if binary_ok:
                    built_loaded = _read_regular_file_bytes(
                        binary, max_bytes=_GROTH16_BUILD_BINARY_MAX_BYTES
                    )
                    if built_loaded is not None:
                        self._validated_native_binary_identities[
                            binary.resolve(strict=True)
                        ] = (
                            hashlib.sha256(built_loaded[0]).hexdigest(),
                            built_loaded[1].st_size,
                        )
                    capability_ready, capability_status = (
                        self._probe_groth16_binary_capabilities(
                            binary,
                            required_circuit_version=(
                                required_circuit_version
                                or TEST_PASS_GROTH16_CIRCUIT_VERSION
                            ),
                        )
                    )
                    if capability_ready:
                        receipt_write_status, activated_binary = (
                            self._write_groth16_build_receipt(backend, binary)
                        )
                    else:
                        receipt_write_status, activated_binary = (
                            "capability_mismatch",
                            None,
                        )
                else:
                    receipt_write_status, activated_binary = (
                        "binary_missing",
                        None,
                    )
            available = activated_binary is not None
            activation = (
                self._publish_groth16_binary(
                    activated_binary,
                    backend=backend,
                    receipt_path=self._groth16_receipt_path(create=False)[0],
                    supported_circuit_versions=(
                        TEST_PASS_GROTH16_SUPPORTED_SOURCE_VERSIONS
                    ),
                )
                if activated_binary is not None
                else {}
            )
            resolution = ProofReuseCapabilityResolution(
                available=available,
                reason_code=(
                    REASON_AVAILABLE
                    if available
                    else (
                        REASON_GROTH16_CAPABILITY_MISMATCH
                        if receipt_write_status == "capability_mismatch"
                        else REASON_GROTH16_BINARY_MISSING
                    )
                ),
                capability="groth16_native",
                installed=available,
                capability_kind="cargo_native_build",
                fallback_action="DEFERRED",
                diagnostics={
                    "binary_path": (str(activated_binary) if available else ""),
                    "binary_source": "cargo_build" if available else "",
                    "required_circuit_version": required_circuit_version,
                    "required_capability_validated": available,
                    "capability_probe_status": capability_status,
                    "process_started": True,
                    "cargo_locked": True,
                    "build_receipt_status": receipt_write_status,
                    "immutable_source_snapshot": True,
                    "isolated_cargo_environment": True,
                    "trusted_setup_attempted": False,
                    "keys_and_circuit_checked": False,
                    **platform_diagnostics,
                    **activation,
                },
            )
            return self._remember_provision_attempt(provision_key, resolution)

    def inspect_groth16_endpoint(self) -> ProofReuseCapabilityResolution:
        """Validate endpoint configuration without connecting to it."""

        configured = str(self._environ.get(PROOF_REUSE_GROTH16_ENDPOINT_ENV, ""))
        raw = configured.strip()
        if not raw:
            return ProofReuseCapabilityResolution(
                available=False,
                reason_code=REASON_GROTH16_ENDPOINT_UNCONFIGURED,
                capability="groth16_endpoint",
                capability_kind="operator_configuration",
                fallback_action="DEFERRED",
            )
        try:
            endpoint = urlsplit(raw)
            port = endpoint.port
        except (TypeError, ValueError):
            endpoint = None
            port = None
        valid = bool(
            endpoint
            and configured == raw
            and not any(character.isspace() for character in raw)
            and "\\" not in raw
            and endpoint.scheme in {"http", "https"}
            and endpoint.hostname
            and not endpoint.username
            and not endpoint.password
            and not endpoint.fragment
        )
        origin = ""
        if valid and endpoint is not None:
            origin = f"{endpoint.scheme}://{endpoint.hostname}"
            if port is not None:
                origin += f":{port}"
        return ProofReuseCapabilityResolution(
            available=valid,
            reason_code=(
                REASON_AVAILABLE if valid else REASON_GROTH16_ENDPOINT_UNCONFIGURED
            ),
            capability="groth16_endpoint",
            capability_kind="operator_configuration",
            fallback_action="DEFERRED",
            diagnostics={"endpoint_origin": origin, "network_attempted": False},
        )

    def _groth16_artifacts_root(self) -> tuple[Path | None, bool]:
        configured = str(
            self._environ.get(DATASETS_GROTH16_ARTIFACTS_ROOT_ENV, "")
        ).strip()
        if configured:
            try:
                requested = Path(configured).expanduser()
                if requested.is_symlink():
                    return None, False
                root = requested.resolve(strict=True)
            except (OSError, RuntimeError):
                return None, False
            if not root.is_dir():
                return None, False
            expected_marker = _reviewed_groth16_artifacts_marker(root)
            configured_marker = str(
                self._environ.get(_GROTH16_REVIEWED_ARTIFACTS_MARKER_ENV, "")
            ).strip()
            return (
                root,
                root == self._reviewed_groth16_artifacts_root
                or bool(expected_marker and configured_marker == expected_marker),
            )
        backend = self._validated_groth16_backend_dir()
        if backend is None:
            return None, True
        return backend / "artifacts", True

    def inspect_groth16_keys(self) -> ProofReuseCapabilityResolution:
        """Inspect key pairs independently of binary/circuit availability."""

        root, reviewed = self._groth16_artifacts_root()
        available_versions: list[int] = []
        proving_versions: list[int] = []
        manifest_verifying_versions: tuple[int, ...] = ()
        manifest_proving_versions: tuple[int, ...] = ()
        manifest_reason = "artifact_manifest_not_inspected"
        manifest_digest = ""
        if root is not None:
            for version in (1, 2, 3):
                vk_relative = f"v{version}/verifying_key.bin"
                pk_relative = f"v{version}/proving_key.bin"
                vk = root / vk_relative
                pk = root / pk_relative
                if reviewed:
                    expected_vk = DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256.get(
                        vk_relative
                    )
                    expected_pk = DATASETS_GROTH16_REVIEWED_ARTIFACTS_SHA256.get(
                        pk_relative
                    )
                    vk_ok = bool(
                        expected_vk
                        and _sha256_regular_file(vk, max_bytes=64 * 1024 * 1024)
                        == expected_vk
                    )
                    pk_ok = bool(
                        expected_pk
                        and _sha256_regular_file(pk, max_bytes=64 * 1024 * 1024)
                        == expected_pk
                    )
                else:
                    vk_digest = _sha256_regular_file(vk, max_bytes=64 * 1024 * 1024)
                    pk_digest = _sha256_regular_file(pk, max_bytes=64 * 1024 * 1024)
                    vk_ok = bool(vk_digest)
                    pk_ok = bool(pk_digest)
                if vk_ok:
                    available_versions.append(version)
                if pk_ok:
                    proving_versions.append(version)
            try:
                from .publication import inspect_pinned_groth16_artifact_versions

                (
                    manifest_verifying_versions,
                    manifest_proving_versions,
                    manifest_reason,
                    manifest_digest,
                ) = inspect_pinned_groth16_artifact_versions(
                    artifacts_root=root,
                    environ=self._environ,
                )
            except Exception:
                manifest_reason = "artifact_manifest_inspection_failed"
            available_versions.extend(
                version
                for version in manifest_verifying_versions
                if version not in available_versions
            )
            proving_versions.extend(
                version
                for version in manifest_proving_versions
                if version not in proving_versions
            )
        return ProofReuseCapabilityResolution(
            available=bool(available_versions),
            reason_code=(
                REASON_AVAILABLE if available_versions else REASON_GROTH16_KEYS_MISSING
            ),
            capability="groth16_keys",
            capability_kind="cryptographic_artifacts",
            fallback_action="DEFERRED",
            diagnostics={
                "verifying_key_versions": available_versions,
                "proving_key_versions": proving_versions,
                "manifest_verifying_key_versions": list(
                    manifest_verifying_versions
                ),
                "manifest_proving_key_versions": list(manifest_proving_versions),
                "artifact_manifest_reason": manifest_reason,
                "artifact_manifest_sha256": manifest_digest,
                "reviewed_bundled_artifacts": reviewed,
                "v4_arbitrary_key_presence_authoritative": False,
                "auto_setup_attempted": False,
            },
        )

    def inspect_groth16_circuit(self) -> ProofReuseCapabilityResolution:
        """Inspect an explicit circuit binding; never auto-select a circuit."""

        circuit_ref = str(
            self._environ.get(PROOF_REUSE_GROTH16_CIRCUIT_REF_ENV, "")
        ).strip()
        valid = bool(
            circuit_ref
            and re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,96}@v[1-9][0-9]{0,5}",
                circuit_ref,
            )
        )
        circuit_version = int(circuit_ref.rsplit("@v", 1)[1]) if valid else None
        return ProofReuseCapabilityResolution(
            available=valid,
            reason_code=(
                REASON_AVAILABLE if valid else REASON_GROTH16_CIRCUIT_UNCONFIGURED
            ),
            capability="groth16_circuit",
            capability_kind="versioned_circuit_binding",
            fallback_action="DEFERRED",
            diagnostics={
                "circuit_ref": circuit_ref if valid else "",
                "circuit_version": circuit_version,
            },
        )

    def inspect_groth16_runtime(self) -> dict[str, Any]:
        """Collect local facts; execute only the bounded capability subcommand."""

        endpoint = self.inspect_groth16_endpoint()
        keys = self.inspect_groth16_keys()
        circuit = self.inspect_groth16_circuit()
        circuit_version = circuit.diagnostics.get("circuit_version")
        native = self.ensure_groth16_native_backend(
            consent=False,
            required_circuit_version=(
                int(circuit_version)
                if isinstance(circuit_version, int)
                and not isinstance(circuit_version, bool)
                else TEST_PASS_GROTH16_CIRCUIT_VERSION
            ),
        )
        verifying_versions = set(keys.diagnostics.get("verifying_key_versions", ()))
        proving_versions = set(keys.diagnostics.get("proving_key_versions", ()))
        version_has_verifying_key = circuit_version in verifying_versions
        version_has_proving_key = circuit_version in proving_versions
        native_ready = bool(
            native.available
            and circuit.available
            and version_has_verifying_key
            and version_has_proving_key
        )
        endpoint_ready = bool(
            endpoint.available and circuit.available and version_has_verifying_key
        )
        ready = native_ready or endpoint_ready
        return {
            "interface": "Groth16RuntimeCapabilityStatus@1",
            "ready": ready,
            "readiness_scope": "generic_native_or_endpoint_capability",
            "action": "RUN" if ready else "DEFERRED",
            # Generic runtime readiness is deliberately not SKIP authority;
            # the separate authority probe validates the pinned v4 manifest.
            "test_certificate_authority_ready": False,
            "test_certificate_authority_reason": (
                "requires_separate_test_certificate_authority_probe"
            ),
            "skip_authority": False,
            "native": native.to_dict(),
            "endpoint": endpoint.to_dict(),
            "keys": keys.to_dict(),
            "circuit": circuit.to_dict(),
            "network_attempted": False,
            "process_started": bool(native.diagnostics.get("process_started")),
            "trusted_setup_attempted": False,
            "version_compatibility": {
                "circuit_version": circuit_version,
                "has_verifying_key": version_has_verifying_key,
                "has_proving_key": version_has_proving_key,
                "native_ready": native_ready,
                "endpoint_ready": endpoint_ready,
            },
        }

    def install(self, dependency: ProofReuseDependency) -> bool:
        """Plugin-compatible install entry point (allowlisted only)."""

        resolution = self.ensure_capability(
            getattr(dependency, "module_name", ""),
            dependency=dependency,
        )
        return bool(resolution.available)

    def ensure_all_allowlisted(
        self,
    ) -> dict[str, ProofReuseCapabilityResolution]:
        """Probe every allowlisted capability; never aborts on failure."""

        return {
            module_name: self.ensure_capability(module_name)
            for module_name in PROOF_REUSE_DEPENDENCY_ALLOWLIST
        }


def get_default_lazy_dependency_installer(
    *,
    environ: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> ProofReuseLazyDependencyInstaller:
    return ProofReuseLazyDependencyInstaller(environ=environ, **kwargs)


class AcceleratorProofReuseBootstrap:
    """Narrow zero-config facade for accelerator proof-reuse discovery.

    Implements ``AcceleratorProofReuseBootstrap@1``.  Ordinary / off-mode paths
    only touch the lightweight loader and config surface.  Enabled modes build
    defaults lazily without item monkeypatches or conftest service injection.
    """

    interface: str = ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE
    plugin_module: str = PLUGIN_MODULE
    config_module: str = PROOF_REUSE_CONFIG_MODULE

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        installer: ProofReuseLazyDependencyInstaller | None = None,
    ) -> None:
        self._uses_process_environment = environ is None
        self._environ = dict(os.environ if environ is None else environ)
        self._installer = installer
        self._services: DefaultProofReuseServices | None = None

    @property
    def installer(self) -> ProofReuseLazyDependencyInstaller:
        if self._installer is None:
            self._installer = ProofReuseLazyDependencyInstaller(
                environ=(None if self._uses_process_environment else self._environ)
            )
        return self._installer

    def lightweight_modules(self) -> tuple[str, ...]:
        """Modules that off-mode / ordinary tests may import safely."""

        return (
            "ipfs_accelerate_py.testing.proof_reuse",
            PROOF_REUSE_CONFIG_MODULE,
            "ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies",
        )

    def optional_plugin_modules(self) -> tuple[str, ...]:
        """Return the plugin module when available; empty when absent."""

        try:
            importlib.import_module(PLUGIN_MODULE)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing and (
                missing == PLUGIN_MODULE or PLUGIN_MODULE.startswith(f"{missing}.")
            ):
                return ()
            raise
        return (PLUGIN_MODULE,)

    def is_non_reusing_tooling_mode(self, mode: str | None = None) -> bool:
        """Coverage/mutation/profile/debugger/leak modes stay non-reusing."""

        active = (
            str(mode or self._environ.get("IPFS_TEST_PROOF_REUSE_MODE", "off"))
            .strip()
            .lower()
        )
        if active in {"", "off", "0", "false", "no", "disabled"}:
            return True
        # Explicit tooling environment markers force non-reuse.
        tooling_env = (
            "COVERAGE_PROCESS_START",
            "COV_CORE_SOURCE",
            "PYTEST_XDIST_WORKER",  # workers never install; composition handles
            "MUTMUT_MUTANT_PATH",
            "MUTPEN_MUTANT",
            "PYTEST_CURRENT_TEST_LEAK",
        )
        if any(self._environ.get(name) for name in tooling_env):
            # xdist workers still may reuse; only true tooling markers apply.
            if self._environ.get("PYTEST_XDIST_WORKER") and not any(
                self._environ.get(name)
                for name in tooling_env
                if name != "PYTEST_XDIST_WORKER"
            ):
                return False
            return True
        return False

    def ensure_capability(
        self, capability: str, **kwargs: Any
    ) -> ProofReuseCapabilityResolution:
        return self.installer.ensure_capability(capability, **kwargs)

    def ensure_nltk_data(self, **kwargs: Any) -> ProofReuseCapabilityResolution:
        """Explicit first-use facade for bounded NLTK resource provisioning."""

        return self.installer.ensure_nltk_data(**kwargs)

    def ensure_groth16_native_backend(
        self, **kwargs: Any
    ) -> ProofReuseCapabilityResolution:
        """Explicit first-use facade for the reviewed native Cargo build."""

        return self.installer.ensure_groth16_native_backend(**kwargs)

    def inspect_groth16_runtime(self) -> dict[str, Any]:
        """Collect binary/endpoint/key/circuit facts without external effects."""

        return self.installer.inspect_groth16_runtime()

    def build_default_services(
        self,
        *,
        mode: Any = None,
        root_path: str | os.PathLike[str] | None = None,
        config: Any = None,
        cache_root: str | os.PathLike[str] | None = None,
        lookup: Any = None,
        store: Any = None,
        provider: Any = None,
        issuer: Any = None,
        identity_services: Any = None,
    ) -> DefaultProofReuseServices:
        """Lazily compose defaults without item attributes or conftest injection."""

        installer: Any | None = None
        if self._installer is not None:
            if self._installer.install_permitted():
                installer = self._installer
        elif proof_reuse_install_permitted(self._environ):
            installer = self.installer
        services = compose_default_proof_reuse_services(
            mode=mode,
            root_path=root_path,
            config=config,
            cache_root=cache_root,
            installer=installer,
            identity_services=identity_services,
            lookup=lookup,
            store=store,
            provider=provider,
            issuer=issuer,
            environ=self._environ,
        )
        self._services = services
        return services

    def dependency_manifest_parity_plan(self) -> dict[str, Any]:
        """Describe core vs proof-reuse packaging expectations for CA/ZK deps."""

        plan = self.installer.dependency_plan()
        return {
            "interface": "AcceleratorProofReuseDependencyManifest@1",
            "content_addressing": {
                "declared_as": "core",
                "requirements": ["multiformats>=0.3,<1", "pymultihash>=0.8.2"],
                "proof_reuse_extra": ["multiformats>=0.3,<1"],
            },
            "datasets_zk": {
                "declared_as": "lazy_proof_reuse_only",
                "packaging_extra": "proof-reuse",
                "remote_published": plan.get("remote_source_published"),
                "distribution": DATASETS_VERIFIER_DEPENDENCY.distribution,
                "module_name": DATASETS_VERIFIER_DEPENDENCY.module_name,
                "install_options": list(DATASETS_VERIFIER_DEPENDENCY.pip_options),
            },
            "schema_validation": {
                "declared_as": "core-and-proof-reuse-extra",
                "requirements": ["jsonschema>=4,<5"],
            },
            "nltk_python": {
                "declared_as": "core-and-proof-reuse-extra",
                "requirements": [NLTK_DEPENDENCY.distribution],
            },
            "nltk_data": plan.get("nltk_data"),
            "groth16_native_backend": plan.get("groth16_native_backend"),
            "groth16_runtime_inputs": plan.get("groth16_runtime_inputs"),
            "manifests": (
                "requirements.txt",
                "requirements-proof-reuse.txt",
                "setup.py",
                "pyproject.toml",
                "MANIFEST.in",
            ),
            "lazy_installer_interface": (
                PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE
            ),
        }


_BOOTSTRAP_SINGLETON: AcceleratorProofReuseBootstrap | None = None
_BOOTSTRAP_LOCK = threading.Lock()


def get_proof_reuse_bootstrap(
    *,
    environ: Mapping[str, str] | None = None,
    reset: bool = False,
) -> AcceleratorProofReuseBootstrap:
    """Return the process-wide narrow proof-reuse bootstrap facade."""

    global _BOOTSTRAP_SINGLETON
    if environ is not None:
        return AcceleratorProofReuseBootstrap(environ=environ)
    with _BOOTSTRAP_LOCK:
        if reset or _BOOTSTRAP_SINGLETON is None:
            _BOOTSTRAP_SINGLETON = AcceleratorProofReuseBootstrap()
        return _BOOTSTRAP_SINGLETON


__all__ = [
    "ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE",
    "AcceleratorProofReuseBootstrap",
    "PACKAGE_AUTO_INSTALL_ENV",
    "PLUGIN_MODULE",
    "PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE",
    "ProofReuseCapabilityResolution",
    "ProofReuseLazyDependencyInstaller",
    "REASON_AUTO_INSTALL_DISABLED",
    "REASON_AVAILABLE",
    "REASON_DEPENDENCY_MISSING",
    "REASON_INCOMPATIBLE_VERSION",
    "REASON_INSTALL_FAILED",
    "REASON_GROTH16_BINARY_MISSING",
    "REASON_GROTH16_BUILD_DISABLED",
    "REASON_GROTH16_CIRCUIT_UNCONFIGURED",
    "REASON_GROTH16_ENDPOINT_UNCONFIGURED",
    "REASON_GROTH16_KEYS_MISSING",
    "REASON_GROTH16_SOURCE_INVALID",
    "REASON_GROTH16_TOOLCHAIN_MISSING",
    "REASON_LOCK_TIMEOUT",
    "REASON_NLTK_DATA_MISSING",
    "REASON_NLTK_DOWNLOAD_DISABLED",
    "REASON_NOT_ALLOWLISTED",
    "REASON_OFFLINE_INDEX",
    "REASON_READ_ONLY_ENVIRONMENT",
    "REASON_RESOLVER_FAILURE",
    "REASON_PROVISION_TIMEOUT",
    "get_default_lazy_dependency_installer",
    "get_proof_reuse_bootstrap",
    "package_auto_install_policy_permits",
    "proof_reuse_install_permitted",
    "resolve_capability_module_name",
]
