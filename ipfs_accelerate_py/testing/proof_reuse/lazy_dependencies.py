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

import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Final, Iterator, Mapping

from .services import (
    DATASETS_VERIFIER_DEPENDENCY,
    JSONSCHEMA_DEPENDENCY,
    MULTIFORMATS_DEPENDENCY,
    PROOF_REUSE_AUTO_INSTALL_ENV,
    PROOF_REUSE_CACHE_DIR_ENV,
    PROOF_REUSE_DEPENDENCY_ALLOWLIST,
    AllowlistedPipInstaller,
    DefaultProofReuseServices,
    ProofReuseDependency,
    automatic_install_enabled,
    compose_default_proof_reuse_services,
    proof_reuse_dependency_plan,
)

PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE: Final = (
    "ProofReuseLazyDependencyInstaller@1"
)
ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE: Final = (
    "AcceleratorProofReuseBootstrap@1"
)

PLUGIN_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PROOF_REUSE_CONFIG_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.config"

# Package-level auto-install policy (orthogonal to proof-reuse mode policy).
PACKAGE_AUTO_INSTALL_ENV: Final = "IPFS_ACCEL_AUTO_INSTALL"

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

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"", "0", "false", "no", "off", "disabled"})

_CAPABILITY_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "content_addressing": MULTIFORMATS_DEPENDENCY.module_name,
        "cid": MULTIFORMATS_DEPENDENCY.module_name,
        "multiformats": MULTIFORMATS_DEPENDENCY.module_name,
        "jsonschema": JSONSCHEMA_DEPENDENCY.module_name,
        "certificate_schema": JSONSCHEMA_DEPENDENCY.module_name,
        "datasets_zk": DATASETS_VERIFIER_DEPENDENCY.module_name,
        "datasets_verifier": DATASETS_VERIFIER_DEPENDENCY.module_name,
        "zk_verifier": DATASETS_VERIFIER_DEPENDENCY.module_name,
    }
)

_VERSION_CONSTRAINT_RE: Final = re.compile(
    r"^(?P<name>[A-Za-z0-9_.\-]+)\s*(?P<spec>.*)$"
)
_SPEC_TOKEN_RE: Final = re.compile(
    r"(==|!=|<=|>=|<|>|~=)\s*([0-9]+(?:\.[0-9A-Za-z.*+!~_-]*)?)"
)

_INSTALL_THREAD_LOCK = threading.Lock()


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

    @property
    def action(self) -> str:
        """Tests always continue; reasons are diagnostic only."""

        return "RUN"

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason_code": self.reason_code,
            "capability": self.capability,
            "module_name": self.module_name,
            "distribution": self.distribution,
            "installed": self.installed,
            "action": self.action,
            "diagnostics": dict(self.diagnostics),
        }


def package_auto_install_policy_permits(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the package-level auto-install policy allows installs.

    Controlled by ``IPFS_ACCEL_AUTO_INSTALL``.  When unset, permission defaults
    to true inside a virtual environment and false otherwise (matching
    :mod:`ipfs_accelerate_py.utils.auto_install`).  Invalid values deny.
    """

    source = os.environ if environ is None else environ
    if PACKAGE_AUTO_INSTALL_ENV not in source:
        try:
            return sys.prefix != getattr(sys, "base_prefix", sys.prefix)
        except Exception:
            return False
    value = str(source.get(PACKAGE_AUTO_INSTALL_ENV, "")).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return False


def proof_reuse_install_permitted(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Both proof-reuse and package auto-install policies must permit install."""

    return automatic_install_enabled(environ) and package_auto_install_policy_permits(
        environ
    )


def resolve_capability_module_name(capability: str) -> str:
    """Map a capability alias or module name onto the allowlist key."""

    text = str(capability or "").strip()
    if not text:
        return ""
    aliased = _CAPABILITY_ALIASES.get(text) or _CAPABILITY_ALIASES.get(
        text.lower()
    )
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
    for left_item, right_item in zip(padded_left, padded_right):
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
                upper_compared = _compare_versions(
                    installed_version, upper_version
                )
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


@contextmanager
def _interprocess_install_fence(
    lock_root: str | os.PathLike[str] | None = None,
    *,
    capability_key: str = "global",
) -> Iterator[None]:
    """Serialize installs across processes with an exclusive file lock."""

    if lock_root:
        root = Path(lock_root)
    else:
        root = Path(tempfile.gettempdir()) / "ipfs-accelerate-proof-reuse-install"
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        yield
        return

    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", capability_key)[:120] or "global"
    path = root / f"{safe_key}.lock"
    try:
        handle = path.open("a+", encoding="utf-8")
    except OSError:
        yield
        return

    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        try:
            handle.close()
        except OSError:
            pass


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
        self._runner = runner
        self._timeout_seconds = float(timeout_seconds)
        self._environ = dict(os.environ if environ is None else environ)
        self._importer = importer or importlib.import_module
        self._lock_root = (
            str(lock_root)
            if lock_root is not None
            else self._environ.get(PROOF_REUSE_CACHE_DIR_ENV, "").strip()
            or None
        )
        self._pip = pip_installer or AllowlistedPipInstaller(
            runner=runner,
            timeout_seconds=self._timeout_seconds,
            environ=self._environ,
        )
        self._outcomes: dict[str, ProofReuseCapabilityResolution] = {}
        self._lock = threading.Lock()

    def allowlist(self) -> Mapping[str, ProofReuseDependency]:
        return PROOF_REUSE_DEPENDENCY_ALLOWLIST

    def dependency_plan(self) -> dict[str, Any]:
        return proof_reuse_dependency_plan(self._environ)

    def install_permitted(self) -> bool:
        return proof_reuse_install_permitted(self._environ)

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
    ) -> ProofReuseCapabilityResolution:
        """Ensure one allowlisted proof-reuse capability is importable.

        Never raises for optional-boundary faults.  Returns a typed reason so
        callers always continue with RUN semantics.
        """

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

        with _INSTALL_THREAD_LOCK, _interprocess_install_fence(
            self._lock_root,
            capability_key=selected.module_name,
        ):
            # Re-check under the fence — another process may have installed.
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
                if self._runner is not None:
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
                present = self._check_present(
                    selected, capability=capability_key
                )
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
        # Prefer the allowlisted installer's distribution selection logic.
        distribution = getattr(self._pip, "_selected_distribution", None)
        selected: str | None
        if callable(distribution):
            selected = distribution(dependency)
        else:
            selected = dependency.distribution
        if not selected:
            return False, None, "no matching distribution found", None
        command = (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *dependency.pip_options,
            selected,
        )
        run_environment = dict(self._environ)
        run_environment.update(dict(dependency.install_environment))
        try:
            completed = self._runner(  # type: ignore[misc]
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                env=run_environment,
            )
        except Exception as exc:  # noqa: BLE001
            return False, None, "", exc
        stdout = getattr(completed, "stdout", "") or ""
        stderr = getattr(completed, "stderr", "") or ""
        output = f"{stdout}\n{stderr}".strip()
        code = getattr(completed, "returncode", 1)
        return code == 0, int(code) if code is not None else None, output, None

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
        self._environ = dict(os.environ if environ is None else environ)
        self._installer = installer
        self._services: DefaultProofReuseServices | None = None

    @property
    def installer(self) -> ProofReuseLazyDependencyInstaller:
        if self._installer is None:
            self._installer = ProofReuseLazyDependencyInstaller(
                environ=self._environ
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
                missing == PLUGIN_MODULE
                or PLUGIN_MODULE.startswith(f"{missing}.")
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

        installer: Any | None = self.installer
        if not self.installer.install_permitted():
            installer = None
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
                "install_options": list(
                    DATASETS_VERIFIER_DEPENDENCY.pip_options
                ),
            },
            "schema_validation": {
                "declared_as": "proof-reuse-extra",
                "requirements": ["jsonschema>=4,<5"],
            },
            "manifests": (
                "requirements.txt",
                "requirements-proof-reuse.txt",
                "setup.py",
                "pyproject.toml",
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
    "REASON_NOT_ALLOWLISTED",
    "REASON_OFFLINE_INDEX",
    "REASON_READ_ONLY_ENVIRONMENT",
    "REASON_RESOLVER_FAILURE",
    "get_default_lazy_dependency_installer",
    "get_proof_reuse_bootstrap",
    "package_auto_install_policy_permits",
    "proof_reuse_install_permitted",
    "resolve_capability_module_name",
]
