"""Lazy, allowlisted service assembly for pytest proof reuse.

This module is deliberately inert when imported.  It does not inspect the
environment, create a cache, import an optional provider, install a package,
or perform a proof operation.  :mod:`.plugin` constructs a resolver only after
proof reuse has been explicitly enabled and calls :meth:`resolve` once during
``pytest_configure``.

The resolver has a closed dependency vocabulary.  An installer can only be
asked for one of those entries, and a failed import/install/construction
returns an unavailable bundle.  Callers must consequently execute tests.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

PROOF_REUSE_AUTO_INSTALL_ENV: Final = (
    "IPFS_TEST_PROOF_REUSE_AUTO_INSTALL"
)
PROOF_REUSE_CACHE_DIR_ENV: Final = "IPFS_TEST_PROOF_REUSE_CACHE_DIR"

MULTIFORMATS_MODULE: Final = "multiformats"
DATASETS_VERIFIER_MODULE: Final = (
    "ipfs_datasets_py.logic.zkp.test_execution_certificate"
)
STORE_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.proof.test_certificate_store"
)
PROVIDER_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.integrations."
    "ipfs_datasets_test_certificate_provider"
)
LOOKUP_MODULE: Final = "ipfs_accelerate_py.testing.proof_reuse.lookup"


@dataclass(frozen=True, slots=True)
class ProofReuseDependency:
    """One exact optional module and its controlled installation spec."""

    module_name: str
    distribution: str
    required_symbols: tuple[str, ...] = ()
    unavailable_reason: str = "plugin_unavailable"


MULTIFORMATS_DEPENDENCY: Final = ProofReuseDependency(
    module_name=MULTIFORMATS_MODULE,
    distribution="multiformats>=0.3,<1",
    required_symbols=("CID", "multihash"),
    unavailable_reason="cid_provider_unavailable",
)
DATASETS_VERIFIER_DEPENDENCY: Final = ProofReuseDependency(
    module_name=DATASETS_VERIFIER_MODULE,
    distribution="ipfs_datasets_py>=0.2,<0.3",
    required_symbols=("verify_test_execution_certificate",),
    unavailable_reason="certificate_provider_unavailable",
)

_DEPENDENCIES: Final = (
    MULTIFORMATS_DEPENDENCY,
    DATASETS_VERIFIER_DEPENDENCY,
)
PROOF_REUSE_DEPENDENCY_ALLOWLIST: Final[
    Mapping[str, ProofReuseDependency]
] = MappingProxyType(
    {dependency.module_name: dependency for dependency in _DEPENDENCIES}
)

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"", "0", "false", "no", "off"})


def automatic_install_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return the explicit auto-install opt-in without mutating anything."""

    source = os.environ if environ is None else environ
    value = str(source.get(PROOF_REUSE_AUTO_INSTALL_ENV, "")).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    # An invalid policy is never interpreted as permission to install.
    return False


@dataclass(frozen=True, slots=True)
class ProofReuseServiceResolution:
    """All-or-nothing result of one lazy service assembly attempt."""

    available: bool
    reason_code: str
    lookup: Any = None
    store: Any = None
    provider: Any = None
    installed_modules: tuple[str, ...] = ()

    @classmethod
    def unavailable(cls, reason_code: str) -> ProofReuseServiceResolution:
        return cls(available=False, reason_code=reason_code)


class AllowlistedPipInstaller:
    """Bounded pip installer for the closed proof-reuse dependency set.

    Nothing invokes this class unless proof reuse is enabled *and* the
    ``IPFS_TEST_PROOF_REUSE_AUTO_INSTALL`` opt-in is true.  Tests and embedding
    applications can inject a different installer instead.  Each dependency
    receives at most one process attempt per installer instance.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1 <= float(timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must be between 1 and 600")
        self._runner = runner or subprocess.run
        self._timeout_seconds = float(timeout_seconds)
        self._outcomes: dict[str, bool] = {}
        self._lock = threading.Lock()

    def install(self, dependency: ProofReuseDependency) -> bool:
        allowed = PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(
            getattr(dependency, "module_name", "")
        )
        if allowed != dependency:
            return False
        with self._lock:
            previous = self._outcomes.get(dependency.module_name)
            if previous is not None:
                return previous
            command = (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                dependency.distribution,
            )
            try:
                completed = self._runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                )
                succeeded = getattr(completed, "returncode", 1) == 0
            except Exception:
                succeeded = False
            self._outcomes[dependency.module_name] = succeeded
            return succeeded


def _is_requested_module_absence(
    exc: ModuleNotFoundError,
    requested: str,
) -> bool:
    missing = str(getattr(exc, "name", "") or "")
    return bool(
        missing
        and (missing == requested or requested.startswith(f"{missing}."))
    )


def _installer_callable(installer: Any) -> Callable[[Any], Any] | None:
    install = getattr(installer, "install", None)
    if callable(install):
        return install
    if callable(installer):
        return installer
    return None


class LazyProofReuseServiceResolver:
    """Resolve and construct proof-reuse services exactly once.

    The importer and installer are injectable so unit tests and managed
    environments do not need network or process access.  No arbitrary module
    name or package spec can reach the installer.
    """

    def __init__(
        self,
        *,
        importer: Callable[[str], Any] | None = None,
        installer: Any = None,
    ) -> None:
        if importer is not None and not callable(importer):
            raise TypeError("importer must be callable")
        if installer is not None and _installer_callable(installer) is None:
            raise TypeError("installer must be callable or expose install()")
        self._importer = importer or importlib.import_module
        self._installer = installer
        self._resolution: ProofReuseServiceResolution | None = None
        self._lock = threading.Lock()

    def _load_dependency(
        self,
        dependency: ProofReuseDependency,
    ) -> tuple[Any | None, bool]:
        if (
            PROOF_REUSE_DEPENDENCY_ALLOWLIST.get(dependency.module_name)
            != dependency
        ):
            return None, False
        try:
            module = self._importer(dependency.module_name)
        except ModuleNotFoundError as exc:
            if not _is_requested_module_absence(
                exc,
                dependency.module_name,
            ):
                return None, False
            install = _installer_callable(self._installer)
            if install is None:
                return None, False
            try:
                installed = install(dependency) is True
            except Exception:
                return None, False
            if not installed:
                return None, False
            importlib.invalidate_caches()
            try:
                module = self._importer(dependency.module_name)
            except Exception:
                return None, False
            was_installed = True
        except Exception:
            return None, False
        else:
            was_installed = False

        if any(
            getattr(module, symbol, None) is None
            for symbol in dependency.required_symbols
        ):
            return None, False
        return module, was_installed

    def _resolve_once(
        self,
        cache_root: str | os.PathLike[str],
    ) -> ProofReuseServiceResolution:
        installed_modules: list[str] = []
        for dependency in _DEPENDENCIES:
            _module, installed = self._load_dependency(dependency)
            if _module is None:
                return ProofReuseServiceResolution.unavailable(
                    dependency.unavailable_reason
                )
            if installed:
                installed_modules.append(dependency.module_name)

        try:
            store_module = self._importer(STORE_MODULE)
            provider_module = self._importer(PROVIDER_MODULE)
            lookup_module = self._importer(LOOKUP_MODULE)
            store_type = store_module.TestCertificateStore
            provider_type = provider_module.IpfsDatasetsTestCertificateProvider
            lookup_type = lookup_module.ProofReuseLookup
        except Exception:
            return ProofReuseServiceResolution.unavailable(
                "plugin_unavailable"
            )

        try:
            provider = provider_type()
            capabilities = provider.capabilities()
            if getattr(capabilities, "prove_on_lookup", None) is not False:
                return ProofReuseServiceResolution.unavailable(
                    "certificate_provider_unavailable"
                )
        except Exception:
            return ProofReuseServiceResolution.unavailable(
                "certificate_provider_unavailable"
            )

        try:
            root = Path(cache_root)
            store = store_type(root)
            if not all(
                callable(getattr(store, name, None))
                for name in ("lookup", "put_candidate", "put_receipt")
            ):
                return ProofReuseServiceResolution.unavailable(
                    "cache_unavailable"
                )
        except Exception:
            return ProofReuseServiceResolution.unavailable(
                "cache_unavailable"
            )

        try:
            lookup = lookup_type(store=store, provider=provider)
        except Exception:
            return ProofReuseServiceResolution.unavailable(
                "plugin_unavailable"
            )
        return ProofReuseServiceResolution(
            available=True,
            reason_code="",
            lookup=lookup,
            store=store,
            provider=provider,
            installed_modules=tuple(installed_modules),
        )

    def resolve(
        self,
        *,
        cache_root: str | os.PathLike[str],
    ) -> ProofReuseServiceResolution:
        """Return the memoized all-or-nothing service bundle."""

        if self._resolution is not None:
            return self._resolution
        with self._lock:
            if self._resolution is None:
                self._resolution = self._resolve_once(cache_root)
        return self._resolution


__all__ = [
    "AllowlistedPipInstaller",
    "DATASETS_VERIFIER_DEPENDENCY",
    "DATASETS_VERIFIER_MODULE",
    "LOOKUP_MODULE",
    "LazyProofReuseServiceResolver",
    "MULTIFORMATS_DEPENDENCY",
    "MULTIFORMATS_MODULE",
    "PROOF_REUSE_AUTO_INSTALL_ENV",
    "PROOF_REUSE_CACHE_DIR_ENV",
    "PROOF_REUSE_DEPENDENCY_ALLOWLIST",
    "PROVIDER_MODULE",
    "ProofReuseDependency",
    "ProofReuseServiceResolution",
    "STORE_MODULE",
    "automatic_install_enabled",
]
