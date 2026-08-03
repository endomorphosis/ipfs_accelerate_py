"""Lazy, allowlisted service assembly for pytest proof reuse.

This module is deliberately inert when imported.  It does not inspect the
environment, create a cache, import an optional provider, install a package,
or perform a proof operation.  :mod:`.plugin` constructs a resolver only after
proof reuse has been explicitly enabled and an exact item identity needs a
lookup or publication.

The resolver has a closed dependency vocabulary.  An installer can only be
asked for one of those entries, and a failed import/install/construction
returns an unavailable bundle.  Callers must consequently execute tests.
"""

from __future__ import annotations

import hashlib
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
PROOF_REUSE_DATASETS_SOURCE_ENV: Final = (
    "IPFS_TEST_PROOF_REUSE_DATASETS_SOURCE"
)

DATASETS_VERIFIER_REVISION: Final = (
    "399297053dfb0ca8edec13e08dd8c8c299169f41"
)
DATASETS_VERIFIER_SOURCE_SHA256: Final = (
    "d6345cdb49840da9961ffd5550af128552afcc7d8b3ad1812dda6b9eb9b331fb"
)
DATASETS_VERIFIER_DISTRIBUTION: Final = "ipfs_datasets_py==0.2.0"
DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED: Final = False
DATASETS_VERIFIER_RELEASE_BLOCKER: Final = (
    "datasets_verifier_revision_unpublished"
)

MULTIFORMATS_MODULE: Final = "multiformats"
JSONSCHEMA_MODULE: Final = "jsonschema"
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
    pip_options: tuple[str, ...] = ()
    install_environment: tuple[tuple[str, str], ...] = ()
    packaging_source: str = "lazy_only"


MULTIFORMATS_DEPENDENCY: Final = ProofReuseDependency(
    module_name=MULTIFORMATS_MODULE,
    distribution="multiformats>=0.3,<1",
    required_symbols=("CID", "multihash"),
    unavailable_reason="cid_provider_unavailable",
    packaging_source="requirements-proof-reuse.txt",
)
JSONSCHEMA_DEPENDENCY: Final = ProofReuseDependency(
    module_name=JSONSCHEMA_MODULE,
    distribution="jsonschema>=4,<5",
    required_symbols=("validators",),
    unavailable_reason="certificate_provider_unavailable",
    packaging_source="requirements-proof-reuse.txt",
)
DATASETS_VERIFIER_DEPENDENCY: Final = ProofReuseDependency(
    module_name=DATASETS_VERIFIER_MODULE,
    distribution=DATASETS_VERIFIER_DISTRIBUTION,
    required_symbols=("verify_test_execution_certificate",),
    unavailable_reason="certificate_provider_unavailable",
    # ipfs_datasets_py has a reverse dependency on ipfs_accelerate_py.  A
    # normal dependency-resolving install would recurse through the whole
    # application stack and can select a second accelerator checkout.  The
    # verifier path is self-contained, so install exactly the reviewed local
    # provider source without resolving that reverse dependency.
    # Reinstall when an older/editable distribution claims the same 0.2.0
    # version but does not contain this verifier module.
    pip_options=(
        "--no-deps",
        "--force-reinstall",
        "--no-build-isolation",
    ),
    install_environment=(
        ("GIT_TERMINAL_PROMPT", "0"),
        ("IPFS_DATASETS_AUTO_INSTALL", "false"),
        ("IPFS_DATASETS_PY_AUTO_GROTH16_BUILD", "0"),
        ("IPFS_DATASETS_PY_AUTO_NLTK_DOWNLOAD", "0"),
        ("IPFS_DATASETS_PY_INCLUDE_VCS_DEPENDENCIES", "0"),
    ),
    packaging_source="reviewed_local_source_only",
)

_DEPENDENCIES: Final = (
    MULTIFORMATS_DEPENDENCY,
    JSONSCHEMA_DEPENDENCY,
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
    """Return the lazy-install policy without mutating anything.

    Installation is enabled by default, but it remains behind the active-use
    boundary: no resolver or installer is constructed until proof reuse is
    enabled and an exact lookup/publication needs its services.  Operators can
    disable every process/network attempt with
    ``IPFS_TEST_PROOF_REUSE_AUTO_INSTALL=0``.  Invalid values deny permission.
    """

    source = os.environ if environ is None else environ
    if PROOF_REUSE_AUTO_INSTALL_ENV not in source:
        return True
    value = str(source.get(PROOF_REUSE_AUTO_INSTALL_ENV, "")).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    # An invalid policy is never interpreted as permission to install.
    return False


def proof_reuse_dependency_plan(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe the complete bounded Python dependency plan.

    This is pure introspection: it does not import a provider, inspect package
    metadata, touch the cache, or start a process.  Groth16/ProveKit binaries,
    keys, circuits, endpoints, and shared caches are external capabilities,
    not Python packages; their absence continues to mean ``RUN``.
    """

    source = os.environ if environ is None else environ
    configured_datasets_source = bool(
        str(source.get(PROOF_REUSE_DATASETS_SOURCE_ENV, "")).strip()
    )
    return {
        "interface": "ProofReuseDependencyPlan@1",
        "lazy": True,
        "cold_import_inert": True,
        "fail_open_to_run": True,
        "automatic_install_enabled": automatic_install_enabled(environ),
        "disable_environment_variable": PROOF_REUSE_AUTO_INSTALL_ENV,
        "datasets_source_environment_variable": (
            PROOF_REUSE_DATASETS_SOURCE_ENV
        ),
        "datasets_requested_source": (
            "configured_local_path"
            if configured_datasets_source
            else "reviewed_integration_sibling"
        ),
        "datasets_reviewed_revision": DATASETS_VERIFIER_REVISION,
        "datasets_verifier_source_sha256": (
            DATASETS_VERIFIER_SOURCE_SHA256
        ),
        "remote_source_published": (
            DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED
        ),
        "release_blocker": DATASETS_VERIFIER_RELEASE_BLOCKER,
        "dependencies": [
            {
                "module_name": dependency.module_name,
                "distribution": dependency.distribution,
                "required_symbols": list(dependency.required_symbols),
                "pip_options": list(dependency.pip_options),
                "packaging_source": dependency.packaging_source,
            }
            for dependency in _DEPENDENCIES
        ],
        "external_capabilities": [
            "groth16_endpoint_or_binary",
            "groth16_verifying_key_and_circuit",
            "provekit_binary_and_artifacts",
            "shared_cache_or_local_cache_directory",
        ],
        "external_capability_absence_action": "run",
        "runtime_activation": {
            "automatic_plugin_discovery": True,
            "ordinary_enabled_run_effective_action": "run",
            "default_identity_services_injected": False,
            "default_identity_service_factory_configured": False,
            "production_identity_injector_configured": False,
            "required_identity_providers": [
                "repository_forest_provider",
                "analysis_index_provider",
                "component_inputs_provider",
                "policy_inputs_provider",
                "runtime_evidence_provider",
            ],
            "default_identity_compiler_available": True,
            "candidate_context_store_configured": False,
            "two_stage_candidate_revalidation_configured": False,
            "lookup_requires_exact_execution_key_before_candidate_read": True,
            "runtime_trace_attribute_producer_configured": False,
            "post_pass_runtime_trace_capture_configured": False,
            "post_pass_receipt_requires_runtime_trace": False,
            "deferred_request_builder_configured": False,
            "deferred_request_transport_compatible": False,
            "deferred_certificate_issuer_configured": False,
            "issuer_in_lazy_service_bundle": False,
            "issuer_in_lazy_service_resolution": False,
            "candidate_certificate_publication_configured": False,
            "authoritative_candidate_publication_configured": False,
            "receipt_content_identity_profiles_conformant": False,
            "receipt_content_identity_gap": (
                "accelerator_cidv1_dag_json_vs_datasets_sha256"
            ),
            "receipt_content_identity_profiles": {
                "accelerator": "cidv1-base32-dag-json-sha2-256",
                "datasets_statement": "sha256-canonical-json-v1",
                "exact_conformance": False,
            },
            "ordinary_warm_skip_path_complete": False,
            "missing_provider_action": "run",
            "completion_authority": False,
            "activation_blocker_codes": [
                "identity_services_unconfigured",
                "candidate_lookup_identity_cycle",
                "post_pass_runtime_trace_unproduced",
                "runtime_trace_not_required_for_receipt",
                "receipt_cid_profile_mismatch",
                "deferred_request_builder_unconfigured",
                "deferred_request_transport_type_mismatch",
                "issuer_unconfigured",
                "authoritative_candidate_not_published",
            ],
            "required_implementation_sequence": [
                {
                    "goals": ["PTR-G020", "PTR-G030", "PTR-G060"],
                    "work": "production_current_identity_provider_factory",
                },
                {
                    "goals": ["PTR-G030", "PTR-G060"],
                    "work": "controlled_current_runtime_preflight_provider",
                },
                {
                    "goals": ["PTR-G010", "PTR-G040", "PTR-G050"],
                    "work": "cross_package_receipt_cid_profile_conformance",
                },
                {
                    "goals": ["PTR-G040", "PTR-G050", "PTR-G060"],
                    "work": "deferred_request_issuer_and_candidate_publication",
                },
                {
                    "goals": [
                        "PTR-G060",
                        "PTR-G080",
                        "PTR-G090",
                        "PTR-G100",
                    ],
                    "work": "unwired_cross_repository_cold_warm_e2e",
                },
                {
                    "goals": ["PTR-G110"],
                    "work": "activated_warm_benchmark_and_rollout_evidence",
                },
            ],
        },
    }


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

    Nothing invokes this class unless proof reuse is enabled, an exact identity
    requires services, and automatic installation has not been disabled.
    Tests and embedding applications can inject a different installer instead.
    Each dependency receives at most one process attempt per installer instance.
    """

    def __init__(
        self,
        *,
        runner: Callable[..., Any] | None = None,
        timeout_seconds: float = 120.0,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 1 <= float(timeout_seconds) <= 600
        ):
            raise ValueError("timeout_seconds must be between 1 and 600")
        self._runner = runner or subprocess.run
        self._timeout_seconds = float(timeout_seconds)
        self._environ = dict(os.environ if environ is None else environ)
        self._outcomes: dict[str, bool] = {}
        self._lock = threading.Lock()

    def _selected_distribution(
        self,
        dependency: ProofReuseDependency,
    ) -> str | None:
        """Choose a validated local datasets checkout; never guess remotely."""

        if dependency.module_name != DATASETS_VERIFIER_MODULE:
            return dependency.distribution
        configured = str(
            self._environ.get(PROOF_REUSE_DATASETS_SOURCE_ENV, "")
        ).strip()
        if configured:
            return self._validated_local_datasets_distribution(
                configured,
                require_reviewed_revision=True,
            )
        try:
            integration_sibling = (
                Path(__file__).resolve().parents[3].parent / "ipfs_datasets"
            )
        except (IndexError, OSError, RuntimeError):
            return None
        return self._validated_local_datasets_distribution(
            integration_sibling,
            require_reviewed_revision=True,
        )

    @staticmethod
    def _detached_git_head(source: Path) -> str | None:
        """Read a detached submodule HEAD without starting a git process."""

        dot_git = source / ".git"
        try:
            if dot_git.is_file():
                pointer = dot_git.read_text(encoding="utf-8").strip()
                prefix = "gitdir:"
                if not pointer.lower().startswith(prefix):
                    return None
                git_dir = Path(pointer[len(prefix) :].strip())
                if not git_dir.is_absolute():
                    git_dir = (source / git_dir).resolve(strict=True)
            elif dot_git.is_dir():
                git_dir = dot_git
            else:
                return None
            head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        except (OSError, RuntimeError):
            return None
        if len(head) != 40 or any(
            character not in "0123456789abcdef" for character in head
        ):
            return None
        return head

    def _validated_local_datasets_distribution(
        self,
        configured: str | os.PathLike[str],
        *,
        require_reviewed_revision: bool,
    ) -> str | None:
        try:
            source = Path(configured).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            return None
        if not source.is_dir():
            return None
        if not any(
            (source / marker).is_file()
            for marker in ("pyproject.toml", "setup.py")
        ):
            return None
        verifier = (
            source
            / "ipfs_datasets_py"
            / "logic"
            / "zkp"
            / "test_execution_certificate.py"
        )
        if not verifier.is_file() or verifier.is_symlink():
            return None
        try:
            if verifier.stat().st_size > 2 * 1024 * 1024:
                return None
            verifier_digest = hashlib.sha256(verifier.read_bytes()).hexdigest()
        except OSError:
            return None
        if verifier_digest != DATASETS_VERIFIER_SOURCE_SHA256:
            return None
        if require_reviewed_revision and (
            self._detached_git_head(source) != DATASETS_VERIFIER_REVISION
        ):
            return None
        return f"ipfs_datasets_py @ {source.as_uri()}"

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
            distribution = self._selected_distribution(dependency)
            if distribution is None:
                self._outcomes[dependency.module_name] = False
                return False
            command = (
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                *dependency.pip_options,
                distribution,
            )
            run_environment = dict(self._environ)
            run_environment.update(dict(dependency.install_environment))
            try:
                completed = self._runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    env=run_environment,
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
    "DATASETS_VERIFIER_DISTRIBUTION",
    "DATASETS_VERIFIER_MODULE",
    "DATASETS_VERIFIER_REVISION",
    "DATASETS_VERIFIER_RELEASE_BLOCKER",
    "DATASETS_VERIFIER_REMOTE_SOURCE_PUBLISHED",
    "DATASETS_VERIFIER_SOURCE_SHA256",
    "LOOKUP_MODULE",
    "JSONSCHEMA_DEPENDENCY",
    "JSONSCHEMA_MODULE",
    "LazyProofReuseServiceResolver",
    "MULTIFORMATS_DEPENDENCY",
    "MULTIFORMATS_MODULE",
    "PROOF_REUSE_AUTO_INSTALL_ENV",
    "PROOF_REUSE_CACHE_DIR_ENV",
    "PROOF_REUSE_DATASETS_SOURCE_ENV",
    "PROOF_REUSE_DEPENDENCY_ALLOWLIST",
    "PROVIDER_MODULE",
    "ProofReuseDependency",
    "ProofReuseServiceResolution",
    "STORE_MODULE",
    "automatic_install_enabled",
    "proof_reuse_dependency_plan",
]
