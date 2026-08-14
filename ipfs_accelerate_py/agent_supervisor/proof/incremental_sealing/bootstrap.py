"""Hermetic IncrementalProofSealer bootstrap with explicit dependency injection (IPS-044).

Cold import of this module is inert: no files, keys, subprocesses, package
installs, network sockets, or daemon paths are opened.  Callers construct an
:class:`IncrementalSealingBootstrap` and inject every durable dependency
explicitly.  Store roots never default to ``~``, ``$XDG_*``, ``~/.ipfs``, or
any daemon-managed location.

Evidence: ``ips/import-hermeticity@1``.

Interfaces: ``IncrementalSealingBootstrap``, ``bootstrap_dependencies``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

IMPORT_HERMETICITY_EVIDENCE: Final[str] = "ips/import-hermeticity@1"
BOOTSTRAP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "hermetic-bootstrap@1"
)
BOOTSTRAP_INTERFACE: Final[str] = "IncrementalSealingBootstrap@1"
CONTRACT_VERSION: Final[int] = 1

# Paths that ordinary import and unbound construction must never touch.
FORBIDDEN_DEFAULT_ROOT_MARKERS: Final[tuple[str, ...]] = (
    "~",
    "$HOME",
    "$XDG_CACHE_HOME",
    "$XDG_CONFIG_HOME",
    "$XDG_DATA_HOME",
    "$XDG_STATE_HOME",
    "~/.ipfs",
    "$IPFS_PATH",
)


class BootstrapError(ValueError):
    """Fail-closed bootstrap contract violation."""


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class BootstrapDependencies:
    """Explicit dependency bundle for IncrementalProofSealer wiring.

    Every field is optional at construction.  Missing durable dependencies stay
    unbound until the caller supplies them; nothing is auto-discovered from the
    environment, home directory, or package install state.
    """

    store_root: Path | str | None = None
    branch_id: str = "main"
    create_store: bool = False
    store: Any | None = None
    wal: Any | None = None
    pointers: Any | None = None
    admission_policy: Any | None = None
    evidence_verifier: Any | None = None
    trusted_policy: Any | None = None
    crash_injector: Callable[..., Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _require_text(self.branch_id, "branch_id"))
        if type(self.create_store) is not bool:
            raise BootstrapError("create_store must be a boolean")
        if self.metadata is None or not isinstance(self.metadata, Mapping):
            raise BootstrapError("metadata must be a mapping")
        # Never silently expand a home/XDG/daemon default root.
        if self.store_root is not None:
            text = str(self.store_root).strip()
            if not text:
                raise BootstrapError("store_root must not be empty when provided")
            expanded_markers = ("~", "$HOME", "$XDG_", "$IPFS_PATH")
            if any(token in text for token in expanded_markers):
                raise BootstrapError(
                    "store_root must be an explicit absolute path; "
                    "home/XDG/IPFS_PATH expansion is forbidden at bootstrap"
                )


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    """Machine-readable status of a hermetic bootstrap attempt."""

    schema: str
    interface: str
    evidence: str
    bound: bool
    store_root: str | None
    branch_id: str
    create_store: bool
    has_store: bool
    has_wal: bool
    has_pointers: bool
    has_admission_policy: bool
    has_evidence_verifier: bool
    has_trusted_policy: bool
    import_side_effects: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence": self.evidence,
            "bound": self.bound,
            "store_root": self.store_root,
            "branch_id": self.branch_id,
            "create_store": self.create_store,
            "has_store": self.has_store,
            "has_wal": self.has_wal,
            "has_pointers": self.has_pointers,
            "has_admission_policy": self.has_admission_policy,
            "has_evidence_verifier": self.has_evidence_verifier,
            "has_trusted_policy": self.has_trusted_policy,
            "import_side_effects": list(self.import_side_effects),
            "reasons": list(self.reasons),
            "contract_version": CONTRACT_VERSION,
        }


class IncrementalSealingBootstrap:
    """Explicit dependency-injection surface for IncrementalProofSealer.

    Construction alone never opens storage, spawns processes, generates keys,
    installs packages, or contacts a network/daemon.  Durable components are
    built only when the caller requests them with an explicit store root.
    """

    __test__ = False

    def __init__(
        self,
        dependencies: BootstrapDependencies | None = None,
        **kwargs: Any,
    ) -> None:
        if dependencies is not None and kwargs:
            raise BootstrapError(
                "pass either BootstrapDependencies or keyword fields, not both"
            )
        if dependencies is None:
            dependencies = BootstrapDependencies(**kwargs)
        elif not isinstance(dependencies, BootstrapDependencies):
            raise BootstrapError("dependencies must be BootstrapDependencies")
        self._deps = dependencies
        # Cached instances remain None until explicitly built.
        self._sealer: Any | None = None
        self._verifier: Any | None = None

    @property
    def dependencies(self) -> BootstrapDependencies:
        return self._deps

    @property
    def bound(self) -> bool:
        return self._deps.store_root is not None or self._deps.store is not None

    def with_dependencies(
        self, dependencies: BootstrapDependencies
    ) -> IncrementalSealingBootstrap:
        """Return a new bootstrap bound to the given dependency bundle."""

        if not isinstance(dependencies, BootstrapDependencies):
            raise BootstrapError("dependencies must be BootstrapDependencies")
        return IncrementalSealingBootstrap(dependencies)

    def bind_store_root(
        self,
        store_root: Path | str,
        *,
        branch_id: str | None = None,
        create_store: bool | None = None,
    ) -> IncrementalSealingBootstrap:
        """Return a new bootstrap with an explicit store root (no I/O)."""

        deps = BootstrapDependencies(
            store_root=store_root,
            branch_id=branch_id if branch_id is not None else self._deps.branch_id,
            create_store=(
                create_store if create_store is not None else self._deps.create_store
            ),
            store=self._deps.store,
            wal=self._deps.wal,
            pointers=self._deps.pointers,
            admission_policy=self._deps.admission_policy,
            evidence_verifier=self._deps.evidence_verifier,
            trusted_policy=self._deps.trusted_policy,
            crash_injector=self._deps.crash_injector,
            metadata=dict(self._deps.metadata),
        )
        return IncrementalSealingBootstrap(deps)

    def report(self) -> BootstrapReport:
        """Describe the current binding without performing side effects."""

        reasons: list[str] = []
        if not self.bound:
            reasons.append("store_root and store are unbound; no durable I/O possible")
        else:
            reasons.append("explicit store binding present")
        reasons.append("import and construction remain free of network/process/key I/O")
        return BootstrapReport(
            schema=BOOTSTRAP_SCHEMA,
            interface=BOOTSTRAP_INTERFACE,
            evidence=IMPORT_HERMETICITY_EVIDENCE,
            bound=self.bound,
            store_root=(
                str(self._deps.store_root) if self._deps.store_root is not None else None
            ),
            branch_id=self._deps.branch_id,
            create_store=self._deps.create_store,
            has_store=self._deps.store is not None,
            has_wal=self._deps.wal is not None,
            has_pointers=self._deps.pointers is not None,
            has_admission_policy=self._deps.admission_policy is not None,
            has_evidence_verifier=self._deps.evidence_verifier is not None,
            has_trusted_policy=self._deps.trusted_policy is not None,
            import_side_effects=(),
            reasons=tuple(reasons),
        )

    def build_evidence_verifier(self) -> Any:
        """Construct or return the injected evidence verifier (lazy import)."""

        if self._deps.evidence_verifier is not None:
            self._verifier = self._deps.evidence_verifier
            return self._verifier
        if self._verifier is not None:
            return self._verifier
        from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
            EvidenceVerifier,
        )

        self._verifier = EvidenceVerifier(policy=self._deps.admission_policy)
        return self._verifier

    def build_sealer(self) -> Any:
        """Construct the atomic sealer from injected dependencies.

        Requires an explicit ``store_root`` or pre-built ``store``.  This is the
        first point at which durable store I/O may occur, and only when the
        caller opts in via ``create_store`` or an already-materialized store.
        """

        if self._sealer is not None:
            return self._sealer
        if self._deps.store is None and self._deps.store_root is None:
            raise BootstrapError(
                "build_sealer requires an explicit store_root or injected store; "
                "no home/XDG/daemon default is available"
            )
        from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer import (
            IncrementalProofSealer,
        )

        self._sealer = IncrementalProofSealer(
            self._deps.store_root,
            branch_id=self._deps.branch_id,
            create=self._deps.create_store,
            crash_injector=self._deps.crash_injector,
            store=self._deps.store,
            wal=self._deps.wal,
            pointers=self._deps.pointers,
        )
        return self._sealer

    def build_migration_context(self) -> Mapping[str, Any]:
        """Return a pure mapping describing migration wiring (no I/O)."""

        return {
            "schema": BOOTSTRAP_SCHEMA,
            "evidence": IMPORT_HERMETICITY_EVIDENCE,
            "bound": self.bound,
            "admission_policy_present": self._deps.admission_policy is not None,
            "verifier_present": (
                self._deps.evidence_verifier is not None or self._verifier is not None
            ),
            "cache_admission_requires_current_policy_verification": True,
            "forbidden_default_roots": list(FORBIDDEN_DEFAULT_ROOT_MARKERS),
        }


def bootstrap_dependencies(**kwargs: Any) -> BootstrapDependencies:
    """Construct a :class:`BootstrapDependencies` bundle (pure; no I/O)."""

    return BootstrapDependencies(**kwargs)


def hermetic_bootstrap(
    dependencies: BootstrapDependencies | None = None,
    **kwargs: Any,
) -> IncrementalSealingBootstrap:
    """Return an unbound or explicitly bound hermetic bootstrap (no I/O)."""

    return IncrementalSealingBootstrap(dependencies, **kwargs)


def assert_import_is_hermetic() -> dict[str, Any]:
    """Return the static hermeticity contract for ordinary package import.

    This helper is pure and does not probe the filesystem or network.  Tests
    that need runtime enforcement should combine it with subprocess isolation.
    """

    return {
        "schema": BOOTSTRAP_SCHEMA,
        "evidence": IMPORT_HERMETICITY_EVIDENCE,
        "interface": BOOTSTRAP_INTERFACE,
        "ordinary_import_side_effects": [],
        "forbidden_on_import": [
            "file_create",
            "key_generate",
            "key_download",
            "subprocess",
            "package_install",
            "network_access",
            "daemon_dependency",
            "home_or_xdg_default_store",
        ],
        "store_root_policy": "explicit_injection_only",
        "contract_version": CONTRACT_VERSION,
    }


__all__ = (
    "BOOTSTRAP_INTERFACE",
    "BOOTSTRAP_SCHEMA",
    "CONTRACT_VERSION",
    "FORBIDDEN_DEFAULT_ROOT_MARKERS",
    "IMPORT_HERMETICITY_EVIDENCE",
    "BootstrapDependencies",
    "BootstrapError",
    "BootstrapReport",
    "IncrementalSealingBootstrap",
    "assert_import_is_hermetic",
    "bootstrap_dependencies",
    "hermetic_bootstrap",
)
