"""Runtime adapters for datasets, harness, verification, storage, and sealer (SCG-023).

Normalizes frozen canonical surfaces into narrow governor runtime views.
Capability probes are lazy version/fingerprint checks that return typed
unavailable results. Missing, stale, or incompatible capability fails closed.

The incremental verification planner Merkle ``VerificationCommitment`` is
structural non-ZK evidence only. It can never satisfy
:class:`IncrementalSealerCapability` or stand in for a released
``IncrementalProofSealer`` / full-checkpoint / delta seal public API.

Importing this module performs no I/O, opens no sockets or processes, and
loads none of the upstream implementation packages until a probe or adapter
construction explicitly requests them.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Evidence / interface constants
# ---------------------------------------------------------------------------

SCG_RUNTIME_ADAPTERS_EVIDENCE: Final[str] = "scg/runtime-adapters@1"

DATASETS_ADAPTER_ID: Final[str] = "scg-governor-datasets-adapter@1"
HARNESS_ADAPTER_ID: Final[str] = "scg-governor-harness-adapter@1"
VERIFICATION_ADAPTER_ID: Final[str] = "scg-governor-verification-adapter@1"
STORE_ADAPTER_ID: Final[str] = "scg-governor-store-adapter@1"
SEALER_CAPABILITY_ID: Final[str] = "scg-incremental-sealer-capability@1"

DATASETS_ADAPTER_INTERFACE: Final[str] = "GovernorDatasetsAdapter@1"
HARNESS_ADAPTER_INTERFACE: Final[str] = "GovernorHarnessAdapter@1"
VERIFICATION_ADAPTER_INTERFACE: Final[str] = "GovernorVerificationAdapter@1"
STORE_ADAPTER_INTERFACE: Final[str] = "GovernorStoreAdapter@1"
SEALER_CAPABILITY_INTERFACE: Final[str] = "IncrementalSealerCapability@1"

# Sealed upstream pins (must match SCG-018 / harness / kit / IVP freezes).
EXPECTED_DATASETS_API_SCHEMA: Final[str] = (
    "ipfs-datasets.software-contracts.semantic-governor-public-api@1"
)
EXPECTED_DATASETS_PACKAGE_INTERFACE: Final[str] = "SemanticGovernorPublicApi@1"
EXPECTED_DATASETS_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_governor"
)

EXPECTED_HARNESS_INTERFACE: Final[str] = "SemanticCompressionHarness@1"
EXPECTED_HARNESS_SCHEMA: Final[str] = (
    "ipfs-accelerate.semantic-compression-harness@1"
)
EXPECTED_HARNESS_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_state.harness"
)
EXPECTED_HARNESS_PACKAGE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_state"
)

EXPECTED_STORE_INTERFACE: Final[str] = "SemanticGovernorStore@1"
EXPECTED_STORE_SCHEMA: Final[str] = (
    "ipfs-kit.semantic-governor-store.contracts@1"
)
EXPECTED_STORE_ARTIFACT_INTERFACE: Final[str] = "DurableSemanticGovernorStore@1"
EXPECTED_STORE_CONTRACTS_MODULE: Final[str] = (
    "ipfs_kit_py.semantic_governor_store.contracts"
)
EXPECTED_STORE_ARTIFACTS_MODULE: Final[str] = (
    "ipfs_kit_py.semantic_governor_store.artifacts"
)
EXPECTED_STORE_PACKAGE: Final[str] = "ipfs_kit_py.semantic_governor_store"

EXPECTED_VERIFICATION_PACKAGE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification"
)
EXPECTED_VERIFICATION_CONTRACTS_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.contracts"
)
EXPECTED_VERIFICATION_COMMITMENT_INTERFACE: Final[str] = (
    "VerificationCommitment@1"
)
EXPECTED_VERIFICATION_COMMITMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-commitment@1"
)
EXPECTED_VERIFICATION_PLAN_INTERFACE: Final[str] = "VerificationPlan@1"
EXPECTED_VERIFICATION_BUNDLE_INTERFACE: Final[str] = "VerificationBundle@1"

# Required public analysis entry points from SCG-018.
REQUIRED_DATASETS_APIS: Final[tuple[str, ...]] = (
    "build_context_coverage_manifest",
    "evaluate_context_sufficiency",
    "diagnose_omission",
    "plan_context_expansion",
    "update_calibration",
    "propose_rule_change",
)
SUPPORTING_DATASETS_APIS: Final[tuple[str, ...]] = (
    "detect_instruction_like_content",
    "apply_trusted_decision",
    "merge_calibration_profiles",
    "validate_rule_proposal",
)

REQUIRED_HARNESS_EXPORTS: Final[tuple[str, ...]] = (
    "SemanticCompressionHarness",
    "run_semantic_patch_loop",
    "HARNESS_LOOP_INTERFACE",
    "HARNESS_LOOP_SCHEMA",
)

REQUIRED_VERIFICATION_EXPORTS: Final[tuple[str, ...]] = (
    "create_verification_plan",
    "IncrementalVerificationPlanner",
    "build_verification_commitment",
    "VerificationCommitment",
    "VerificationBundle",
    "VerificationPlan",
    "VerificationReceiptCache",
    "choose_model_route",
)

REQUIRED_STORE_CONTRACT_EXPORTS: Final[tuple[str, ...]] = (
    "SEMANTIC_GOVERNOR_STORE_INTERFACE",
    "SEMANTIC_GOVERNOR_STORE_SCHEMA",
    "SemanticGovernorStore",
)
REQUIRED_STORE_ARTIFACT_EXPORTS: Final[tuple[str, ...]] = (
    "ARTIFACT_MODULE_INTERFACE",
    "DurableSemanticGovernorStore",
)

# Candidate *public* modules that may host a released sealer. Never probe
# unfinished private IPS development packages.
_SEALER_PUBLIC_CANDIDATE_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.proof_sealer",
    "ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer",
    "ipfs_kit_py.proof_sealer",
    "ipfs_kit_py.incremental_proof_sealer",
)

_SEALER_PUBLIC_SYMBOLS: Final[tuple[str, ...]] = (
    "IncrementalProofSealer",
    "DeltaSeal",
    "build_delta_seal",
    "publish_delta_seal",
    "FullCheckpointSeal",
    "create_full_checkpoint",
    "publish_full_checkpoint",
)

# Symbols / interfaces that are explicitly not sealers.
_IVP_NON_SEALER_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "VerificationCommitment",
        "build_verification_commitment",
        "VERIFICATION_COMMITMENT_INTERFACE",
        "VERIFICATION_COMMITMENT_SCHEMA",
    }
)

_MAX_DIAGNOSTIC: Final[int] = 512


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class CapabilityStatus(str, Enum):
    """Closed capability disposition."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INCOMPATIBLE = "incompatible"
    STALE = "stale"
    MISSING = "missing"


class SealStatus(str, Enum):
    """Closed seal status for governor artifacts."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Typed failures
# ---------------------------------------------------------------------------


class GovernorAdapterError(ValueError):
    """Closed adapter validation failure (schema/export/binding)."""


class GovernorCapabilityUnavailable(RuntimeError):
    """Required governor surface capability is missing, stale, or incompatible.

    Callers must treat this as fail-closed unavailability, never as success or
    as an implicit fallback to a weaker evidence kind.
    """

    def __init__(
        self,
        operation: str,
        reason_code: str,
        diagnostic: str,
        *,
        adapter_id: str,
        retryable: bool = False,
        status: str = CapabilityStatus.UNAVAILABLE.value,
    ) -> None:
        self.operation = operation
        self.reason_code = _token(reason_code, "reason_code")
        self.diagnostic = str(diagnostic)[:_MAX_DIAGNOSTIC]
        self.adapter_id = adapter_id
        self.retryable = bool(retryable)
        self.status = _enum_value(status, CapabilityStatus, "status")
        super().__init__(
            f"{adapter_id}:{operation}:{self.reason_code}: {self.diagnostic}"
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "adapter_id": self.adapter_id,
                "operation": self.operation,
                "reason_code": self.reason_code,
                "diagnostic": self.diagnostic,
                "retryable": self.retryable,
                "status": self.status,
                "available": False,
            }
        )


# ---------------------------------------------------------------------------
# Capability records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurfaceCapability:
    """Closed capability witness for one adapted surface."""

    available: bool
    adapter_id: str
    interface_id: str
    schema: str
    operations: tuple[str, ...]
    fingerprints: Mapping[str, str]
    status: str = CapabilityStatus.AVAILABLE.value
    reason_code: str | None = None
    diagnostic: str | None = None
    retryable: bool = False

    def require_available(self, operation: str) -> None:
        if not self.available:
            raise GovernorCapabilityUnavailable(
                operation,
                self.reason_code or "capability_unavailable",
                self.diagnostic or f"{self.adapter_id} unavailable",
                adapter_id=self.adapter_id,
                retryable=self.retryable,
                status=self.status,
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "available": self.available,
                "adapter_id": self.adapter_id,
                "interface_id": self.interface_id,
                "schema": self.schema,
                "operations": list(self.operations),
                "fingerprints": dict(self.fingerprints),
                "status": self.status,
                "reason_code": self.reason_code,
                "diagnostic": self.diagnostic,
                "retryable": self.retryable,
            }
        )


@dataclass(frozen=True, slots=True)
class IncrementalSealerCapability:
    """Released incremental / full-checkpoint proof-sealer capability.

    Normative invariants:

    * Missing sealer → ``seal_status=unavailable`` (typed, not silent).
    * IVP ``VerificationCommitment`` can never set ``available=True``.
    * ``is_zk`` is only true when a released sealer public API claims ZK.
    * Unfinished private IPS modules are never imported.
    """

    available: bool
    adapter_id: str = SEALER_CAPABILITY_ID
    interface_id: str = SEALER_CAPABILITY_INTERFACE
    seal_status: str = SealStatus.UNAVAILABLE.value
    status: str = CapabilityStatus.UNAVAILABLE.value
    is_zk: bool = False
    is_full_or_delta_seal: bool = False
    can_be_satisfied_by_ivp_commitment: bool = False
    operations: tuple[str, ...] = ()
    fingerprints: Mapping[str, str] = MappingProxyType({})
    public_module: str | None = None
    reason_code: str | None = "sealer_unavailable"
    diagnostic: str | None = (
        "released IncrementalProofSealer public API is not present"
    )
    retryable: bool = False

    def __post_init__(self) -> None:
        # Force the IVP non-substitution invariant even if a caller forges fields.
        object.__setattr__(self, "can_be_satisfied_by_ivp_commitment", False)
        if self.available and self.seal_status != SealStatus.AVAILABLE.value:
            object.__setattr__(self, "seal_status", SealStatus.AVAILABLE.value)
        if not self.available and self.seal_status == SealStatus.AVAILABLE.value:
            object.__setattr__(self, "seal_status", SealStatus.UNAVAILABLE.value)

    def require_available(self, operation: str = "seal") -> None:
        if not self.available:
            raise GovernorCapabilityUnavailable(
                operation,
                self.reason_code or "sealer_unavailable",
                self.diagnostic or "incremental sealer unavailable",
                adapter_id=self.adapter_id,
                retryable=self.retryable,
                status=self.status,
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "available": self.available,
                "adapter_id": self.adapter_id,
                "interface_id": self.interface_id,
                "seal_status": self.seal_status,
                "status": self.status,
                "is_zk": self.is_zk,
                "is_full_or_delta_seal": self.is_full_or_delta_seal,
                "can_be_satisfied_by_ivp_commitment": False,
                "operations": list(self.operations),
                "fingerprints": dict(self.fingerprints),
                "public_module": self.public_module,
                "reason_code": self.reason_code,
                "diagnostic": self.diagnostic,
                "retryable": self.retryable,
            }
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _token(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GovernorAdapterError(f"{name} must be a nonempty trimmed string")
    if any(ch.isspace() for ch in value):
        raise GovernorAdapterError(f"{name} must not contain whitespace")
    if len(value) > 256:
        raise GovernorAdapterError(f"{name} exceeds 256 characters")
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise GovernorAdapterError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _text_pin(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise GovernorAdapterError(f"{name} must be a nonempty trimmed string")
    return value


def _attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _require_exports(module: Any, names: Sequence[str], label: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise GovernorCapabilityUnavailable(
            "load",
            "missing_exports",
            f"{label} missing required exports: {', '.join(missing)}",
            adapter_id=label,
            retryable=False,
            status=CapabilityStatus.MISSING.value,
        )


def _check_pin(
    actual: Any,
    expected: str,
    name: str,
    *,
    adapter_id: str,
    stale_if_older: bool = True,
) -> None:
    text = _text_pin(actual, name)
    if text == expected:
        return
    # Versioned "@N" mismatch: treat lower major as stale, otherwise incompatible.
    actual_major = _schema_major(text)
    expected_major = _schema_major(expected)
    if (
        stale_if_older
        and actual_major is not None
        and expected_major is not None
        and actual_major < expected_major
    ):
        raise GovernorCapabilityUnavailable(
            "load",
            "stale_capability",
            f"{name} is {text!r}, expected sealed {expected!r}",
            adapter_id=adapter_id,
            retryable=False,
            status=CapabilityStatus.STALE.value,
        )
    raise GovernorCapabilityUnavailable(
        "load",
        "incompatible_capability",
        f"{name} is {text!r}, expected sealed {expected!r}",
        adapter_id=adapter_id,
        retryable=False,
        status=CapabilityStatus.INCOMPATIBLE.value,
    )


def _schema_major(value: str) -> int | None:
    if "@" not in value:
        return None
    suffix = value.rsplit("@", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _unavailable_capability(
    *,
    adapter_id: str,
    interface_id: str,
    schema: str,
    reason_code: str,
    diagnostic: str,
    status: str = CapabilityStatus.UNAVAILABLE.value,
    retryable: bool = False,
    operations: tuple[str, ...] = (),
    fingerprints: Mapping[str, str] | None = None,
) -> SurfaceCapability:
    return SurfaceCapability(
        available=False,
        adapter_id=adapter_id,
        interface_id=interface_id,
        schema=schema,
        operations=operations,
        fingerprints=MappingProxyType(dict(fingerprints or {})),
        status=status,
        reason_code=reason_code,
        diagnostic=diagnostic[:_MAX_DIAGNOSTIC],
        retryable=retryable,
    )


def _import_module(module_name: str, *, adapter_id: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # ImportError and ambient layout failures
        raise GovernorCapabilityUnavailable(
            "load",
            "import_failed",
            f"import of {module_name!r} failed: {exc}",
            adapter_id=adapter_id,
            retryable=True,
            status=CapabilityStatus.MISSING.value,
        ) from exc


def _try_import_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _callable_map(
    module: Any, names: Iterable[str]
) -> Mapping[str, Callable[..., Any]]:
    out: dict[str, Callable[..., Any]] = {}
    for name in names:
        value = _attr(module, name)
        if not callable(value):
            raise GovernorCapabilityUnavailable(
                "load",
                "missing_exports",
                f"required export {name!r} is not callable",
                adapter_id=DATASETS_ADAPTER_ID,
                retryable=False,
                status=CapabilityStatus.MISSING.value,
            )
        out[name] = value
    return MappingProxyType(out)


# ---------------------------------------------------------------------------
# Datasets adapter
# ---------------------------------------------------------------------------


def probe_datasets_capability(
    *, surface: Any | None = None
) -> SurfaceCapability:
    """Probe the datasets semantic-governor public API (lazy, fail-closed)."""

    try:
        module = surface if surface is not None else _import_module(
            EXPECTED_DATASETS_MODULE, adapter_id=DATASETS_ADAPTER_ID
        )
        _require_exports(
            module,
            (
                "SEMANTIC_GOVERNOR_API_SCHEMA",
                "SEMANTIC_GOVERNOR_PACKAGE_INTERFACE",
                "REQUIRED_PUBLIC_APIS",
            )
            + REQUIRED_DATASETS_APIS,
            DATASETS_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "SEMANTIC_GOVERNOR_API_SCHEMA"),
            EXPECTED_DATASETS_API_SCHEMA,
            "SEMANTIC_GOVERNOR_API_SCHEMA",
            adapter_id=DATASETS_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "SEMANTIC_GOVERNOR_PACKAGE_INTERFACE"),
            EXPECTED_DATASETS_PACKAGE_INTERFACE,
            "SEMANTIC_GOVERNOR_PACKAGE_INTERFACE",
            adapter_id=DATASETS_ADAPTER_ID,
        )
        declared = tuple(_attr(module, "REQUIRED_PUBLIC_APIS") or ())
        missing = [name for name in REQUIRED_DATASETS_APIS if name not in declared]
        if missing:
            raise GovernorCapabilityUnavailable(
                "load",
                "missing_exports",
                f"REQUIRED_PUBLIC_APIS missing {', '.join(missing)}",
                adapter_id=DATASETS_ADAPTER_ID,
                retryable=False,
                status=CapabilityStatus.MISSING.value,
            )
        ops = tuple(REQUIRED_DATASETS_APIS) + tuple(
            name for name in SUPPORTING_DATASETS_APIS if hasattr(module, name)
        )
        fingerprints = MappingProxyType(
            {
                "api_schema": EXPECTED_DATASETS_API_SCHEMA,
                "package_interface": EXPECTED_DATASETS_PACKAGE_INTERFACE,
                "module": EXPECTED_DATASETS_MODULE,
            }
        )
        return SurfaceCapability(
            available=True,
            adapter_id=DATASETS_ADAPTER_ID,
            interface_id=DATASETS_ADAPTER_INTERFACE,
            schema=EXPECTED_DATASETS_API_SCHEMA,
            operations=ops,
            fingerprints=fingerprints,
            status=CapabilityStatus.AVAILABLE.value,
        )
    except GovernorCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=DATASETS_ADAPTER_ID,
            interface_id=DATASETS_ADAPTER_INTERFACE,
            schema=EXPECTED_DATASETS_API_SCHEMA,
            reason_code=exc.reason_code,
            diagnostic=exc.diagnostic,
            status=exc.status,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_DATASETS_MODULE},
        )


class GovernorDatasetsAdapter:
    """Narrow runtime view over the datasets semantic-governor public API."""

    __slots__ = ("_surface", "_capability", "_apis")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_datasets_capability(
                surface=surface
            )
        else:
            self._surface = None
            self._capability = capability
        self._apis: Mapping[str, Callable[..., Any]] | None = None

    @property
    def capability(self) -> SurfaceCapability:
        if self._capability is None:
            self._ensure_loaded()
        assert self._capability is not None
        return self._capability

    def _ensure_loaded(self) -> Any:
        if self._surface is None:
            self._surface = _import_module(
                EXPECTED_DATASETS_MODULE, adapter_id=DATASETS_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_datasets_capability(surface=self._surface)
        self._capability.require_available("load")
        if self._apis is None:
            self._apis = _callable_map(self._surface, REQUIRED_DATASETS_APIS)
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    def api(self, name: str) -> Callable[..., Any]:
        self._ensure_loaded()
        assert self._apis is not None
        if name not in self._apis:
            # Supporting APIs may be present without being primary.
            surface = self._surface
            value = _attr(surface, name)
            if not callable(value):
                raise GovernorCapabilityUnavailable(
                    name,
                    "missing_exports",
                    f"datasets API {name!r} is not available",
                    adapter_id=DATASETS_ADAPTER_ID,
                    retryable=False,
                    status=CapabilityStatus.MISSING.value,
                )
            return value
        return self._apis[name]

    def evaluate_context_sufficiency(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("evaluate_context_sufficiency")(*args, **kwargs)

    def diagnose_omission(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("diagnose_omission")(*args, **kwargs)

    def plan_context_expansion(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("plan_context_expansion")(*args, **kwargs)

    def update_calibration(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("update_calibration")(*args, **kwargs)

    def propose_rule_change(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("propose_rule_change")(*args, **kwargs)

    def build_context_coverage_manifest(self, *args: Any, **kwargs: Any) -> Any:
        return self.api("build_context_coverage_manifest")(*args, **kwargs)

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": DATASETS_ADAPTER_ID,
                "interface_id": DATASETS_ADAPTER_INTERFACE,
                "capability": cap.to_mapping(),
                "operations": list(cap.operations),
                "module": EXPECTED_DATASETS_MODULE,
            }
        )


def load_datasets_adapter(surface: Any | None = None) -> GovernorDatasetsAdapter:
    """Load the datasets adapter and fail closed if capability is unavailable."""

    adapter = GovernorDatasetsAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Harness adapter
# ---------------------------------------------------------------------------


def probe_harness_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe the semantic-compression harness surface."""

    try:
        module = surface if surface is not None else _import_module(
            EXPECTED_HARNESS_MODULE, adapter_id=HARNESS_ADAPTER_ID
        )
        _require_exports(module, REQUIRED_HARNESS_EXPORTS, HARNESS_ADAPTER_ID)
        _check_pin(
            _attr(module, "HARNESS_LOOP_INTERFACE"),
            EXPECTED_HARNESS_INTERFACE,
            "HARNESS_LOOP_INTERFACE",
            adapter_id=HARNESS_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "HARNESS_LOOP_SCHEMA"),
            EXPECTED_HARNESS_SCHEMA,
            "HARNESS_LOOP_SCHEMA",
            adapter_id=HARNESS_ADAPTER_ID,
        )
        harness_cls = _attr(module, "SemanticCompressionHarness")
        if not isinstance(harness_cls, type):
            raise GovernorCapabilityUnavailable(
                "load",
                "missing_exports",
                "SemanticCompressionHarness is not a type",
                adapter_id=HARNESS_ADAPTER_ID,
                retryable=False,
                status=CapabilityStatus.MISSING.value,
            )
        fingerprints = MappingProxyType(
            {
                "interface": EXPECTED_HARNESS_INTERFACE,
                "schema": EXPECTED_HARNESS_SCHEMA,
                "module": EXPECTED_HARNESS_MODULE,
            }
        )
        return SurfaceCapability(
            available=True,
            adapter_id=HARNESS_ADAPTER_ID,
            interface_id=HARNESS_ADAPTER_INTERFACE,
            schema=EXPECTED_HARNESS_SCHEMA,
            operations=REQUIRED_HARNESS_EXPORTS,
            fingerprints=fingerprints,
            status=CapabilityStatus.AVAILABLE.value,
        )
    except GovernorCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=HARNESS_ADAPTER_ID,
            interface_id=HARNESS_ADAPTER_INTERFACE,
            schema=EXPECTED_HARNESS_SCHEMA,
            reason_code=exc.reason_code,
            diagnostic=exc.diagnostic,
            status=exc.status,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_HARNESS_MODULE},
        )


class GovernorHarnessAdapter:
    """Narrow runtime view over ``SemanticCompressionHarness``."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_harness_capability(
                surface=surface
            )
        else:
            self._surface = None
            self._capability = capability

    @property
    def capability(self) -> SurfaceCapability:
        if self._capability is None:
            self._ensure_loaded()
        assert self._capability is not None
        return self._capability

    def _ensure_loaded(self) -> Any:
        if self._surface is None:
            self._surface = _import_module(
                EXPECTED_HARNESS_MODULE, adapter_id=HARNESS_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_harness_capability(surface=self._surface)
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def harness_class(self) -> type:
        surface = self._ensure_loaded()
        return surface.SemanticCompressionHarness

    def run_semantic_patch_loop(self, *args: Any, **kwargs: Any) -> Any:
        surface = self._ensure_loaded()
        return surface.run_semantic_patch_loop(*args, **kwargs)

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": HARNESS_ADAPTER_ID,
                "interface_id": HARNESS_ADAPTER_INTERFACE,
                "capability": cap.to_mapping(),
                "harness_interface": EXPECTED_HARNESS_INTERFACE,
                "harness_schema": EXPECTED_HARNESS_SCHEMA,
                "module": EXPECTED_HARNESS_MODULE,
            }
        )


def load_harness_adapter(surface: Any | None = None) -> GovernorHarnessAdapter:
    adapter = GovernorHarnessAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Verification adapter
# ---------------------------------------------------------------------------


def probe_verification_capability(
    *, surface: Any | None = None
) -> SurfaceCapability:
    """Probe the incremental-verification public surface."""

    try:
        module = surface if surface is not None else _import_module(
            EXPECTED_VERIFICATION_PACKAGE, adapter_id=VERIFICATION_ADAPTER_ID
        )
        _require_exports(
            module, REQUIRED_VERIFICATION_EXPORTS, VERIFICATION_ADAPTER_ID
        )
        # Commitment pins live on contracts; package re-exports may be lazy.
        contracts = _attr(module, "contracts")
        if contracts is None:
            contracts = _try_import_module(EXPECTED_VERIFICATION_CONTRACTS_MODULE)
        if contracts is not None:
            iface = _attr(contracts, "VERIFICATION_COMMITMENT_INTERFACE")
            schema = _attr(contracts, "VERIFICATION_COMMITMENT_SCHEMA")
            if iface is not None:
                _check_pin(
                    iface,
                    EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
                    "VERIFICATION_COMMITMENT_INTERFACE",
                    adapter_id=VERIFICATION_ADAPTER_ID,
                )
            if schema is not None:
                _check_pin(
                    schema,
                    EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
                    "VERIFICATION_COMMITMENT_SCHEMA",
                    adapter_id=VERIFICATION_ADAPTER_ID,
                )
        fingerprints = MappingProxyType(
            {
                "commitment_interface": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
                "commitment_schema": EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
                "package": EXPECTED_VERIFICATION_PACKAGE,
                "is_proof_sealer": "false",
                "is_zk": "false",
            }
        )
        return SurfaceCapability(
            available=True,
            adapter_id=VERIFICATION_ADAPTER_ID,
            interface_id=VERIFICATION_ADAPTER_INTERFACE,
            schema=EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            operations=REQUIRED_VERIFICATION_EXPORTS,
            fingerprints=fingerprints,
            status=CapabilityStatus.AVAILABLE.value,
        )
    except GovernorCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=VERIFICATION_ADAPTER_ID,
            interface_id=VERIFICATION_ADAPTER_INTERFACE,
            schema=EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            reason_code=exc.reason_code,
            diagnostic=exc.diagnostic,
            status=exc.status,
            retryable=exc.retryable,
            fingerprints={"package": EXPECTED_VERIFICATION_PACKAGE},
        )


class GovernorVerificationAdapter:
    """Narrow runtime view over IVP planning, bundles, and commitments.

    The adapter exposes structural verification evidence. It never promotes a
    ``VerificationCommitment`` to sealer or ZK authority.
    """

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_verification_capability(
                surface=surface
            )
        else:
            self._surface = None
            self._capability = capability

    @property
    def capability(self) -> SurfaceCapability:
        if self._capability is None:
            self._ensure_loaded()
        assert self._capability is not None
        return self._capability

    def _ensure_loaded(self) -> Any:
        if self._surface is None:
            self._surface = _import_module(
                EXPECTED_VERIFICATION_PACKAGE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
        if self._capability is None:
            self._capability = probe_verification_capability(
                surface=self._surface
            )
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    def create_verification_plan(self, *args: Any, **kwargs: Any) -> Any:
        surface = self._ensure_loaded()
        return surface.create_verification_plan(*args, **kwargs)

    def build_verification_commitment(self, *args: Any, **kwargs: Any) -> Any:
        """Build a structural non-ZK commitment (never a sealer)."""

        surface = self._ensure_loaded()
        return surface.build_verification_commitment(*args, **kwargs)

    def choose_model_route(self, *args: Any, **kwargs: Any) -> Any:
        surface = self._ensure_loaded()
        return surface.choose_model_route(*args, **kwargs)

    def commitment_is_proof_sealer(self) -> bool:
        """IVP commitments are never proof sealers."""

        return False

    def commitment_is_zk(self) -> bool:
        """IVP commitments are explicitly non-ZK."""

        return False

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": VERIFICATION_ADAPTER_ID,
                "interface_id": VERIFICATION_ADAPTER_INTERFACE,
                "capability": cap.to_mapping(),
                "commitment_interface": EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
                "commitment_is_proof_sealer": False,
                "commitment_is_zk": False,
                "can_satisfy_sealer_capability": False,
                "package": EXPECTED_VERIFICATION_PACKAGE,
            }
        )


def load_verification_adapter(
    surface: Any | None = None,
) -> GovernorVerificationAdapter:
    adapter = GovernorVerificationAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Store adapter
# ---------------------------------------------------------------------------


def probe_store_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe the kit durable semantic-governor store surface."""

    try:
        if surface is not None:
            contracts = surface
            artifacts = surface
        else:
            contracts = _import_module(
                EXPECTED_STORE_CONTRACTS_MODULE, adapter_id=STORE_ADAPTER_ID
            )
            artifacts = _import_module(
                EXPECTED_STORE_ARTIFACTS_MODULE, adapter_id=STORE_ADAPTER_ID
            )
        _require_exports(
            contracts, REQUIRED_STORE_CONTRACT_EXPORTS, STORE_ADAPTER_ID
        )
        _require_exports(
            artifacts, REQUIRED_STORE_ARTIFACT_EXPORTS, STORE_ADAPTER_ID
        )
        _check_pin(
            _attr(contracts, "SEMANTIC_GOVERNOR_STORE_INTERFACE"),
            EXPECTED_STORE_INTERFACE,
            "SEMANTIC_GOVERNOR_STORE_INTERFACE",
            adapter_id=STORE_ADAPTER_ID,
        )
        _check_pin(
            _attr(contracts, "SEMANTIC_GOVERNOR_STORE_SCHEMA"),
            EXPECTED_STORE_SCHEMA,
            "SEMANTIC_GOVERNOR_STORE_SCHEMA",
            adapter_id=STORE_ADAPTER_ID,
        )
        _check_pin(
            _attr(artifacts, "ARTIFACT_MODULE_INTERFACE"),
            EXPECTED_STORE_ARTIFACT_INTERFACE,
            "ARTIFACT_MODULE_INTERFACE",
            adapter_id=STORE_ADAPTER_ID,
        )
        store_cls = _attr(artifacts, "DurableSemanticGovernorStore")
        if not isinstance(store_cls, type):
            raise GovernorCapabilityUnavailable(
                "load",
                "missing_exports",
                "DurableSemanticGovernorStore is not a type",
                adapter_id=STORE_ADAPTER_ID,
                retryable=False,
                status=CapabilityStatus.MISSING.value,
            )
        for method_name in ("put_artifact", "get_verified_artifact"):
            if not callable(getattr(store_cls, method_name, None)):
                raise GovernorCapabilityUnavailable(
                    "load",
                    "missing_exports",
                    f"DurableSemanticGovernorStore missing {method_name}",
                    adapter_id=STORE_ADAPTER_ID,
                    retryable=False,
                    status=CapabilityStatus.MISSING.value,
                )
        fingerprints = MappingProxyType(
            {
                "interface": EXPECTED_STORE_INTERFACE,
                "schema": EXPECTED_STORE_SCHEMA,
                "artifact_interface": EXPECTED_STORE_ARTIFACT_INTERFACE,
                "contracts_module": EXPECTED_STORE_CONTRACTS_MODULE,
                "artifacts_module": EXPECTED_STORE_ARTIFACTS_MODULE,
            }
        )
        return SurfaceCapability(
            available=True,
            adapter_id=STORE_ADAPTER_ID,
            interface_id=STORE_ADAPTER_INTERFACE,
            schema=EXPECTED_STORE_SCHEMA,
            operations=(
                "DurableSemanticGovernorStore",
                "put_artifact",
                "get_verified_artifact",
                "SEMANTIC_GOVERNOR_STORE_INTERFACE",
            ),
            fingerprints=fingerprints,
            status=CapabilityStatus.AVAILABLE.value,
        )
    except GovernorCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=STORE_ADAPTER_ID,
            interface_id=STORE_ADAPTER_INTERFACE,
            schema=EXPECTED_STORE_SCHEMA,
            reason_code=exc.reason_code,
            diagnostic=exc.diagnostic,
            status=exc.status,
            retryable=exc.retryable,
            fingerprints={"package": EXPECTED_STORE_PACKAGE},
        )


class GovernorStoreAdapter:
    """Narrow runtime view over durable governor artifact storage."""

    __slots__ = ("_contracts", "_artifacts", "_capability")

    def __init__(
        self,
        *,
        contracts: Any | None = None,
        artifacts: Any | None = None,
        surface: Any | None = None,
        capability: SurfaceCapability | None = None,
    ) -> None:
        # ``surface`` injects a combined module for tests.
        if surface is not None:
            self._contracts = surface
            self._artifacts = surface
            self._capability = capability or probe_store_capability(
                surface=surface
            )
        else:
            self._contracts = contracts
            self._artifacts = artifacts
            self._capability = capability

    @property
    def capability(self) -> SurfaceCapability:
        if self._capability is None:
            self._ensure_loaded()
        assert self._capability is not None
        return self._capability

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._contracts is None:
            self._contracts = _import_module(
                EXPECTED_STORE_CONTRACTS_MODULE, adapter_id=STORE_ADAPTER_ID
            )
        if self._artifacts is None:
            self._artifacts = _import_module(
                EXPECTED_STORE_ARTIFACTS_MODULE, adapter_id=STORE_ADAPTER_ID
            )
        if self._capability is None:
            # Probe with a combined namespace when both modules loaded separately.
            combined = SimpleNamespace(
                SEMANTIC_GOVERNOR_STORE_INTERFACE=_attr(
                    self._contracts, "SEMANTIC_GOVERNOR_STORE_INTERFACE"
                ),
                SEMANTIC_GOVERNOR_STORE_SCHEMA=_attr(
                    self._contracts, "SEMANTIC_GOVERNOR_STORE_SCHEMA"
                ),
                SemanticGovernorStore=_attr(
                    self._contracts, "SemanticGovernorStore"
                ),
                ARTIFACT_MODULE_INTERFACE=_attr(
                    self._artifacts, "ARTIFACT_MODULE_INTERFACE"
                ),
                DurableSemanticGovernorStore=_attr(
                    self._artifacts, "DurableSemanticGovernorStore"
                ),
            )
            self._capability = probe_store_capability(surface=combined)
        self._capability.require_available("load")
        return self._contracts, self._artifacts

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def store_class(self) -> type:
        _, artifacts = self._ensure_loaded()
        return artifacts.DurableSemanticGovernorStore

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": STORE_ADAPTER_ID,
                "interface_id": STORE_ADAPTER_INTERFACE,
                "capability": cap.to_mapping(),
                "store_interface": EXPECTED_STORE_INTERFACE,
                "store_schema": EXPECTED_STORE_SCHEMA,
                "artifact_interface": EXPECTED_STORE_ARTIFACT_INTERFACE,
            }
        )


def load_store_adapter(
    *,
    contracts: Any | None = None,
    artifacts: Any | None = None,
    surface: Any | None = None,
) -> GovernorStoreAdapter:
    adapter = GovernorStoreAdapter(
        contracts=contracts, artifacts=artifacts, surface=surface
    )
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Incremental sealer capability
# ---------------------------------------------------------------------------


def _sealer_unavailable(
    *,
    reason_code: str,
    diagnostic: str,
    status: str = CapabilityStatus.UNAVAILABLE.value,
    public_module: str | None = None,
    operations: tuple[str, ...] = (),
    fingerprints: Mapping[str, str] | None = None,
    retryable: bool = False,
) -> IncrementalSealerCapability:
    return IncrementalSealerCapability(
        available=False,
        adapter_id=SEALER_CAPABILITY_ID,
        interface_id=SEALER_CAPABILITY_INTERFACE,
        seal_status=SealStatus.UNAVAILABLE.value,
        status=status,
        is_zk=False,
        is_full_or_delta_seal=False,
        can_be_satisfied_by_ivp_commitment=False,
        operations=operations,
        fingerprints=MappingProxyType(dict(fingerprints or {})),
        public_module=public_module,
        reason_code=reason_code,
        diagnostic=diagnostic[:_MAX_DIAGNOSTIC],
        retryable=retryable,
    )


def _looks_like_ivp_commitment(evidence: Any) -> bool:
    if evidence is None:
        return False
    if type(evidence) is str:
        text = evidence.strip()
        return text in _IVP_NON_SEALER_SYMBOLS or text in {
            EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
            EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            "VerificationCommitment@1",
        }
    name = getattr(evidence, "__name__", None)
    if type(name) is str and name in _IVP_NON_SEALER_SYMBOLS:
        return True
    cls_name = type(evidence).__name__
    if cls_name in _IVP_NON_SEALER_SYMBOLS:
        return True
    # Dataclass / mapping shaped commitments.
    if isinstance(evidence, Mapping):
        schema = evidence.get("schema") or evidence.get("interface_id")
        if schema in {
            EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
            EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        }:
            return True
        if evidence.get("kind") == "verification_commitment":
            return True
    iface = _attr(evidence, "interface_id") or _attr(evidence, "schema")
    if iface in {
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    }:
        return True
    return False


def reject_ivp_commitment_as_sealer(evidence: Any) -> None:
    """Fail closed when IVP commitment evidence is offered as sealer capability."""

    if _looks_like_ivp_commitment(evidence):
        raise GovernorCapabilityUnavailable(
            "seal",
            "ivp_commitment_not_sealer",
            "VerificationCommitment is structural non-ZK evidence and cannot "
            "satisfy IncrementalSealerCapability",
            adapter_id=SEALER_CAPABILITY_ID,
            retryable=False,
            status=CapabilityStatus.INCOMPATIBLE.value,
        )


def sealer_capability_from_evidence(
    evidence: Any,
) -> IncrementalSealerCapability:
    """Interpret candidate evidence as sealer capability (fail-closed).

    IVP commitments and related symbols always yield typed unavailability with
    reason ``ivp_commitment_not_sealer``.
    """

    if evidence is None:
        return _sealer_unavailable(
            reason_code="sealer_unavailable",
            diagnostic="no sealer evidence provided",
            status=CapabilityStatus.MISSING.value,
        )
    if _looks_like_ivp_commitment(evidence):
        return _sealer_unavailable(
            reason_code="ivp_commitment_not_sealer",
            diagnostic=(
                "VerificationCommitment cannot satisfy IncrementalSealerCapability"
            ),
            status=CapabilityStatus.INCOMPATIBLE.value,
            fingerprints={
                "rejected_evidence": "VerificationCommitment",
                "is_zk": "false",
                "is_proof_sealer": "false",
            },
        )
    # An explicit released sealer surface may be injected for tests.
    if isinstance(evidence, IncrementalSealerCapability):
        # Re-construct to re-assert IVP invariant.
        return IncrementalSealerCapability(
            available=evidence.available,
            adapter_id=evidence.adapter_id,
            interface_id=evidence.interface_id,
            seal_status=evidence.seal_status,
            status=evidence.status,
            is_zk=evidence.is_zk,
            is_full_or_delta_seal=evidence.is_full_or_delta_seal,
            can_be_satisfied_by_ivp_commitment=False,
            operations=evidence.operations,
            fingerprints=MappingProxyType(dict(evidence.fingerprints)),
            public_module=evidence.public_module,
            reason_code=evidence.reason_code,
            diagnostic=evidence.diagnostic,
            retryable=evidence.retryable,
        )
    if isinstance(evidence, ModuleType) or hasattr(evidence, "__dict__"):
        return _capability_from_sealer_module(evidence)
    return _sealer_unavailable(
        reason_code="incompatible_capability",
        diagnostic=f"unsupported sealer evidence type {type(evidence).__name__}",
        status=CapabilityStatus.INCOMPATIBLE.value,
    )


def _capability_from_sealer_module(module: Any) -> IncrementalSealerCapability:
    present = tuple(
        name for name in _SEALER_PUBLIC_SYMBOLS if hasattr(module, name)
    )
    if not present:
        return _sealer_unavailable(
            reason_code="missing_exports",
            diagnostic=(
                "candidate sealer module exposes none of the released public "
                f"symbols: {', '.join(_SEALER_PUBLIC_SYMBOLS)}"
            ),
            status=CapabilityStatus.MISSING.value,
            public_module=getattr(module, "__name__", None),
        )
    # Optional pin fields when a released surface declares them.
    for pin_name, expected in (
        ("INCREMENTAL_PROOF_SEALER_INTERFACE", "IncrementalProofSealer@1"),
        ("DELTA_SEAL_INTERFACE", "DeltaSeal@1"),
        ("FULL_CHECKPOINT_SEAL_INTERFACE", "FullCheckpointSeal@1"),
    ):
        actual = _attr(module, pin_name)
        if actual is None:
            continue
        try:
            _check_pin(
                actual,
                expected,
                pin_name,
                adapter_id=SEALER_CAPABILITY_ID,
            )
        except GovernorCapabilityUnavailable as exc:
            return _sealer_unavailable(
                reason_code=exc.reason_code,
                diagnostic=exc.diagnostic,
                status=exc.status,
                public_module=getattr(module, "__name__", None),
                operations=present,
            )
    is_zk = bool(_attr(module, "IS_ZK_SEALER") or _attr(module, "is_zk"))
    return IncrementalSealerCapability(
        available=True,
        adapter_id=SEALER_CAPABILITY_ID,
        interface_id=SEALER_CAPABILITY_INTERFACE,
        seal_status=SealStatus.AVAILABLE.value,
        status=CapabilityStatus.AVAILABLE.value,
        is_zk=is_zk,
        is_full_or_delta_seal=True,
        can_be_satisfied_by_ivp_commitment=False,
        operations=present,
        fingerprints=MappingProxyType(
            {
                "public_module": str(getattr(module, "__name__", "injected")),
                "symbols": ",".join(present),
            }
        ),
        public_module=getattr(module, "__name__", None),
        reason_code=None,
        diagnostic=None,
        retryable=False,
    )


def probe_incremental_sealer_capability(
    *,
    surface: Any | None = None,
    candidate_modules: Sequence[str] | None = None,
) -> IncrementalSealerCapability:
    """Probe for a released IncrementalProofSealer without private IPS imports.

    When no public candidate module is present, returns typed
    ``seal_status=unavailable``. Never treats IVP commitment as success.
    """

    if surface is not None:
        if _looks_like_ivp_commitment(surface):
            return sealer_capability_from_evidence(surface)
        return _capability_from_sealer_module(surface)

    modules = tuple(candidate_modules or _SEALER_PUBLIC_CANDIDATE_MODULES)
    last_error: str | None = None
    for module_name in modules:
        # Deliberately avoid importing unfinished private IPS packages.
        # Only exact public candidate paths are considered.
        if "private" in module_name or "._" in module_name:
            continue
        module = _try_import_module(module_name)
        if module is None:
            continue
        capability = _capability_from_sealer_module(module)
        if capability.available:
            return capability
        last_error = capability.diagnostic
    return _sealer_unavailable(
        reason_code="sealer_unavailable",
        diagnostic=(
            last_error
            or "released IncrementalProofSealer / FullCheckpointSeal / "
            "DeltaSeal public API is not present on this tree"
        ),
        status=CapabilityStatus.UNAVAILABLE.value,
        fingerprints=MappingProxyType(
            {
                "candidates": ",".join(modules),
                "ivp_commitment_may_substitute": "false",
            }
        ),
    )


# ---------------------------------------------------------------------------
# Aggregate runtime facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GovernorRuntimeAdapters:
    """Bundle of probed runtime adapters and sealer capability."""

    datasets: GovernorDatasetsAdapter
    harness: GovernorHarnessAdapter
    verification: GovernorVerificationAdapter
    store: GovernorStoreAdapter
    sealer: IncrementalSealerCapability

    def require_execution_surfaces(self) -> None:
        """Fail closed unless datasets, harness, verification, and store are available.

        Sealer unavailability is typed and does not block non-seal execution.
        """

        self.datasets.require_available()
        self.harness.require_available()
        self.verification.require_available()
        self.store.require_available()

    def require_sealer(self) -> IncrementalSealerCapability:
        self.sealer.require_available()
        return self.sealer

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "evidence_id": SCG_RUNTIME_ADAPTERS_EVIDENCE,
                "datasets": self.datasets.capability.to_mapping(),
                "harness": self.harness.capability.to_mapping(),
                "verification": self.verification.capability.to_mapping(),
                "store": self.store.capability.to_mapping(),
                "sealer": self.sealer.to_mapping(),
            }
        )


def load_runtime_adapters(
    *,
    datasets_surface: Any | None = None,
    harness_surface: Any | None = None,
    verification_surface: Any | None = None,
    store_surface: Any | None = None,
    sealer_surface: Any | None = None,
    require_sealer: bool = False,
) -> GovernorRuntimeAdapters:
    """Load all runtime adapters; fail closed on required surface unavailability.

    Sealer remains optional unless ``require_sealer=True``.
    """

    datasets = load_datasets_adapter(datasets_surface)
    harness = load_harness_adapter(harness_surface)
    verification = load_verification_adapter(verification_surface)
    store = load_store_adapter(surface=store_surface)
    sealer = probe_incremental_sealer_capability(surface=sealer_surface)
    if require_sealer:
        sealer.require_available()
    return GovernorRuntimeAdapters(
        datasets=datasets,
        harness=harness,
        verification=verification,
        store=store,
        sealer=sealer,
    )


__all__ = [
    "SCG_RUNTIME_ADAPTERS_EVIDENCE",
    "DATASETS_ADAPTER_ID",
    "HARNESS_ADAPTER_ID",
    "VERIFICATION_ADAPTER_ID",
    "STORE_ADAPTER_ID",
    "SEALER_CAPABILITY_ID",
    "DATASETS_ADAPTER_INTERFACE",
    "HARNESS_ADAPTER_INTERFACE",
    "VERIFICATION_ADAPTER_INTERFACE",
    "STORE_ADAPTER_INTERFACE",
    "SEALER_CAPABILITY_INTERFACE",
    "EXPECTED_DATASETS_API_SCHEMA",
    "EXPECTED_DATASETS_PACKAGE_INTERFACE",
    "EXPECTED_HARNESS_INTERFACE",
    "EXPECTED_HARNESS_SCHEMA",
    "EXPECTED_STORE_INTERFACE",
    "EXPECTED_STORE_SCHEMA",
    "EXPECTED_STORE_ARTIFACT_INTERFACE",
    "EXPECTED_VERIFICATION_COMMITMENT_INTERFACE",
    "EXPECTED_VERIFICATION_COMMITMENT_SCHEMA",
    "REQUIRED_DATASETS_APIS",
    "CapabilityStatus",
    "SealStatus",
    "GovernorAdapterError",
    "GovernorCapabilityUnavailable",
    "SurfaceCapability",
    "IncrementalSealerCapability",
    "GovernorDatasetsAdapter",
    "GovernorHarnessAdapter",
    "GovernorVerificationAdapter",
    "GovernorStoreAdapter",
    "GovernorRuntimeAdapters",
    "probe_datasets_capability",
    "probe_harness_capability",
    "probe_verification_capability",
    "probe_store_capability",
    "probe_incremental_sealer_capability",
    "load_datasets_adapter",
    "load_harness_adapter",
    "load_verification_adapter",
    "load_store_adapter",
    "load_runtime_adapters",
    "reject_ivp_commitment_as_sealer",
    "sealer_capability_from_evidence",
]
