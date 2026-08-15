"""Bind released canonical authorities for adversarial assurance (AAE-039).

Adapters probe exact released index, capsule, context, verification, policy,
state, storage, and sealer interfaces. Missing or drifted authority is reported
as ``typed_unavailable`` — never simulated success and never substituted by a
weaker surface (including IVP ``VerificationCommitment`` for sealer).

Cold import is side-effect free: no Git, store, process, network, or filesystem
operations run at import time. Upstream packages load only when a probe or
adapter construction explicitly requests them.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, Final, Mapping, Sequence

# ---------------------------------------------------------------------------
# Evidence / interface constants
# ---------------------------------------------------------------------------

AAE_RUNTIME_ADAPTERS_EVIDENCE: Final[str] = "aae/runtime-adapters@1"

INDEX_ADAPTER_ID: Final[str] = "aae-index-adapter@1"
CAPSULE_ADAPTER_ID: Final[str] = "aae-capsule-adapter@1"
CONTEXT_ADAPTER_ID: Final[str] = "aae-context-adapter@1"
VERIFICATION_ADAPTER_ID: Final[str] = "aae-verification-adapter@1"
POLICY_ADAPTER_ID: Final[str] = "aae-policy-adapter@1"
STATE_ADAPTER_ID: Final[str] = "aae-state-adapter@1"
STORAGE_ADAPTER_ID: Final[str] = "aae-storage-adapter@1"
SEALER_ADAPTER_ID: Final[str] = "aae-sealer-adapter@1"

INDEX_ADAPTER_INTERFACE: Final[str] = "AssuranceIndexAdapter@1"
CAPSULE_ADAPTER_INTERFACE: Final[str] = "AssuranceCapsuleAdapter@1"
CONTEXT_ADAPTER_INTERFACE: Final[str] = "AssuranceContextAdapter@1"
VERIFICATION_ADAPTER_INTERFACE: Final[str] = "AssuranceVerificationAdapter@1"
POLICY_ADAPTER_INTERFACE: Final[str] = "AssurancePolicyAdapter@1"
STATE_ADAPTER_INTERFACE: Final[str] = "AssuranceStateAdapter@1"
STORAGE_ADAPTER_INTERFACE: Final[str] = "AssuranceStorageAdapter@1"
SEALER_ADAPTER_INTERFACE: Final[str] = "AssuranceSealerAdapter@1"

# Released pin expectations (exact interface / schema strings).
EXPECTED_INDEX_INTERFACE: Final[str] = "IncrementalSemanticIndex@1"
EXPECTED_INDEX_SCHEMA: Final[str] = (
    "ipfs-datasets.software-contracts.semantic-index@2"
)
EXPECTED_INDEX_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_index"
)
EXPECTED_INDEX_MODELS_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_index.models"
)

EXPECTED_CAPSULE_INTERFACE: Final[str] = "SemanticCapsuleCompiler@1"
EXPECTED_CAPSULE_SCHEMA: Final[str] = (
    "ipfs-datasets.software-contracts.semantic-capsule-compiler@1"
)
EXPECTED_CAPSULE_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state"
)
EXPECTED_CAPSULE_COMPILER_MODULE: Final[str] = (
    "ipfs_datasets_py.logic.software_contracts.semantic_state.capsules"
)

EXPECTED_CONTEXT_INTERFACE: Final[str] = "ContextPack@1"
EXPECTED_CONTEXT_SCHEMA: Final[str] = (
    "ipfs-accelerate.context-pack-result@1"
)
EXPECTED_CONTEXT_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack"
)

EXPECTED_VERIFICATION_PUBLIC_INTERFACE: Final[str] = (
    "IncrementalVerificationPublicApi@1"
)
EXPECTED_VERIFICATION_PLANNER_INTERFACE: Final[str] = (
    "IncrementalVerificationPlanner@1"
)
EXPECTED_VERIFICATION_EXECUTOR_INTERFACE: Final[str] = "VerificationExecutor@1"
EXPECTED_VERIFICATION_CACHE_INTERFACE: Final[str] = (
    "VerificationReceiptCache@1"
)
EXPECTED_MODEL_ROUTE_INTERFACE: Final[str] = "ModelRoutePlanner@1"
EXPECTED_VERIFICATION_COMMITMENT_INTERFACE: Final[str] = (
    "VerificationCommitment@1"
)
EXPECTED_VERIFICATION_COMMITMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-commitment@1"
)
EXPECTED_VERIFICATION_PACKAGE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification"
)
EXPECTED_VERIFICATION_EXECUTOR_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.executor"
)
EXPECTED_VERIFICATION_PLANNER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.planner"
)
EXPECTED_VERIFICATION_CACHE_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.receipt_cache"
)
EXPECTED_MODEL_ROUTE_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.verification.model_route"
)

EXPECTED_POLICY_INTERFACE: Final[str] = "AssurancePolicyRepository@1"
EXPECTED_POLICY_SCHEMA: Final[str] = (
    "ipfs-kit.adversarial-assurance-store.policy-cas@1"
)
EXPECTED_POLICY_MODULE: Final[str] = (
    "ipfs_kit_py.adversarial_assurance_store"
)
EXPECTED_POLICY_IMPL_MODULE: Final[str] = (
    "ipfs_kit_py.adversarial_assurance_store.policy"
)

EXPECTED_STATE_ROOT_ADAPTER_INTERFACE: Final[str] = "DurableStateRootAdapter@1"
EXPECTED_STATE_ROOTS_INTERFACE: Final[str] = "DurableStateRoots@1"
EXPECTED_CAMPAIGN_STATE_INTERFACE: Final[str] = "MutationCampaignState@1"
EXPECTED_STATE_ROOT_ADAPTER_MODULE: Final[str] = (
    "ipfs_kit_py.mcp_server.mcplusplus.state_root_adapter"
)
EXPECTED_STATE_ROOTS_MODULE: Final[str] = (
    "ipfs_kit_py.mcp_server.mcplusplus.state_root_contracts"
)
EXPECTED_ASSURANCE_STORE_PACKAGE: Final[str] = (
    "ipfs_kit_py.adversarial_assurance_store"
)

EXPECTED_STORAGE_COORDINATION_INTERFACE: Final[str] = (
    "DurableCoordinationStore@1"
)
EXPECTED_STORAGE_ARTIFACT_INTERFACE: Final[str] = (
    "DurableAssuranceArtifactStore@1"
)
EXPECTED_STORAGE_PACKAGE_INTERFACE: Final[str] = "AdversarialAssuranceStore@1"
EXPECTED_STORAGE_SCHEMA: Final[str] = (
    "ipfs-kit.adversarial-assurance-store.contracts@1"
)
EXPECTED_COORDINATION_MODULE: Final[str] = (
    "ipfs_kit_py.mcp_server.mcplusplus.coordination_storage"
)

# Exact AAE-006 released sealer bindings (module path per public symbol).
SEALER_API_BINDINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "IncrementalProofSealer": (
            "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer"
        ),
        "FullCheckpointSeal": (
            "ipfs_accelerate_py.agent_supervisor.proof"
            ".incremental_sealing.full_checkpoint"
        ),
        "DeltaSeal": (
            "ipfs_accelerate_py.agent_supervisor.proof"
            ".incremental_sealing.delta_seal"
        ),
        "create_full_checkpoint": (
            "ipfs_accelerate_py.agent_supervisor.proof"
            ".incremental_sealing.full_checkpoint"
        ),
        "publish_full_checkpoint": (
            "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer"
        ),
        "build_delta_seal": (
            "ipfs_accelerate_py.agent_supervisor.proof"
            ".incremental_sealing.delta_seal"
        ),
        "publish_delta_seal": (
            "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer"
        ),
    }
)
EXPECTED_SEALER_INTERFACE: Final[str] = "IncrementalProofSealer@1"
EXPECTED_SEALER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/atomic-sealer@1"
)
EXPECTED_SEALER_MODULE: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer"
)

AUTHORITY_KEYS: Final[tuple[str, ...]] = (
    "index",
    "capsule",
    "context",
    "verification",
    "policy",
    "state",
    "storage",
    "sealer",
)

REQUIRED_VERIFICATION_EXPORTS: Final[tuple[str, ...]] = (
    "IncrementalVerificationPlanner",
    "VerificationReceiptCache",
    "ModelRoutePlanner",
    "create_verification_plan",
    "choose_model_route",
    "build_verification_commitment",
    "VerificationCommitment",
)

REQUIRED_VERIFICATION_EXECUTOR_EXPORTS: Final[tuple[str, ...]] = (
    "VerificationExecutor",
)

_IVP_NON_SEALER_SYMBOLS: Final[frozenset[str]] = frozenset(
    {
        "VerificationCommitment",
        "build_verification_commitment",
        "VERIFICATION_COMMITMENT_INTERFACE",
        "VERIFICATION_COMMITMENT_SCHEMA",
        EXPECTED_VERIFICATION_COMMITMENT_INTERFACE,
        EXPECTED_VERIFICATION_COMMITMENT_SCHEMA,
    }
)

_MAX_DIAGNOSTIC: Final[int] = 512


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class AuthorityStatus(str, Enum):
    """Public closed status mapping for released authorities."""

    AVAILABLE = "available"
    TYPED_UNAVAILABLE = "typed_unavailable"


class CapabilityReason(str, Enum):
    """Closed reason codes under typed unavailability / availability."""

    OK = "ok"
    MISSING = "missing"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    IMPORT_FAILED = "import_failed"
    MISSING_EXPORTS = "missing_exports"
    STALE_CAPABILITY = "stale_capability"
    INCOMPATIBLE_CAPABILITY = "incompatible_capability"
    IVP_COMMITMENT_NOT_SEALER = "ivp_commitment_not_sealer"
    SEALER_UNAVAILABLE = "sealer_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"


class SealStatus(str, Enum):
    """Closed seal availability status."""

    AVAILABLE = "available"
    TYPED_UNAVAILABLE = "typed_unavailable"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Typed failures
# ---------------------------------------------------------------------------


class AssuranceAdapterError(ValueError):
    """Closed adapter validation failure (schema/export/binding)."""


class AssuranceCapabilityUnavailable(RuntimeError):
    """Required authority is missing, drifted, or typed unavailable.

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
        authority: str = "",
        retryable: bool = False,
        status: str = AuthorityStatus.TYPED_UNAVAILABLE.value,
    ) -> None:
        self.operation = str(operation or "use")
        self.reason_code = _token(reason_code, "reason_code")
        self.diagnostic = str(diagnostic)[:_MAX_DIAGNOSTIC]
        self.adapter_id = adapter_id
        self.authority = str(authority or "")
        self.retryable = bool(retryable)
        self.status = _enum_value(status, AuthorityStatus, "status")
        super().__init__(
            f"{adapter_id}:{self.operation}:{self.reason_code}: {self.diagnostic}"
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "adapter_id": self.adapter_id,
                "authority": self.authority,
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
    """Closed capability witness for one adapted authority surface."""

    available: bool
    adapter_id: str
    interface_id: str
    schema: str
    authority: str
    operations: tuple[str, ...]
    fingerprints: Mapping[str, str]
    status: str = AuthorityStatus.AVAILABLE.value
    reason_code: str | None = None
    diagnostic: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        status = _enum_value(self.status, AuthorityStatus, "status")
        if self.available and status != AuthorityStatus.AVAILABLE.value:
            status = AuthorityStatus.AVAILABLE.value
        if not self.available and status == AuthorityStatus.AVAILABLE.value:
            status = AuthorityStatus.TYPED_UNAVAILABLE.value
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "fingerprints",
            MappingProxyType(dict(self.fingerprints or {})),
        )
        object.__setattr__(self, "operations", tuple(self.operations or ()))

    def require_available(self, operation: str = "use") -> None:
        if not self.available:
            raise AssuranceCapabilityUnavailable(
                operation,
                self.reason_code or CapabilityReason.CAPABILITY_UNAVAILABLE.value,
                self.diagnostic or f"{self.adapter_id} typed unavailable",
                adapter_id=self.adapter_id,
                authority=self.authority,
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
                "authority": self.authority,
                "operations": list(self.operations),
                "fingerprints": dict(self.fingerprints),
                "status": self.status,
                "reason_code": self.reason_code,
                "diagnostic": self.diagnostic,
                "retryable": self.retryable,
            }
        )


@dataclass(frozen=True, slots=True)
class SealerCapability:
    """Released IncrementalProofSealer / full-checkpoint / delta capability.

    Normative invariants:

    * Missing sealer → ``status=typed_unavailable`` / ``seal_status=typed_unavailable``.
    * IVP ``VerificationCommitment`` can never set ``available=True``.
    * Only exact AAE-006 released module bindings are admitted as public APIs.
    """

    available: bool
    adapter_id: str = SEALER_ADAPTER_ID
    interface_id: str = SEALER_ADAPTER_INTERFACE
    authority: str = "sealer"
    seal_status: str = SealStatus.TYPED_UNAVAILABLE.value
    status: str = AuthorityStatus.TYPED_UNAVAILABLE.value
    is_zk: bool = False
    is_full_or_delta_seal: bool = False
    can_be_satisfied_by_ivp_commitment: bool = False
    operations: tuple[str, ...] = ()
    bindings: Mapping[str, str] = MappingProxyType({})
    fingerprints: Mapping[str, str] = MappingProxyType({})
    public_module: str | None = None
    reason_code: str | None = CapabilityReason.SEALER_UNAVAILABLE.value
    diagnostic: str | None = (
        "released IncrementalProofSealer public API is typed unavailable"
    )
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "can_be_satisfied_by_ivp_commitment", False)
        if self.available:
            object.__setattr__(self, "status", AuthorityStatus.AVAILABLE.value)
            object.__setattr__(self, "seal_status", SealStatus.AVAILABLE.value)
        else:
            if self.status == AuthorityStatus.AVAILABLE.value:
                object.__setattr__(
                    self, "status", AuthorityStatus.TYPED_UNAVAILABLE.value
                )
            if self.seal_status == SealStatus.AVAILABLE.value:
                object.__setattr__(
                    self, "seal_status", SealStatus.TYPED_UNAVAILABLE.value
                )
        object.__setattr__(
            self, "bindings", MappingProxyType(dict(self.bindings or {}))
        )
        object.__setattr__(
            self, "fingerprints", MappingProxyType(dict(self.fingerprints or {}))
        )
        object.__setattr__(self, "operations", tuple(self.operations or ()))

    def require_available(self, operation: str = "seal") -> None:
        if not self.available:
            raise AssuranceCapabilityUnavailable(
                operation,
                self.reason_code or CapabilityReason.SEALER_UNAVAILABLE.value,
                self.diagnostic or "incremental sealer typed unavailable",
                adapter_id=self.adapter_id,
                authority=self.authority,
                retryable=self.retryable,
                status=self.status,
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "available": self.available,
                "adapter_id": self.adapter_id,
                "interface_id": self.interface_id,
                "authority": self.authority,
                "seal_status": self.seal_status,
                "status": self.status,
                "is_zk": self.is_zk,
                "is_full_or_delta_seal": self.is_full_or_delta_seal,
                "can_be_satisfied_by_ivp_commitment": False,
                "operations": list(self.operations),
                "bindings": dict(self.bindings),
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
        raise AssuranceAdapterError(f"{name} must be a nonempty trimmed string")
    if any(ch.isspace() for ch in value):
        raise AssuranceAdapterError(f"{name} must not contain whitespace")
    if len(value) > 256:
        raise AssuranceAdapterError(f"{name} exceeds 256 characters")
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as exc:
        raise AssuranceAdapterError(
            f"{name} has unsupported value {value!r}"
        ) from exc


def _text_pin(value: Any, name: str) -> str:
    if type(value) is not str or not value or not value.strip():
        raise AssuranceAdapterError(f"{name} must be a nonempty trimmed string")
    return value.strip()


def _attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _require_exports(module: Any, names: Sequence[str], *, adapter_id: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise AssuranceCapabilityUnavailable(
            "load",
            CapabilityReason.MISSING_EXPORTS.value,
            f"{adapter_id} missing required exports: {', '.join(missing)}",
            adapter_id=adapter_id,
            retryable=False,
            status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        )


def _schema_major(value: str) -> int | None:
    if "@" not in value:
        return None
    suffix = value.rsplit("@", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


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
    actual_major = _schema_major(text)
    expected_major = _schema_major(expected)
    if (
        stale_if_older
        and actual_major is not None
        and expected_major is not None
        and actual_major < expected_major
    ):
        raise AssuranceCapabilityUnavailable(
            "load",
            CapabilityReason.STALE_CAPABILITY.value,
            f"{name} is {text!r}, expected sealed {expected!r}",
            adapter_id=adapter_id,
            retryable=False,
            status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        )
    raise AssuranceCapabilityUnavailable(
        "load",
        CapabilityReason.INCOMPATIBLE_CAPABILITY.value,
        f"{name} is {text!r}, expected sealed {expected!r}",
        adapter_id=adapter_id,
        retryable=False,
        status=AuthorityStatus.TYPED_UNAVAILABLE.value,
    )


def _unavailable_capability(
    *,
    adapter_id: str,
    interface_id: str,
    schema: str,
    authority: str,
    reason_code: str,
    diagnostic: str,
    retryable: bool = False,
    operations: tuple[str, ...] = (),
    fingerprints: Mapping[str, str] | None = None,
) -> SurfaceCapability:
    return SurfaceCapability(
        available=False,
        adapter_id=adapter_id,
        interface_id=interface_id,
        schema=schema,
        authority=authority,
        operations=operations,
        fingerprints=MappingProxyType(dict(fingerprints or {})),
        status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        reason_code=reason_code,
        diagnostic=diagnostic[:_MAX_DIAGNOSTIC],
        retryable=retryable,
    )


def _import_module(module_name: str, *, adapter_id: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # ImportError and ambient layout failures
        raise AssuranceCapabilityUnavailable(
            "load",
            CapabilityReason.IMPORT_FAILED.value,
            f"import of {module_name!r} failed: {exc}",
            adapter_id=adapter_id,
            retryable=True,
            status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        ) from exc


def _try_import_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _reason_from_exc(exc: AssuranceCapabilityUnavailable) -> str:
    return exc.reason_code or CapabilityReason.CAPABILITY_UNAVAILABLE.value


# ---------------------------------------------------------------------------
# Index adapter
# ---------------------------------------------------------------------------


def probe_index_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe IncrementalSemanticIndex released surface."""

    try:
        if surface is not None:
            module = surface
            models = surface
        else:
            module = _import_module(
                EXPECTED_INDEX_MODULE, adapter_id=INDEX_ADAPTER_ID
            )
            models = _import_module(
                EXPECTED_INDEX_MODELS_MODULE, adapter_id=INDEX_ADAPTER_ID
            )
        _require_exports(
            module, ("IncrementalSemanticIndex",), adapter_id=INDEX_ADAPTER_ID
        )
        index_cls = _attr(module, "IncrementalSemanticIndex")
        if not isinstance(index_cls, type):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "IncrementalSemanticIndex is not a type",
                adapter_id=INDEX_ADAPTER_ID,
            )
        schema = _attr(models, "SEMANTIC_INDEX_SCHEMA")
        if schema is not None:
            _check_pin(
                schema,
                EXPECTED_INDEX_SCHEMA,
                "SEMANTIC_INDEX_SCHEMA",
                adapter_id=INDEX_ADAPTER_ID,
            )
        interface_pin = _attr(module, "INCREMENTAL_SEMANTIC_INDEX_INTERFACE")
        if interface_pin is not None:
            _check_pin(
                interface_pin,
                EXPECTED_INDEX_INTERFACE,
                "INCREMENTAL_SEMANTIC_INDEX_INTERFACE",
                adapter_id=INDEX_ADAPTER_ID,
            )
        return SurfaceCapability(
            available=True,
            adapter_id=INDEX_ADAPTER_ID,
            interface_id=INDEX_ADAPTER_INTERFACE,
            schema=EXPECTED_INDEX_SCHEMA,
            authority="index",
            operations=("IncrementalSemanticIndex",),
            fingerprints=MappingProxyType(
                {
                    "interface": EXPECTED_INDEX_INTERFACE,
                    "schema": EXPECTED_INDEX_SCHEMA,
                    "module": EXPECTED_INDEX_MODULE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=INDEX_ADAPTER_ID,
            interface_id=INDEX_ADAPTER_INTERFACE,
            schema=EXPECTED_INDEX_SCHEMA,
            authority="index",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_INDEX_MODULE},
        )


class AssuranceIndexAdapter:
    """Narrow runtime view over IncrementalSemanticIndex."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_index_capability(surface=surface)
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
                EXPECTED_INDEX_MODULE, adapter_id=INDEX_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_index_capability(surface=self._surface)
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def index_class(self) -> type:
        surface = self._ensure_loaded()
        return surface.IncrementalSemanticIndex

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": INDEX_ADAPTER_ID,
                "interface_id": INDEX_ADAPTER_INTERFACE,
                "authority": "index",
                "capability": cap.to_mapping(),
                "released_interface": EXPECTED_INDEX_INTERFACE,
            }
        )


def load_index_adapter(surface: Any | None = None) -> AssuranceIndexAdapter:
    adapter = AssuranceIndexAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Capsule adapter
# ---------------------------------------------------------------------------


def probe_capsule_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe SemanticCapsuleCompiler@1 functional surface."""

    try:
        if surface is not None:
            module = surface
            capsules = surface
        else:
            module = _import_module(
                EXPECTED_CAPSULE_MODULE, adapter_id=CAPSULE_ADAPTER_ID
            )
            capsules = _import_module(
                EXPECTED_CAPSULE_COMPILER_MODULE, adapter_id=CAPSULE_ADAPTER_ID
            )
        _require_exports(
            module,
            ("compile_semantic_capsule",),
            adapter_id=CAPSULE_ADAPTER_ID,
        )
        if not callable(_attr(module, "compile_semantic_capsule")):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "compile_semantic_capsule is not callable",
                adapter_id=CAPSULE_ADAPTER_ID,
            )
        interface = _attr(capsules, "SEMANTIC_CAPSULE_COMPILER_INTERFACE")
        schema = _attr(capsules, "SEMANTIC_CAPSULE_COMPILER_SCHEMA")
        if interface is not None:
            _check_pin(
                interface,
                EXPECTED_CAPSULE_INTERFACE,
                "SEMANTIC_CAPSULE_COMPILER_INTERFACE",
                adapter_id=CAPSULE_ADAPTER_ID,
            )
        if schema is not None:
            _check_pin(
                schema,
                EXPECTED_CAPSULE_SCHEMA,
                "SEMANTIC_CAPSULE_COMPILER_SCHEMA",
                adapter_id=CAPSULE_ADAPTER_ID,
            )
        ops = ["compile_semantic_capsule"]
        if callable(_attr(module, "compile_semantic_capsules")):
            ops.append("compile_semantic_capsules")
        return SurfaceCapability(
            available=True,
            adapter_id=CAPSULE_ADAPTER_ID,
            interface_id=CAPSULE_ADAPTER_INTERFACE,
            schema=EXPECTED_CAPSULE_SCHEMA,
            authority="capsule",
            operations=tuple(ops),
            fingerprints=MappingProxyType(
                {
                    "interface": EXPECTED_CAPSULE_INTERFACE,
                    "schema": EXPECTED_CAPSULE_SCHEMA,
                    "module": EXPECTED_CAPSULE_MODULE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=CAPSULE_ADAPTER_ID,
            interface_id=CAPSULE_ADAPTER_INTERFACE,
            schema=EXPECTED_CAPSULE_SCHEMA,
            authority="capsule",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_CAPSULE_MODULE},
        )


class AssuranceCapsuleAdapter:
    """Narrow runtime view over SemanticCapsuleCompiler@1."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_capsule_capability(
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
                EXPECTED_CAPSULE_MODULE, adapter_id=CAPSULE_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_capsule_capability(surface=self._surface)
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    def compile_semantic_capsule(self, *args: Any, **kwargs: Any) -> Any:
        surface = self._ensure_loaded()
        return surface.compile_semantic_capsule(*args, **kwargs)

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": CAPSULE_ADAPTER_ID,
                "interface_id": CAPSULE_ADAPTER_INTERFACE,
                "authority": "capsule",
                "capability": cap.to_mapping(),
                "released_interface": EXPECTED_CAPSULE_INTERFACE,
            }
        )


def load_capsule_adapter(surface: Any | None = None) -> AssuranceCapsuleAdapter:
    adapter = AssuranceCapsuleAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Context adapter
# ---------------------------------------------------------------------------


def probe_context_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe ContextPacker / pack_context released surface."""

    try:
        module = (
            surface
            if surface is not None
            else _import_module(
                EXPECTED_CONTEXT_MODULE, adapter_id=CONTEXT_ADAPTER_ID
            )
        )
        _require_exports(
            module,
            ("ContextPacker", "pack_context", "CONTEXT_PACK_INTERFACE"),
            adapter_id=CONTEXT_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "CONTEXT_PACK_INTERFACE"),
            EXPECTED_CONTEXT_INTERFACE,
            "CONTEXT_PACK_INTERFACE",
            adapter_id=CONTEXT_ADAPTER_ID,
        )
        if not isinstance(_attr(module, "ContextPacker"), type):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "ContextPacker is not a type",
                adapter_id=CONTEXT_ADAPTER_ID,
            )
        if not callable(_attr(module, "pack_context")):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "pack_context is not callable",
                adapter_id=CONTEXT_ADAPTER_ID,
            )
        return SurfaceCapability(
            available=True,
            adapter_id=CONTEXT_ADAPTER_ID,
            interface_id=CONTEXT_ADAPTER_INTERFACE,
            schema=EXPECTED_CONTEXT_SCHEMA,
            authority="context",
            operations=("ContextPacker", "pack_context"),
            fingerprints=MappingProxyType(
                {
                    "interface": EXPECTED_CONTEXT_INTERFACE,
                    "schema": EXPECTED_CONTEXT_SCHEMA,
                    "module": EXPECTED_CONTEXT_MODULE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=CONTEXT_ADAPTER_ID,
            interface_id=CONTEXT_ADAPTER_INTERFACE,
            schema=EXPECTED_CONTEXT_SCHEMA,
            authority="context",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_CONTEXT_MODULE},
        )


class AssuranceContextAdapter:
    """Narrow runtime view over ContextPacker / pack_context."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_context_capability(
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
                EXPECTED_CONTEXT_MODULE, adapter_id=CONTEXT_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_context_capability(surface=self._surface)
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    def pack_context(self, *args: Any, **kwargs: Any) -> Any:
        surface = self._ensure_loaded()
        return surface.pack_context(*args, **kwargs)

    @property
    def packer_class(self) -> type:
        surface = self._ensure_loaded()
        return surface.ContextPacker

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": CONTEXT_ADAPTER_ID,
                "interface_id": CONTEXT_ADAPTER_INTERFACE,
                "authority": "context",
                "capability": cap.to_mapping(),
                "released_interface": EXPECTED_CONTEXT_INTERFACE,
            }
        )


def load_context_adapter(surface: Any | None = None) -> AssuranceContextAdapter:
    adapter = AssuranceContextAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Verification adapter
# ---------------------------------------------------------------------------


def probe_verification_capability(
    *, surface: Any | None = None
) -> SurfaceCapability:
    """Probe IVP planner/executor/cache/route surfaces (non-sealer)."""

    try:
        if surface is not None:
            package = surface
            executor = surface
            planner = surface
            cache = surface
            route = surface
        else:
            package = _import_module(
                EXPECTED_VERIFICATION_PACKAGE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
            executor = _import_module(
                EXPECTED_VERIFICATION_EXECUTOR_MODULE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
            planner = _import_module(
                EXPECTED_VERIFICATION_PLANNER_MODULE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
            cache = _import_module(
                EXPECTED_VERIFICATION_CACHE_MODULE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
            route = _import_module(
                EXPECTED_MODEL_ROUTE_MODULE,
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
        _require_exports(
            package,
            REQUIRED_VERIFICATION_EXPORTS,
            adapter_id=VERIFICATION_ADAPTER_ID,
        )
        _require_exports(
            executor,
            REQUIRED_VERIFICATION_EXECUTOR_EXPORTS,
            adapter_id=VERIFICATION_ADAPTER_ID,
        )
        public_iface = _attr(package, "PUBLIC_API_INTERFACE")
        if public_iface is not None:
            _check_pin(
                public_iface,
                EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
                "PUBLIC_API_INTERFACE",
                adapter_id=VERIFICATION_ADAPTER_ID,
            )
        for module, attr, expected in (
            (planner, "VERIFICATION_PLANNER_INTERFACE", EXPECTED_VERIFICATION_PLANNER_INTERFACE),
            (executor, "VERIFICATION_EXECUTOR_INTERFACE", EXPECTED_VERIFICATION_EXECUTOR_INTERFACE),
            (cache, "VERIFICATION_RECEIPT_CACHE_INTERFACE", EXPECTED_VERIFICATION_CACHE_INTERFACE),
            (route, "MODEL_ROUTE_PLANNER_INTERFACE", EXPECTED_MODEL_ROUTE_INTERFACE),
        ):
            pin = _attr(module, attr)
            if pin is not None:
                _check_pin(
                    pin, expected, attr, adapter_id=VERIFICATION_ADAPTER_ID
                )
        operations = tuple(REQUIRED_VERIFICATION_EXPORTS) + tuple(
            REQUIRED_VERIFICATION_EXECUTOR_EXPORTS
        )
        return SurfaceCapability(
            available=True,
            adapter_id=VERIFICATION_ADAPTER_ID,
            interface_id=VERIFICATION_ADAPTER_INTERFACE,
            schema=EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
            authority="verification",
            operations=operations,
            fingerprints=MappingProxyType(
                {
                    "public_interface": EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
                    "planner_interface": EXPECTED_VERIFICATION_PLANNER_INTERFACE,
                    "executor_interface": EXPECTED_VERIFICATION_EXECUTOR_INTERFACE,
                    "cache_interface": EXPECTED_VERIFICATION_CACHE_INTERFACE,
                    "route_interface": EXPECTED_MODEL_ROUTE_INTERFACE,
                    "commitment_is_proof_sealer": "false",
                    "package": EXPECTED_VERIFICATION_PACKAGE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=VERIFICATION_ADAPTER_ID,
            interface_id=VERIFICATION_ADAPTER_INTERFACE,
            schema=EXPECTED_VERIFICATION_PUBLIC_INTERFACE,
            authority="verification",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"package": EXPECTED_VERIFICATION_PACKAGE},
        )


class AssuranceVerificationAdapter:
    """Narrow runtime view over IVP verification authorities.

    Structural commitments are never promoted to sealer or ZK authority.
    """

    __slots__ = ("_surface", "_executor", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._executor = surface
            self._capability = capability or probe_verification_capability(
                surface=surface
            )
        else:
            self._surface = None
            self._executor = None
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
        if self._executor is None:
            # Injected combined surfaces reuse package; live path loads executor.
            if hasattr(self._surface, "VerificationExecutor"):
                self._executor = self._surface
            else:
                self._executor = _import_module(
                    EXPECTED_VERIFICATION_EXECUTOR_MODULE,
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
        return False

    def commitment_is_zk(self) -> bool:
        return False

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": VERIFICATION_ADAPTER_ID,
                "interface_id": VERIFICATION_ADAPTER_INTERFACE,
                "authority": "verification",
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
) -> AssuranceVerificationAdapter:
    adapter = AssuranceVerificationAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Policy adapter
# ---------------------------------------------------------------------------


def probe_policy_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe AssurancePolicyRepository@1 CAS surface."""

    try:
        module = (
            surface
            if surface is not None
            else _import_module(
                EXPECTED_POLICY_MODULE, adapter_id=POLICY_ADAPTER_ID
            )
        )
        _require_exports(
            module,
            (
                "AssurancePolicyRepository",
                "ASSURANCE_POLICY_REPOSITORY_INTERFACE",
                "POLICY_CAS_SCHEMA",
            ),
            adapter_id=POLICY_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "ASSURANCE_POLICY_REPOSITORY_INTERFACE"),
            EXPECTED_POLICY_INTERFACE,
            "ASSURANCE_POLICY_REPOSITORY_INTERFACE",
            adapter_id=POLICY_ADAPTER_ID,
        )
        _check_pin(
            _attr(module, "POLICY_CAS_SCHEMA"),
            EXPECTED_POLICY_SCHEMA,
            "POLICY_CAS_SCHEMA",
            adapter_id=POLICY_ADAPTER_ID,
        )
        return SurfaceCapability(
            available=True,
            adapter_id=POLICY_ADAPTER_ID,
            interface_id=POLICY_ADAPTER_INTERFACE,
            schema=EXPECTED_POLICY_SCHEMA,
            authority="policy",
            operations=(
                "AssurancePolicyRepository",
                "ASSURANCE_POLICY_REPOSITORY_INTERFACE",
                "POLICY_CAS_SCHEMA",
            ),
            fingerprints=MappingProxyType(
                {
                    "interface": EXPECTED_POLICY_INTERFACE,
                    "schema": EXPECTED_POLICY_SCHEMA,
                    "module": EXPECTED_POLICY_MODULE,
                    "production_policy_change": "false",
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=POLICY_ADAPTER_ID,
            interface_id=POLICY_ADAPTER_INTERFACE,
            schema=EXPECTED_POLICY_SCHEMA,
            authority="policy",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_POLICY_MODULE},
        )


class AssurancePolicyAdapter:
    """Narrow runtime view over policy CAS (never auto-promotes)."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_policy_capability(
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
                EXPECTED_POLICY_MODULE, adapter_id=POLICY_ADAPTER_ID
            )
        if self._capability is None:
            self._capability = probe_policy_capability(surface=self._surface)
        self._capability.require_available("load")
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def repository_class(self) -> type:
        surface = self._ensure_loaded()
        return surface.AssurancePolicyRepository

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": POLICY_ADAPTER_ID,
                "interface_id": POLICY_ADAPTER_INTERFACE,
                "authority": "policy",
                "capability": cap.to_mapping(),
                "released_interface": EXPECTED_POLICY_INTERFACE,
                "production_policy_change": False,
            }
        )


def load_policy_adapter(surface: Any | None = None) -> AssurancePolicyAdapter:
    adapter = AssurancePolicyAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# State adapter
# ---------------------------------------------------------------------------


def probe_state_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe DurableStateRootAdapter / campaign state surfaces."""

    try:
        if surface is not None:
            adapter_mod = surface
            roots_mod = surface
            store_pkg = surface
        else:
            adapter_mod = _import_module(
                EXPECTED_STATE_ROOT_ADAPTER_MODULE,
                adapter_id=STATE_ADAPTER_ID,
            )
            roots_mod = _import_module(
                EXPECTED_STATE_ROOTS_MODULE, adapter_id=STATE_ADAPTER_ID
            )
            store_pkg = _import_module(
                EXPECTED_ASSURANCE_STORE_PACKAGE, adapter_id=STATE_ADAPTER_ID
            )
        _require_exports(
            adapter_mod,
            ("DurableStateRootAdapter",),
            adapter_id=STATE_ADAPTER_ID,
        )
        _require_exports(
            roots_mod, ("DurableStateRoots",), adapter_id=STATE_ADAPTER_ID
        )
        if not isinstance(_attr(adapter_mod, "DurableStateRootAdapter"), type):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "DurableStateRootAdapter is not a type",
                adapter_id=STATE_ADAPTER_ID,
            )
        for method_name in (
            "compare_and_swap_root",
            "current_root",
            "get_verified",
            "put_verified",
        ):
            if not callable(
                getattr(
                    _attr(adapter_mod, "DurableStateRootAdapter"),
                    method_name,
                    None,
                )
            ):
                raise AssuranceCapabilityUnavailable(
                    "load",
                    CapabilityReason.MISSING_EXPORTS.value,
                    f"DurableStateRootAdapter missing {method_name}",
                    adapter_id=STATE_ADAPTER_ID,
                )
        campaign_iface = _attr(store_pkg, "CAMPAIGN_STATE_INTERFACE")
        if campaign_iface is not None:
            _check_pin(
                campaign_iface,
                EXPECTED_CAMPAIGN_STATE_INTERFACE,
                "CAMPAIGN_STATE_INTERFACE",
                adapter_id=STATE_ADAPTER_ID,
            )
        return SurfaceCapability(
            available=True,
            adapter_id=STATE_ADAPTER_ID,
            interface_id=STATE_ADAPTER_INTERFACE,
            schema=EXPECTED_CAMPAIGN_STATE_INTERFACE,
            authority="state",
            operations=(
                "DurableStateRootAdapter",
                "DurableStateRoots",
                "compare_and_swap_root",
                "current_root",
            ),
            fingerprints=MappingProxyType(
                {
                    "state_root_adapter": EXPECTED_STATE_ROOT_ADAPTER_INTERFACE,
                    "state_roots": EXPECTED_STATE_ROOTS_INTERFACE,
                    "campaign_state": EXPECTED_CAMPAIGN_STATE_INTERFACE,
                    "adapter_module": EXPECTED_STATE_ROOT_ADAPTER_MODULE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=STATE_ADAPTER_ID,
            interface_id=STATE_ADAPTER_INTERFACE,
            schema=EXPECTED_CAMPAIGN_STATE_INTERFACE,
            authority="state",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_STATE_ROOT_ADAPTER_MODULE},
        )


class AssuranceStateAdapter:
    """Narrow runtime view over durable state roots / campaign state."""

    __slots__ = ("_surface", "_capability")

    def __init__(
        self,
        surface: Any | None = None,
        *,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._surface = surface
            self._capability = capability or probe_state_capability(
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
        if self._capability is None:
            # Live probe imports adapter + roots + store package modules.
            if self._surface is None:
                self._capability = probe_state_capability()
                self._surface = _import_module(
                    EXPECTED_STATE_ROOT_ADAPTER_MODULE,
                    adapter_id=STATE_ADAPTER_ID,
                )
            else:
                self._capability = probe_state_capability(surface=self._surface)
        self._capability.require_available("load")
        if self._surface is None:
            self._surface = _import_module(
                EXPECTED_STATE_ROOT_ADAPTER_MODULE,
                adapter_id=STATE_ADAPTER_ID,
            )
        return self._surface

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def state_root_adapter_class(self) -> type:
        surface = self._ensure_loaded()
        return surface.DurableStateRootAdapter

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": STATE_ADAPTER_ID,
                "interface_id": STATE_ADAPTER_INTERFACE,
                "authority": "state",
                "capability": cap.to_mapping(),
                "released_interface": EXPECTED_STATE_ROOT_ADAPTER_INTERFACE,
            }
        )


def load_state_adapter(surface: Any | None = None) -> AssuranceStateAdapter:
    adapter = AssuranceStateAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Storage adapter
# ---------------------------------------------------------------------------


def probe_storage_capability(*, surface: Any | None = None) -> SurfaceCapability:
    """Probe DurableCoordinationStore + assurance artifact store."""

    try:
        if surface is not None:
            coordination = surface
            store_pkg = surface
        else:
            coordination = _import_module(
                EXPECTED_COORDINATION_MODULE, adapter_id=STORAGE_ADAPTER_ID
            )
            store_pkg = _import_module(
                EXPECTED_ASSURANCE_STORE_PACKAGE, adapter_id=STORAGE_ADAPTER_ID
            )
        _require_exports(
            coordination,
            ("DurableCoordinationStore",),
            adapter_id=STORAGE_ADAPTER_ID,
        )
        _require_exports(
            store_pkg,
            (
                "DurableAssuranceArtifactStore",
                "ARTIFACT_MODULE_INTERFACE",
                "PACKAGE_INTERFACE",
                "ASSURANCE_ARTIFACT_STORE_SCHEMA",
            ),
            adapter_id=STORAGE_ADAPTER_ID,
        )
        if not isinstance(
            _attr(coordination, "DurableCoordinationStore"), type
        ):
            raise AssuranceCapabilityUnavailable(
                "load",
                CapabilityReason.MISSING_EXPORTS.value,
                "DurableCoordinationStore is not a type",
                adapter_id=STORAGE_ADAPTER_ID,
            )
        _check_pin(
            _attr(store_pkg, "ARTIFACT_MODULE_INTERFACE"),
            EXPECTED_STORAGE_ARTIFACT_INTERFACE,
            "ARTIFACT_MODULE_INTERFACE",
            adapter_id=STORAGE_ADAPTER_ID,
        )
        _check_pin(
            _attr(store_pkg, "PACKAGE_INTERFACE"),
            EXPECTED_STORAGE_PACKAGE_INTERFACE,
            "PACKAGE_INTERFACE",
            adapter_id=STORAGE_ADAPTER_ID,
        )
        _check_pin(
            _attr(store_pkg, "ASSURANCE_ARTIFACT_STORE_SCHEMA"),
            EXPECTED_STORAGE_SCHEMA,
            "ASSURANCE_ARTIFACT_STORE_SCHEMA",
            adapter_id=STORAGE_ADAPTER_ID,
        )
        return SurfaceCapability(
            available=True,
            adapter_id=STORAGE_ADAPTER_ID,
            interface_id=STORAGE_ADAPTER_INTERFACE,
            schema=EXPECTED_STORAGE_SCHEMA,
            authority="storage",
            operations=(
                "DurableCoordinationStore",
                "DurableAssuranceArtifactStore",
                "recover_assurance_campaigns",
            ),
            fingerprints=MappingProxyType(
                {
                    "coordination_interface": EXPECTED_STORAGE_COORDINATION_INTERFACE,
                    "artifact_interface": EXPECTED_STORAGE_ARTIFACT_INTERFACE,
                    "package_interface": EXPECTED_STORAGE_PACKAGE_INTERFACE,
                    "schema": EXPECTED_STORAGE_SCHEMA,
                    "coordination_module": EXPECTED_COORDINATION_MODULE,
                }
            ),
            status=AuthorityStatus.AVAILABLE.value,
            reason_code=CapabilityReason.OK.value,
        )
    except AssuranceCapabilityUnavailable as exc:
        return _unavailable_capability(
            adapter_id=STORAGE_ADAPTER_ID,
            interface_id=STORAGE_ADAPTER_INTERFACE,
            schema=EXPECTED_STORAGE_SCHEMA,
            authority="storage",
            reason_code=_reason_from_exc(exc),
            diagnostic=exc.diagnostic,
            retryable=exc.retryable,
            fingerprints={"module": EXPECTED_COORDINATION_MODULE},
        )


class AssuranceStorageAdapter:
    """Narrow runtime view over coordination + assurance artifact storage."""

    __slots__ = ("_coordination", "_store_pkg", "_capability")

    def __init__(
        self,
        *,
        surface: Any | None = None,
        capability: SurfaceCapability | None = None,
    ) -> None:
        if surface is not None:
            self._coordination = surface
            self._store_pkg = surface
            self._capability = capability or probe_storage_capability(
                surface=surface
            )
        else:
            self._coordination = None
            self._store_pkg = None
            self._capability = capability

    @property
    def capability(self) -> SurfaceCapability:
        if self._capability is None:
            self._ensure_loaded()
        assert self._capability is not None
        return self._capability

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._capability is None:
            if self._coordination is None and self._store_pkg is None:
                self._capability = probe_storage_capability()
            else:
                combined = SimpleNamespace(
                    DurableCoordinationStore=_attr(
                        self._coordination, "DurableCoordinationStore"
                    )
                    if self._coordination is not None
                    else None,
                    DurableAssuranceArtifactStore=_attr(
                        self._store_pkg, "DurableAssuranceArtifactStore"
                    )
                    if self._store_pkg is not None
                    else None,
                    ARTIFACT_MODULE_INTERFACE=_attr(
                        self._store_pkg, "ARTIFACT_MODULE_INTERFACE"
                    )
                    if self._store_pkg is not None
                    else None,
                    PACKAGE_INTERFACE=_attr(self._store_pkg, "PACKAGE_INTERFACE")
                    if self._store_pkg is not None
                    else None,
                    ASSURANCE_ARTIFACT_STORE_SCHEMA=_attr(
                        self._store_pkg, "ASSURANCE_ARTIFACT_STORE_SCHEMA"
                    )
                    if self._store_pkg is not None
                    else None,
                )
                self._capability = probe_storage_capability(surface=combined)
        self._capability.require_available("load")
        if self._coordination is None:
            self._coordination = _import_module(
                EXPECTED_COORDINATION_MODULE, adapter_id=STORAGE_ADAPTER_ID
            )
        if self._store_pkg is None:
            self._store_pkg = _import_module(
                EXPECTED_ASSURANCE_STORE_PACKAGE, adapter_id=STORAGE_ADAPTER_ID
            )
        return self._coordination, self._store_pkg

    def require_available(self) -> SurfaceCapability:
        cap = self.capability
        cap.require_available("use")
        return cap

    @property
    def coordination_store_class(self) -> type:
        coordination, _ = self._ensure_loaded()
        return coordination.DurableCoordinationStore

    @property
    def artifact_store_class(self) -> type:
        _, store_pkg = self._ensure_loaded()
        return store_pkg.DurableAssuranceArtifactStore

    def runtime_view(self) -> Mapping[str, Any]:
        cap = self.require_available()
        self._ensure_loaded()
        return MappingProxyType(
            {
                "adapter_id": STORAGE_ADAPTER_ID,
                "interface_id": STORAGE_ADAPTER_INTERFACE,
                "authority": "storage",
                "capability": cap.to_mapping(),
                "coordination_interface": EXPECTED_STORAGE_COORDINATION_INTERFACE,
                "artifact_interface": EXPECTED_STORAGE_ARTIFACT_INTERFACE,
            }
        )


def load_storage_adapter(surface: Any | None = None) -> AssuranceStorageAdapter:
    adapter = AssuranceStorageAdapter(surface=surface)
    adapter.require_available()
    return adapter


# ---------------------------------------------------------------------------
# Sealer capability (AAE-006 released bindings)
# ---------------------------------------------------------------------------


def _sealer_unavailable(
    *,
    reason_code: str,
    diagnostic: str,
    public_module: str | None = None,
    operations: tuple[str, ...] = (),
    bindings: Mapping[str, str] | None = None,
    fingerprints: Mapping[str, str] | None = None,
    retryable: bool = False,
) -> SealerCapability:
    return SealerCapability(
        available=False,
        adapter_id=SEALER_ADAPTER_ID,
        interface_id=SEALER_ADAPTER_INTERFACE,
        authority="sealer",
        seal_status=SealStatus.TYPED_UNAVAILABLE.value,
        status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        is_zk=False,
        is_full_or_delta_seal=False,
        can_be_satisfied_by_ivp_commitment=False,
        operations=operations,
        bindings=MappingProxyType(dict(bindings or {})),
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
        return text in _IVP_NON_SEALER_SYMBOLS
    name = getattr(evidence, "__name__", None)
    if type(name) is str and name in _IVP_NON_SEALER_SYMBOLS:
        return True
    cls_name = type(evidence).__name__
    if cls_name in _IVP_NON_SEALER_SYMBOLS:
        return True
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
        raise AssuranceCapabilityUnavailable(
            "seal",
            CapabilityReason.IVP_COMMITMENT_NOT_SEALER.value,
            "VerificationCommitment is structural non-ZK evidence and cannot "
            "satisfy released IncrementalProofSealer bindings",
            adapter_id=SEALER_ADAPTER_ID,
            authority="sealer",
            retryable=False,
            status=AuthorityStatus.TYPED_UNAVAILABLE.value,
        )


def sealer_capability_from_evidence(evidence: Any) -> SealerCapability:
    """Interpret candidate evidence as sealer capability (fail-closed)."""

    if evidence is None:
        return _sealer_unavailable(
            reason_code=CapabilityReason.SEALER_UNAVAILABLE.value,
            diagnostic="no sealer evidence provided",
        )
    if _looks_like_ivp_commitment(evidence):
        return _sealer_unavailable(
            reason_code=CapabilityReason.IVP_COMMITMENT_NOT_SEALER.value,
            diagnostic=(
                "VerificationCommitment cannot satisfy IncrementalProofSealer"
            ),
            fingerprints={
                "rejected_evidence": "VerificationCommitment",
                "is_zk": "false",
                "is_proof_sealer": "false",
            },
        )
    if isinstance(evidence, SealerCapability):
        return SealerCapability(
            available=evidence.available,
            adapter_id=evidence.adapter_id,
            interface_id=evidence.interface_id,
            authority=evidence.authority,
            seal_status=evidence.seal_status,
            status=evidence.status,
            is_zk=evidence.is_zk,
            is_full_or_delta_seal=evidence.is_full_or_delta_seal,
            can_be_satisfied_by_ivp_commitment=False,
            operations=evidence.operations,
            bindings=MappingProxyType(dict(evidence.bindings)),
            fingerprints=MappingProxyType(dict(evidence.fingerprints)),
            public_module=evidence.public_module,
            reason_code=evidence.reason_code,
            diagnostic=evidence.diagnostic,
            retryable=evidence.retryable,
        )
    if isinstance(evidence, ModuleType) or hasattr(evidence, "__dict__"):
        return _capability_from_sealer_surface(evidence)
    return _sealer_unavailable(
        reason_code=CapabilityReason.INCOMPATIBLE_CAPABILITY.value,
        diagnostic=f"unsupported sealer evidence type {type(evidence).__name__}",
    )


def _capability_from_sealer_surface(surface: Any) -> SealerCapability:
    present = tuple(
        name for name in SEALER_API_BINDINGS if hasattr(surface, name)
    )
    if not present:
        return _sealer_unavailable(
            reason_code=CapabilityReason.MISSING_EXPORTS.value,
            diagnostic=(
                "candidate sealer surface exposes none of the released public "
                f"symbols: {', '.join(SEALER_API_BINDINGS)}"
            ),
            public_module=getattr(surface, "__name__", None),
        )
    for pin_name, expected in (
        ("SEALER_INTERFACE", EXPECTED_SEALER_INTERFACE),
        ("INCREMENTAL_PROOF_SEALER_INTERFACE", EXPECTED_SEALER_INTERFACE),
        ("SEALER_SCHEMA", EXPECTED_SEALER_SCHEMA),
    ):
        actual = _attr(surface, pin_name)
        if actual is None:
            continue
        try:
            _check_pin(
                actual, expected, pin_name, adapter_id=SEALER_ADAPTER_ID
            )
        except AssuranceCapabilityUnavailable as exc:
            return _sealer_unavailable(
                reason_code=_reason_from_exc(exc),
                diagnostic=exc.diagnostic,
                public_module=getattr(surface, "__name__", None),
                operations=present,
            )
    required_core = (
        "IncrementalProofSealer",
        "FullCheckpointSeal",
        "DeltaSeal",
        "create_full_checkpoint",
        "publish_full_checkpoint",
        "build_delta_seal",
        "publish_delta_seal",
    )
    missing_core = [name for name in required_core if name not in present]
    if missing_core:
        return _sealer_unavailable(
            reason_code=CapabilityReason.MISSING_EXPORTS.value,
            diagnostic=(
                "released sealer surface missing core symbols: "
                + ", ".join(missing_core)
            ),
            public_module=getattr(surface, "__name__", None),
            operations=present,
        )
    is_zk = bool(_attr(surface, "IS_ZK_SEALER") or _attr(surface, "is_zk"))
    bindings = {
        name: SEALER_API_BINDINGS[name]
        for name in present
        if name in SEALER_API_BINDINGS
    }
    return SealerCapability(
        available=True,
        adapter_id=SEALER_ADAPTER_ID,
        interface_id=SEALER_ADAPTER_INTERFACE,
        authority="sealer",
        seal_status=SealStatus.AVAILABLE.value,
        status=AuthorityStatus.AVAILABLE.value,
        is_zk=is_zk,
        is_full_or_delta_seal=True,
        can_be_satisfied_by_ivp_commitment=False,
        operations=present,
        bindings=MappingProxyType(bindings),
        fingerprints=MappingProxyType(
            {
                "public_module": str(
                    getattr(surface, "__name__", "injected")
                ),
                "symbols": ",".join(present),
                "interface": EXPECTED_SEALER_INTERFACE,
            }
        ),
        public_module=getattr(surface, "__name__", None),
        reason_code=CapabilityReason.OK.value,
        diagnostic=None,
        retryable=False,
    )


def probe_sealer_capability(
    *,
    surface: Any | None = None,
    require_exact_bindings: bool = True,
) -> SealerCapability:
    """Probe exact AAE-006 released sealer bindings (lazy, fail-closed).

    When ``require_exact_bindings`` is true (default), each public symbol is
    loaded from its released module path. Injected test surfaces may set
    ``require_exact_bindings=False`` or pass ``surface=``.
    """

    if surface is not None:
        if _looks_like_ivp_commitment(surface):
            return sealer_capability_from_evidence(surface)
        return _capability_from_sealer_surface(surface)

    if not require_exact_bindings:
        module = _try_import_module(EXPECTED_SEALER_MODULE)
        if module is None:
            return _sealer_unavailable(
                reason_code=CapabilityReason.IMPORT_FAILED.value,
                diagnostic=f"import of {EXPECTED_SEALER_MODULE!r} failed",
                retryable=True,
            )
        return _capability_from_sealer_surface(module)

    resolved: dict[str, Any] = {}
    bindings_hit: dict[str, str] = {}
    for symbol, module_name in SEALER_API_BINDINGS.items():
        module = _try_import_module(module_name)
        if module is None:
            return _sealer_unavailable(
                reason_code=CapabilityReason.IMPORT_FAILED.value,
                diagnostic=(
                    f"released sealer binding import failed for {symbol} "
                    f"from {module_name!r}"
                ),
                bindings=dict(SEALER_API_BINDINGS),
                retryable=True,
            )
        value = _attr(module, symbol)
        if value is None:
            return _sealer_unavailable(
                reason_code=CapabilityReason.MISSING_EXPORTS.value,
                diagnostic=(
                    f"released module {module_name!r} missing symbol {symbol}"
                ),
                public_module=module_name,
                bindings=dict(SEALER_API_BINDINGS),
            )
        resolved[symbol] = value
        bindings_hit[symbol] = module_name

    # Pin interface/schema from the sealer module.
    sealer_mod = _try_import_module(EXPECTED_SEALER_MODULE)
    if sealer_mod is not None:
        for pin_name, expected in (
            ("SEALER_INTERFACE", EXPECTED_SEALER_INTERFACE),
            ("SEALER_SCHEMA", EXPECTED_SEALER_SCHEMA),
        ):
            actual = _attr(sealer_mod, pin_name)
            if actual is None:
                continue
            try:
                _check_pin(
                    actual, expected, pin_name, adapter_id=SEALER_ADAPTER_ID
                )
            except AssuranceCapabilityUnavailable as exc:
                return _sealer_unavailable(
                    reason_code=_reason_from_exc(exc),
                    diagnostic=exc.diagnostic,
                    public_module=EXPECTED_SEALER_MODULE,
                    operations=tuple(resolved),
                    bindings=bindings_hit,
                )

    surface_ns = SimpleNamespace(
        __name__=EXPECTED_SEALER_MODULE,
        SEALER_INTERFACE=EXPECTED_SEALER_INTERFACE,
        SEALER_SCHEMA=EXPECTED_SEALER_SCHEMA,
        **resolved,
    )
    capability = _capability_from_sealer_surface(surface_ns)
    if capability.available:
        return SealerCapability(
            available=True,
            adapter_id=SEALER_ADAPTER_ID,
            interface_id=SEALER_ADAPTER_INTERFACE,
            authority="sealer",
            seal_status=SealStatus.AVAILABLE.value,
            status=AuthorityStatus.AVAILABLE.value,
            is_zk=capability.is_zk,
            is_full_or_delta_seal=True,
            can_be_satisfied_by_ivp_commitment=False,
            operations=tuple(resolved),
            bindings=MappingProxyType(bindings_hit),
            fingerprints=MappingProxyType(
                {
                    "public_module": EXPECTED_SEALER_MODULE,
                    "symbols": ",".join(resolved),
                    "interface": EXPECTED_SEALER_INTERFACE,
                    "binding_source": "aae-006-prerequisite-receipt",
                }
            ),
            public_module=EXPECTED_SEALER_MODULE,
            reason_code=CapabilityReason.OK.value,
            diagnostic=None,
            retryable=False,
        )
    return capability


# ---------------------------------------------------------------------------
# Aggregate runtime facade
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AssuranceRuntimeAdapters:
    """Bundle of probed AAE runtime authorities."""

    index: AssuranceIndexAdapter
    capsule: AssuranceCapsuleAdapter
    context: AssuranceContextAdapter
    verification: AssuranceVerificationAdapter
    policy: AssurancePolicyAdapter
    state: AssuranceStateAdapter
    storage: AssuranceStorageAdapter
    sealer: SealerCapability

    def authority_status_map(self) -> Mapping[str, Mapping[str, Any]]:
        """Closed status mapping for all eight released authorities."""

        return MappingProxyType(
            {
                "index": self.index.capability.to_mapping(),
                "capsule": self.capsule.capability.to_mapping(),
                "context": self.context.capability.to_mapping(),
                "verification": self.verification.capability.to_mapping(),
                "policy": self.policy.capability.to_mapping(),
                "state": self.state.capability.to_mapping(),
                "storage": self.storage.capability.to_mapping(),
                "sealer": self.sealer.to_mapping(),
            }
        )

    def require_execution_surfaces(self) -> None:
        """Fail closed unless non-optional campaign execution surfaces are available.

        Sealer unavailability is typed and does not block non-seal composition
        when callers opt out of ``require_sealer``.
        """

        self.index.require_available()
        self.capsule.require_available()
        self.context.require_available()
        self.verification.require_available()
        self.policy.require_available()
        self.state.require_available()
        self.storage.require_available()

    def require_sealer(self) -> SealerCapability:
        self.sealer.require_available()
        return self.sealer

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "evidence_id": AAE_RUNTIME_ADAPTERS_EVIDENCE,
                "authorities": self.authority_status_map(),
                "index": self.index.capability.to_mapping(),
                "capsule": self.capsule.capability.to_mapping(),
                "context": self.context.capability.to_mapping(),
                "verification": self.verification.capability.to_mapping(),
                "policy": self.policy.capability.to_mapping(),
                "state": self.state.capability.to_mapping(),
                "storage": self.storage.capability.to_mapping(),
                "sealer": self.sealer.to_mapping(),
            }
        )


def probe_all_authorities(
    *,
    index_surface: Any | None = None,
    capsule_surface: Any | None = None,
    context_surface: Any | None = None,
    verification_surface: Any | None = None,
    policy_surface: Any | None = None,
    state_surface: Any | None = None,
    storage_surface: Any | None = None,
    sealer_surface: Any | None = None,
) -> Mapping[str, Mapping[str, Any]]:
    """Probe all eight authorities without requiring availability."""

    return MappingProxyType(
        {
            "index": probe_index_capability(surface=index_surface).to_mapping(),
            "capsule": probe_capsule_capability(
                surface=capsule_surface
            ).to_mapping(),
            "context": probe_context_capability(
                surface=context_surface
            ).to_mapping(),
            "verification": probe_verification_capability(
                surface=verification_surface
            ).to_mapping(),
            "policy": probe_policy_capability(
                surface=policy_surface
            ).to_mapping(),
            "state": probe_state_capability(surface=state_surface).to_mapping(),
            "storage": probe_storage_capability(
                surface=storage_surface
            ).to_mapping(),
            "sealer": probe_sealer_capability(
                surface=sealer_surface
            ).to_mapping(),
        }
    )


def load_runtime_adapters(
    *,
    index_surface: Any | None = None,
    capsule_surface: Any | None = None,
    context_surface: Any | None = None,
    verification_surface: Any | None = None,
    policy_surface: Any | None = None,
    state_surface: Any | None = None,
    storage_surface: Any | None = None,
    sealer_surface: Any | None = None,
    require_sealer: bool = False,
    require_execution: bool = True,
) -> AssuranceRuntimeAdapters:
    """Load runtime adapters; fail closed on required surface unavailability.

    Sealer remains optional unless ``require_sealer=True``.
    """

    index = AssuranceIndexAdapter(surface=index_surface)
    capsule = AssuranceCapsuleAdapter(surface=capsule_surface)
    context = AssuranceContextAdapter(surface=context_surface)
    verification = AssuranceVerificationAdapter(surface=verification_surface)
    policy = AssurancePolicyAdapter(surface=policy_surface)
    state = AssuranceStateAdapter(surface=state_surface)
    storage = AssuranceStorageAdapter(surface=storage_surface)
    sealer = probe_sealer_capability(surface=sealer_surface)

    runtime = AssuranceRuntimeAdapters(
        index=index,
        capsule=capsule,
        context=context,
        verification=verification,
        policy=policy,
        state=state,
        storage=storage,
        sealer=sealer,
    )
    if require_execution:
        runtime.require_execution_surfaces()
    if require_sealer:
        runtime.require_sealer()
    return runtime


__all__ = [
    "AAE_RUNTIME_ADAPTERS_EVIDENCE",
    "AUTHORITY_KEYS",
    "INDEX_ADAPTER_ID",
    "CAPSULE_ADAPTER_ID",
    "CONTEXT_ADAPTER_ID",
    "VERIFICATION_ADAPTER_ID",
    "POLICY_ADAPTER_ID",
    "STATE_ADAPTER_ID",
    "STORAGE_ADAPTER_ID",
    "SEALER_ADAPTER_ID",
    "INDEX_ADAPTER_INTERFACE",
    "CAPSULE_ADAPTER_INTERFACE",
    "CONTEXT_ADAPTER_INTERFACE",
    "VERIFICATION_ADAPTER_INTERFACE",
    "POLICY_ADAPTER_INTERFACE",
    "STATE_ADAPTER_INTERFACE",
    "STORAGE_ADAPTER_INTERFACE",
    "SEALER_ADAPTER_INTERFACE",
    "SEALER_API_BINDINGS",
    "EXPECTED_INDEX_INTERFACE",
    "EXPECTED_CAPSULE_INTERFACE",
    "EXPECTED_CONTEXT_INTERFACE",
    "EXPECTED_VERIFICATION_PUBLIC_INTERFACE",
    "EXPECTED_POLICY_INTERFACE",
    "EXPECTED_STATE_ROOT_ADAPTER_INTERFACE",
    "EXPECTED_STORAGE_PACKAGE_INTERFACE",
    "EXPECTED_SEALER_INTERFACE",
    "EXPECTED_VERIFICATION_COMMITMENT_INTERFACE",
    "EXPECTED_VERIFICATION_COMMITMENT_SCHEMA",
    "AuthorityStatus",
    "CapabilityReason",
    "SealStatus",
    "AssuranceAdapterError",
    "AssuranceCapabilityUnavailable",
    "SurfaceCapability",
    "SealerCapability",
    "AssuranceIndexAdapter",
    "AssuranceCapsuleAdapter",
    "AssuranceContextAdapter",
    "AssuranceVerificationAdapter",
    "AssurancePolicyAdapter",
    "AssuranceStateAdapter",
    "AssuranceStorageAdapter",
    "AssuranceRuntimeAdapters",
    "probe_index_capability",
    "probe_capsule_capability",
    "probe_context_capability",
    "probe_verification_capability",
    "probe_policy_capability",
    "probe_state_capability",
    "probe_storage_capability",
    "probe_sealer_capability",
    "probe_all_authorities",
    "sealer_capability_from_evidence",
    "reject_ivp_commitment_as_sealer",
    "load_index_adapter",
    "load_capsule_adapter",
    "load_context_adapter",
    "load_verification_adapter",
    "load_policy_adapter",
    "load_state_adapter",
    "load_storage_adapter",
    "load_runtime_adapters",
]
