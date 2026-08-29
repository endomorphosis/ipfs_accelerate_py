"""Governed patch lifecycle coordinator (PCCE-021).

This module is the sole accepted-patch lifecycle authority. Adapters and CLI
may call it but may not reproduce or skip a stage. Importing this module
performs no I/O, network, process, or filesystem mutation and does not bind a
model provider or search sibling checkouts.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.proof_context.compatibility import (
    CompatibilityError,
    reject_mock,
    reject_mutable_ref,
)
from ipfs_accelerate_py.proof_context.errors import (
    ERRORS,
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofContextError,
    SchemaMismatchError,
    UnknownFieldError,
    error_status,
    from_provider_error,
)
from ipfs_accelerate_py.proof_context.policy import (
    COMPATIBILITY_MATRIX_CONTENT_ID,
    FORBIDDEN_EVIDENCE,
    LIVE_MODES,
    MODES,
    PCCE_006_CONTENT_ID,
    POLICY_CID,
    PROVENANCES,
    SIMULATION_WATERMARK,
    PolicyError,
    admit_cid,
    admit_evidence,
    admit_mode,
    admit_provenance,
)
from ipfs_accelerate_py.proof_context.results import (
    IDENTITY_FIELDS,
    RESULT_STATE_CID,
    STATUSES,
    ResultIdentities,
    admit_status,
    is_success,
    is_terminal,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
LIFECYCLE_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/lifecycle"
LIFECYCLE_RECORD_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/lifecycle-record"
STAGE_ARTIFACT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/stage-artifact"
CHECKPOINT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/lifecycle-checkpoint"
CONTRACT_VERSION: Final[str] = "0.1"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"
SOLE_AUTHORITY: Final[bool] = True
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
PROVIDER_BOUND: Final[bool] = False

# Frozen governed sequence. Every accepted production/supervised patch must
# traverse this exact order. Adapters cannot skip, reorder, or short-circuit it.
STAGES: Final[tuple[str, ...]] = (
    "identify-operator",
    "resolve-repository",
    "scan-semantic",
    "invalidate",
    "context-pack",
    "sufficiency",
    "route",
    "proposal",
    "scope-check",
    "isolated-apply",
    "impact",
    "incremental-verify",
    "escalate",
    "assurance",
    "seal",
    "disposition",
)

STAGE_CONTRACTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "identify-operator": "pcce/proof-context/v0.1/task-specification",
        "resolve-repository": "pcce/proof-context/v0.1/repository-state",
        "scan-semantic": "pcce/proof-context/v0.1/semantic-capsule",
        "invalidate": "pcce/proof-context/v0.1/invalidation-plan",
        "context-pack": "pcce/proof-context/v0.1/context-pack",
        "sufficiency": "pcce/proof-context/v0.1/context-pack",
        "route": "pcce/proof-context/v0.1/model-route-decision",
        "proposal": "pcce/proof-context/v0.1/patch-proposal",
        "scope-check": "pcce/proof-context/v0.1/patch-proposal",
        "isolated-apply": "pcce/proof-context/v0.1/execution-receipt",
        "impact": "pcce/proof-context/v0.1/repository-state",
        "incremental-verify": "pcce/proof-context/v0.1/verification-plan",
        "escalate": "pcce/proof-context/v0.1/context-pack",
        "assurance": "pcce/proof-context/v0.1/qualification-result",
        "seal": "pcce/proof-context/v0.1/incremental-seal",
        "disposition": "pcce/proof-context/v0.1/qualification-result",
    }
)

APPLY_STAGE: Final[str] = "isolated-apply"
SEAL_STAGE: Final[str] = "seal"
DISPOSITION_STAGE: Final[str] = "disposition"
VERIFY_STAGE: Final[str] = "incremental-verify"
PROTECTED_REFS: Final[tuple[str, ...]] = (
    "main",
    "master",
    "head",
    "origin/main",
    "origin/master",
)
PUBLICATION_REQUIREMENTS: Final[tuple[str, ...]] = (
    "sealed",
    "live-provenance",
    "all-stages",
    "policy-admitted",
    "lease-valid",
    "fence-valid",
)
STOP_PUBLICATION_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if status != "succeeded"
)
_BIND_ONCE_FIELDS: Final[tuple[str, ...]] = (
    "patch_id",
    "artifact_id",
    "lease_id",
    "fence_id",
    "worktree_id",
)
_STABLE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "operator_id",
    "repository_id",
    "repository_state_cid",
    "task_id",
    "run_id",
    "trace_id",
    "contract_version",
)
_PORT_FIELDS: Final[tuple[str, ...]] = (
    "operator",
    "repository",
    "semantic",
    "route",
    "proposal",
    "scope",
    "worktree",
    "verification",
    "assurance",
    "sealing",
    "disposition",
    "governance",
    "persistence",
)
_BYPASS_KEYS: Final[frozenset[str]] = frozenset(
    {
        "skip",
        "skip_stages",
        "start_at",
        "bypass",
        "bypass_stages",
        "self_approved",
        "adapter_approved",
    }
)


class LifecycleError(ProofContextError):
    """Fail-closed lifecycle error. Never claims success or publication."""

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        details: Mapping[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details, reason=reason)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze(item) for item in sorted(value, key=repr))
    return value


def _canonicalize(value: Any) -> str:
    if value is None or isinstance(value, (bool, int, str)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, Mapping):
        parts = []
        for key in sorted(str(item) for item in value):
            parts.append(
                json.dumps(str(key), ensure_ascii=False, separators=(",", ":"))
                + ":"
                + _canonicalize(value[key] if key in value else value[str(key)])
            )
        return "{" + ",".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonicalize(item) for item in value) + "]"
    raise LifecycleError(
        f"unsupported lifecycle canonicalization type {type(value).__name__}",
        code="malformed",
    )


def mint_lifecycle_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonicalize(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def admit_stage(stage: Any) -> str:
    if not isinstance(stage, str) or stage not in STAGES:
        raise UnknownFieldError(f"unknown lifecycle stage {stage!r}")
    return stage


def _admit_lifecycle_cid(value: str) -> str:
    try:
        return admit_cid(value)
    except PolicyError as exc:
        raise LifecycleError(
            str(exc),
            code=exc.reason if exc.reason in ERRORS else "pseudo_cid",
        ) from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise MalformedError("payload must be a mapping")


def _protected_ref(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in PROTECTED_REFS:
        return True
    try:
        reject_mutable_ref(value)
    except CompatibilityError:
        return True
    return False


@dataclass(frozen=True)
class LifecycleIdentities:
    """Stable identities bound through every lifecycle stage."""

    operator_id: str
    repository_id: str
    repository_state_cid: str
    task_id: str
    run_id: str
    trace_id: str
    contract_version: str = CONTRACT_VERSION
    patch_id: str | None = None
    artifact_id: str | None = None
    evidence_cid: str | None = None
    lease_id: str | None = None
    fence_id: str | None = None
    worktree_id: str | None = None

    def __post_init__(self) -> None:
        for name in _STABLE_IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MalformedError(f"identity field {name} is required")
        if self.contract_version != CONTRACT_VERSION:
            raise SchemaMismatchError(
                f"contract version {self.contract_version!r} is not {CONTRACT_VERSION}"
            )
        object.__setattr__(
            self, "repository_state_cid", _admit_lifecycle_cid(self.repository_state_cid)
        )
        if self.evidence_cid is not None:
            object.__setattr__(
                self, "evidence_cid", _admit_lifecycle_cid(self.evidence_cid)
            )

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "operator_id": self.operator_id,
                "repository_id": self.repository_id,
                "repository_state_cid": self.repository_state_cid,
                "task_id": self.task_id,
                "run_id": self.run_id,
                "trace_id": self.trace_id,
                "contract_version": self.contract_version,
                "patch_id": self.patch_id,
                "artifact_id": self.artifact_id,
                "evidence_cid": self.evidence_cid,
                "lease_id": self.lease_id,
                "fence_id": self.fence_id,
                "worktree_id": self.worktree_id,
            }
        )

    def to_result_identities(self) -> ResultIdentities:
        return ResultIdentities(
            repository_id=self.repository_id,
            repository_state_cid=self.repository_state_cid,
            task_id=self.task_id,
            run_id=self.run_id,
            trace_id=self.trace_id,
            evidence_cid=self.evidence_cid,
            patch_id=self.patch_id,
            artifact_id=self.artifact_id,
            contract_version=self.contract_version,
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> LifecycleIdentities:
        if not isinstance(payload, Mapping):
            raise MalformedError("identities must be a mapping")
        extra = set(payload) - {
            "operator_id",
            "repository_id",
            "repository_state_cid",
            "task_id",
            "run_id",
            "trace_id",
            "contract_version",
            "patch_id",
            "artifact_id",
            "evidence_cid",
            "lease_id",
            "fence_id",
            "worktree_id",
        }
        if extra:
            raise UnknownFieldError(f"unknown identity field {sorted(extra)[0]!r}")
        return cls(
            operator_id=str(payload.get("operator_id") or ""),
            repository_id=str(payload.get("repository_id") or ""),
            repository_state_cid=str(payload.get("repository_state_cid") or ""),
            task_id=str(payload.get("task_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            contract_version=str(payload.get("contract_version") or CONTRACT_VERSION),
            patch_id=_optional_str(payload.get("patch_id")),
            artifact_id=_optional_str(payload.get("artifact_id")),
            evidence_cid=_optional_str(payload.get("evidence_cid")),
            lease_id=_optional_str(payload.get("lease_id")),
            fence_id=_optional_str(payload.get("fence_id")),
            worktree_id=_optional_str(payload.get("worktree_id")),
        )


def merge_identities(
    current: LifecycleIdentities,
    incoming: LifecycleIdentities,
) -> LifecycleIdentities:
    for name in _STABLE_IDENTITY_FIELDS:
        if getattr(incoming, name) != getattr(current, name):
            raise IdentityInconsistentError(
                f"identity field {name} drifted",
                details={"field": name},
            )
    updates: dict[str, str | None] = {}
    for name in _BIND_ONCE_FIELDS:
        existing = getattr(current, name)
        proposed = getattr(incoming, name)
        if existing and proposed and existing != proposed:
            raise IdentityInconsistentError(
                f"identity field {name} is already bound",
                details={"field": name},
            )
        updates[name] = proposed or existing
    return replace(current, **updates)


@dataclass(frozen=True)
class StageArtifact:
    """Typed identity-bound artifact consumed and emitted by one stage."""

    schema: str
    stage: str
    status: str
    identities: LifecycleIdentities
    artifact_cid: str
    provenance: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    inbound_cid: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", admit_stage(self.stage))
        object.__setattr__(self, "payload", _freeze(self.payload))
        if self.schema != STAGE_ARTIFACT_SCHEMA:
            raise SchemaMismatchError(
                f"stage artifact schema {self.schema!r} is not {STAGE_ARTIFACT_SCHEMA}"
            )
        admit_status(self.status)
        admit_provenance(self.provenance)
        object.__setattr__(self, "artifact_cid", _admit_lifecycle_cid(self.artifact_cid))
        if self.inbound_cid is not None:
            object.__setattr__(self, "inbound_cid", _admit_lifecycle_cid(self.inbound_cid))
        if self.error is not None and self.error not in ERRORS:
            raise UnknownFieldError(f"unknown error {self.error!r}")
        if self.status == "succeeded" and self.error is not None:
            raise BoundaryViolationError("succeeded stage artifacts cannot carry an error")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "stage": self.stage,
                "status": self.status,
                "identities": dict(self.identities.to_mapping()),
                "artifact_cid": self.artifact_cid,
                "provenance": self.provenance,
                "payload": dict(self.payload) if isinstance(self.payload, Mapping) else self.payload,
                "inbound_cid": self.inbound_cid,
                "error": self.error,
                "contract": STAGE_CONTRACTS[self.stage],
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StageArtifact:
        if not isinstance(payload, Mapping):
            raise MalformedError("stage artifact must be a mapping")
        identities_raw = payload.get("identities")
        if not isinstance(identities_raw, Mapping):
            raise MalformedError("stage artifact identities are required")
        return cls(
            schema=str(payload.get("schema") or STAGE_ARTIFACT_SCHEMA),
            stage=str(payload.get("stage") or ""),
            status=str(payload.get("status") or ""),
            identities=LifecycleIdentities.from_mapping(identities_raw),
            artifact_cid=str(payload.get("artifact_cid") or ""),
            provenance=str(payload.get("provenance") or "live"),
            payload=payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {},
            inbound_cid=_optional_str(payload.get("inbound_cid")),
            error=_optional_str(payload.get("error")),
        )


@dataclass(frozen=True)
class GovernanceReceipts:
    """Lease, fence, worktree, and schedule receipts bound to a run."""

    lease: Mapping[str, Any]
    fence: Mapping[str, Any]
    worktree: Mapping[str, Any]
    schedule: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lease", _freeze(self.lease))
        object.__setattr__(self, "fence", _freeze(self.fence))
        object.__setattr__(self, "worktree", _freeze(self.worktree))
        object.__setattr__(self, "schedule", _freeze(self.schedule))

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "lease": dict(self.lease) if isinstance(self.lease, Mapping) else self.lease,
                "fence": dict(self.fence) if isinstance(self.fence, Mapping) else self.fence,
                "worktree": dict(self.worktree)
                if isinstance(self.worktree, Mapping)
                else self.worktree,
                "schedule": dict(self.schedule)
                if isinstance(self.schedule, Mapping)
                else self.schedule,
            }
        )


@dataclass(frozen=True)
class LifecycleRecord:
    """Terminal lifecycle result. Publication is coordinator-owned."""

    schema: str
    status: str
    identities: LifecycleIdentities
    mode: str
    provenance: str
    published: bool
    sealed: bool
    stages: tuple[str, ...]
    artifacts: tuple[StageArtifact, ...]
    governance: GovernanceReceipts
    evidence_cid: str
    error: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    accepted: bool = field(init=False, default=False)
    terminal: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.schema != LIFECYCLE_RECORD_SCHEMA:
            raise SchemaMismatchError(
                f"lifecycle record schema {self.schema!r} is not {LIFECYCLE_RECORD_SCHEMA}"
            )
        admit_status(self.status)
        admit_mode(self.mode)
        admit_provenance(self.provenance)
        object.__setattr__(self, "evidence_cid", _admit_lifecycle_cid(self.evidence_cid))
        if self.error is not None and self.error not in ERRORS:
            raise UnknownFieldError(f"unknown error {self.error!r}")
        if self.published and not (
            self.mode in LIVE_MODES
            and self.status == "succeeded"
            and self.sealed
            and self.provenance == "live"
            and tuple(artifact.stage for artifact in self.artifacts) == STAGES
        ):
            raise BoundaryViolationError(
                "publication requires a complete sealed live production/supervised run"
            )
        if self.published and self.error is not None:
            raise BoundaryViolationError("published results cannot carry an error")
        if self.status == "succeeded" and self.provenance == "simulated":
            raise BoundaryViolationError("simulated results cannot be labeled succeeded")
        object.__setattr__(
            self,
            "accepted",
            bool(self.published and is_success(self.status, provenance=self.provenance)),
        )
        object.__setattr__(self, "terminal", is_terminal(self.status) or not self.published)

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "contract_version": CONTRACT_VERSION,
                "status": self.status,
                "mode": self.mode,
                "provenance": self.provenance,
                "published": self.published,
                "sealed": self.sealed,
                "accepted": self.accepted,
                "error": self.error,
                "identities": dict(self.identities.to_mapping()),
                "stages": self.stages,
                "trace": [dict(artifact.to_mapping()) for artifact in self.artifacts],
                "governance": dict(self.governance.to_mapping()),
                "evidence_cid": self.evidence_cid,
                "payload": dict(self.payload) if isinstance(self.payload, Mapping) else self.payload,
                "lifecycle_cid": LIFECYCLE_CID,
                "policy_cid": POLICY_CID,
                "result_state_cid": RESULT_STATE_CID,
            }
        )

    def to_checkpoint(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": CHECKPOINT_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                "mode": self.mode,
                "identities": dict(self.identities.to_mapping()),
                "completed": [dict(artifact.to_mapping()) for artifact in self.artifacts],
                "governance": dict(self.governance.to_mapping()),
                "status": self.status,
                "published": self.published,
            }
        )


@runtime_checkable
class OperatorPort(Protocol):
    def identify(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class RepositoryPort(Protocol):
    def resolve(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class SemanticPort(Protocol):
    def scan(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact: ...

    def invalidate(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...

    def context_pack(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...

    def sufficiency(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...

    def impact(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...

    def escalate(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class RoutePort(Protocol):
    def route(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact: ...


@runtime_checkable
class ProposalPort(Protocol):
    def propose(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact: ...


@runtime_checkable
class ScopePort(Protocol):
    def check(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact: ...


@runtime_checkable
class WorktreePort(Protocol):
    def apply(
        self,
        identities: LifecycleIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact: ...

    def discard(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class VerificationPort(Protocol):
    def verify(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class AssurancePort(Protocol):
    def assure(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class SealingPort(Protocol):
    def seal(self, identities: LifecycleIdentities, repository: Path) -> StageArtifact: ...


@runtime_checkable
class DispositionPort(Protocol):
    def decide(
        self, identities: LifecycleIdentities, repository: Path
    ) -> StageArtifact: ...


@runtime_checkable
class GovernancePort(Protocol):
    def acquire_lease(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]: ...

    def acquire_fence(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]: ...

    def admit_schedule(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]: ...

    def check_cancellation(
        self, identities: LifecycleIdentities, repository: Path
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class PersistencePort(Protocol):
    def persist(
        self,
        artifact: StageArtifact | Mapping[str, Any],
        *,
        published: bool,
    ) -> Mapping[str, Any]: ...


_PORT_PROTOCOLS: Final[Mapping[str, type]] = MappingProxyType(
    {
        "operator": OperatorPort,
        "repository": RepositoryPort,
        "semantic": SemanticPort,
        "route": RoutePort,
        "proposal": ProposalPort,
        "scope": ScopePort,
        "worktree": WorktreePort,
        "verification": VerificationPort,
        "assurance": AssurancePort,
        "sealing": SealingPort,
        "disposition": DispositionPort,
        "governance": GovernancePort,
        "persistence": PersistencePort,
    }
)


@dataclass(frozen=True)
class LifecyclePorts:
    """Injected canonical ports. The coordinator does not construct backends."""

    operator: OperatorPort
    repository: RepositoryPort
    semantic: SemanticPort
    route: RoutePort
    proposal: ProposalPort
    scope: ScopePort
    worktree: WorktreePort
    verification: VerificationPort
    assurance: AssurancePort
    sealing: SealingPort
    disposition: DispositionPort
    governance: GovernancePort
    persistence: PersistencePort


def _is_lifecycle_ports(value: Any) -> bool:
    """Accept LifecyclePorts across importlib.reload of this module."""

    if isinstance(value, LifecyclePorts):
        return True
    return type(value).__name__ == "LifecyclePorts" and all(
        hasattr(value, name) for name in _PORT_FIELDS
    )


def _admit_ports(ports: LifecyclePorts, *, mode: str) -> LifecyclePorts:
    if not _is_lifecycle_ports(ports):
        raise MalformedError("ports must be LifecyclePorts")
    if not isinstance(ports, LifecyclePorts):
        ports = LifecyclePorts(
            **{name: getattr(ports, name) for name in _PORT_FIELDS}
        )
    admitted: dict[str, Any] = {}
    for name in _PORT_FIELDS:
        port = getattr(ports, name)
        if port is None:
            raise LifecycleError(
                f"canonical port {name} is unavailable",
                code="unavailable_capability",
                details={"capability": name},
            )
        if mode in LIVE_MODES:
            try:
                reject_mock(port)
            except CompatibilityError as exc:
                raise BoundaryViolationError(
                    "production and supervised reject mock ports",
                    details={"capability": name},
                ) from exc
        protocol = _PORT_PROTOCOLS[name]
        if not isinstance(port, protocol):
            raise LifecycleError(
                f"canonical port {name} does not implement {protocol.__name__}",
                code="unavailable_capability",
                details={"capability": name},
            )
        admitted[name] = port
    return LifecyclePorts(**admitted)


def _truthy(value: Any) -> bool:
    return value is True or value == "true" or value == 1


def _continues(stage: str, status: str) -> bool:
    if status == "succeeded":
        return True
    # Post-verify expansion is the next frozen stage; all other non-success
    # statuses stop publication immediately.
    return stage == VERIFY_STAGE and status == "context_insufficient"


def _is_lifecycle_identities(value: Any) -> bool:
    """Accept LifecycleIdentities across importlib.reload of this module."""

    if isinstance(value, LifecycleIdentities):
        return True
    return (
        type(value).__name__ == "LifecycleIdentities"
        and hasattr(value, "operator_id")
        and hasattr(value, "to_mapping")
        and callable(getattr(value, "to_mapping", None))
    )


def _error_for_status(status: str, explicit: str | None) -> str | None:
    if status == "succeeded":
        return None
    if explicit:
        return explicit
    if status in ERRORS:
        return status
    mapped = {
        "rejected": "boundary_violation",
        "stale": "stale_root",
        "invalid": "malformed",
        "simulated": "simulated_promoted",
        "unavailable": "unavailable_capability",
        "model_escalation_required": "context_insufficient",
        "human_review_required": "human_review_required",
    }
    return mapped.get(status, "boundary_violation")


class PatchLifecycle:
    """Sole accepted-patch lifecycle authority. Restart-aware and fail-closed."""

    schema = LIFECYCLE_SCHEMA
    contract_version = CONTRACT_VERSION
    sole_authority = SOLE_AUTHORITY
    sibling_layout_required = SIBLING_LAYOUT_REQUIRED
    provider_bound = PROVIDER_BOUND

    def __init__(
        self,
        repository: Path,
        *,
        ports: LifecyclePorts,
        identities: LifecycleIdentities,
        mode: str,
    ) -> None:
        self._repository = repository
        self._ports = ports
        self._identities = identities
        self._mode = mode
        self._applied = False
        self._canonical_head: str | None = None
        self._persisted: list[Mapping[str, Any]] = []
        self._governance: GovernanceReceipts | None = None

    @classmethod
    def open(
        cls,
        repository: str | Path,
        *,
        ports: LifecyclePorts,
        identities: LifecycleIdentities,
        mode: str = "production",
    ) -> PatchLifecycle:
        try:
            admitted_mode = admit_mode(mode)
        except PolicyError as exc:
            raise LifecycleError(
                str(exc),
                code=exc.reason if exc.reason in ERRORS else "unknown_field",
            ) from exc
        if not _is_lifecycle_identities(identities):
            raise MalformedError("identities must be LifecycleIdentities")
        admitted_ports = _admit_ports(ports, mode=admitted_mode)
        root = Path(repository)
        if not root.is_dir():
            raise LifecycleError(
                "repository must be an ordinary directory",
                code="malformed",
                details={"stage": "resolve-repository"},
            )
        return cls(
            root,
            ports=admitted_ports,
            identities=identities,
            mode=admitted_mode,
        )

    @property
    def repository(self) -> Path:
        return self._repository

    @property
    def ports(self) -> LifecyclePorts:
        return self._ports

    @property
    def identities(self) -> LifecycleIdentities:
        return self._identities

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def persisted_evidence(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._persisted)

    def run(
        self,
        proposal: Mapping[str, Any] | None = None,
        *,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> LifecycleRecord:
        self._reject_bypass(proposal)
        self._reject_bypass(checkpoint)
        if proposal is not None:
            self._check_payload_identities(proposal)
        try:
            if checkpoint is not None:
                self._preload_checkpoint_identities(checkpoint)
            self._governance = self._reuse_or_acquire_governance(checkpoint)
            completed = self._admit_checkpoint(checkpoint)
        except BoundaryViolationError:
            raise
        except ProofContextError as exc:
            return self._early_stop(exc)
        start = len(completed)
        inbound = completed[-1] if completed else None
        self._applied = any(
            artifact.stage == APPLY_STAGE and artifact.status == "succeeded"
            for artifact in completed
        )
        if start and tuple(artifact.stage for artifact in completed) != STAGES[:start]:
            raise BoundaryViolationError(
                "checkpoint is not a prefix of the frozen lifecycle",
                details={"stage": completed[-1].stage if completed else "identify-operator"},
            )
        if start >= len(STAGES):
            return self._finalize(completed, status=completed[-1].status)
        for stage in STAGES[start:]:
            halt = self._halt_signal(stage)
            if halt is not None:
                return self._stop(completed, halt, inbound=inbound, stage=stage)
            try:
                raw = self._invoke(stage, proposal)
                artifact = self._admit_artifact(stage, raw, inbound)
                self._identities = artifact.identities
                self._stage_gates(artifact)
                self._persist(artifact, published=False)
                completed = (*completed, artifact)
                if artifact.stage == APPLY_STAGE and artifact.status == "succeeded":
                    self._applied = True
                    self._bind_worktree(artifact)
                inbound = artifact
            except ProofContextError as exc:
                return self._stop(
                    completed,
                    {
                        "status": exc.status,
                        "error": exc.code,
                        "message": str(exc),
                    },
                    inbound=inbound,
                    stage=stage,
                )
            except PolicyError as exc:
                code = exc.reason if exc.reason in ERRORS else "boundary_violation"
                return self._stop(
                    completed,
                    {"status": error_status(code), "error": code, "message": str(exc)},
                    inbound=inbound,
                    stage=stage,
                )
            except Exception as exc:  # noqa: BLE001 - map provider faults into taxonomy
                typed = from_provider_error(exc)
                return self._stop(
                    completed,
                    {
                        "status": typed.status,
                        "error": typed.code,
                        "message": str(typed),
                    },
                    inbound=inbound,
                    stage=stage,
                )
            if not _continues(stage, artifact.status):
                return self._stop(
                    completed,
                    {
                        "status": artifact.status,
                        "error": artifact.error,
                        "message": artifact.stage,
                    },
                    inbound=artifact,
                    stage=stage,
                    already_recorded=True,
                )
            if stage == "escalate":
                unresolved = self._unresolved_escalation(completed)
                if unresolved is not None:
                    return self._stop(
                        completed,
                        unresolved,
                        inbound=artifact,
                        stage=stage,
                        already_recorded=True,
                    )
        return self._finalize(completed, status=completed[-1].status)

    def resume(self, checkpoint: Mapping[str, Any]) -> LifecycleRecord:
        return self.run(checkpoint=checkpoint)

    def _reject_bypass(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise MalformedError("payload must be a mapping")
        for key in _BYPASS_KEYS:
            if key in payload and payload[key]:
                raise BoundaryViolationError(
                    "adapters cannot bypass or self-approve a lifecycle stage",
                    details={"reason": key},
                )

    def _check_payload_identities(self, payload: Mapping[str, Any]) -> None:
        raw = payload.get("identities")
        if isinstance(raw, LifecycleIdentities):
            self._identities = merge_identities(self._identities, raw)
        elif isinstance(raw, Mapping):
            self._identities = merge_identities(
                self._identities,
                LifecycleIdentities.from_mapping(raw),
            )
        for name in (*_STABLE_IDENTITY_FIELDS, *_BIND_ONCE_FIELDS):
            if name not in payload:
                continue
            expected = getattr(self._identities, name)
            actual = payload[name]
            if actual is None or expected is None:
                continue
            if str(actual) != str(expected):
                raise IdentityInconsistentError(
                    f"payload identity field {name} drifted",
                    details={"field": name},
                )

    def _early_stop(self, exc: ProofContextError) -> LifecycleRecord:
        if self._governance is None:
            self._governance = GovernanceReceipts(
                lease={"valid": False, "lease_id": self._identities.lease_id},
                fence={"valid": False, "fence_id": self._identities.fence_id},
                worktree={},
                schedule={"admitted": False},
            )
        stage = STAGES[0]
        artifact = StageArtifact(
            schema=STAGE_ARTIFACT_SCHEMA,
            stage=stage,
            status=exc.status,
            identities=self._identities,
            artifact_cid=self._stop_cid(stage),
            provenance="live" if self._mode != "simulation" else "simulated",
            payload={"halted": True, "published": False, "message": str(exc)},
            error=exc.code,
        )
        try:
            self._persist(artifact, published=False)
        except Exception:  # noqa: BLE001 - persistence failure still cannot publish
            if self._identities.evidence_cid is None:
                self._identities = replace(
                    self._identities,
                    evidence_cid=self._stop_cid("evidence"),
                )
        return self._record(
            (artifact,),
            status=exc.status,
            error=exc.code,
            published=False,
            sealed=False,
        )

    def _preload_checkpoint_identities(self, checkpoint: Mapping[str, Any]) -> None:
        raw = checkpoint.get("identities")
        if isinstance(raw, Mapping):
            self._identities = merge_identities(
                self._identities,
                LifecycleIdentities.from_mapping(raw),
            )

    def _reuse_or_acquire_governance(
        self,
        checkpoint: Mapping[str, Any] | None,
    ) -> GovernanceReceipts:
        if checkpoint is None:
            return self._acquire_governance()
        raw = checkpoint.get("governance")
        if not isinstance(raw, Mapping):
            return self._acquire_governance()
        lease = raw.get("lease") if isinstance(raw.get("lease"), Mapping) else {}
        fence = raw.get("fence") if isinstance(raw.get("fence"), Mapping) else {}
        if not lease.get("valid") or not lease.get("lease_id"):
            return self._acquire_governance()
        if not fence.get("valid") or not fence.get("fence_id"):
            return self._acquire_governance()
        updates = {
            "lease_id": str(lease["lease_id"]),
            "fence_id": str(fence["fence_id"]),
        }
        current_lease = self._identities.lease_id
        current_fence = self._identities.fence_id
        if current_lease and current_lease != updates["lease_id"]:
            raise IdentityInconsistentError(
                "checkpoint lease identity drifted",
                details={"field": "lease_id"},
            )
        if current_fence and current_fence != updates["fence_id"]:
            raise IdentityInconsistentError(
                "checkpoint fence identity drifted",
                details={"field": "fence_id"},
            )
        self._identities = replace(self._identities, **updates)
        receipts = GovernanceReceipts(
            lease=lease,
            fence=fence,
            worktree=raw.get("worktree") if isinstance(raw.get("worktree"), Mapping) else {},
            schedule=raw.get("schedule")
            if isinstance(raw.get("schedule"), Mapping)
            else {"admitted": True},
        )
        self._persist_mapping(
            {
                "kind": "governance",
                "published": False,
                "reused": True,
                "lease": dict(lease),
                "fence": dict(fence),
            },
            published=False,
        )
        return receipts

    def _acquire_governance(self) -> GovernanceReceipts:
        ports = self._ports.governance
        schedule = _as_mapping(
            ports.admit_schedule(self._identities, self._repository)
        )
        if schedule.get("admitted") is False or schedule.get("status") in STOP_PUBLICATION_STATUSES:
            raise LifecycleError(
                "schedule did not admit this run",
                code=str(schedule.get("error") or "unavailable_capability"),
                details={"stage": "identify-operator", "capability": "schedule"},
            )
        lease = _as_mapping(ports.acquire_lease(self._identities, self._repository))
        fence = _as_mapping(ports.acquire_fence(self._identities, self._repository))
        if lease.get("valid") is False or not lease.get("lease_id"):
            raise LifecycleError(
                "lease is not valid",
                code="unavailable_capability",
                details={"stage": "identify-operator", "lease_id": str(lease.get("lease_id") or "")},
            )
        if fence.get("valid") is False or not fence.get("fence_id"):
            raise LifecycleError(
                "fence is not valid",
                code="unavailable_capability",
                details={"stage": "identify-operator"},
            )
        updates = {
            "lease_id": str(lease["lease_id"]),
            "fence_id": str(fence["fence_id"]),
        }
        self._identities = replace(self._identities, **updates)
        receipts = GovernanceReceipts(
            lease=lease,
            fence=fence,
            worktree={},
            schedule=schedule,
        )
        self._persist_mapping(
            {
                "kind": "governance",
                "published": False,
                "lease": dict(lease),
                "fence": dict(fence),
                "schedule": dict(schedule),
            },
            published=False,
        )
        return receipts

    def _halt_signal(self, stage: str) -> Mapping[str, Any] | None:
        signal = _as_mapping(
            self._ports.governance.check_cancellation(self._identities, self._repository)
        )
        status = signal.get("status", "succeeded")
        if status in {"timeout", "cancelled", "unavailable"}:
            return {
                "status": status,
                "error": _error_for_status(str(status), _optional_str(signal.get("error"))),
                "message": f"{stage} halted",
            }
        lease = self._governance.lease if self._governance else {}
        fence = self._governance.fence if self._governance else {}
        if lease.get("valid") is False or fence.get("valid") is False:
            return {
                "status": "stale",
                "error": "stale_root",
                "message": "lease or fence expired",
            }
        return None

    def _invoke(
        self,
        stage: str,
        proposal: Mapping[str, Any] | None,
    ) -> StageArtifact:
        identities = self._identities
        repository = self._repository
        ports = self._ports
        if stage == "identify-operator":
            return ports.operator.identify(identities, repository)
        if stage == "resolve-repository":
            return ports.repository.resolve(identities, repository)
        if stage == "scan-semantic":
            return ports.semantic.scan(identities, repository)
        if stage == "invalidate":
            return ports.semantic.invalidate(identities, repository)
        if stage == "context-pack":
            return ports.semantic.context_pack(identities, repository)
        if stage == "sufficiency":
            return ports.semantic.sufficiency(identities, repository)
        if stage == "route":
            return ports.route.route(identities, repository)
        if stage == "proposal":
            return ports.proposal.propose(identities, repository, proposal)
        if stage == "scope-check":
            return ports.scope.check(identities, repository, proposal)
        if stage == "isolated-apply":
            return ports.worktree.apply(identities, repository, proposal)
        if stage == "impact":
            return ports.semantic.impact(identities, repository)
        if stage == "incremental-verify":
            return ports.verification.verify(identities, repository)
        if stage == "escalate":
            return ports.semantic.escalate(identities, repository)
        if stage == "assurance":
            return ports.assurance.assure(identities, repository)
        if stage == "seal":
            return ports.sealing.seal(identities, repository)
        if stage == "disposition":
            return ports.disposition.decide(identities, repository)
        raise UnknownFieldError(f"unknown lifecycle stage {stage!r}")

    def _admit_artifact(
        self,
        stage: str,
        raw: Any,
        inbound: StageArtifact | None,
    ) -> StageArtifact:
        if not isinstance(raw, StageArtifact):
            if type(raw).__name__ == "StageArtifact" and hasattr(raw, "to_mapping"):
                raw = StageArtifact.from_mapping(raw.to_mapping())
            else:
                raise MalformedError("lifecycle ports must return StageArtifact")
        if raw.stage != stage:
            raise IdentityInconsistentError(
                f"port returned {raw.stage!r} for {stage!r}",
                details={"stage": stage},
            )
        expected_inbound = inbound.artifact_cid if inbound is not None else None
        if inbound is None and raw.inbound_cid is not None:
            raise IdentityInconsistentError(
                "first stage cannot consume a prior artifact",
                details={"stage": stage},
            )
        if inbound is not None and raw.inbound_cid not in {None, expected_inbound}:
            raise IdentityInconsistentError(
                "stage inbound CID does not match the prior artifact",
                details={"stage": stage},
            )
        merged = merge_identities(self._identities, raw.identities)
        if self._mode in LIVE_MODES and raw.provenance == "simulated":
            raise LifecycleError(
                "simulated evidence cannot enter production or supervised modes",
                code="simulated_promoted",
                details={"stage": stage},
            )
        if self._mode in LIVE_MODES and raw.status == "simulated":
            raise LifecycleError(
                "simulated status cannot enter production or supervised modes",
                code="simulated_promoted",
                details={"stage": stage},
            )
        return replace(
            raw,
            identities=merged,
            inbound_cid=expected_inbound,
            error=_error_for_status(raw.status, raw.error) if raw.status != "succeeded" else None,
        )

    def _stage_gates(self, artifact: StageArtifact) -> None:
        payload = artifact.payload if isinstance(artifact.payload, Mapping) else {}
        if _truthy(payload.get("self_approved")) or _truthy(payload.get("adapter_approved")):
            raise BoundaryViolationError(
                "an adapter cannot approve its own patch",
                details={"stage": artifact.stage},
            )
        adapter_id = _optional_str(payload.get("adapter_id"))
        approver_id = _optional_str(payload.get("approver_id"))
        if adapter_id and approver_id and adapter_id == approver_id:
            raise BoundaryViolationError(
                "an adapter cannot approve its own patch",
                details={"stage": artifact.stage},
            )
        if payload.get("credentials") or payload.get("mutation_authority"):
            raise BoundaryViolationError(
                "route or proposal artifacts must not carry credentials or mutation authority",
                details={"stage": artifact.stage},
            )
        if artifact.stage == APPLY_STAGE:
            self._apply_gates(payload)
        if artifact.stage == VERIFY_STAGE:
            self._verify_gates(payload)
        if artifact.stage == SEAL_STAGE and artifact.status == "succeeded":
            if _truthy(payload.get("unsealed")) or payload.get("sealed") is False:
                raise BoundaryViolationError(
                    "production and supervised cannot accept an unsealed patch",
                    details={"stage": artifact.stage},
                )
            seal_cid = _optional_str(payload.get("seal_cid"))
            if not seal_cid:
                raise BoundaryViolationError(
                    "seal stage must emit a seal identity",
                    details={"stage": artifact.stage},
                )
        if artifact.stage == DISPOSITION_STAGE and artifact.status == "succeeded":
            if self._mode in LIVE_MODES and not _truthy(payload.get("sealed")):
                if not _optional_str(payload.get("seal_cid")):
                    raise BoundaryViolationError(
                        "production and supervised cannot accept an unsealed patch",
                        details={"stage": artifact.stage},
                    )

    def _apply_gates(self, payload: Mapping[str, Any]) -> None:
        if payload.get("disposable") is False:
            raise BoundaryViolationError(
                "apply must use an isolated disposable worktree",
                details={"stage": APPLY_STAGE},
            )
        target = _optional_str(payload.get("target_ref")) or _optional_str(
            payload.get("canonical_ref")
        )
        if target and _protected_ref(target) and _truthy(payload.get("mutates_canonical")):
            raise BoundaryViolationError(
                "protected canonical branches cannot be mutated",
                details={"stage": APPLY_STAGE},
            )
        if _truthy(payload.get("canonical_mutated")):
            raise BoundaryViolationError(
                "protected canonical branches cannot be mutated",
                details={"stage": APPLY_STAGE},
            )
        if _truthy(payload.get("concealed_partial_effect")):
            raise BoundaryViolationError(
                "partial effects cannot be concealed",
                details={"stage": APPLY_STAGE, "reason": "partial_effect"},
            )

    def _verify_gates(self, payload: Mapping[str, Any]) -> None:
        if _truthy(payload.get("selected_independently")):
            raise BoundaryViolationError(
                "tests and proofs cannot be selected independently of canonical planners",
                details={"stage": VERIFY_STAGE},
            )
        planner = _optional_str(payload.get("planner_authority"))
        if planner and planner != "canonical":
            raise BoundaryViolationError(
                "tests and proofs cannot be selected independently of canonical planners",
                details={"stage": VERIFY_STAGE},
            )

    def _unresolved_escalation(
        self,
        completed: Sequence[StageArtifact],
    ) -> Mapping[str, Any] | None:
        verify = next(
            (item for item in completed if item.stage == VERIFY_STAGE),
            None,
        )
        escalate = next(
            (item for item in completed if item.stage == "escalate"),
            None,
        )
        if verify is None or verify.status == "succeeded":
            return None
        payload = escalate.payload if escalate is not None else {}
        if escalate is not None and escalate.status == "succeeded" and _truthy(
            payload.get("resolved")
        ):
            return None
        return {
            "status": verify.status,
            "error": verify.error or "context_insufficient",
            "message": "escalation did not resolve insufficient context",
        }

    def _bind_worktree(self, artifact: StageArtifact) -> None:
        payload = artifact.payload if isinstance(artifact.payload, Mapping) else {}
        worktree_id = _optional_str(payload.get("worktree_id"))
        if worktree_id:
            if (
                self._identities.worktree_id
                and self._identities.worktree_id != worktree_id
            ):
                raise IdentityInconsistentError(
                    "identity field worktree_id is already bound",
                    details={"field": "worktree_id"},
                )
            self._identities = replace(self._identities, worktree_id=worktree_id)
        self._canonical_head = _optional_str(payload.get("canonical_head"))
        current = self._governance or GovernanceReceipts(
            lease={"valid": bool(self._identities.lease_id), "lease_id": self._identities.lease_id},
            fence={"valid": bool(self._identities.fence_id), "fence_id": self._identities.fence_id},
            worktree={},
            schedule={"admitted": True},
        )
        self._governance = GovernanceReceipts(
            lease=current.lease,
            fence=current.fence,
            worktree={
                "worktree_id": worktree_id,
                "disposable": payload.get("disposable", True),
                "canonical_mutated": False,
                "canonical_head": self._canonical_head,
                "receipt_cid": artifact.artifact_cid,
            },
            schedule=current.schedule,
        )

    def _persist(self, artifact: StageArtifact, *, published: bool) -> Mapping[str, Any]:
        receipt = _as_mapping(
            self._ports.persistence.persist(artifact, published=published)
        )
        evidence_cid = _optional_str(receipt.get("evidence_cid"))
        if evidence_cid:
            self._identities = replace(
                self._identities,
                evidence_cid=_admit_lifecycle_cid(evidence_cid),
            )
        frozen = MappingProxyType(
            {
                "stage": artifact.stage,
                "status": artifact.status,
                "published": published,
                "artifact_cid": artifact.artifact_cid,
                "evidence_cid": evidence_cid,
            }
        )
        self._persisted.append(frozen)
        return receipt

    def _persist_mapping(self, payload: Mapping[str, Any], *, published: bool) -> None:
        receipt = _as_mapping(
            self._ports.persistence.persist(payload, published=published)
        )
        self._persisted.append(
            MappingProxyType({"kind": payload.get("kind"), "published": published, **dict(receipt)})
        )

    def _discard_worktree(self) -> bool:
        if not self._applied:
            return True
        try:
            result = _as_mapping(
                self._ports.worktree.discard(self._identities, self._repository)
            )
        except Exception:  # noqa: BLE001 - discard failure is a partial effect
            return False
        return result.get("discarded") is True

    def _stop(
        self,
        completed: Sequence[StageArtifact],
        halt: Mapping[str, Any],
        *,
        inbound: StageArtifact | None,
        stage: str,
        already_recorded: bool = False,
    ) -> LifecycleRecord:
        status = admit_status(str(halt.get("status") or "rejected"))
        error = _error_for_status(status, _optional_str(halt.get("error")))
        discarded = self._discard_worktree()
        if self._applied and not discarded:
            status = "partial_effect"
            error = "partial_effect"
        artifacts = tuple(completed)
        if not already_recorded or not artifacts or artifacts[-1].stage != stage:
            stop_artifact = StageArtifact(
                schema=STAGE_ARTIFACT_SCHEMA,
                stage=stage if stage in STAGES else artifacts[-1].stage if artifacts else STAGES[0],
                status=status,
                identities=self._identities,
                artifact_cid=self._stop_cid(stage),
                provenance="live" if self._mode != "simulation" else "simulated",
                payload={
                    "halted": True,
                    "published": False,
                    "applied": self._applied,
                    "discarded": discarded,
                    "message": halt.get("message"),
                },
                inbound_cid=inbound.artifact_cid if inbound is not None else None,
                error=error,
            )
            if not already_recorded:
                self._persist(stop_artifact, published=False)
                artifacts = (*artifacts, stop_artifact)
        elif artifacts:
            self._persist(artifacts[-1], published=False)
        return self._record(
            artifacts,
            status=status,
            error=error,
            published=False,
            sealed=False,
        )

    def _stop_cid(self, stage: str) -> str:
        return mint_lifecycle_cid(
            {
                "stage": stage,
                "run_id": self._identities.run_id,
                "trace_id": self._identities.trace_id,
                "kind": "halt",
            }
        )

    def _finalize(
        self,
        completed: Sequence[StageArtifact],
        *,
        status: str,
    ) -> LifecycleRecord:
        artifacts = tuple(completed)
        stages = tuple(artifact.stage for artifact in artifacts)
        sealed = any(
            artifact.stage == SEAL_STAGE and artifact.status == "succeeded"
            for artifact in artifacts
        )
        seal_cid = None
        for artifact in artifacts:
            if artifact.stage == SEAL_STAGE:
                payload = artifact.payload if isinstance(artifact.payload, Mapping) else {}
                seal_cid = _optional_str(payload.get("seal_cid")) or artifact.artifact_cid
        disposition = artifacts[-1] if artifacts else None
        final_status = disposition.status if disposition is not None else status
        provenance = disposition.provenance if disposition is not None else "live"
        error = disposition.error if disposition is not None else _error_for_status(final_status, None)
        if self._mode == "simulation":
            provenance = "simulated"
            if final_status == "succeeded":
                final_status = "simulated"
                error = "simulated_promoted"
        published = False
        if stages == STAGES and final_status == "succeeded" and sealed:
            evidence = {
                "mode": self._mode,
                "provenance": provenance,
                "status": "succeeded",
                "artifact_cid": disposition.artifact_cid if disposition else self._identities.repository_state_cid,
                "seal_cid": seal_cid,
                "sealed": True,
                "signature": "lifecycle-coordinator",
                "signature_required": self._mode in LIVE_MODES,
                "self_approved": False,
                "parents": [artifact.artifact_cid for artifact in artifacts],
            }
            try:
                decision = admit_evidence(self._mode, evidence)
            except PolicyError as exc:
                code = exc.reason if exc.reason in ERRORS else "boundary_violation"
                return self._record(
                    artifacts,
                    status=error_status(code),
                    error=code,
                    published=False,
                    sealed=sealed,
                )
            published = bool(
                decision.accepted
                and decision.admitted
                and self._mode in LIVE_MODES
                and provenance == "live"
            )
            if published:
                error = None
            elif self._mode in LIVE_MODES:
                final_status = decision.status
                error = decision.error or "boundary_violation"
                published = False
        if published:
            if disposition is not None:
                self._persist(disposition, published=True)
        else:
            if self._applied:
                discarded = self._discard_worktree()
                if not discarded:
                    final_status = "partial_effect"
                    error = "partial_effect"
                    published = False
        return self._record(
            artifacts,
            status=final_status,
            error=error,
            published=published,
            sealed=sealed,
        )

    def _record(
        self,
        artifacts: Sequence[StageArtifact],
        *,
        status: str,
        error: str | None,
        published: bool,
        sealed: bool,
    ) -> LifecycleRecord:
        evidence_cid = self._identities.evidence_cid
        if evidence_cid is None:
            evidence_cid = mint_lifecycle_cid(
                {
                    "run_id": self._identities.run_id,
                    "trace_id": self._identities.trace_id,
                    "stages": [artifact.stage for artifact in artifacts],
                }
            )
            self._identities = replace(self._identities, evidence_cid=evidence_cid)
        if published and status == "succeeded" and self._identities.patch_id is None:
            raise IdentityInconsistentError(
                "published results must bind patch identity",
                details={"reason": "patch_id"},
            )
        provenance = "simulated" if self._mode == "simulation" else "live"
        if published:
            provenance = "live"
        identities = self._identities
        if published or is_terminal(status):
            # Terminal/published results bind evidence; patch identity is
            # required only for patch-bearing statuses.
            if identities.evidence_cid is None:
                identities = replace(identities, evidence_cid=evidence_cid)
        governance = self._governance or GovernanceReceipts(
            lease={"valid": False, "lease_id": identities.lease_id},
            fence={"valid": False, "fence_id": identities.fence_id},
            worktree={},
            schedule={"admitted": False},
        )
        return LifecycleRecord(
            schema=LIFECYCLE_RECORD_SCHEMA,
            status=status,
            identities=identities,
            mode=self._mode,
            provenance=provenance,
            published=published,
            sealed=sealed,
            stages=tuple(artifact.stage for artifact in artifacts),
            artifacts=tuple(artifacts),
            governance=governance,
            evidence_cid=evidence_cid,
            error=None if published else error,
            payload={
                "published": published,
                "applied": self._applied,
                "sole_authority": True,
                "persisted": [dict(item) for item in self._persisted],
            },
        )

    def _admit_checkpoint(
        self,
        checkpoint: Mapping[str, Any] | None,
    ) -> tuple[StageArtifact, ...]:
        if checkpoint is None:
            return ()
        payload = _as_mapping(checkpoint)
        schema = payload.get("schema")
        if schema not in {None, CHECKPOINT_SCHEMA}:
            raise SchemaMismatchError(
                f"checkpoint schema {schema!r} is not {CHECKPOINT_SCHEMA}"
            )
        if payload.get("mode") not in {None, self._mode}:
            raise IdentityInconsistentError(
                "checkpoint mode does not match the lifecycle mode",
                details={"field": "mode"},
            )
        raw_identities = payload.get("identities")
        if isinstance(raw_identities, Mapping):
            self._identities = merge_identities(
                self._identities,
                LifecycleIdentities.from_mapping(raw_identities),
            )
        completed_raw = payload.get("completed") or payload.get("artifacts") or ()
        if not isinstance(completed_raw, (list, tuple)):
            raise MalformedError("checkpoint completed stages must be a sequence")
        artifacts: list[StageArtifact] = []
        inbound: StageArtifact | None = None
        for index, item in enumerate(completed_raw):
            artifact = item if isinstance(item, StageArtifact) else StageArtifact.from_mapping(item)
            expected = STAGES[index] if index < len(STAGES) else None
            if artifact.stage != expected:
                raise BoundaryViolationError(
                    "checkpoint skips or reorders a frozen lifecycle stage",
                    details={"stage": artifact.stage},
                )
            admitted = self._admit_artifact(artifact.stage, artifact, inbound)
            artifacts.append(admitted)
            inbound = admitted
            self._identities = admitted.identities
            if admitted.stage == APPLY_STAGE and admitted.status == "succeeded":
                self._applied = True
                self._bind_worktree(admitted)
            if admitted.status != "succeeded" and not _continues(admitted.stage, admitted.status):
                break
        if payload.get("published") is True and tuple(item.stage for item in artifacts) != STAGES:
            raise BoundaryViolationError(
                "incomplete checkpoints cannot claim publication",
                details={"stage": artifacts[-1].stage if artifacts else STAGES[0]},
            )
        return tuple(artifacts)


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": LIFECYCLE_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "stages": STAGES,
        "stage_contracts": dict(STAGE_CONTRACTS),
        "modes": MODES,
        "live_modes": tuple(sorted(LIVE_MODES)),
        "provenances": PROVENANCES,
        "statuses": STATUSES,
        "stop_publication_statuses": STOP_PUBLICATION_STATUSES,
        "publication_requirements": PUBLICATION_REQUIREMENTS,
        "protected_refs": PROTECTED_REFS,
        "sole_authority": SOLE_AUTHORITY,
        "restart_aware": True,
        "sibling_layout_required": SIBLING_LAYOUT_REQUIRED,
        "provider_bound": PROVIDER_BOUND,
        "simulation_watermark": SIMULATION_WATERMARK,
        "forbidden_evidence": FORBIDDEN_EVIDENCE,
        "identity_fields": ("operator_id", *IDENTITY_FIELDS, "lease_id", "fence_id", "worktree_id"),
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "policy_cid": POLICY_CID,
        "result_state_cid": RESULT_STATE_CID,
    }
)
LIFECYCLE_CID: Final[str] = mint_lifecycle_cid(_DESCRIPTOR_BODY)
LIFECYCLE_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": LIFECYCLE_CID}
)


def lifecycle_descriptor() -> Mapping[str, Any]:
    return LIFECYCLE_DESCRIPTOR


def lifecycle_cid() -> str:
    return LIFECYCLE_CID


def frozen_lifecycle() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "stages": STAGES,
            "cid": LIFECYCLE_CID,
            "sole_authority": SOLE_AUTHORITY,
            "pcce_006_content_id": PCCE_006_CONTENT_ID,
            "policy_cid": POLICY_CID,
            "result_state_cid": RESULT_STATE_CID,
        }
    )


__all__ = [
    "APPLY_STAGE",
    "CHECKPOINT_SCHEMA",
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "DISPOSITION_STAGE",
    "LIFECYCLE_CID",
    "LIFECYCLE_DESCRIPTOR",
    "LIFECYCLE_RECORD_SCHEMA",
    "LIFECYCLE_SCHEMA",
    "LIVE_MODES",
    "MODES",
    "PCCE_006_CONTENT_ID",
    "POLICY_CID",
    "PROTECTED_REFS",
    "PROVIDER_BOUND",
    "PUBLICATION_REQUIREMENTS",
    "RESULT_STATE_CID",
    "SCHEMA",
    "SEAL_STAGE",
    "SIBLING_LAYOUT_REQUIRED",
    "SOLE_AUTHORITY",
    "STAGE_ARTIFACT_SCHEMA",
    "STAGE_CONTRACTS",
    "STAGES",
    "STATUSES",
    "STOP_PUBLICATION_STATUSES",
    "VERIFY_STAGE",
    "AssurancePort",
    "DispositionPort",
    "GovernancePort",
    "GovernanceReceipts",
    "LifecycleError",
    "LifecycleIdentities",
    "LifecyclePorts",
    "LifecycleRecord",
    "OperatorPort",
    "PatchLifecycle",
    "PersistencePort",
    "ProposalPort",
    "RepositoryPort",
    "RoutePort",
    "ScopePort",
    "SealingPort",
    "SemanticPort",
    "StageArtifact",
    "VerificationPort",
    "WorktreePort",
    "admit_stage",
    "frozen_lifecycle",
    "lifecycle_cid",
    "lifecycle_descriptor",
    "merge_identities",
    "mint_lifecycle_cid",
]
