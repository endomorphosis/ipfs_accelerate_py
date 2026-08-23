"""Versioned procedure registry with authorized expected-old CAS.

The registry owns lifecycle metadata, not procedure authority.  Procedure
bodies stay content-addressed by CID and are never persisted here.  Promotion,
rollback, and revocation each require an independent authorization, an
expected-old head, and an exact target.  A procedure cannot promote itself;
a certificate never grants promotion.

Durable metadata/events are written through an injected store so existing
DuckDB artifact/event owners remain the persistence authorities.  Tests use
the in-memory store.  History is immutable; corruption is quarantined and
recovered from intact snapshots.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Protocol

from ..proof.formal_verification_contracts import content_identity
from .certificate import (
    CertificateAdmission,
    CurrentCertificateContext,
    ProcedureCertificateVerifier,
)
from .contracts import (
    ArtifactBindings,
    ArtifactState,
    ProcedureCertificate,
    ProcedureContractError,
    ProcedureDeprecationReceipt as DeprecationReceiptArtifact,
    ProcedurePromotionReceipt as PromotionReceiptArtifact,
    ProcedureRegistryRevision as RegistryRevisionArtifact,
    ProcedureRollbackReceipt as RollbackReceiptArtifact,
    ProcedureVersion,
    RiskClass,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _strings,
    _text,
)
from .verifier import FORBIDDEN_SELF_PRODUCERS


REGISTRY_REVISION: Final[str] = "ProcedureRegistry@1"
REGISTRY_REVISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/registry-revision@1"
)
REGISTRY_CAS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/registry-cas@1"
)
REGISTRY_AUTHORIZATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/registry-authorization@1"
)
REGISTRY_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/registry-event@1"
)
DRIFT_ACTOR_ID: Final[str] = "procedure-registry-drift-monitor@1"
EMPTY_REVISION_ID: Final[str] = ""
MAX_REGISTRY_PROCEDURES: Final[int] = 4_096
MAX_REVISIONS_PER_PROCEDURE: Final[int] = 128
MAX_LOOKUP_RESULTS: Final[int] = 256
MAX_EVENTS: Final[int] = 8_192
MAX_QUARANTINE: Final[int] = 1_024

_RISK_ORDER: Final[dict[RiskClass, int]] = {
    RiskClass.OBSERVATION_ONLY: 0,
    RiskClass.REVERSIBLE_LOCAL: 1,
    RiskClass.REPOSITORY_WRITE: 2,
    RiskClass.PUBLIC_CONTRACT: 3,
    RiskClass.AUTHORITY_OR_SECURITY: 4,
}

_SELF_ACTOR_ALIASES: Final[frozenset[str]] = FORBIDDEN_SELF_PRODUCERS | frozenset(
    {
        "self-promoted",
        "self-promotion",
        "self-authorized",
        "procedure-self",
    }
)


class RegistryLifecycleState(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed registry states from the procedure-compiler plan."""

    CANDIDATE = "candidate"
    DEVELOPMENT = "development"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    DEGRADED = "degraded"
    STALE = "stale"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class RegistryOperation(str, Enum):  # noqa: UP042 - package supports Python 3.8
    REGISTER = "register"
    ADVANCE = "advance"
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    REVOKE = "revoke"
    DEMOTE = "demote"
    RECOVER = "recover"


class RegistryCASOutcome(str, Enum):  # noqa: UP042 - package supports Python 3.8
    COMMITTED = "committed"
    STALE = "stale"
    CONFLICT = "conflict"
    NOOP = "noop"
    QUARANTINED = "quarantined"


class ProcedureRegistryError(ProcedureContractError):
    """A registry request, authorization, or stored row is unsafe."""


class RegistryAuthorizationError(ProcedureRegistryError):
    """Independent authorization is missing, mismatched, or self-granted."""


class RegistryCASError(ProcedureRegistryError):
    """Expected-old compare-and-swap lost or was unauthorized."""

    def __init__(self, message: str, *, cas: "RegistryCAS" | None = None) -> None:
        super().__init__(message)
        self.cas = cas


class RegistryCorruptionError(ProcedureRegistryError):
    """Stored metadata failed integrity checks and was quarantined."""

    def __init__(
        self,
        message: str,
        *,
        procedure_id: str = "",
        revision_id: str = "",
        reason_code: str = "corrupt_revision",
    ) -> None:
        super().__init__(message)
        self.procedure_id = procedure_id
        self.revision_id = revision_id
        self.reason_code = reason_code


class RegistryNotFoundError(ProcedureRegistryError):
    """Requested procedure, revision, or certificate is absent."""


REGISTER_STATES: Final[frozenset[RegistryLifecycleState]] = frozenset(
    {
        RegistryLifecycleState.CANDIDATE,
        RegistryLifecycleState.DEVELOPMENT,
        RegistryLifecycleState.SHADOW,
        RegistryLifecycleState.REJECTED,
    }
)
PROMOTABLE_STATES: Final[frozenset[RegistryLifecycleState]] = frozenset(
    {
        RegistryLifecycleState.CANDIDATE,
        RegistryLifecycleState.DEVELOPMENT,
        RegistryLifecycleState.SHADOW,
    }
)
TERMINAL_STATES: Final[frozenset[RegistryLifecycleState]] = frozenset(
    {
        RegistryLifecycleState.REVOKED,
        RegistryLifecycleState.REJECTED,
    }
)
USABLE_STATES: Final[frozenset[RegistryLifecycleState]] = frozenset(
    {RegistryLifecycleState.PROMOTED}
)
_ADVANCE_ORDER: Final[dict[RegistryLifecycleState, frozenset[RegistryLifecycleState]]] = {
    RegistryLifecycleState.CANDIDATE: frozenset(
        {
            RegistryLifecycleState.DEVELOPMENT,
            RegistryLifecycleState.SHADOW,
            RegistryLifecycleState.REJECTED,
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.REVOKED,
        }
    ),
    RegistryLifecycleState.DEVELOPMENT: frozenset(
        {
            RegistryLifecycleState.SHADOW,
            RegistryLifecycleState.REJECTED,
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.REVOKED,
        }
    ),
    RegistryLifecycleState.SHADOW: frozenset(
        {
            RegistryLifecycleState.DEGRADED,
            RegistryLifecycleState.REJECTED,
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.REVOKED,
        }
    ),
    RegistryLifecycleState.PROMOTED: frozenset(
        {
            RegistryLifecycleState.SUPERSEDED,
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.DEGRADED,
            RegistryLifecycleState.REVOKED,
        }
    ),
    RegistryLifecycleState.DEGRADED: frozenset(
        {
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.REVOKED,
            RegistryLifecycleState.SUPERSEDED,
            RegistryLifecycleState.SHADOW,
        }
    ),
    RegistryLifecycleState.STALE: frozenset(
        {
            RegistryLifecycleState.REVOKED,
            RegistryLifecycleState.SUPERSEDED,
            RegistryLifecycleState.REJECTED,
        }
    ),
    RegistryLifecycleState.SUPERSEDED: frozenset({RegistryLifecycleState.REVOKED}),
    RegistryLifecycleState.REVOKED: frozenset(),
    RegistryLifecycleState.REJECTED: frozenset(),
}


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ProcedureRegistryError("{} must be a boolean".format(field_name))
    return value


def _state(value: Any) -> RegistryLifecycleState:
    return _enum(value, RegistryLifecycleState, "state")


def _operation(value: Any) -> RegistryOperation:
    return _enum(value, RegistryOperation, "operation")


def _version_tuple(version: ProcedureVersion) -> tuple[int, int, int]:
    return (version.major, version.minor, version.patch)


def _risk_rank(value: RiskClass) -> int:
    return _RISK_ORDER[value]


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(item for item in values if item)))


def _certificate_subject_actors(certificate: ProcedureCertificate) -> frozenset[str]:
    identities = {
        certificate.content_id,
        certificate.procedure_cid,
        certificate.task_family_cid,
        certificate.counterexample_set_cid,
        certificate.held_out_evaluation_cid,
        certificate.shadow_evaluation_cid,
        *certificate.source_episode_cids,
        *certificate.specification_cids,
        *certificate.proof_receipt_cids,
        *certificate.test_receipt_cids,
        *certificate.adversarial_assurance_cids,
    }
    return frozenset(item.lower() for item in identities if item)


def _self_actors(procedure_id: str, procedure_cid: str, certificate: ProcedureCertificate) -> frozenset[str]:
    return _SELF_ACTOR_ALIASES | _certificate_subject_actors(certificate) | frozenset(
        {
            procedure_id.lower(),
            procedure_cid.lower(),
        }
    )


@dataclass(frozen=True)
class RegistryAuthorization:
    """Independent admission for one exact registry mutation."""

    actor_id: str
    decision_cid: str
    operation: RegistryOperation
    target_procedure_cid: str
    expected_old_revision_id: str = EMPTY_REVISION_ID
    target_revision_id: str = EMPTY_REVISION_ID
    granted: bool = False
    issued_at_ms: int = 0
    schema: str = REGISTRY_AUTHORIZATION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id"))
        object.__setattr__(self, "decision_cid", _identifier(self.decision_cid, "decision_cid"))
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(
            self,
            "target_procedure_cid",
            _identifier(self.target_procedure_cid, "target_procedure_cid"),
        )
        object.__setattr__(
            self,
            "expected_old_revision_id",
            _identifier(
                self.expected_old_revision_id,
                "expected_old_revision_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "target_revision_id",
            _identifier(self.target_revision_id, "target_revision_id", required=False),
        )
        object.__setattr__(self, "granted", _bool(self.granted, "granted"))
        object.__setattr__(
            self, "issued_at_ms", _nonnegative_int(self.issued_at_ms, "issued_at_ms")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != REGISTRY_AUTHORIZATION_SCHEMA:
            raise ProcedureRegistryError("unsupported registry authorization schema")
        if not self.granted:
            raise RegistryAuthorizationError("registry authorization is not granted")

    @property
    def authorization_cid(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "decision_cid": self.decision_cid,
            "expected_old_revision_id": self.expected_old_revision_id,
            "granted": True,
            "issued_at_ms": self.issued_at_ms,
            "operation": self.operation.value,
            "schema": self.schema,
            "target_procedure_cid": self.target_procedure_cid,
            "target_revision_id": self.target_revision_id,
        }


@dataclass(frozen=True)
class ProcedureRegistryRevision:
    """Immutable catalog row.  Procedure IR is referenced by CID only."""

    procedure_id: str
    procedure_cid: str
    certificate_cid: str
    task_family_cid: str
    version: ProcedureVersion
    state: RegistryLifecycleState
    bindings: ArtifactBindings
    capability_ids: tuple[str, ...]
    risk_ceiling: RiskClass
    repository_families: tuple[str, ...]
    supported_language_classes: tuple[str, ...]
    supported_framework_classes: tuple[str, ...]
    generation: int
    predecessor_revision_id: str
    rollback_target_revision_id: str
    expected_old_revision_id: str
    actor_id: str
    authorization_cid: str
    operation: RegistryOperation
    created_at_ms: int
    schema: str = REGISTRY_REVISION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "procedure_id", _identifier(self.procedure_id, "procedure_id"))
        object.__setattr__(
            self, "procedure_cid", _identifier(self.procedure_cid, "procedure_cid")
        )
        object.__setattr__(
            self, "certificate_cid", _identifier(self.certificate_cid, "certificate_cid")
        )
        object.__setattr__(
            self, "task_family_cid", _identifier(self.task_family_cid, "task_family_cid")
        )
        object.__setattr__(
            self, "version", _nested(self.version, ProcedureVersion, "version")
        )
        object.__setattr__(self, "state", _state(self.state))
        object.__setattr__(
            self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings")
        )
        object.__setattr__(
            self,
            "capability_ids",
            _strings(self.capability_ids, "capability_ids", identifiers=True),
        )
        object.__setattr__(
            self, "risk_ceiling", _enum(self.risk_ceiling, RiskClass, "risk_ceiling")
        )
        for name in (
            "repository_families",
            "supported_language_classes",
            "supported_framework_classes",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        object.__setattr__(
            self, "generation", _positive_int(self.generation, "generation", maximum=1_000_000)
        )
        for name in (
            "predecessor_revision_id",
            "rollback_target_revision_id",
            "expected_old_revision_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id"))
        object.__setattr__(
            self,
            "authorization_cid",
            _identifier(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != REGISTRY_REVISION_SCHEMA:
            raise ProcedureRegistryError("unsupported registry revision schema")
        if self.procedure_cid == self.actor_id:
            raise RegistryAuthorizationError("a procedure cannot author its own registry revision")

    @property
    def revision_id(self) -> str:
        return content_identity(self._identity_payload())

    @property
    def lifecycle_usable(self) -> bool:
        return self.state in USABLE_STATES

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "actor_id": self.actor_id,
            "authorization_cid": self.authorization_cid,
            "bindings": self.bindings.to_dict(),
            "capability_ids": self.capability_ids,
            "certificate_cid": self.certificate_cid,
            "created_at_ms": self.created_at_ms,
            "expected_old_revision_id": self.expected_old_revision_id,
            "generation": self.generation,
            "operation": self.operation.value,
            "predecessor_revision_id": self.predecessor_revision_id,
            "procedure_cid": self.procedure_cid,
            "procedure_id": self.procedure_id,
            "repository_families": self.repository_families,
            "risk_ceiling": self.risk_ceiling.value,
            "rollback_target_revision_id": self.rollback_target_revision_id,
            "schema": self.schema,
            "state": self.state.value,
            "supported_framework_classes": self.supported_framework_classes,
            "supported_language_classes": self.supported_language_classes,
            "task_family_cid": self.task_family_cid,
            "version": self.version.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["revision_id"] = self.revision_id
        payload["lifecycle_usable"] = self.lifecycle_usable
        return payload

    def to_artifact(self) -> RegistryRevisionArtifact:
        return RegistryRevisionArtifact(
            bindings=self.bindings,
            artifact_version=self.generation,
            state=_artifact_state(self.state),
            subject_cid=self.procedure_cid,
            reference_cids=_sorted_unique(
                (
                    self.revision_id,
                    self.certificate_cid,
                    self.authorization_cid,
                    self.predecessor_revision_id,
                    self.rollback_target_revision_id,
                )
            ),
            labels=(self.procedure_id, self.state.value, self.operation.value),
            facts={
                "generation": self.generation,
                "procedure_id": self.procedure_id,
                "revision_id": self.revision_id,
                "task_family_cid": self.task_family_cid,
                "version": self.version.semantic_version,
            },
            created_at_ms=self.created_at_ms,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProcedureRegistryRevision":
        if not isinstance(payload, Mapping):
            raise ProcedureRegistryError("registry revision must be an object")
        record = cls(
            procedure_id=payload.get("procedure_id", ""),
            procedure_cid=payload.get("procedure_cid", ""),
            certificate_cid=payload.get("certificate_cid", ""),
            task_family_cid=payload.get("task_family_cid", ""),
            version=payload.get("version", {}),
            state=payload.get("state", RegistryLifecycleState.CANDIDATE),
            bindings=payload.get("bindings", {}),
            capability_ids=payload.get("capability_ids", ()),
            risk_ceiling=payload.get("risk_ceiling", RiskClass.OBSERVATION_ONLY),
            repository_families=payload.get("repository_families", ()),
            supported_language_classes=payload.get("supported_language_classes", ()),
            supported_framework_classes=payload.get("supported_framework_classes", ()),
            generation=payload.get("generation", 1),
            predecessor_revision_id=payload.get("predecessor_revision_id", ""),
            rollback_target_revision_id=payload.get("rollback_target_revision_id", ""),
            expected_old_revision_id=payload.get("expected_old_revision_id", ""),
            actor_id=payload.get("actor_id", ""),
            authorization_cid=payload.get("authorization_cid", ""),
            operation=payload.get("operation", RegistryOperation.REGISTER),
            created_at_ms=payload.get("created_at_ms", 0),
            schema=payload.get("schema", REGISTRY_REVISION_SCHEMA),
        )
        claimed = payload.get("revision_id")
        if claimed is not None and claimed != record.revision_id:
            raise RegistryCorruptionError(
                "stored revision identity does not match canonical content",
                procedure_id=record.procedure_id,
                revision_id=str(claimed),
                reason_code="revision_cid_mismatch",
            )
        return record


def _artifact_state(state: RegistryLifecycleState) -> ArtifactState:
    return ArtifactState(state.value)


@dataclass(frozen=True)
class RegistryCAS:
    """Authorized expected-old compare-and-swap receipt for one head."""

    accepted: bool
    stale: bool
    outcome: RegistryCASOutcome
    operation: RegistryOperation
    procedure_id: str
    expected_old_revision_id: str
    observed_revision_id: str
    target_procedure_cid: str
    target_revision_id: str
    new_revision_id: str
    rollback_target_revision_id: str
    authorization_cid: str
    actor_id: str
    generation: int
    reason_code: str
    schema: str = REGISTRY_CAS_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(self, "stale", _bool(self.stale, "stale"))
        object.__setattr__(
            self, "outcome", _enum(self.outcome, RegistryCASOutcome, "outcome")
        )
        object.__setattr__(self, "operation", _operation(self.operation))
        object.__setattr__(self, "procedure_id", _identifier(self.procedure_id, "procedure_id"))
        for name in (
            "expected_old_revision_id",
            "observed_revision_id",
            "target_revision_id",
            "new_revision_id",
            "rollback_target_revision_id",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "target_procedure_cid",
            _identifier(self.target_procedure_cid, "target_procedure_cid"),
        )
        object.__setattr__(
            self,
            "authorization_cid",
            _identifier(self.authorization_cid, "authorization_cid"),
        )
        object.__setattr__(self, "actor_id", _identifier(self.actor_id, "actor_id"))
        object.__setattr__(
            self, "generation", _nonnegative_int(self.generation, "generation")
        )
        object.__setattr__(self, "reason_code", _identifier(self.reason_code, "reason_code"))
        object.__setattr__(self, "schema", _text(self.schema, "schema"))
        if self.schema != REGISTRY_CAS_SCHEMA:
            raise ProcedureRegistryError("unsupported registry CAS schema")
        if self.accepted and self.stale:
            raise ProcedureRegistryError("accepted CAS cannot be stale")
        if self.stale and self.outcome not in {
            RegistryCASOutcome.STALE,
            RegistryCASOutcome.CONFLICT,
        }:
            raise ProcedureRegistryError("stale CAS must use a stale or conflict outcome")
        if self.accepted and self.outcome not in {
            RegistryCASOutcome.COMMITTED,
            RegistryCASOutcome.NOOP,
        }:
            raise ProcedureRegistryError("accepted CAS must commit or no-op")
        if self.accepted and not self.new_revision_id:
            raise ProcedureRegistryError("accepted CAS must bind the exact new revision")
        if self.accepted and not self.target_procedure_cid:
            raise ProcedureRegistryError("accepted CAS must bind the exact target")

    @property
    def cas_id(self) -> str:
        return content_identity(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "actor_id": self.actor_id,
            "authorization_cid": self.authorization_cid,
            "expected_old_revision_id": self.expected_old_revision_id,
            "generation": self.generation,
            "new_revision_id": self.new_revision_id,
            "observed_revision_id": self.observed_revision_id,
            "operation": self.operation.value,
            "outcome": self.outcome.value,
            "procedure_id": self.procedure_id,
            "reason_code": self.reason_code,
            "rollback_target_revision_id": self.rollback_target_revision_id,
            "schema": self.schema,
            "stale": self.stale,
            "target_procedure_cid": self.target_procedure_cid,
            "target_revision_id": self.target_revision_id,
        }

    def to_artifact(self, bindings: ArtifactBindings, *, created_at_ms: int) -> Any:
        if self.operation is RegistryOperation.PROMOTE:
            artifact_cls = PromotionReceiptArtifact
            artifact_state = (
                ArtifactState.PROMOTED if self.accepted else ArtifactState.REJECTED
            )
        elif self.operation is RegistryOperation.ROLLBACK:
            artifact_cls = RollbackReceiptArtifact
            artifact_state = (
                ArtifactState.PROMOTED if self.accepted else ArtifactState.REJECTED
            )
        elif self.operation is RegistryOperation.REVOKE:
            artifact_cls = DeprecationReceiptArtifact
            artifact_state = (
                ArtifactState.REVOKED if self.accepted else ArtifactState.REJECTED
            )
        else:
            artifact_cls = RegistryRevisionArtifact
            artifact_state = ArtifactState.CANDIDATE if self.accepted else ArtifactState.REJECTED
        references = _sorted_unique(
            (
                self.cas_id,
                self.new_revision_id,
                self.observed_revision_id,
                self.expected_old_revision_id,
                self.rollback_target_revision_id,
                self.authorization_cid,
            )
        )
        return artifact_cls(
            bindings=bindings,
            artifact_version=max(self.generation, 1),
            state=artifact_state,
            subject_cid=self.target_procedure_cid,
            reference_cids=references,
            labels=(self.procedure_id, self.operation.value, self.outcome.value),
            facts={
                "accepted": self.accepted,
                "cas_id": self.cas_id,
                "expected_old_revision_id": self.expected_old_revision_id,
                "new_revision_id": self.new_revision_id,
                "reason_code": self.reason_code,
                "stale": self.stale,
                "target_revision_id": self.target_revision_id,
            },
            created_at_ms=created_at_ms,
        )


@dataclass(frozen=True)
class RegistryFilter:
    """Closed, exact lookup constraints.  Empty fields are unconstrained."""

    procedure_id: str = ""
    procedure_cid: str = ""
    task_family_cid: str = ""
    states: tuple[RegistryLifecycleState, ...] = ()
    capability_ids: tuple[str, ...] = ()
    max_risk: RiskClass | None = None
    environment_id: str = ""
    repository_id: str = ""
    tree_id: str = ""
    policy_revision: str = ""
    language_classes: tuple[str, ...] = ()
    framework_classes: tuple[str, ...] = ()
    repository_families: tuple[str, ...] = ()
    version: ProcedureVersion | None = None
    usable_only: bool = True

    def __post_init__(self) -> None:
        for name in (
            "procedure_id",
            "procedure_cid",
            "task_family_cid",
            "environment_id",
            "repository_id",
            "tree_id",
            "policy_revision",
        ):
            object.__setattr__(
                self, name, _identifier(getattr(self, name), name, required=False)
            )
        states = tuple(
            _state(item) for item in (self.states or ())
        )
        object.__setattr__(self, "states", states)
        for name in (
            "capability_ids",
            "language_classes",
            "framework_classes",
            "repository_families",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), name, identifiers=True),
            )
        if self.max_risk is not None:
            object.__setattr__(
                self, "max_risk", _enum(self.max_risk, RiskClass, "max_risk")
            )
        if self.version is not None:
            object.__setattr__(
                self, "version", _nested(self.version, ProcedureVersion, "version")
            )
        object.__setattr__(self, "usable_only", _bool(self.usable_only, "usable_only"))


@dataclass(frozen=True)
class RegistryMutation:
    """Committed mutation: new row, CAS receipt, and optional prior head."""

    revision: ProcedureRegistryRevision
    cas: RegistryCAS
    previous: ProcedureRegistryRevision | None = None
    certificate_admission: CertificateAdmission | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cas": self.cas.to_dict(),
            "previous_revision_id": (
                "" if self.previous is None else self.previous.revision_id
            ),
            "revision": self.revision.to_dict(),
        }


class ProcedureRegistryStore(Protocol):
    """Metadata owner.  Implementations wrap existing artifact/event stores."""

    def exclusive(self) -> AbstractContextManager[None]: ...

    def procedure_ids(self) -> tuple[str, ...]: ...

    def get_head_id(self, procedure_id: str) -> str: ...

    def set_head_id(self, procedure_id: str, revision_id: str) -> None: ...

    def put_revision(self, revision: ProcedureRegistryRevision) -> None: ...

    def get_revision(self, revision_id: str) -> ProcedureRegistryRevision | None: ...

    def list_revision_ids(self, procedure_id: str) -> tuple[str, ...]: ...

    def put_certificate(self, certificate: ProcedureCertificate) -> None: ...

    def get_certificate(self, certificate_cid: str) -> ProcedureCertificate | None: ...

    def append_event(self, event: Mapping[str, Any]) -> None: ...

    def events(self) -> tuple[Mapping[str, Any], ...]: ...

    def quarantine(self, record: Mapping[str, Any]) -> None: ...

    def quarantined(self) -> tuple[Mapping[str, Any], ...]: ...


class InMemoryProcedureRegistryStore:
    """Process-local metadata/event owner used by tests and dry-run callers.

    DuckDB control-plane owners remain the durable authority in production; this
    store never opens a control file and never stores procedure bodies.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._heads: dict[str, str] = {}
        self._revisions: dict[str, dict[str, Any]] = {}
        self._revisions_by_procedure: dict[str, list[str]] = {}
        self._certificates: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        self._quarantine: list[dict[str, Any]] = []

    @contextmanager
    def exclusive(self) -> Any:
        with self._lock:
            yield

    def procedure_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._heads) | set(self._revisions_by_procedure)))

    def get_head_id(self, procedure_id: str) -> str:
        return self._heads.get(procedure_id, EMPTY_REVISION_ID)

    def set_head_id(self, procedure_id: str, revision_id: str) -> None:
        if len(self._heads) >= MAX_REGISTRY_PROCEDURES and procedure_id not in self._heads:
            raise ProcedureRegistryError("registry procedure bound exhausted")
        self._heads[procedure_id] = revision_id

    def put_revision(self, revision: ProcedureRegistryRevision) -> None:
        if not isinstance(revision, ProcedureRegistryRevision):
            raise ProcedureRegistryError("store received a non-revision")
        payload = revision.to_dict()
        revision_id = revision.revision_id
        existing = self._revisions.get(revision_id)
        if existing is not None and existing != payload:
            raise RegistryCorruptionError(
                "content-addressed revision collision",
                procedure_id=revision.procedure_id,
                revision_id=revision_id,
                reason_code="revision_collision",
            )
        ids = self._revisions_by_procedure.setdefault(revision.procedure_id, [])
        if len(ids) >= MAX_REVISIONS_PER_PROCEDURE and revision_id not in ids:
            raise ProcedureRegistryError("registry revision bound exhausted")
        self._revisions[revision_id] = payload
        if revision_id not in ids:
            ids.append(revision_id)

    def get_revision(self, revision_id: str) -> ProcedureRegistryRevision | None:
        if not revision_id:
            return None
        payload = self._revisions.get(revision_id)
        if payload is None:
            return None
        try:
            record = ProcedureRegistryRevision.from_dict(payload)
        except RegistryCorruptionError as exc:
            self.quarantine(
                {
                    "kind": "revision",
                    "revision_id": revision_id,
                    "reason_code": getattr(exc, "reason_code", "corrupt_revision"),
                }
            )
            raise
        except ProcedureContractError as exc:
            self.quarantine(
                {
                    "kind": "revision",
                    "revision_id": revision_id,
                    "reason_code": getattr(exc, "reason_code", "corrupt_revision"),
                }
            )
            raise RegistryCorruptionError(
                "stored revision is corrupt",
                revision_id=revision_id,
                reason_code="corrupt_revision",
            ) from exc
        if record.revision_id != revision_id:
            self.quarantine(
                {
                    "kind": "revision",
                    "revision_id": revision_id,
                    "reason_code": "revision_cid_mismatch",
                }
            )
            raise RegistryCorruptionError(
                "stored revision identity does not match canonical content",
                procedure_id=record.procedure_id,
                revision_id=revision_id,
                reason_code="revision_cid_mismatch",
            )
        return record

    def list_revision_ids(self, procedure_id: str) -> tuple[str, ...]:
        return tuple(self._revisions_by_procedure.get(procedure_id, ()))

    def put_certificate(self, certificate: ProcedureCertificate) -> None:
        if not isinstance(certificate, ProcedureCertificate):
            raise ProcedureRegistryError("store received a non-certificate")
        self._certificates[certificate.content_id] = certificate.to_dict()

    def get_certificate(self, certificate_cid: str) -> ProcedureCertificate | None:
        payload = self._certificates.get(certificate_cid)
        if payload is None:
            return None
        try:
            certificate = ProcedureCertificate.from_dict(payload)
        except ProcedureContractError as exc:
            self.quarantine(
                {
                    "kind": "certificate",
                    "certificate_cid": certificate_cid,
                    "reason_code": "corrupt_certificate",
                }
            )
            raise RegistryCorruptionError(
                "stored certificate is corrupt",
                revision_id=certificate_cid,
                reason_code="corrupt_certificate",
            ) from exc
        if certificate.content_id != certificate_cid:
            self.quarantine(
                {
                    "kind": "certificate",
                    "certificate_cid": certificate_cid,
                    "reason_code": "certificate_cid_mismatch",
                }
            )
            raise RegistryCorruptionError(
                "stored certificate identity does not match canonical content",
                revision_id=certificate_cid,
                reason_code="certificate_cid_mismatch",
            )
        return certificate

    def append_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise ProcedureRegistryError("registry event must be an object")
        if len(self._events) >= MAX_EVENTS:
            raise ProcedureRegistryError("registry event bound exhausted")
        self._events.append(dict(event))

    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(item) for item in self._events)

    def quarantine(self, record: Mapping[str, Any]) -> None:
        if len(self._quarantine) >= MAX_QUARANTINE:
            return
        self._quarantine.append(dict(record))

    def quarantined(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(MappingProxyType(item) for item in self._quarantine)

    def corrupt_revision_payload(self, revision_id: str, field: str, value: Any) -> None:
        """Test helper: mutate a stored snapshot so integrity checks fail."""

        payload = self._revisions.get(revision_id)
        if payload is None:
            raise RegistryNotFoundError("revision is not stored")
        mutated = dict(payload)
        mutated[field] = value
        self._revisions[revision_id] = mutated

    def drop_head(self, procedure_id: str) -> None:
        """Test helper: forget the current pointer while leaving history."""

        self._heads.pop(procedure_id, None)


def _matches_filter(
    revision: ProcedureRegistryRevision,
    query: RegistryFilter,
    *,
    currently_usable: bool,
) -> bool:
    if query.procedure_id and revision.procedure_id != query.procedure_id:
        return False
    if query.procedure_cid and revision.procedure_cid != query.procedure_cid:
        return False
    if query.task_family_cid and revision.task_family_cid != query.task_family_cid:
        return False
    if query.states and revision.state not in query.states:
        return False
    if query.usable_only and not currently_usable:
        return False
    if query.capability_ids and not set(query.capability_ids).issubset(
        set(revision.capability_ids)
    ):
        return False
    if query.max_risk is not None and _risk_rank(revision.risk_ceiling) > _risk_rank(
        query.max_risk
    ):
        return False
    if query.environment_id and revision.bindings.environment_id != query.environment_id:
        return False
    if query.repository_id and revision.bindings.repository_id != query.repository_id:
        return False
    if query.tree_id and revision.bindings.tree_id != query.tree_id:
        return False
    if (
        query.policy_revision
        and revision.bindings.policy_revision != query.policy_revision
    ):
        return False
    if query.language_classes and not set(query.language_classes).issubset(
        set(revision.supported_language_classes)
    ):
        return False
    if query.framework_classes and not set(query.framework_classes).issubset(
        set(revision.supported_framework_classes)
    ):
        return False
    if query.repository_families and not set(query.repository_families).issubset(
        set(revision.repository_families)
    ):
        return False
    if query.version is not None and _version_tuple(revision.version) != _version_tuple(
        query.version
    ):
        return False
    return True


def _sort_key(revision: ProcedureRegistryRevision) -> tuple[Any, ...]:
    major, minor, patch = _version_tuple(revision.version)
    return (
        revision.task_family_cid,
        revision.procedure_id,
        -major,
        -minor,
        -patch,
        -revision.generation,
        revision.procedure_cid,
        revision.revision_id,
    )


class ProcedureRegistry:
    """Deterministic versioned catalog with authorized CAS mutations."""

    revision: Final[str] = REGISTRY_REVISION

    def __init__(
        self,
        verifier: ProcedureCertificateVerifier,
        context_provider: Callable[[], CurrentCertificateContext],
        store: ProcedureRegistryStore | None = None,
        *,
        clock_ms: Callable[[], int] | None = None,
        event_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not isinstance(verifier, ProcedureCertificateVerifier):
            raise ProcedureRegistryError("registry requires a ProcedureCertificateVerifier")
        if not callable(context_provider):
            raise ProcedureRegistryError("registry requires a current certificate context provider")
        self._verifier = verifier
        self._context_provider = context_provider
        self._store: ProcedureRegistryStore = store or InMemoryProcedureRegistryStore()
        self._clock_ms = clock_ms or (lambda: self._context_provider().now_ms)
        self._event_sink = event_sink

    @property
    def store(self) -> ProcedureRegistryStore:
        return self._store

    def register(
        self,
        *,
        procedure_id: str,
        procedure_cid: str,
        certificate: ProcedureCertificate,
        authorization: RegistryAuthorization,
        initial_state: RegistryLifecycleState = RegistryLifecycleState.CANDIDATE,
        capability_ids: Sequence[str] = (),
        expected_old_revision_id: str = EMPTY_REVISION_ID,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        procedure_cid = _identifier(procedure_cid, "procedure_cid")
        state = _state(initial_state)
        if state not in REGISTER_STATES:
            raise ProcedureRegistryError("a procedure cannot register itself as promoted")
        if state is RegistryLifecycleState.PROMOTED:
            raise ProcedureRegistryError("registration cannot promote")
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        admission = self._require_current_certificate(
            certificate, procedure_cid=procedure_cid
        )
        self._require_authorization(
            authorization,
            operation=RegistryOperation.REGISTER,
            procedure_id=procedure_id,
            procedure_cid=procedure_cid,
            certificate=certificate,
            expected_old_revision_id=expected_old_revision_id,
        )
        with self._store.exclusive():
            head = self._load_head(procedure_id)
            if head is not None and head.procedure_cid == procedure_cid:
                if head.state is state and head.certificate_cid == certificate.content_id:
                    cas = self._cas_record(
                        accepted=True,
                        stale=False,
                        outcome=RegistryCASOutcome.NOOP,
                        operation=RegistryOperation.REGISTER,
                        procedure_id=procedure_id,
                        expected_old=head.revision_id,
                        observed=head.revision_id,
                        target_procedure_cid=procedure_cid,
                        target_revision_id=head.revision_id,
                        new_revision_id=head.revision_id,
                        rollback_target=head.rollback_target_revision_id,
                        authorization=authorization,
                        generation=head.generation,
                        reason_code="idempotent_register",
                    )
                    return RegistryMutation(
                        revision=head,
                        cas=cas,
                        previous=head,
                        certificate_admission=admission,
                    )
                raise ProcedureRegistryError("procedure CID is already registered")
            if head is not None and head.state is RegistryLifecycleState.PROMOTED:
                observed = head.revision_id
                if expected_old_revision_id != observed:
                    cas = self._stale_cas(
                        operation=RegistryOperation.REGISTER,
                        procedure_id=procedure_id,
                        expected_old=expected_old_revision_id,
                        observed=observed,
                        target_procedure_cid=procedure_cid,
                        authorization=authorization,
                        generation=head.generation,
                    )
                    raise RegistryCASError(
                        "register expected-old revision does not match the current head",
                        cas=cas,
                    )
                generation = self._next_generation(procedure_id)
                predecessor = observed
                expected_old = observed
            else:
                observed = EMPTY_REVISION_ID if head is None else head.revision_id
                if expected_old_revision_id != observed:
                    cas = self._stale_cas(
                        operation=RegistryOperation.REGISTER,
                        procedure_id=procedure_id,
                        expected_old=expected_old_revision_id,
                        observed=observed,
                        target_procedure_cid=procedure_cid,
                        authorization=authorization,
                        generation=0 if head is None else head.generation,
                    )
                    raise RegistryCASError(
                        "register expected-old revision does not match the current head",
                        cas=cas,
                    )
                generation = 1 if head is None else head.generation + 1
                predecessor = observed
                expected_old = observed
            revision = self._build_revision(
                procedure_id=procedure_id,
                procedure_cid=procedure_cid,
                certificate=certificate,
                state=state,
                capability_ids=capability_ids,
                generation=generation,
                predecessor_revision_id=predecessor,
                rollback_target_revision_id=EMPTY_REVISION_ID,
                expected_old_revision_id=expected_old,
                authorization=authorization,
                operation=RegistryOperation.REGISTER,
                created_at_ms=created_at,
            )
            move_head = head is None or head.state is not RegistryLifecycleState.PROMOTED
            cas = self._commit(
                procedure_id=procedure_id,
                expected_old=expected_old if move_head else (head.revision_id if head else EMPTY_REVISION_ID),
                revision=revision,
                authorization=authorization,
                move_head=move_head,
                observed=observed if move_head else (head.revision_id if head else EMPTY_REVISION_ID),
            )
            self._store.put_certificate(certificate)
            return RegistryMutation(
                revision=revision,
                cas=cas,
                previous=head,
                certificate_admission=admission,
            )

    def advance(
        self,
        *,
        procedure_id: str,
        next_state: RegistryLifecycleState,
        authorization: RegistryAuthorization,
        expected_old_revision_id: str,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        next_state = _state(next_state)
        if next_state is RegistryLifecycleState.PROMOTED:
            raise ProcedureRegistryError("advance cannot promote; use promote()")
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        with self._store.exclusive():
            head = self._require_head(procedure_id)
            certificate = self._require_stored_certificate(head.certificate_cid)
            admission = self._require_current_certificate(
                certificate, procedure_cid=head.procedure_cid
            )
            self._require_authorization(
                authorization,
                operation=RegistryOperation.ADVANCE,
                procedure_id=procedure_id,
                procedure_cid=head.procedure_cid,
                certificate=certificate,
                expected_old_revision_id=expected_old_revision_id,
                target_revision_id=head.revision_id,
            )
            self._require_transition(head.state, next_state)
            revision = self._successor(
                head,
                state=next_state,
                authorization=authorization,
                operation=RegistryOperation.ADVANCE,
                created_at_ms=created_at,
                expected_old_revision_id=expected_old_revision_id,
                rollback_target_revision_id=head.rollback_target_revision_id,
            )
            cas = self._commit(
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                revision=revision,
                authorization=authorization,
                move_head=True,
                observed=head.revision_id,
            )
            return RegistryMutation(
                revision=revision,
                cas=cas,
                previous=head,
                certificate_admission=admission,
            )

    def promote(
        self,
        *,
        procedure_id: str,
        target_procedure_cid: str,
        authorization: RegistryAuthorization,
        expected_old_revision_id: str,
        rollback_target_revision_id: str = EMPTY_REVISION_ID,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        target_procedure_cid = _identifier(target_procedure_cid, "target_procedure_cid")
        rollback_target_revision_id = _identifier(
            rollback_target_revision_id, "rollback_target_revision_id", required=False
        )
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        with self._store.exclusive():
            head = self._load_head(procedure_id)
            observed = EMPTY_REVISION_ID if head is None else head.revision_id
            self._raise_if_stale(
                operation=RegistryOperation.PROMOTE,
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                observed=observed,
                target_procedure_cid=target_procedure_cid,
                authorization=authorization,
                generation=0 if head is None else head.generation,
                target_revision_id=authorization.target_revision_id,
                rollback_target=rollback_target_revision_id,
            )
            source = self._locate_target(procedure_id, target_procedure_cid)
            certificate = self._require_stored_certificate(source.certificate_cid)
            admission = self._require_current_certificate(
                certificate, procedure_cid=target_procedure_cid
            )
            if admission.grants_promotion or admission.grants_authority:
                raise RegistryAuthorizationError("certificate admission cannot grant promotion")
            self._require_authorization(
                authorization,
                operation=RegistryOperation.PROMOTE,
                procedure_id=procedure_id,
                procedure_cid=target_procedure_cid,
                certificate=certificate,
                expected_old_revision_id=expected_old_revision_id,
                target_revision_id=source.revision_id,
            )
            if source.state not in PROMOTABLE_STATES:
                raise ProcedureRegistryError(
                    "only candidate, development, or shadow revisions can be promoted"
                )
            prior_promoted = self._current_promoted(procedure_id)
            required_rollback = (
                EMPTY_REVISION_ID
                if prior_promoted is None
                else prior_promoted.revision_id
            )
            if rollback_target_revision_id != required_rollback:
                raise ProcedureRegistryError(
                    "promotion requires the exact rollback target"
                )
            revision = self._successor(
                source,
                state=RegistryLifecycleState.PROMOTED,
                authorization=authorization,
                operation=RegistryOperation.PROMOTE,
                created_at_ms=created_at,
                expected_old_revision_id=expected_old_revision_id,
                rollback_target_revision_id=rollback_target_revision_id,
                generation=self._next_generation(procedure_id),
                predecessor_revision_id=observed,
            )
            cas = self._commit(
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                revision=revision,
                authorization=authorization,
                move_head=True,
                observed=observed,
            )
            return RegistryMutation(
                revision=revision,
                cas=cas,
                previous=head,
                certificate_admission=admission,
            )

    def rollback(
        self,
        *,
        procedure_id: str,
        target_revision_id: str,
        authorization: RegistryAuthorization,
        expected_old_revision_id: str,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        target_revision_id = _identifier(target_revision_id, "target_revision_id")
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        with self._store.exclusive():
            head = self._require_head(procedure_id)
            self._raise_if_stale(
                operation=RegistryOperation.ROLLBACK,
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                observed=head.revision_id,
                target_procedure_cid=head.procedure_cid,
                authorization=authorization,
                generation=head.generation,
                target_revision_id=target_revision_id,
                rollback_target=head.rollback_target_revision_id,
            )
            target = self._require_revision(target_revision_id)
            if target.procedure_id != procedure_id:
                raise ProcedureRegistryError("rollback target belongs to a different procedure")
            if target.revision_id == head.revision_id:
                raise ProcedureRegistryError("rollback target must differ from the current head")
            if (
                head.rollback_target_revision_id
                and head.rollback_target_revision_id != target.revision_id
            ):
                raise ProcedureRegistryError("rollback target is not the exact recorded target")
            certificate = self._require_stored_certificate(target.certificate_cid)
            admission = self._require_current_certificate(
                certificate, procedure_cid=target.procedure_cid
            )
            self._require_authorization(
                authorization,
                operation=RegistryOperation.ROLLBACK,
                procedure_id=procedure_id,
                procedure_cid=target.procedure_cid,
                certificate=certificate,
                expected_old_revision_id=expected_old_revision_id,
                target_revision_id=target.revision_id,
            )
            revision = self._successor(
                target,
                state=RegistryLifecycleState.PROMOTED,
                authorization=authorization,
                operation=RegistryOperation.ROLLBACK,
                created_at_ms=created_at,
                expected_old_revision_id=expected_old_revision_id,
                rollback_target_revision_id=head.revision_id,
                generation=self._next_generation(procedure_id),
                predecessor_revision_id=head.revision_id,
            )
            cas = self._commit(
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                revision=revision,
                authorization=authorization,
                move_head=True,
                observed=head.revision_id,
            )
            return RegistryMutation(
                revision=revision,
                cas=cas,
                previous=head,
                certificate_admission=admission,
            )

    def revoke(
        self,
        *,
        procedure_id: str,
        target_procedure_cid: str,
        authorization: RegistryAuthorization,
        expected_old_revision_id: str,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        target_procedure_cid = _identifier(target_procedure_cid, "target_procedure_cid")
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        with self._store.exclusive():
            head = self._require_head(procedure_id)
            self._raise_if_stale(
                operation=RegistryOperation.REVOKE,
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                observed=head.revision_id,
                target_procedure_cid=target_procedure_cid,
                authorization=authorization,
                generation=head.generation,
                target_revision_id=authorization.target_revision_id,
                rollback_target=head.rollback_target_revision_id,
            )
            if head.procedure_cid != target_procedure_cid:
                raise ProcedureRegistryError("revocation target is not the current head procedure")
            if head.state in TERMINAL_STATES:
                raise ProcedureRegistryError("terminal revisions cannot be revoked again")
            certificate = self._require_stored_certificate(head.certificate_cid)
            self._require_authorization(
                authorization,
                operation=RegistryOperation.REVOKE,
                procedure_id=procedure_id,
                procedure_cid=target_procedure_cid,
                certificate=certificate,
                expected_old_revision_id=expected_old_revision_id,
                target_revision_id=head.revision_id,
            )
            revision = self._successor(
                head,
                state=RegistryLifecycleState.REVOKED,
                authorization=authorization,
                operation=RegistryOperation.REVOKE,
                created_at_ms=created_at,
                expected_old_revision_id=expected_old_revision_id,
                rollback_target_revision_id=head.rollback_target_revision_id,
            )
            cas = self._commit(
                procedure_id=procedure_id,
                expected_old=expected_old_revision_id,
                revision=revision,
                authorization=authorization,
                move_head=True,
                observed=head.revision_id,
            )
            return RegistryMutation(revision=revision, cas=cas, previous=head)

    def demote(
        self,
        *,
        procedure_id: str,
        expected_old_revision_id: str,
        reason_state: RegistryLifecycleState = RegistryLifecycleState.STALE,
        authorization: RegistryAuthorization | None = None,
        now_ms: int | None = None,
    ) -> RegistryMutation:
        procedure_id = _identifier(procedure_id, "procedure_id")
        reason_state = _state(reason_state)
        created_at = _nonnegative_int(
            self._clock_ms() if now_ms is None else now_ms, "now_ms"
        )
        with self._store.exclusive():
            return self._demote_locked(
                procedure_id=procedure_id,
                expected_old_revision_id=expected_old_revision_id,
                reason_state=reason_state,
                authorization=authorization,
                created_at_ms=created_at,
            )

    def recover(self, procedure_id: str) -> ProcedureRegistryRevision:
        procedure_id = _identifier(procedure_id, "procedure_id")
        with self._store.exclusive():
            intact = self._intact_revisions(procedure_id)
            if not intact:
                raise RegistryCorruptionError(
                    "no intact revision remains for recovery",
                    procedure_id=procedure_id,
                    reason_code="unrecoverable",
                )
            chosen = max(intact, key=lambda item: (item.generation, item.revision_id))
            self._store.set_head_id(procedure_id, chosen.revision_id)
            self._emit_event(
                {
                    "schema": REGISTRY_EVENT_SCHEMA,
                    "operation": RegistryOperation.RECOVER.value,
                    "procedure_id": procedure_id,
                    "revision_id": chosen.revision_id,
                    "reason_code": "recovered_from_intact_history",
                }
            )
            return chosen

    def get(self, procedure_id: str, *, demote_stale: bool = True) -> ProcedureRegistryRevision:
        procedure_id = _identifier(procedure_id, "procedure_id")
        with self._store.exclusive():
            head = self._require_head(procedure_id)
            if demote_stale:
                head = self._maybe_demote_stale(head)
            return head

    def get_revision(self, revision_id: str) -> ProcedureRegistryRevision:
        return self._require_revision(_identifier(revision_id, "revision_id"))

    def history(self, procedure_id: str) -> tuple[ProcedureRegistryRevision, ...]:
        procedure_id = _identifier(procedure_id, "procedure_id")
        with self._store.exclusive():
            records = self._intact_revisions(procedure_id)
            return tuple(sorted(records, key=lambda item: (item.generation, item.revision_id)))

    def lookup_exact(
        self,
        procedure_cid: str,
        *,
        bindings: ArtifactBindings | None = None,
        usable_only: bool = True,
    ) -> ProcedureRegistryRevision | None:
        procedure_cid = _identifier(procedure_cid, "procedure_cid")
        query = RegistryFilter(
            procedure_cid=procedure_cid,
            repository_id="" if bindings is None else bindings.repository_id,
            tree_id="" if bindings is None else bindings.tree_id,
            policy_revision="" if bindings is None else bindings.policy_revision,
            environment_id="" if bindings is None else bindings.environment_id,
            usable_only=usable_only,
            states=() if not usable_only else (RegistryLifecycleState.PROMOTED,),
        )
        matches = self.filter(query)
        return matches[0] if matches else None

    def lookup_family(
        self,
        task_family_cid: str,
        query: RegistryFilter | None = None,
    ) -> tuple[ProcedureRegistryRevision, ...]:
        task_family_cid = _identifier(task_family_cid, "task_family_cid")
        base = query or RegistryFilter()
        merged = RegistryFilter(
            procedure_id=base.procedure_id,
            procedure_cid=base.procedure_cid,
            task_family_cid=task_family_cid,
            states=base.states,
            capability_ids=base.capability_ids,
            max_risk=base.max_risk,
            environment_id=base.environment_id,
            repository_id=base.repository_id,
            tree_id=base.tree_id,
            policy_revision=base.policy_revision,
            language_classes=base.language_classes,
            framework_classes=base.framework_classes,
            repository_families=base.repository_families,
            version=base.version,
            usable_only=base.usable_only,
        )
        return self.filter(merged)

    def choose_version(
        self,
        procedure_id: str,
        version: ProcedureVersion | None = None,
        *,
        usable_only: bool = True,
    ) -> ProcedureRegistryRevision | None:
        matches = self.filter(
            RegistryFilter(
                procedure_id=_identifier(procedure_id, "procedure_id"),
                version=version,
                usable_only=usable_only,
            )
        )
        return matches[0] if matches else None

    def filter(self, query: RegistryFilter | None = None) -> tuple[ProcedureRegistryRevision, ...]:
        query = query or RegistryFilter()
        if not isinstance(query, RegistryFilter):
            raise ProcedureRegistryError("filter query must be a RegistryFilter")
        with self._store.exclusive():
            selected: dict[str, ProcedureRegistryRevision] = {}
            for procedure_id in self._store.procedure_ids():
                try:
                    records = self._intact_revisions(procedure_id)
                except RegistryCorruptionError:
                    continue
                for record in records:
                    currently_usable = self._currently_usable(record)
                    if not _matches_filter(
                        record, query, currently_usable=currently_usable
                    ):
                        continue
                    previous = selected.get(record.procedure_cid)
                    if previous is None or _sort_key(record) < _sort_key(previous):
                        selected[record.procedure_cid] = record
            ordered = tuple(sorted(selected.values(), key=_sort_key))
            if len(ordered) > MAX_LOOKUP_RESULTS:
                raise ProcedureRegistryError("lookup result bound exhausted")
            return ordered

    def status(self) -> Mapping[str, Any]:
        with self._store.exclusive():
            heads = []
            for procedure_id in self._store.procedure_ids():
                try:
                    head = self._load_head(procedure_id)
                except RegistryCorruptionError:
                    heads.append(
                        {
                            "procedure_id": procedure_id,
                            "state": "quarantined",
                            "revision_id": self._store.get_head_id(procedure_id),
                        }
                    )
                    continue
                if head is None:
                    continue
                heads.append(
                    {
                        "procedure_id": procedure_id,
                        "procedure_cid": head.procedure_cid,
                        "revision_id": head.revision_id,
                        "state": head.state.value,
                        "generation": head.generation,
                        "usable": self._currently_usable(head),
                    }
                )
            payload = {
                "revision": self.revision,
                "procedure_count": len(heads),
                "event_count": len(self._store.events()),
                "quarantine_count": len(self._store.quarantined()),
                "heads": tuple(sorted(heads, key=lambda item: item["procedure_id"])),
            }
            return MappingProxyType(payload)

    def _currently_usable(self, revision: ProcedureRegistryRevision) -> bool:
        if not revision.lifecycle_usable:
            return False
        if self._store.get_head_id(revision.procedure_id) != revision.revision_id:
            return False
        try:
            certificate = self._store.get_certificate(revision.certificate_cid)
        except RegistryCorruptionError:
            return False
        if certificate is None:
            return False
        admission = self._verifier.verify(certificate, self._context_provider())
        return bool(admission.accepted and admission.usable)

    def _maybe_demote_stale(
        self, head: ProcedureRegistryRevision
    ) -> ProcedureRegistryRevision:
        if head.state in TERMINAL_STATES or head.state in {
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.SUPERSEDED,
        }:
            return head
        try:
            certificate = self._store.get_certificate(head.certificate_cid)
        except RegistryCorruptionError:
            certificate = None
        if certificate is None:
            return head
        admission = self._verifier.verify(certificate, self._context_provider())
        if admission.accepted and admission.usable:
            return head
        mutation = self._demote_locked(
            procedure_id=head.procedure_id,
            expected_old_revision_id=head.revision_id,
            reason_state=RegistryLifecycleState.STALE,
            authorization=None,
            created_at_ms=self._clock_ms(),
        )
        return mutation.revision

    def _demote_locked(
        self,
        *,
        procedure_id: str,
        expected_old_revision_id: str,
        reason_state: RegistryLifecycleState,
        authorization: RegistryAuthorization | None,
        created_at_ms: int,
    ) -> RegistryMutation:
        if reason_state not in {
            RegistryLifecycleState.STALE,
            RegistryLifecycleState.DEGRADED,
            RegistryLifecycleState.SUPERSEDED,
        }:
            raise ProcedureRegistryError("demote must use stale, degraded, or superseded")
        head = self._require_head(procedure_id)
        certificate = self._require_stored_certificate(head.certificate_cid)
        if authorization is None:
            authorization = RegistryAuthorization(
                actor_id=DRIFT_ACTOR_ID,
                decision_cid="registry-stale-certificate-demote",
                operation=RegistryOperation.DEMOTE,
                target_procedure_cid=head.procedure_cid,
                expected_old_revision_id=expected_old_revision_id,
                target_revision_id=head.revision_id,
                granted=True,
                issued_at_ms=created_at_ms,
            )
        self._require_authorization(
            authorization,
            operation=RegistryOperation.DEMOTE,
            procedure_id=procedure_id,
            procedure_cid=head.procedure_cid,
            certificate=certificate,
            expected_old_revision_id=expected_old_revision_id,
            target_revision_id=head.revision_id,
            allow_drift_actor=True,
        )
        self._require_transition(head.state, reason_state)
        revision = self._successor(
            head,
            state=reason_state,
            authorization=authorization,
            operation=RegistryOperation.DEMOTE,
            created_at_ms=created_at_ms,
            expected_old_revision_id=expected_old_revision_id,
            rollback_target_revision_id=head.rollback_target_revision_id,
        )
        cas = self._commit(
            procedure_id=procedure_id,
            expected_old=expected_old_revision_id,
            revision=revision,
            authorization=authorization,
            move_head=True,
            observed=head.revision_id,
        )
        return RegistryMutation(revision=revision, cas=cas, previous=head)

    def _require_current_certificate(
        self,
        certificate: ProcedureCertificate,
        *,
        procedure_cid: str,
    ) -> CertificateAdmission:
        if not isinstance(certificate, ProcedureCertificate):
            raise ProcedureRegistryError("certificate must be a ProcedureCertificate")
        if certificate.procedure_cid != procedure_cid:
            raise ProcedureRegistryError("certificate does not bind the exact procedure CID")
        if certificate.state is ArtifactState.PROMOTED:
            raise RegistryAuthorizationError("a certificate cannot assert promotion")
        admission = self._verifier.verify(certificate, self._context_provider())
        if not admission.accepted or not admission.usable:
            raise ProcedureRegistryError(
                "certificate is not independently current: {}".format(admission.reason_code.value)
            )
        if admission.grants_promotion or admission.grants_authority:
            raise RegistryAuthorizationError("certificate admission cannot grant promotion")
        return admission

    def _require_authorization(
        self,
        authorization: RegistryAuthorization,
        *,
        operation: RegistryOperation,
        procedure_id: str,
        procedure_cid: str,
        certificate: ProcedureCertificate,
        expected_old_revision_id: str,
        target_revision_id: str = EMPTY_REVISION_ID,
        allow_drift_actor: bool = False,
    ) -> None:
        if not isinstance(authorization, RegistryAuthorization):
            raise RegistryAuthorizationError("mutation requires RegistryAuthorization")
        if authorization.operation is not operation:
            raise RegistryAuthorizationError("authorization operation does not match the mutation")
        if authorization.target_procedure_cid != procedure_cid:
            raise RegistryAuthorizationError("authorization does not bind the exact procedure CID")
        if authorization.expected_old_revision_id != expected_old_revision_id:
            raise RegistryAuthorizationError("authorization does not bind the expected-old CAS token")
        if operation in {
            RegistryOperation.PROMOTE,
            RegistryOperation.ROLLBACK,
            RegistryOperation.REVOKE,
            RegistryOperation.ADVANCE,
            RegistryOperation.DEMOTE,
        }:
            if not authorization.target_revision_id:
                raise RegistryAuthorizationError(
                    "authorization must bind the exact target revision"
                )
            if authorization.target_revision_id != target_revision_id:
                raise RegistryAuthorizationError(
                    "authorization does not bind the exact target revision"
                )
        if authorization.decision_cid == certificate.content_id:
            raise RegistryAuthorizationError("a certificate cannot authorize registry mutation")
        actor = authorization.actor_id.lower()
        forbidden = _self_actors(procedure_id, procedure_cid, certificate)
        if actor in forbidden:
            raise RegistryAuthorizationError("a procedure cannot promote or mutate itself")
        if allow_drift_actor and authorization.actor_id == DRIFT_ACTOR_ID:
            if operation is RegistryOperation.PROMOTE:
                raise RegistryAuthorizationError("drift actor cannot promote")
            return
        if authorization.actor_id == DRIFT_ACTOR_ID and operation is RegistryOperation.PROMOTE:
            raise RegistryAuthorizationError("drift actor cannot promote")

    def _require_transition(
        self,
        current: RegistryLifecycleState,
        nxt: RegistryLifecycleState,
    ) -> None:
        allowed = _ADVANCE_ORDER.get(current, frozenset())
        if nxt not in allowed:
            raise ProcedureRegistryError(
                "closed lifecycle forbids {} -> {}".format(current.value, nxt.value)
            )

    def _build_revision(
        self,
        *,
        procedure_id: str,
        procedure_cid: str,
        certificate: ProcedureCertificate,
        state: RegistryLifecycleState,
        capability_ids: Sequence[str],
        generation: int,
        predecessor_revision_id: str,
        rollback_target_revision_id: str,
        expected_old_revision_id: str,
        authorization: RegistryAuthorization,
        operation: RegistryOperation,
        created_at_ms: int,
    ) -> ProcedureRegistryRevision:
        return ProcedureRegistryRevision(
            procedure_id=procedure_id,
            procedure_cid=procedure_cid,
            certificate_cid=certificate.content_id,
            task_family_cid=certificate.task_family_cid,
            version=certificate.procedure_version,
            state=state,
            bindings=certificate.bindings,
            capability_ids=tuple(capability_ids),
            risk_ceiling=certificate.risk_ceiling,
            repository_families=certificate.repository_families,
            supported_language_classes=certificate.supported_language_classes,
            supported_framework_classes=certificate.supported_framework_classes,
            generation=generation,
            predecessor_revision_id=predecessor_revision_id,
            rollback_target_revision_id=rollback_target_revision_id,
            expected_old_revision_id=expected_old_revision_id,
            actor_id=authorization.actor_id,
            authorization_cid=authorization.authorization_cid,
            operation=operation,
            created_at_ms=created_at_ms,
        )

    def _successor(
        self,
        source: ProcedureRegistryRevision,
        *,
        state: RegistryLifecycleState,
        authorization: RegistryAuthorization,
        operation: RegistryOperation,
        created_at_ms: int,
        expected_old_revision_id: str,
        rollback_target_revision_id: str,
        generation: int | None = None,
        predecessor_revision_id: str | None = None,
    ) -> ProcedureRegistryRevision:
        self._require_stored_certificate(source.certificate_cid)
        return ProcedureRegistryRevision(
            procedure_id=source.procedure_id,
            procedure_cid=source.procedure_cid,
            certificate_cid=source.certificate_cid,
            task_family_cid=source.task_family_cid,
            version=source.version,
            state=state,
            bindings=source.bindings,
            capability_ids=source.capability_ids,
            risk_ceiling=source.risk_ceiling,
            repository_families=source.repository_families,
            supported_language_classes=source.supported_language_classes,
            supported_framework_classes=source.supported_framework_classes,
            generation=source.generation + 1 if generation is None else generation,
            predecessor_revision_id=(
                source.revision_id if predecessor_revision_id is None else predecessor_revision_id
            ),
            rollback_target_revision_id=rollback_target_revision_id,
            expected_old_revision_id=expected_old_revision_id,
            actor_id=authorization.actor_id,
            authorization_cid=authorization.authorization_cid,
            operation=operation,
            created_at_ms=created_at_ms,
        )

    def _commit(
        self,
        *,
        procedure_id: str,
        expected_old: str,
        revision: ProcedureRegistryRevision,
        authorization: RegistryAuthorization,
        move_head: bool,
        observed: str,
    ) -> RegistryCAS:
        if move_head:
            self._raise_if_stale(
                operation=revision.operation,
                procedure_id=procedure_id,
                expected_old=expected_old,
                observed=observed,
                target_procedure_cid=revision.procedure_cid,
                authorization=authorization,
                generation=revision.generation,
                target_revision_id=revision.revision_id,
                rollback_target=revision.rollback_target_revision_id,
            )
        self._store.put_revision(revision)
        if move_head:
            self._store.set_head_id(procedure_id, revision.revision_id)
        cas = self._cas_record(
            accepted=True,
            stale=False,
            outcome=RegistryCASOutcome.COMMITTED,
            operation=revision.operation,
            procedure_id=procedure_id,
            expected_old=expected_old,
            observed=observed,
            target_procedure_cid=revision.procedure_cid,
            target_revision_id=revision.revision_id,
            new_revision_id=revision.revision_id,
            rollback_target=revision.rollback_target_revision_id,
            authorization=authorization,
            generation=revision.generation,
            reason_code="committed",
        )
        self._emit_event(
            {
                "schema": REGISTRY_EVENT_SCHEMA,
                "cas": cas.to_dict(),
                "revision_id": revision.revision_id,
                "artifact": revision.to_artifact().to_dict(),
                "receipt": cas.to_artifact(
                    revision.bindings, created_at_ms=revision.created_at_ms
                ).to_dict(),
            }
        )
        return cas

    def _cas_record(
        self,
        *,
        accepted: bool,
        stale: bool,
        outcome: RegistryCASOutcome,
        operation: RegistryOperation,
        procedure_id: str,
        expected_old: str,
        observed: str,
        target_procedure_cid: str,
        target_revision_id: str,
        new_revision_id: str,
        rollback_target: str,
        authorization: RegistryAuthorization,
        generation: int,
        reason_code: str,
    ) -> RegistryCAS:
        return RegistryCAS(
            accepted=accepted,
            stale=stale,
            outcome=outcome,
            operation=operation,
            procedure_id=procedure_id,
            expected_old_revision_id=expected_old,
            observed_revision_id=observed,
            target_procedure_cid=target_procedure_cid,
            target_revision_id=target_revision_id,
            new_revision_id=new_revision_id,
            rollback_target_revision_id=rollback_target,
            authorization_cid=authorization.authorization_cid,
            actor_id=authorization.actor_id,
            generation=generation,
            reason_code=reason_code,
        )

    def _raise_if_stale(
        self,
        *,
        operation: RegistryOperation,
        procedure_id: str,
        expected_old: str,
        observed: str,
        target_procedure_cid: str,
        authorization: RegistryAuthorization,
        generation: int,
        target_revision_id: str = EMPTY_REVISION_ID,
        rollback_target: str = EMPTY_REVISION_ID,
    ) -> None:
        if expected_old == observed:
            return
        cas = self._stale_cas(
            operation=operation,
            procedure_id=procedure_id,
            expected_old=expected_old,
            observed=observed,
            target_procedure_cid=target_procedure_cid,
            authorization=authorization,
            generation=generation,
            target_revision_id=target_revision_id,
            rollback_target=rollback_target,
        )
        self._emit_event(cas.to_dict())
        raise RegistryCASError(
            "expected-old revision does not match the current head",
            cas=cas,
        )

    def _stale_cas(
        self,
        *,
        operation: RegistryOperation,
        procedure_id: str,
        expected_old: str,
        observed: str,
        target_procedure_cid: str,
        authorization: RegistryAuthorization,
        generation: int,
        target_revision_id: str = EMPTY_REVISION_ID,
        rollback_target: str = EMPTY_REVISION_ID,
    ) -> RegistryCAS:
        return self._cas_record(
            accepted=False,
            stale=True,
            outcome=RegistryCASOutcome.STALE,
            operation=operation,
            procedure_id=procedure_id,
            expected_old=expected_old,
            observed=observed,
            target_procedure_cid=target_procedure_cid,
            target_revision_id=target_revision_id,
            new_revision_id=EMPTY_REVISION_ID,
            rollback_target=rollback_target,
            authorization=authorization,
            generation=generation,
            reason_code="stale_expected_old",
        )

    def _load_head(self, procedure_id: str) -> ProcedureRegistryRevision | None:
        head_id = self._store.get_head_id(procedure_id)
        if not head_id:
            return None
        return self._require_revision(head_id)

    def _require_head(self, procedure_id: str) -> ProcedureRegistryRevision:
        head = self._load_head(procedure_id)
        if head is None:
            raise RegistryNotFoundError("procedure is not registered")
        return head

    def _require_revision(self, revision_id: str) -> ProcedureRegistryRevision:
        record = self._store.get_revision(revision_id)
        if record is None:
            raise RegistryNotFoundError("revision is not registered")
        return record

    def _require_stored_certificate(self, certificate_cid: str) -> ProcedureCertificate:
        certificate = self._store.get_certificate(certificate_cid)
        if certificate is None:
            raise RegistryNotFoundError("certificate is not stored")
        return certificate

    def _locate_target(
        self, procedure_id: str, procedure_cid: str
    ) -> ProcedureRegistryRevision:
        records = self._intact_revisions(procedure_id)
        matches = [item for item in records if item.procedure_cid == procedure_cid]
        if not matches:
            raise RegistryNotFoundError("exact procedure CID is not registered")
        return max(matches, key=lambda item: (item.generation, item.revision_id))

    def _current_promoted(self, procedure_id: str) -> ProcedureRegistryRevision | None:
        head = self._load_head(procedure_id)
        if head is not None and head.state is RegistryLifecycleState.PROMOTED:
            return head
        promoted = [
            item
            for item in self._intact_revisions(procedure_id)
            if item.state is RegistryLifecycleState.PROMOTED
        ]
        if not promoted:
            return None
        return max(promoted, key=lambda item: (item.generation, item.revision_id))

    def _next_generation(self, procedure_id: str) -> int:
        records = self._intact_revisions(procedure_id)
        if not records:
            return 1
        return max(item.generation for item in records) + 1

    def _intact_revisions(self, procedure_id: str) -> tuple[ProcedureRegistryRevision, ...]:
        intact: list[ProcedureRegistryRevision] = []
        for revision_id in self._store.list_revision_ids(procedure_id):
            try:
                record = self._store.get_revision(revision_id)
            except RegistryCorruptionError:
                continue
            if record is not None:
                intact.append(record)
        return tuple(intact)

    def _emit_event(self, event: Mapping[str, Any]) -> None:
        self._store.append_event(event)
        if self._event_sink is not None:
            self._event_sink(event)


__all__ = [
    "DRIFT_ACTOR_ID",
    "EMPTY_REVISION_ID",
    "InMemoryProcedureRegistryStore",
    "PROMOTABLE_STATES",
    "ProcedureRegistry",
    "ProcedureRegistryError",
    "ProcedureRegistryRevision",
    "ProcedureRegistryStore",
    "REGISTER_STATES",
    "REGISTRY_CAS_SCHEMA",
    "REGISTRY_REVISION",
    "REGISTRY_REVISION_SCHEMA",
    "RegistryAuthorization",
    "RegistryAuthorizationError",
    "RegistryCAS",
    "RegistryCASError",
    "RegistryCASOutcome",
    "RegistryCorruptionError",
    "RegistryFilter",
    "RegistryLifecycleState",
    "RegistryMutation",
    "RegistryNotFoundError",
    "RegistryOperation",
    "TERMINAL_STATES",
    "USABLE_STATES",
]
