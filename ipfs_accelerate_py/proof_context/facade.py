"""Provider-neutral ProofCarryingContextEngine facade (PCCE-020).

The facade owns no semantic, persistence, routing, verification, assurance,
or sealing implementations. Canonical ports are injected. Importing this
module performs no I/O, network, process, or filesystem mutation and does
not bind a model provider or search sibling checkouts.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from ipfs_accelerate_py.proof_context.compatibility import (
    reject_mock,
    reject_pseudo_cid,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
ENGINE_RECORD_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/engine-record"
CONTRACT_VERSION: Final[str] = "0.1"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"
INTERFACE: Final[str] = "ProofCarryingContextEngine@0.1"
SIBLING_LAYOUT_REQUIRED: Final[bool] = False
PROVIDER_BOUND: Final[bool] = False

EPIC_A_GATE_TASK: Final[str] = "PCCE-011"
EPIC_A_GATE_CONTENT_ID: Final[str] = (
    "sha256:da97b6cd1307bf42b2109b2647a79e64f0c754f85e19ed1d73f549b580b14ce6"
)
PCCE_006_CONTENT_ID: Final[str] = (
    "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
)
COMPATIBILITY_MATRIX_CONTENT_ID: Final[str] = (
    "sha256:bfe49d9f3b6d2f472ae58d369b2138fc4e8e6320fccdd181e07a5564e075e920"
)

MODES: Final[tuple[str, ...]] = (
    "production",
    "supervised",
    "evaluation",
    "simulation",
)
LIVE_MODES: Final[frozenset[str]] = frozenset({"production", "supervised"})
PROVENANCES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
STATUSES: Final[tuple[str, ...]] = (
    "succeeded",
    "rejected",
    "verification_failed",
    "proof_failed",
    "assurance_failed",
    "context_insufficient",
    "model_escalation_required",
    "human_review_required",
    "unavailable",
    "timeout",
    "cancelled",
    "invalid",
    "stale",
    "simulated",
    "infrastructure_failure",
    "partial_effect",
    "repair_required",
)

OPERATIONS: Final[tuple[str, ...]] = (
    "open",
    "scan",
    "status",
    "plan",
    "context-pack",
    "route",
    "run",
    "verify",
    "expand-context",
    "assurance",
    "seal",
    "report",
    "resume",
)
INSTANCE_OPERATIONS: Final[tuple[str, ...]] = OPERATIONS[1:]

_METHOD_TO_OPERATION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scan": "scan",
        "status": "status",
        "plan": "plan",
        "context_pack": "context-pack",
        "route": "route",
        "run": "run",
        "verify": "verify",
        "expand_context": "expand-context",
        "assurance": "assurance",
        "seal": "seal",
        "report": "report",
        "resume": "resume",
    }
)
_STABLE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_state_cid",
    "task_id",
    "run_id",
    "trace_id",
    "contract_version",
)
_BIND_ONCE_IDENTITY_FIELDS: Final[tuple[str, ...]] = ("patch_id", "artifact_id")
_PORT_FIELDS: Final[tuple[str, ...]] = (
    "semantic",
    "persistence",
    "route",
    "execution",
    "verification",
    "assurance",
    "sealing",
    "report",
)

OPERATION_CONTRACTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scan": "pcce/proof-context/v0.1/repository-state",
        "plan": "pcce/proof-context/v0.1/invalidation-plan",
        "context-pack": "pcce/proof-context/v0.1/context-pack",
        "route": "pcce/proof-context/v0.1/model-route-decision",
        "run": "pcce/proof-context/v0.1/execution-receipt",
        "verify": "pcce/proof-context/v0.1/verification-plan",
        "expand-context": "pcce/proof-context/v0.1/context-pack",
        "assurance": "pcce/proof-context/v0.1/qualification-result",
        "seal": "pcce/proof-context/v0.1/incremental-seal",
        "status": "pcce/proof-context/v0.1/execution-receipt",
        "report": "pcce/proof-context/v0.1/qualification-result",
        "resume": "pcce/proof-context/v0.1/execution-receipt",
    }
)


class FacadeError(RuntimeError):
    """Fail-closed facade error. Reason is a closed v0.1 error code."""

    reason = "invalid"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        if reason is not None:
            self.reason = reason


def _freeze_mapping(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if payload is None:
        return MappingProxyType({})
    return MappingProxyType(dict(payload))


def _canonical_operation(value: str) -> str:
    normalized = value.replace("_", "-")
    if normalized not in OPERATIONS:
        raise FacadeError(
            f"unknown facade operation {value!r}",
            reason="unknown_field",
        )
    return normalized


@dataclass(frozen=True)
class EngineIdentities:
    """Stable identities threaded through every facade operation."""

    repository_id: str
    repository_state_cid: str
    task_id: str
    run_id: str
    trace_id: str
    contract_version: str = CONTRACT_VERSION
    patch_id: str | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for name in _STABLE_IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise FacadeError(
                    f"identity field {name} is required",
                    reason="malformed",
                )
        if self.contract_version != CONTRACT_VERSION:
            raise FacadeError(
                f"contract version {self.contract_version!r} is not {CONTRACT_VERSION}",
                reason="schema_mismatch",
            )
        reject_pseudo_cid(self.repository_state_cid)

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "repository_id": self.repository_id,
                "repository_state_cid": self.repository_state_cid,
                "task_id": self.task_id,
                "run_id": self.run_id,
                "trace_id": self.trace_id,
                "contract_version": self.contract_version,
                "patch_id": self.patch_id,
                "artifact_id": self.artifact_id,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> EngineIdentities:
        return cls(
            repository_id=str(payload["repository_id"]),
            repository_state_cid=str(payload["repository_state_cid"]),
            task_id=str(payload["task_id"]),
            run_id=str(payload["run_id"]),
            trace_id=str(payload["trace_id"]),
            contract_version=str(payload.get("contract_version", CONTRACT_VERSION)),
            patch_id=_optional_str(payload.get("patch_id")),
            artifact_id=_optional_str(payload.get("artifact_id")),
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


@dataclass(frozen=True)
class EngineRecord:
    """Typed identity-bound result returned by every instance operation."""

    schema: str
    operation: str
    status: str
    identities: EngineIdentities
    artifact_cid: str
    provenance: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _canonical_operation(self.operation))
        object.__setattr__(self, "payload", _freeze_mapping(self.payload))
        if self.status not in STATUSES:
            raise FacadeError(
                f"unknown status {self.status!r}",
                reason="unknown_field",
            )
        if self.provenance not in PROVENANCES:
            raise FacadeError(
                f"unknown provenance {self.provenance!r}",
                reason="unknown_field",
            )
        if not self.schema:
            raise FacadeError("engine record schema is required", reason="malformed")
        reject_pseudo_cid(self.artifact_cid)

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "operation": self.operation,
                "status": self.status,
                "identities": dict(self.identities.to_mapping()),
                "artifact_cid": self.artifact_cid,
                "provenance": self.provenance,
                "payload": dict(self.payload),
                "contract": OPERATION_CONTRACTS.get(self.operation),
            }
        )


@runtime_checkable
class SemanticPort(Protocol):
    def scan(self, identities: EngineIdentities, repository: Path) -> EngineRecord: ...

    def plan(self, identities: EngineIdentities, repository: Path) -> EngineRecord: ...

    def context_pack(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...

    def expand_context(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...


@runtime_checkable
class PersistencePort(Protocol):
    def resume(
        self,
        identities: EngineIdentities,
        repository: Path,
        checkpoint: Mapping[str, Any] | None = None,
    ) -> EngineRecord: ...


@runtime_checkable
class RoutePort(Protocol):
    def route(self, identities: EngineIdentities, repository: Path) -> EngineRecord: ...


@runtime_checkable
class ExecutionPort(Protocol):
    def run(
        self,
        identities: EngineIdentities,
        repository: Path,
        proposal: Mapping[str, Any] | None = None,
    ) -> EngineRecord: ...


@runtime_checkable
class VerificationPort(Protocol):
    def verify(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...


@runtime_checkable
class AssurancePort(Protocol):
    def assurance(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...


@runtime_checkable
class SealingPort(Protocol):
    def seal(self, identities: EngineIdentities, repository: Path) -> EngineRecord: ...


@runtime_checkable
class ReportPort(Protocol):
    def status(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...

    def report(
        self, identities: EngineIdentities, repository: Path
    ) -> EngineRecord: ...


_PORT_PROTOCOLS: Final[Mapping[str, type]] = MappingProxyType(
    {
        "semantic": SemanticPort,
        "persistence": PersistencePort,
        "route": RoutePort,
        "execution": ExecutionPort,
        "verification": VerificationPort,
        "assurance": AssurancePort,
        "sealing": SealingPort,
        "report": ReportPort,
    }
)


@dataclass(frozen=True)
class EnginePorts:
    """Injected canonical ports. The facade does not construct backends."""

    semantic: SemanticPort
    persistence: PersistencePort
    route: RoutePort
    execution: ExecutionPort
    verification: VerificationPort
    assurance: AssurancePort
    sealing: SealingPort
    report: ReportPort


def _snapshot_callable(fn: Any) -> Mapping[str, Any]:
    signature = inspect.signature(fn)
    parameters: list[str] = []
    keyword_only: list[str] = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        parameters.append(name)
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            keyword_only.append(name)
    annotation = signature.return_annotation
    if annotation is inspect.Signature.empty:
        return_name = None
    elif isinstance(annotation, str):
        return_name = annotation
    else:
        return_name = getattr(annotation, "__name__", str(annotation))
    return MappingProxyType(
        {
            "parameters": tuple(parameters),
            "keyword_only": tuple(keyword_only),
            "return": return_name,
        }
    )


def public_signature_snapshot() -> Mapping[str, Mapping[str, Any]]:
    """Stable public method snapshot. Used as PCCE-020 API evidence."""

    snapshot: dict[str, Mapping[str, Any]] = {
        "open": _snapshot_callable(ProofCarryingContextEngine.open),
    }
    for method_name in _METHOD_TO_OPERATION:
        snapshot[method_name] = _snapshot_callable(
            getattr(ProofCarryingContextEngine, method_name)
        )
    return MappingProxyType(snapshot)


class ProofCarryingContextEngine:
    """Stable provider-neutral engine surface over injected canonical ports."""

    schema = SCHEMA
    interface = INTERFACE
    contract_version = CONTRACT_VERSION
    sibling_layout_required = SIBLING_LAYOUT_REQUIRED
    provider_bound = PROVIDER_BOUND

    def __init__(
        self,
        repository: Path,
        *,
        ports: EnginePorts,
        identities: EngineIdentities,
        mode: str,
    ) -> None:
        self._repository = repository
        self._ports = ports
        self._identities = identities
        self._mode = mode

    @classmethod
    def open(
        cls,
        repository: str | Path,
        *,
        ports: EnginePorts,
        identities: EngineIdentities,
        mode: str = "production",
    ) -> ProofCarryingContextEngine:
        if mode not in MODES:
            raise FacadeError(f"unknown runtime mode {mode!r}", reason="unknown_field")
        if not isinstance(identities, EngineIdentities):
            raise FacadeError("identities must be EngineIdentities", reason="malformed")
        admitted_ports = _admit_ports(ports, mode=mode)
        root = _admit_repository(repository)
        return cls(
            root,
            ports=admitted_ports,
            identities=identities,
            mode=mode,
        )

    @property
    def repository(self) -> Path:
        return self._repository

    @property
    def ports(self) -> EnginePorts:
        return self._ports

    @property
    def identities(self) -> EngineIdentities:
        return self._identities

    @property
    def mode(self) -> str:
        return self._mode

    def scan(self) -> EngineRecord:
        return self._call("scan", self._ports.semantic.scan)

    def status(self) -> EngineRecord:
        return self._call("status", self._ports.report.status)

    def plan(self) -> EngineRecord:
        return self._call("plan", self._ports.semantic.plan)

    def context_pack(self) -> EngineRecord:
        return self._call("context-pack", self._ports.semantic.context_pack)

    def route(self) -> EngineRecord:
        record = self._call("route", self._ports.route.route)
        _reject_bound_provider(record)
        return record

    def run(self, proposal: Mapping[str, Any] | None = None) -> EngineRecord:
        self._check_payload_identities(proposal)
        return self._call("run", self._ports.execution.run, extra=proposal)

    def verify(self) -> EngineRecord:
        return self._call("verify", self._ports.verification.verify)

    def expand_context(self) -> EngineRecord:
        return self._call("expand-context", self._ports.semantic.expand_context)

    def assurance(self) -> EngineRecord:
        return self._call("assurance", self._ports.assurance.assurance)

    def seal(self) -> EngineRecord:
        return self._call("seal", self._ports.sealing.seal)

    def report(self) -> EngineRecord:
        return self._call("report", self._ports.report.report)

    def resume(self, checkpoint: Mapping[str, Any] | None = None) -> EngineRecord:
        self._check_payload_identities(checkpoint)
        return self._call("resume", self._ports.persistence.resume, extra=checkpoint)

    def _call(
        self,
        operation: str,
        fn: Any,
        extra: Mapping[str, Any] | None = None,
    ) -> EngineRecord:
        canonical = _canonical_operation(operation)
        if extra is None:
            record = fn(self._identities, self._repository)
        else:
            record = fn(self._identities, self._repository, extra)
        return self._admit(canonical, record)

    def _admit(self, operation: str, record: Any) -> EngineRecord:
        if not isinstance(record, EngineRecord):
            raise FacadeError(
                "canonical ports must return EngineRecord",
                reason="malformed",
            )
        if record.operation != operation:
            raise FacadeError(
                f"port returned {record.operation!r} for {operation!r}",
                reason="identity_inconsistent",
            )
        merged = self._merge_identities(record.identities)
        self._admit_mode(record)
        admitted = replace(
            record,
            identities=merged,
            schema=record.schema or ENGINE_RECORD_SCHEMA,
        )
        self._identities = merged
        return admitted

    def _merge_identities(self, incoming: EngineIdentities) -> EngineIdentities:
        if not isinstance(incoming, EngineIdentities):
            raise FacadeError(
                "port result identities must be EngineIdentities",
                reason="malformed",
            )
        current = self._identities
        for name in _STABLE_IDENTITY_FIELDS:
            if getattr(incoming, name) != getattr(current, name):
                raise FacadeError(
                    f"identity field {name} drifted",
                    reason="identity_inconsistent",
                )
        updates: dict[str, str | None] = {}
        for name in _BIND_ONCE_IDENTITY_FIELDS:
            existing = getattr(current, name)
            proposed = getattr(incoming, name)
            if existing and proposed and existing != proposed:
                raise FacadeError(
                    f"identity field {name} is already bound",
                    reason="identity_inconsistent",
                )
            updates[name] = proposed or existing
        return replace(current, **updates)

    def _admit_mode(self, record: EngineRecord) -> None:
        if self._mode in LIVE_MODES and record.provenance == "simulated":
            raise FacadeError(
                "simulated evidence cannot enter production or supervised modes",
                reason="simulated_promoted",
            )
        if self._mode in LIVE_MODES and record.status == "simulated":
            raise FacadeError(
                "simulated status cannot enter production or supervised modes",
                reason="simulated_promoted",
            )

    def _check_payload_identities(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            raise FacadeError("payload must be a mapping", reason="malformed")
        raw = payload.get("identities")
        if isinstance(raw, EngineIdentities):
            self._merge_identities(raw)
        elif isinstance(raw, Mapping):
            self._merge_identities(EngineIdentities.from_mapping(raw))
        for name in (*_STABLE_IDENTITY_FIELDS, *_BIND_ONCE_IDENTITY_FIELDS):
            if name not in payload:
                continue
            expected = getattr(self._identities, name)
            actual = payload[name]
            if actual is None:
                continue
            if expected is None:
                continue
            if str(actual) != str(expected):
                raise FacadeError(
                    f"payload identity field {name} drifted",
                    reason="identity_inconsistent",
                )


def _admit_repository(repository: str | Path) -> Path:
    root = Path(repository)
    if not root.is_dir():
        raise FacadeError(
            "repository must be an ordinary directory",
            reason="invalid",
        )
    return root


def _admit_ports(ports: EnginePorts, *, mode: str) -> EnginePorts:
    if not isinstance(ports, EnginePorts):
        raise FacadeError("ports must be EnginePorts", reason="malformed")
    admitted: dict[str, Any] = {}
    for name in _PORT_FIELDS:
        port = getattr(ports, name)
        if port is None:
            raise FacadeError(
                f"canonical port {name} is unavailable",
                reason="unavailable_capability",
            )
        if mode in LIVE_MODES:
            reject_mock(port)
        protocol = _PORT_PROTOCOLS[name]
        if not isinstance(port, protocol):
            raise FacadeError(
                f"canonical port {name} does not implement {protocol.__name__}",
                reason="unavailable_capability",
            )
        admitted[name] = port
    return EnginePorts(**admitted)


def _reject_bound_provider(record: EngineRecord) -> None:
    credentials = record.payload.get("credentials")
    if credentials:
        raise FacadeError(
            "route decisions must not carry provider credentials",
            reason="boundary_violation",
        )
    mutation_authority = record.payload.get("mutation_authority")
    if mutation_authority:
        raise FacadeError(
            "route decisions must not carry mutation authority",
            reason="boundary_violation",
        )


__all__ = [
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "ENGINE_RECORD_SCHEMA",
    "EPIC_A_GATE_CONTENT_ID",
    "EPIC_A_GATE_TASK",
    "INTERFACE",
    "INSTANCE_OPERATIONS",
    "MODES",
    "OPERATION_CONTRACTS",
    "OPERATIONS",
    "PCCE_006_CONTENT_ID",
    "PROVIDER_BOUND",
    "PROVENANCES",
    "SCHEMA",
    "SIBLING_LAYOUT_REQUIRED",
    "STATUSES",
    "AssurancePort",
    "EngineIdentities",
    "EnginePorts",
    "EngineRecord",
    "ExecutionPort",
    "FacadeError",
    "PersistencePort",
    "ProofCarryingContextEngine",
    "ReportPort",
    "RoutePort",
    "SealingPort",
    "SemanticPort",
    "VerificationPort",
    "public_signature_snapshot",
]
