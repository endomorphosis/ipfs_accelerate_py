"""Typed v0.1 proof-context result records and closed transitions (PCCE-023).

Status wire values remain the frozen MCP++ taxonomy. Runtime records add
retryability, partial-effect, human-review, and repair behavior without
widening the accepted vocabulary. Importing this module performs no I/O,
network, process, or filesystem mutation.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.proof_context.compatibility import (
    CompatibilityError,
    reject_pseudo_cid,
)
from ipfs_accelerate_py.proof_context.errors import (
    ERROR_TAXONOMY_CONTENT_ID,
    ERRORS,
    BoundaryViolationError,
    IdentityInconsistentError,
    MalformedError,
    ProofContextError,
    PseudoCidError,
    SchemaMismatchError,
    SimulatedPromotedError,
    UnknownFieldError,
    admit_error,
    error_for,
    error_status,
)

SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1"
RESULT_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/result"
RESULT_DESCRIPTOR_SCHEMA: Final[str] = "ipfs-accelerate.proof-context.v0.1/result-state"
CONTRACT_VERSION: Final[str] = "0.1"
CONTRACT_SCHEMA_PREFIX: Final[str] = "pcce/proof-context/v0.1/"
STATUS_TAXONOMY_CONTENT_ID: Final[str] = (
    "sha256:5f206feebb6213d3a1c113e37373ac8402003170cea609035ec9b871ca9fdd19"
)
PCCE_006_CONTENT_ID: Final[str] = (
    "sha256:b5503d2c2ec22e34091b3f747241fbde0519a9f0b213a03e0456a8f980a43f37"
)
COMPATIBILITY_MATRIX_CONTENT_ID: Final[str] = (
    "sha256:bfe49d9f3b6d2f472ae58d369b2138fc4e8e6320fccdd181e07a5564e075e920"
)

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

PROVENANCES: Final[tuple[str, ...]] = ("live", "replayed", "simulated")
START: Final[None] = None
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_state_cid",
    "task_id",
    "run_id",
    "trace_id",
    "patch_id",
    "evidence_cid",
)
REQUIRED_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_id",
    "repository_state_cid",
    "task_id",
    "run_id",
    "trace_id",
)
PATCH_BEARING_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "succeeded",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "partial_effect",
        "repair_required",
    }
)
FAILURE_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "rejected",
        "verification_failed",
        "proof_failed",
        "assurance_failed",
        "invalid",
        "stale",
    }
)
# status -> (disposition, terminal, retryable, escalation, human_review, repair, partial_effect)
# disposition is None only for succeeded; never used to collapse unavailable.
_STATUS_ROWS: Final[tuple[tuple[str, str | None, bool, bool, bool, bool, bool, bool], ...]] = (
    ("succeeded", None, True, False, False, False, False, False),
    ("rejected", "reject", True, False, False, False, False, False),
    ("verification_failed", "reject", True, False, False, False, False, False),
    ("proof_failed", "reject", True, False, False, False, False, False),
    ("assurance_failed", "reject", True, False, False, False, False, False),
    ("context_insufficient", "escalation", False, False, True, False, False, False),
    ("model_escalation_required", "escalation", False, False, True, False, False, False),
    ("human_review_required", "review", False, False, False, True, False, False),
    ("unavailable", "retry", False, True, False, False, False, False),
    ("timeout", "retry", False, True, False, False, False, False),
    ("cancelled", "reject", True, False, False, False, False, False),
    ("invalid", "reject", True, False, False, False, False, False),
    ("stale", "reject", True, False, False, False, False, False),
    ("simulated", "reject", True, False, False, False, False, False),
    ("infrastructure_failure", "retry", False, True, False, False, False, False),
    ("partial_effect", "repair", False, False, False, False, True, True),
    ("repair_required", "repair", False, False, False, False, True, False),
)

_STATUS_INDEX: Final[Mapping[str, int]] = MappingProxyType(
    {name: index for index, name in enumerate(STATUSES)}
)


def _ordered(*names: str) -> tuple[str, ...]:
    unique = tuple(dict.fromkeys(names))
    unknown = [name for name in unique if name not in _STATUS_INDEX]
    if unknown:
        raise ValueError(f"transition target {unknown[0]!r} is not in the frozen taxonomy")
    return tuple(sorted(unique, key=lambda name: _STATUS_INDEX[name]))


def _semantics_map() -> Mapping[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for (
        status,
        disposition,
        terminal,
        retryable,
        escalation,
        human_review,
        repair,
        partial_effect,
    ) in _STATUS_ROWS:
        rows[status] = MappingProxyType(
            {
                "status": status,
                "disposition": disposition,
                "terminal": terminal,
                "retryable": retryable,
                "escalation": escalation,
                "human_review": human_review,
                "repair": repair,
                "partial_effect": partial_effect,
                "accepted": status == "succeeded",
                "failed": status in FAILURE_STATUSES,
                "unavailable": status == "unavailable",
            }
        )
    return MappingProxyType(rows)


STATUS_SEMANTICS: Final[Mapping[str, Mapping[str, Any]]] = _semantics_map()
TERMINAL_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if STATUS_SEMANTICS[status]["terminal"]
)
RETRYABLE_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if STATUS_SEMANTICS[status]["retryable"]
)
ESCALATION_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if STATUS_SEMANTICS[status]["escalation"]
)
REVIEW_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if STATUS_SEMANTICS[status]["human_review"]
)
REPAIR_STATUSES: Final[tuple[str, ...]] = tuple(
    status for status in STATUSES if STATUS_SEMANTICS[status]["repair"]
)

# Closed, deterministic legal transitions. Self-reissue is always legal.
# Unavailable cannot be relabeled success or failure.
_TRANSITION_TARGETS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "succeeded": _ordered("succeeded"),
        "rejected": _ordered("rejected"),
        "verification_failed": _ordered("verification_failed"),
        "proof_failed": _ordered("proof_failed"),
        "assurance_failed": _ordered("assurance_failed"),
        "context_insufficient": _ordered(
            "context_insufficient",
            "succeeded",
            "model_escalation_required",
            "human_review_required",
            "rejected",
            "cancelled",
            "unavailable",
            "timeout",
        ),
        "model_escalation_required": _ordered(
            "model_escalation_required",
            "succeeded",
            "human_review_required",
            "rejected",
            "cancelled",
            "verification_failed",
            "proof_failed",
            "assurance_failed",
            "context_insufficient",
            "timeout",
        ),
        "human_review_required": _ordered(
            "human_review_required",
            "succeeded",
            "rejected",
            "cancelled",
            "repair_required",
        ),
        "unavailable": _ordered(
            "unavailable",
            "cancelled",
            "human_review_required",
            "timeout",
            "infrastructure_failure",
        ),
        "timeout": _ordered(
            "timeout",
            "succeeded",
            "cancelled",
            "unavailable",
            "infrastructure_failure",
            "human_review_required",
            "repair_required",
            "partial_effect",
        ),
        "cancelled": _ordered("cancelled"),
        "invalid": _ordered("invalid"),
        "stale": _ordered("stale"),
        "simulated": _ordered("simulated"),
        "infrastructure_failure": _ordered(
            "infrastructure_failure",
            "succeeded",
            "timeout",
            "unavailable",
            "cancelled",
            "human_review_required",
            "repair_required",
            "partial_effect",
        ),
        "partial_effect": _ordered(
            "partial_effect",
            "repair_required",
            "human_review_required",
            "cancelled",
        ),
        "repair_required": _ordered(
            "repair_required",
            "succeeded",
            "human_review_required",
            "rejected",
            "cancelled",
            "partial_effect",
        ),
    }
)


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
    raise MalformedError(
        f"unsupported result canonicalization type {type(value).__name__}"
    )


def mint_result_cid(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonicalize(value).encode("utf-8")).digest()
    raw = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(raw).decode("ascii").lower().rstrip("=")


def admit_status(status: Any) -> str:
    if not isinstance(status, str) or status not in STATUSES:
        raise UnknownFieldError(f"unknown status {status!r}")
    return status


def admit_provenance(provenance: Any) -> str:
    if not isinstance(provenance, str) or provenance not in PROVENANCES:
        raise UnknownFieldError(f"unknown provenance {provenance!r}")
    return provenance


def status_semantics(status: str) -> Mapping[str, Any]:
    return STATUS_SEMANTICS[admit_status(status)]


def classify_status(status: str) -> str | None:
    """Retry, escalation, review, repair, reject, or None for succeeded."""

    disposition = status_semantics(status)["disposition"]
    return str(disposition) if disposition is not None else None


def is_terminal(status: str) -> bool:
    return bool(status_semantics(status)["terminal"])


def is_success(status: str, *, provenance: str = "live") -> bool:
    admit_provenance(provenance)
    return admit_status(status) == "succeeded" and provenance == "live"


def is_failure(status: str) -> bool:
    return admit_status(status) in FAILURE_STATUSES


def is_unavailable(status: str) -> bool:
    return admit_status(status) == "unavailable"


def is_legal_transition(source: str | None, target: str) -> bool:
    admitted_target = admit_status(target)
    if source is START:
        return True
    admitted_source = admit_status(source)
    return admitted_target in _TRANSITION_TARGETS[admitted_source]


def admit_transition(source: str | None, target: str) -> str:
    """Return `target` iff the closed table admits the edge. Deterministic."""

    admitted_target = admit_status(target)
    if source is START:
        return admitted_target
    admitted_source = admit_status(source)
    allowed = _TRANSITION_TARGETS[admitted_source]
    if admitted_target not in allowed:
        raise BoundaryViolationError(
            f"illegal transition {admitted_source} -> {admitted_target}",
            details={"stage": admitted_source, "reason": admitted_target},
        )
    return admitted_target


def legal_targets(source: str) -> tuple[str, ...]:
    return _TRANSITION_TARGETS[admit_status(source)]


def transition_table() -> Mapping[str, tuple[str, ...]]:
    return _TRANSITION_TARGETS


def transition_pairs() -> tuple[Mapping[str, str], ...]:
    pairs: list[Mapping[str, str]] = []
    for source in STATUSES:
        for target in _TRANSITION_TARGETS[source]:
            pairs.append(MappingProxyType({"from": source, "to": target}))
    return tuple(pairs)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _require_cid(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedError(f"identity field {field} is required")
    try:
        reject_pseudo_cid(value)
    except CompatibilityError as exc:
        raise PseudoCidError("pseudo-CID is not admitted") from exc
    return value


@dataclass(frozen=True)
class ResultIdentities:
    """Stable identities bound through every result record."""

    repository_id: str
    repository_state_cid: str
    task_id: str
    run_id: str
    trace_id: str
    evidence_cid: str | None = None
    patch_id: str | None = None
    artifact_id: str | None = None
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in REQUIRED_IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MalformedError(f"identity field {name} is required")
        if self.contract_version != CONTRACT_VERSION:
            raise SchemaMismatchError(
                f"contract version {self.contract_version!r} is not {CONTRACT_VERSION}"
            )
        _require_cid(self.repository_state_cid, field="repository_state_cid")
        if self.evidence_cid is not None:
            object.__setattr__(
                self,
                "evidence_cid",
                _require_cid(self.evidence_cid, field="evidence_cid"),
            )
        if self.patch_id is not None and not str(self.patch_id).strip():
            raise MalformedError("identity field patch_id is required")
        if self.artifact_id is not None and not str(self.artifact_id).strip():
            raise MalformedError("identity field artifact_id is required")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "repository_id": self.repository_id,
                "repository_state_cid": self.repository_state_cid,
                "task_id": self.task_id,
                "run_id": self.run_id,
                "trace_id": self.trace_id,
                "patch_id": self.patch_id,
                "evidence_cid": self.evidence_cid,
                "artifact_id": self.artifact_id,
                "contract_version": self.contract_version,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ResultIdentities:
        if not isinstance(payload, Mapping):
            raise MalformedError("identities must be a mapping")
        extra = set(payload) - {
            "repository_id",
            "repository_state_cid",
            "task_id",
            "run_id",
            "trace_id",
            "patch_id",
            "evidence_cid",
            "artifact_id",
            "contract_version",
        }
        if extra:
            raise UnknownFieldError(f"unknown identity field {sorted(extra)[0]!r}")
        return cls(
            repository_id=str(payload.get("repository_id") or ""),
            repository_state_cid=str(payload.get("repository_state_cid") or ""),
            task_id=str(payload.get("task_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            trace_id=str(payload.get("trace_id") or ""),
            evidence_cid=_optional_str(payload.get("evidence_cid")),
            patch_id=_optional_str(payload.get("patch_id")),
            artifact_id=_optional_str(payload.get("artifact_id")),
            contract_version=str(payload.get("contract_version") or CONTRACT_VERSION),
        )


def _bind_terminal_identities(status: str, identities: ResultIdentities) -> None:
    if not is_terminal(status):
        return
    if not identities.evidence_cid:
        raise IdentityInconsistentError(
            "terminal results must bind evidence identity",
            details={"reason": status},
        )
    if status in PATCH_BEARING_STATUSES and not identities.patch_id:
        raise IdentityInconsistentError(
            "terminal patch-bearing results must bind patch identity",
            details={"reason": status},
        )
    mapping = identities.to_mapping()
    for name in IDENTITY_FIELDS:
        if name not in mapping:
            raise IdentityInconsistentError(
                f"terminal results must bind {name}",
                details={"field": name},
            )


@dataclass(frozen=True)
class ResultRecord:
    """Typed identity-bound result. Callers cannot mistake non-success for success."""

    schema: str
    status: str
    identities: ResultIdentities
    provenance: str = "live"
    error: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))
        if self.schema != RESULT_SCHEMA:
            raise SchemaMismatchError(
                f"result schema {self.schema!r} is not {RESULT_SCHEMA}"
            )
        admit_status(self.status)
        admit_provenance(self.provenance)
        if self.error is not None:
            admit_error(self.error)
        semantics = STATUS_SEMANTICS[self.status]
        object.__setattr__(self, "disposition", semantics["disposition"])
        object.__setattr__(self, "terminal", bool(semantics["terminal"]))
        object.__setattr__(self, "retryable", bool(semantics["retryable"]))
        object.__setattr__(self, "escalation", bool(semantics["escalation"]))
        object.__setattr__(self, "human_review", bool(semantics["human_review"]))
        object.__setattr__(self, "repair", bool(semantics["repair"]))
        object.__setattr__(self, "partial_effect", bool(semantics["partial_effect"]))
        object.__setattr__(self, "failed", bool(semantics["failed"]))
        object.__setattr__(self, "unavailable", bool(semantics["unavailable"]))
        accepted = is_success(self.status, provenance=self.provenance)
        if accepted and self.error is not None:
            raise BoundaryViolationError("succeeded results cannot carry an error")
        if self.status == "succeeded" and self.provenance == "simulated":
            raise SimulatedPromotedError("simulated results cannot be labeled succeeded")
        if self.status == "unavailable" and accepted:
            raise BoundaryViolationError("unavailable cannot be collapsed into success")
        if self.status == "unavailable" and self.failed:
            raise BoundaryViolationError("unavailable cannot be collapsed into failure")
        if self.status in {"partial_effect", "repair_required", "human_review_required"} and accepted:
            raise BoundaryViolationError("repair and review results cannot claim success")
        object.__setattr__(self, "accepted", accepted)
        _bind_terminal_identities(self.status, self.identities)
        if self.error is not None:
            mapped_status = error_status(self.error)
            if mapped_status != self.status and self.status not in {
                mapped_status,
                "repair_required",
                "human_review_required",
                "cancelled",
            }:
                # Error codes may accompany a legally reached sibling status,
                # but cannot authorize success.
                if self.status == "succeeded":
                    raise BoundaryViolationError("errors cannot claim success")

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": self.schema,
                "contract_version": CONTRACT_VERSION,
                "status": self.status,
                "provenance": self.provenance,
                "error": self.error,
                "accepted": self.accepted,
                "failed": self.failed,
                "unavailable": self.unavailable,
                "terminal": self.terminal,
                "retryable": self.retryable,
                "escalation": self.escalation,
                "human_review": self.human_review,
                "repair": self.repair,
                "partial_effect": self.partial_effect,
                "disposition": self.disposition,
                "identities": dict(self.identities.to_mapping()),
                "payload": dict(self.payload) if isinstance(self.payload, Mapping) else self.payload,
                "status_taxonomy_content_id": STATUS_TAXONOMY_CONTENT_ID,
                "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
            }
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ResultRecord:
        if not isinstance(payload, Mapping):
            raise MalformedError("result payload must be a mapping")
        extra = set(payload) - {
            "schema",
            "contract_version",
            "status",
            "provenance",
            "error",
            "accepted",
            "failed",
            "unavailable",
            "terminal",
            "retryable",
            "escalation",
            "human_review",
            "repair",
            "partial_effect",
            "disposition",
            "identities",
            "payload",
            "status_taxonomy_content_id",
            "error_taxonomy_content_id",
        }
        if extra:
            raise UnknownFieldError(f"unknown result field {sorted(extra)[0]!r}")
        schema = payload.get("schema", RESULT_SCHEMA)
        identities_raw = payload.get("identities")
        if not isinstance(identities_raw, Mapping):
            raise MalformedError("result identities are required")
        record = cls(
            schema=str(schema),
            status=str(payload.get("status") or ""),
            identities=ResultIdentities.from_mapping(identities_raw),
            provenance=str(payload.get("provenance") or "live"),
            error=_optional_str(payload.get("error")),
            payload=payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {},
        )
        if payload.get("accepted") is True and not record.accepted:
            raise BoundaryViolationError("payload accepted flag cannot claim success")
        return record

    def transition(self, target: str, *, error: str | None = None) -> ResultRecord:
        admitted = admit_transition(self.status, target)
        return ResultRecord(
            schema=RESULT_SCHEMA,
            status=admitted,
            identities=self.identities,
            provenance=self.provenance,
            error=error if error is not None else (None if admitted == "succeeded" else self.error),
            payload=self.payload,
        )


def admit_result(value: Any) -> ResultRecord:
    """Admit a typed result. Generic success dictionaries are rejected."""

    if isinstance(value, ResultRecord):
        return value
    if isinstance(value, ProofContextError):
        raise MalformedError("errors are not result records")
    if not isinstance(value, Mapping):
        raise MalformedError("result must be a ResultRecord or mapping")
    if any(key in value for key in ("ok", "success", "passed", "failed")) and "status" not in value:
        raise MalformedError("generic success dictionaries are not admitted")
    if value.get("ok") is True or value.get("success") is True or value.get("passed") is True:
        if value.get("status") != "succeeded":
            raise BoundaryViolationError("generic success dictionaries are not admitted")
    return ResultRecord.from_mapping(value)


def result_from_error(
    error: ProofContextError | str,
    identities: ResultIdentities,
    *,
    provenance: str = "live",
    payload: Mapping[str, Any] | None = None,
) -> ResultRecord:
    """Project a typed error into a non-success result record."""

    typed = error if isinstance(error, ProofContextError) else error_for(admit_error(error))
    if typed.accepted:
        raise BoundaryViolationError("errors cannot claim success")
    return ResultRecord(
        schema=RESULT_SCHEMA,
        status=typed.status,
        identities=identities,
        provenance=provenance,
        error=typed.code,
        payload=payload or {},
    )


def emit_result(
    status: str,
    identities: ResultIdentities,
    *,
    provenance: str = "live",
    error: str | None = None,
    payload: Mapping[str, Any] | None = None,
    source: str | None = START,
) -> ResultRecord:
    admitted = admit_transition(source, status)
    return ResultRecord(
        schema=RESULT_SCHEMA,
        status=admitted,
        identities=identities,
        provenance=provenance,
        error=error,
        payload=payload or {},
    )


_DESCRIPTOR_BODY: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "schema": RESULT_DESCRIPTOR_SCHEMA,
        "contract_version": CONTRACT_VERSION,
        "contract_schema_prefix": CONTRACT_SCHEMA_PREFIX,
        "statuses": STATUSES,
        "errors": ERRORS,
        "provenances": PROVENANCES,
        "terminal_statuses": TERMINAL_STATUSES,
        "retryable_statuses": RETRYABLE_STATUSES,
        "escalation_statuses": ESCALATION_STATUSES,
        "review_statuses": REVIEW_STATUSES,
        "repair_statuses": REPAIR_STATUSES,
        "failure_statuses": tuple(status for status in STATUSES if status in FAILURE_STATUSES),
        "patch_bearing_statuses": tuple(
            status for status in STATUSES if status in PATCH_BEARING_STATUSES
        ),
        "status_semantics": {status: dict(STATUS_SEMANTICS[status]) for status in STATUSES},
        "transitions": {status: list(_TRANSITION_TARGETS[status]) for status in STATUSES},
        "identity_fields": IDENTITY_FIELDS,
        "pcce_006_content_id": PCCE_006_CONTENT_ID,
        "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
        "status_taxonomy_content_id": STATUS_TAXONOMY_CONTENT_ID,
        "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
    }
)
RESULT_STATE_CID: Final[str] = mint_result_cid(_DESCRIPTOR_BODY)
RESULT_DESCRIPTOR: Final[Mapping[str, Any]] = MappingProxyType(
    {**dict(_DESCRIPTOR_BODY), "cid": RESULT_STATE_CID}
)


def result_descriptor() -> Mapping[str, Any]:
    return RESULT_DESCRIPTOR


def result_state_cid() -> str:
    return RESULT_STATE_CID


def frozen_result_taxonomy() -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "pcce_006_content_id": PCCE_006_CONTENT_ID,
            "compatibility_matrix_content_id": COMPATIBILITY_MATRIX_CONTENT_ID,
            "status_taxonomy_content_id": STATUS_TAXONOMY_CONTENT_ID,
            "error_taxonomy_content_id": ERROR_TAXONOMY_CONTENT_ID,
            "statuses": STATUSES,
            "errors": ERRORS,
            "provenances": PROVENANCES,
            "cid": RESULT_STATE_CID,
        }
    )


__all__ = [
    "COMPATIBILITY_MATRIX_CONTENT_ID",
    "CONTRACT_SCHEMA_PREFIX",
    "CONTRACT_VERSION",
    "ESCALATION_STATUSES",
    "FAILURE_STATUSES",
    "IDENTITY_FIELDS",
    "PCCE_006_CONTENT_ID",
    "PATCH_BEARING_STATUSES",
    "PROVENANCES",
    "REPAIR_STATUSES",
    "RESULT_DESCRIPTOR",
    "RESULT_DESCRIPTOR_SCHEMA",
    "RESULT_SCHEMA",
    "RESULT_STATE_CID",
    "RETRYABLE_STATUSES",
    "REVIEW_STATUSES",
    "SCHEMA",
    "START",
    "STATUSES",
    "STATUS_SEMANTICS",
    "STATUS_TAXONOMY_CONTENT_ID",
    "TERMINAL_STATUSES",
    "ResultIdentities",
    "ResultRecord",
    "admit_provenance",
    "admit_result",
    "admit_status",
    "admit_transition",
    "classify_status",
    "emit_result",
    "frozen_result_taxonomy",
    "is_failure",
    "is_legal_transition",
    "is_success",
    "is_terminal",
    "is_unavailable",
    "legal_targets",
    "mint_result_cid",
    "result_descriptor",
    "result_from_error",
    "result_state_cid",
    "status_semantics",
    "transition_pairs",
    "transition_table",
]
