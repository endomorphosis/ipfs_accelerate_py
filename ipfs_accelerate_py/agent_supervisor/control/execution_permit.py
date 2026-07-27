"""Exact, short-lived execution permits for mutation-capable operations.

Plan admission is intentionally not execution authority.  This module is the
single transport-neutral boundary that turns a current, admitted
``PlanAdmissionRequest`` into a narrowly scoped permit and verifies it
immediately before an effect.

The permit binds the complete :class:`DecisionRequest`, the admitted candidate
graph, all semantic roots, the mandatory dependency closure and context
witness, domain evidence, validation plan, caller, policy, lease, fencing
epoch, expiry, idempotency key, and use bound.  Verification is fail closed and
consumes a use atomically, so a successful check cannot be replayed silently.
It never grants task-completion authority.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..decision_context import ContextCompletenessWitness
from ..decision_contracts import (
    DecisionAuthority,
    DecisionKind,
    DecisionRequest,
    DecisionStage,
)
from ..formal_verification_contracts import canonical_json_bytes, content_identity
from ..ir_constraint_compiler import (
    PlanAdmissionReceipt,
    PlanAdmissionRequest,
    PlanAdmissionVerdict,
    ValidationRequirement,
    compile_plan_admission,
)
from ..semantic_dependency_graph import MandatoryClosure


EXECUTION_PERMIT_VERSION: Final[int] = 1
EXECUTION_PERMIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-permit@1"
)
EXECUTION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-evidence@1"
)
EXECUTION_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-attempt@1"
)
PERMIT_USE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/execution-permit-use@1"
)
DEFAULT_MAX_PERMIT_TTL_MS: Final[int] = 5 * 60 * 1_000
MAX_ALLOWED_USES: Final[int] = 1_024
_ACCEPTED_EVIDENCE_AUTHORITIES: Final[frozenset[str]] = frozenset(
    {"authoritative", "verified", "verified_input"}
)
_COMPLETION_WORDS: Final[frozenset[str]] = frozenset(
    {
        "complete",
        "completion",
        "mark_complete",
        "mark_completed",
        "set_complete",
        "task_complete",
    }
)


class ExecutionPermitError(ValueError):
    """Base class for malformed, denied, stale, or replayed permits."""


class PermitIssuanceError(ExecutionPermitError):
    """The evidence supplied for issuance does not establish exact authority."""


class PermitVerificationCode(str, Enum):
    VALID = "valid"
    UNTRUSTED = "untrusted_permit"
    REPLAYED = "replayed"
    CHANGED_OPERATION = "changed_operation"
    CHANGED_TARGET = "changed_target"
    CHANGED_EFFECT = "changed_effect"
    STALE_ROOT = "stale_root"
    STALE_RECEIPT = "stale_receipt"
    EXPIRED = "expired"
    NOT_YET_VALID = "not_yet_valid"
    LEASE_LOST = "lease_lost"
    FENCE_LOST = "fence_lost"
    CALLER_MISMATCH = "caller_mismatch"
    TASK_MISMATCH = "task_mismatch"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    PATH_BROADENING = "path_broadening"
    PARTIAL_AUTHORITY = "partial_authority"
    MANDATORY_STATE_UNKNOWN = "mandatory_state_unknown"
    MANDATORY_STATE_CONTRADICTORY = "mandatory_state_contradictory"
    COMPLETION_AUTHORITY_FORBIDDEN = "completion_authority_forbidden"
    INVALID_PERMIT = "invalid_permit"


class PermitVerificationError(ExecutionPermitError):
    """A permit cannot authorize the supplied immediate operation."""

    def __init__(self, code: PermitVerificationCode | str, message: str) -> None:
        self.code = PermitVerificationCode(code)
        super().__init__(message)


class PermitReplayError(PermitVerificationError):
    """A permit use sequence has already been consumed or is out of range."""


class MandatoryEvidenceState(str, Enum):
    SATISFIED = "satisfied"
    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"
    STALE = "stale"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ExecutionPermitError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise ExecutionPermitError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise ExecutionPermitError(f"{name} is required")
    return value


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExecutionPermitError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    if maximum is not None and value > maximum:
        raise ExecutionPermitError(f"{name} exceeds its maximum")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ExecutionPermitError(
            "floating point values are not canonical permit data"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ExecutionPermitError("permit mapping keys must be strings")
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _plain(converter())
    converter = getattr(value, "to_record", None)
    if callable(converter):
        return _plain(converter())
    raise ExecutionPermitError(
        f"unsupported permit value: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    plain = _plain(value)
    if isinstance(plain, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in plain.items()}
        )
    if isinstance(plain, list):
        return tuple(_freeze(item) for item in plain)
    return plain


def _strings(
    value: Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ExecutionPermitError(f"{name} must be a sequence")
    result = tuple(sorted({_text(item, name) for item in value}))
    if required and not result:
        raise ExecutionPermitError(f"{name} must not be empty")
    return result


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()


def _artifact_token(artifact: Any) -> str:
    return ":".join(
        (
            str(getattr(artifact, "artifact_id", "") or ""),
            str(getattr(artifact, "cid_v1", "") or ""),
            str(getattr(artifact, "supervisor_digest", "") or ""),
        )
    )


def _semantic_root_tokens(request: DecisionRequest) -> Mapping[str, str]:
    return MappingProxyType(
        {
            root.kind.value: _artifact_token(root.artifact)
            for root in request.semantic_roots
        }
    )


def _semantic_roots_digest(request: DecisionRequest) -> str:
    return _canonical_digest(tuple(root.to_record() for root in request.semantic_roots))


def _repository_tree_aliases(request: DecisionRequest) -> frozenset[str]:
    repository = request.repository_root
    worktree = request.dirty_worktree_root
    return frozenset(
        {
            repository.artifact_id,
            repository.cid_v1,
            repository.supervisor_digest,
            _artifact_token(repository),
            worktree.artifact_id,
            worktree.cid_v1,
            worktree.supervisor_digest,
            _artifact_token(worktree),
        }
    )


def _declared_paths(request: DecisionRequest) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                path
                for target in request.action.targets
                for path in target.repository_paths
            }
            | {
                path
                for effect in request.expected_effects
                for path in effect.repository_paths
            }
        )
    )


def _candidate_records(
    request: PlanAdmissionRequest,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    candidate = _plain(request.candidate_plan)
    raw_actions = candidate.get("actions", candidate.get("tasks", ()))
    if isinstance(raw_actions, Mapping):
        raw_actions = tuple(raw_actions.values())
    actions = tuple(
        item for item in raw_actions if isinstance(item, Mapping)
    )
    selected = tuple(
        item
        for item in actions
        if item.get("action_id", item.get("task_id", item.get("id")))
        == request.decision_request.action.action_id
    )
    if len(selected) != 1:
        raise PermitIssuanceError(
            "candidate graph must contain the exact DecisionRequest action once"
        )
    action = selected[0]
    raw_effects: list[Mapping[str, Any]] = []
    embedded = action.get("effects") or ()
    if isinstance(embedded, Mapping):
        embedded = (embedded,)
    raw_effects.extend(item for item in embedded if isinstance(item, Mapping))
    top = candidate.get("effects") or ()
    if isinstance(top, Mapping):
        top = tuple(top.values())
    raw_effects.extend(
        item
        for item in top
        if isinstance(item, Mapping)
        and item.get("action_id", item.get("task_id"))
        == request.decision_request.action.action_id
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in raw_effects:
        effect_id = str(item.get("effect_id") or "")
        if effect_id:
            normalized = _plain(item)
            previous = by_id.get(effect_id)
            if previous is not None and previous != normalized:
                raise PermitIssuanceError(
                    "candidate graph contains contradictory duplicate effects"
                )
            by_id[effect_id] = normalized
    expected_ids = {
        item.effect_id for item in request.decision_request.expected_effects
    }
    if set(by_id) != expected_ids:
        raise PermitIssuanceError(
            "candidate and DecisionRequest effect identities differ"
        )
    return _freeze(action), tuple(_freeze(by_id[key]) for key in sorted(by_id))


def _check_candidate_binding(
    decision: DecisionRequest,
    action: Mapping[str, Any],
) -> None:
    comparisons = {
        "action_id": decision.action.action_id,
        "action": decision.action.action,
        "principal": decision.principal,
        "requested_authority": decision.requested_authority.value,
    }
    aliases = {
        "tool": decision.action.tool_id,
        "tool_id": decision.action.tool_id,
    }
    for name, expected in comparisons.items():
        if action.get(name) != expected:
            raise PermitIssuanceError(
                f"candidate {name} does not match the DecisionRequest"
            )
    if not any(action.get(name) == expected for name, expected in aliases.items()):
        raise PermitIssuanceError(
            "candidate tool does not match the DecisionRequest"
        )
    if "arguments" not in action or _plain(action["arguments"]) != _plain(
        decision.action.arguments
    ):
        raise PermitIssuanceError(
            "candidate tool arguments do not match the DecisionRequest"
        )
    expected_targets = {item.target_id for item in decision.action.targets}
    raw_targets = action.get("targets")
    if raw_targets is None:
        raw_target = action.get("target")
        candidate_targets = set() if raw_target is None else {raw_target}
    else:
        if isinstance(raw_targets, str) or not isinstance(raw_targets, Sequence):
            raise PermitIssuanceError("candidate targets must be a sequence")
        candidate_targets = set(raw_targets)
    if candidate_targets != expected_targets:
        raise PermitIssuanceError(
            "candidate targets do not exactly match the DecisionRequest"
        )
    candidate_paths = {
        str(path)
        for path in action.get("repository_paths", ())
    }
    if not candidate_paths.issubset(_declared_paths(decision)):
        raise PermitIssuanceError(
            "candidate action broadens the DecisionRequest path scope"
        )


@dataclass(frozen=True)
class ExecutionEvidence:
    """One current, authoritative receipt used by permit issuance."""

    domain: str
    receipt_id: str
    subject_ids: tuple[str, ...] = ()
    semantic_roots: Mapping[str, str] = field(default_factory=dict)
    state: MandatoryEvidenceState = MandatoryEvidenceState.SATISFIED
    authority: str = "authoritative"
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _text(self.domain, "evidence domain"))
        object.__setattr__(
            self, "receipt_id", _text(self.receipt_id, "evidence receipt_id")
        )
        object.__setattr__(
            self,
            "subject_ids",
            _strings(self.subject_ids, "evidence subject_ids"),
        )
        if not isinstance(self.semantic_roots, Mapping):
            raise ExecutionPermitError("evidence semantic_roots must be a mapping")
        roots = {
            _text(key, "evidence root kind"): _text(value, "evidence root")
            for key, value in self.semantic_roots.items()
        }
        object.__setattr__(
            self,
            "semantic_roots",
            MappingProxyType(dict(sorted(roots.items()))),
        )
        object.__setattr__(self, "state", MandatoryEvidenceState(self.state))
        object.__setattr__(
            self, "authority", _text(self.authority, "evidence authority")
        )
        if self.expires_at_ms is not None:
            object.__setattr__(
                self,
                "expires_at_ms",
                _integer(self.expires_at_ms, "evidence expires_at_ms"),
            )

    @property
    def current_and_authoritative(self) -> bool:
        return (
            self.state is MandatoryEvidenceState.SATISFIED
            and self.authority in _ACCEPTED_EVIDENCE_AUTHORITIES
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_EVIDENCE_SCHEMA,
            "version": EXECUTION_PERMIT_VERSION,
            "domain": self.domain,
            "receipt_id": self.receipt_id,
            "subject_ids": list(self.subject_ids),
            "semantic_roots": dict(self.semantic_roots),
            "state": self.state.value,
            "authority": self.authority,
            "expires_at_ms": self.expires_at_ms,
        }

    @property
    def content_id(self) -> str:
        return content_identity(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionEvidence":
        if value.get("schema") != EXECUTION_EVIDENCE_SCHEMA:
            raise ExecutionPermitError("unsupported execution evidence schema")
        if value.get("version") != EXECUTION_PERMIT_VERSION:
            raise ExecutionPermitError("unsupported execution evidence version")
        result = cls(
            domain=value.get("domain", ""),
            receipt_id=value.get("receipt_id", ""),
            subject_ids=tuple(value.get("subject_ids") or ()),
            semantic_roots=value.get("semantic_roots") or {},
            state=value.get("state", MandatoryEvidenceState.UNKNOWN),
            authority=value.get("authority", ""),
            expires_at_ms=value.get("expires_at_ms"),
        )
        if set(value) != set(result.to_dict()):
            raise ExecutionPermitError(
                "execution evidence has missing or unknown fields"
            )
        return result


def _coerce_evidence(values: Any) -> tuple[ExecutionEvidence, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ExecutionPermitError("evidence_receipts must be a sequence")
    result = tuple(
        item
        if isinstance(item, ExecutionEvidence)
        else ExecutionEvidence.from_dict(item)
        for item in values
    )
    result = tuple(sorted(result, key=lambda item: (item.domain, item.receipt_id)))
    ids = tuple(item.receipt_id for item in result)
    if len(ids) != len(set(ids)):
        raise ExecutionPermitError("evidence receipt IDs must be unique")
    return result


@dataclass(frozen=True)
class ExecutionPermit:
    """Immutable authority for one exact declared operation."""

    decision_request: DecisionRequest
    candidate_plan_id: str
    candidate_graph_id: str
    candidate_action: Mapping[str, Any]
    candidate_effects: tuple[Mapping[str, Any], ...]
    admission_request_id: str
    admission_receipt_id: str
    repository_tree_id: str
    semantic_roots: Mapping[str, str]
    mandatory_closure: MandatoryClosure
    context_witness: ContextCompletenessWitness
    evidence_receipts: tuple[ExecutionEvidence, ...]
    validation_plan: tuple[ValidationRequirement, ...]
    caller: str
    policy_id: str
    policy_revision: str
    issued_at_ms: int
    expires_at_ms: int
    allowed_use_count: int
    issuer_id: str = "agent-supervisor:execution-permit-issuer"

    def __post_init__(self) -> None:
        if not isinstance(self.decision_request, DecisionRequest):
            if not isinstance(self.decision_request, Mapping):
                raise ExecutionPermitError(
                    "decision_request must be a DecisionRequest"
                )
            object.__setattr__(
                self,
                "decision_request",
                DecisionRequest.from_dict(self.decision_request),
            )
        for name in (
            "candidate_plan_id",
            "candidate_graph_id",
            "admission_request_id",
            "admission_receipt_id",
            "repository_tree_id",
            "caller",
            "policy_id",
            "policy_revision",
            "issuer_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        if not isinstance(self.candidate_action, Mapping):
            raise ExecutionPermitError("candidate_action must be a mapping")
        object.__setattr__(
            self, "candidate_action", _freeze(self.candidate_action)
        )
        if isinstance(self.candidate_effects, (str, bytes)) or not isinstance(
            self.candidate_effects, Sequence
        ):
            raise ExecutionPermitError("candidate_effects must be a sequence")
        effects = tuple(_freeze(item) for item in self.candidate_effects)
        if any(not isinstance(item, Mapping) for item in effects):
            raise ExecutionPermitError("candidate effects must be mappings")
        object.__setattr__(self, "candidate_effects", effects)
        roots = {
            _text(key, "semantic root kind"): _text(value, "semantic root")
            for key, value in self.semantic_roots.items()
        }
        object.__setattr__(
            self,
            "semantic_roots",
            MappingProxyType(dict(sorted(roots.items()))),
        )
        if not isinstance(self.mandatory_closure, MandatoryClosure):
            if not isinstance(self.mandatory_closure, Mapping):
                raise ExecutionPermitError(
                    "mandatory_closure must be a MandatoryClosure"
                )
            object.__setattr__(
                self,
                "mandatory_closure",
                MandatoryClosure.from_dict(self.mandatory_closure),
            )
        if not isinstance(self.context_witness, ContextCompletenessWitness):
            if not isinstance(self.context_witness, Mapping):
                raise ExecutionPermitError(
                    "context_witness must be a ContextCompletenessWitness"
                )
            object.__setattr__(
                self,
                "context_witness",
                ContextCompletenessWitness.from_dict(self.context_witness),
            )
        object.__setattr__(
            self, "evidence_receipts", _coerce_evidence(self.evidence_receipts)
        )
        validations = tuple(
            item
            if isinstance(item, ValidationRequirement)
            else ValidationRequirement.from_dict(item)
            for item in self.validation_plan
        )
        validations = tuple(
            sorted(validations, key=lambda item: item.requirement_id)
        )
        object.__setattr__(self, "validation_plan", validations)
        object.__setattr__(
            self, "issued_at_ms", _integer(self.issued_at_ms, "issued_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _integer(self.expires_at_ms, "expires_at_ms")
        )
        if self.expires_at_ms <= self.issued_at_ms:
            raise ExecutionPermitError("permit expiry must be after issuance")
        if self.expires_at_ms - self.issued_at_ms > DEFAULT_MAX_PERMIT_TTL_MS:
            raise ExecutionPermitError(
                "permit lifetime exceeds the absolute short-lived TTL bound"
            )
        object.__setattr__(
            self,
            "allowed_use_count",
            _integer(
                self.allowed_use_count,
                "allowed_use_count",
                minimum=1,
                maximum=MAX_ALLOWED_USES,
            ),
        )
        decision = self.decision_request
        if decision.requested_authority is not DecisionAuthority.MUTATION:
            raise ExecutionPermitError(
                "an execution permit requires exact mutation authority"
            )
        if (
            decision.lease_id is None
            or decision.fencing_epoch is None
            or decision.idempotency_key is None
        ):
            raise ExecutionPermitError(
                "execution permit requires lease, fence, and idempotency"
            )
        if self.semantic_roots != _semantic_root_tokens(decision):
            raise ExecutionPermitError(
                "permit semantic roots do not match its DecisionRequest"
            )
        if self.context_witness.decision_request_id != decision.request_id:
            raise ExecutionPermitError(
                "context witness belongs to a different DecisionRequest"
            )
        if self.context_witness.closure_id != self.mandatory_closure.closure_id:
            raise ExecutionPermitError(
                "context witness and mandatory closure differ"
            )
        if self.context_witness.roots_digest != _semantic_roots_digest(decision):
            raise ExecutionPermitError(
                "context witness semantic roots differ from the DecisionRequest"
            )
        _check_candidate_binding(decision, self.candidate_action)
        candidate_effect_ids = {
            str(item.get("effect_id") or "") for item in self.candidate_effects
        }
        expected_effect_ids = {
            item.effect_id for item in decision.expected_effects
        }
        if candidate_effect_ids != expected_effect_ids:
            raise ExecutionPermitError(
                "candidate effects do not exactly match the DecisionRequest"
            )

    @property
    def permit_id(self) -> str:
        return content_identity(self._payload())

    @property
    def content_id(self) -> str:
        return self.permit_id

    @property
    def lease_id(self) -> str:
        assert self.decision_request.lease_id is not None
        return self.decision_request.lease_id

    @property
    def fencing_epoch(self) -> int:
        assert self.decision_request.fencing_epoch is not None
        return self.decision_request.fencing_epoch

    @property
    def idempotency_key(self) -> str:
        assert self.decision_request.idempotency_key is not None
        return self.decision_request.idempotency_key

    @property
    def principal_id(self) -> str:
        return self.decision_request.principal

    @property
    def objective_id(self) -> str:
        return self.decision_request.objective_id

    @property
    def repository_id(self) -> str:
        return self.decision_request.repository_id

    @property
    def worktree_root_id(self) -> str:
        return _artifact_token(self.decision_request.dirty_worktree_root)

    @property
    def declared_paths(self) -> tuple[str, ...]:
        return _declared_paths(self.decision_request)

    @property
    def grants_completion_authority(self) -> bool:
        return False

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def operation_fingerprint(self) -> str:
        return content_identity(
            {
                "decision_request": self.decision_request.to_dict(),
                "candidate_plan_id": self.candidate_plan_id,
                "candidate_graph_id": self.candidate_graph_id,
                "candidate_action": _plain(self.candidate_action),
                "candidate_effects": _plain(self.candidate_effects),
                "repository_tree_id": self.repository_tree_id,
                "semantic_roots": dict(self.semantic_roots),
                "closure_id": self.mandatory_closure.closure_id,
                "context_witness_id": self.context_witness.content_id,
                "evidence_receipt_ids": [
                    item.content_id for item in self.evidence_receipts
                ],
                "validation_plan": [
                    item.to_dict() for item in self.validation_plan
                ],
                "caller": self.caller,
                "policy_id": self.policy_id,
                "policy_revision": self.policy_revision,
                "lease_id": self.lease_id,
                "fencing_epoch": self.fencing_epoch,
                "idempotency_key": self.idempotency_key,
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_PERMIT_SCHEMA,
            "version": EXECUTION_PERMIT_VERSION,
            "decision_request": self.decision_request.to_dict(),
            "decision_request_id": self.decision_request.request_id,
            "candidate_plan_id": self.candidate_plan_id,
            "candidate_graph_id": self.candidate_graph_id,
            "candidate_action": _plain(self.candidate_action),
            "tool_arguments": _plain(self.decision_request.action.arguments),
            "targets": [
                item.to_dict() for item in self.decision_request.action.targets
            ],
            "candidate_effects": _plain(self.candidate_effects),
            "expected_effects": [
                item.to_dict() for item in self.decision_request.expected_effects
            ],
            "admission_request_id": self.admission_request_id,
            "admission_receipt_id": self.admission_receipt_id,
            "repository_id": self.repository_id,
            "repository_path": self.decision_request.repository_path,
            "repository_tree_id": self.repository_tree_id,
            "worktree_root_id": self.worktree_root_id,
            "semantic_roots": dict(self.semantic_roots),
            "mandatory_closure": self.mandatory_closure.to_dict(),
            "context_witness": self.context_witness.to_dict(),
            "evidence_receipts": [
                item.to_dict() for item in self.evidence_receipts
            ],
            "validation_plan": [
                item.to_dict() for item in self.validation_plan
            ],
            "caller": self.caller,
            "principal_id": self.principal_id,
            "objective_id": self.objective_id,
            "objective_revision": self.decision_request.objective_revision,
            "lease_id": self.lease_id,
            "fencing_epoch": self.fencing_epoch,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "idempotency_key": self.idempotency_key,
            "allowed_use_count": self.allowed_use_count,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "issuer_id": self.issuer_id,
            "declared_paths": list(self.declared_paths),
            "operation_fingerprint": self.operation_fingerprint,
            "grants_completion_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "permit_id": self.permit_id}

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPermit":
        if value.get("schema") != EXECUTION_PERMIT_SCHEMA:
            raise ExecutionPermitError("unsupported execution permit schema")
        if value.get("version") != EXECUTION_PERMIT_VERSION:
            raise ExecutionPermitError("unsupported execution permit version")
        result = cls(
            decision_request=DecisionRequest.from_dict(
                value.get("decision_request") or {}
            ),
            candidate_plan_id=value.get("candidate_plan_id", ""),
            candidate_graph_id=value.get("candidate_graph_id", ""),
            candidate_action=value.get("candidate_action") or {},
            candidate_effects=tuple(value.get("candidate_effects") or ()),
            admission_request_id=value.get("admission_request_id", ""),
            admission_receipt_id=value.get("admission_receipt_id", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            semantic_roots=value.get("semantic_roots") or {},
            mandatory_closure=MandatoryClosure.from_dict(
                value.get("mandatory_closure") or {}
            ),
            context_witness=ContextCompletenessWitness.from_dict(
                value.get("context_witness") or {}
            ),
            evidence_receipts=tuple(
                ExecutionEvidence.from_dict(item)
                for item in value.get("evidence_receipts") or ()
            ),
            validation_plan=tuple(
                ValidationRequirement.from_dict(item)
                for item in value.get("validation_plan") or ()
            ),
            caller=value.get("caller", ""),
            policy_id=value.get("policy_id", ""),
            policy_revision=value.get("policy_revision", ""),
            issued_at_ms=value.get("issued_at_ms", -1),
            expires_at_ms=value.get("expires_at_ms", -1),
            allowed_use_count=value.get("allowed_use_count", 0),
            issuer_id=value.get("issuer_id", ""),
        )
        projections = {
            "permit_id": result.permit_id,
            "decision_request_id": result.decision_request.request_id,
            "repository_id": result.repository_id,
            "repository_path": result.decision_request.repository_path,
            "worktree_root_id": result.worktree_root_id,
            "principal_id": result.principal_id,
            "objective_id": result.objective_id,
            "objective_revision": result.decision_request.objective_revision,
            "lease_id": result.lease_id,
            "fencing_epoch": result.fencing_epoch,
            "idempotency_key": result.idempotency_key,
            "operation_fingerprint": result.operation_fingerprint,
            "grants_completion_authority": False,
        }
        for name, expected in projections.items():
            if value.get(name) != expected:
                raise ExecutionPermitError(
                    f"execution permit {name} projection does not match content"
                )
        sequence_projections = {
            "tool_arguments": _plain(result.decision_request.action.arguments),
            "targets": [
                item.to_dict() for item in result.decision_request.action.targets
            ],
            "expected_effects": [
                item.to_dict() for item in result.decision_request.expected_effects
            ],
            "declared_paths": list(result.declared_paths),
        }
        for name, expected in sequence_projections.items():
            if value.get(name) != expected:
                raise ExecutionPermitError(
                    f"execution permit {name} projection does not match content"
                )
        if set(value) != set(result.to_dict()):
            raise ExecutionPermitError(
                "execution permit has missing or unknown fields"
            )
        return result

    @classmethod
    def from_json(cls, value: str) -> "ExecutionPermit":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ExecutionPermitError("execution permit JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ExecutionPermitError("execution permit JSON must be an object")
        result = cls.from_dict(payload)
        if result.to_json() != value:
            raise ExecutionPermitError(
                "execution permit JSON changes during canonical round trip"
            )
        return result


def _derived_evidence(
    admission: PlanAdmissionRequest,
    receipt: PlanAdmissionReceipt,
    additional: Sequence[ExecutionEvidence],
) -> tuple[ExecutionEvidence, ...]:
    roots = dict(_semantic_root_tokens(admission.decision_request))
    records = [
        ExecutionEvidence(
            domain="intent",
            receipt_id=receipt.intent_result_id,
            semantic_roots=roots,
        ),
        *(
            ExecutionEvidence(
                domain="legal",
                receipt_id=receipt_id,
                semantic_roots=roots,
            )
            for receipt_id in receipt.legal_result_ids
        ),
        ExecutionEvidence(
            domain="security_policy",
            receipt_id=admission.security_policy.content_id,
            semantic_roots=roots,
        ),
        *(
            ExecutionEvidence(
                domain="security",
                receipt_id=receipt_id,
                semantic_roots=roots,
            )
            for receipt_id in receipt.security_decision_ids
        ),
        *(
            ExecutionEvidence(
                domain="proof",
                receipt_id=receipt_id,
                semantic_roots=roots,
            )
            for receipt_id in receipt.proof_result_ids
        ),
        *(
            ExecutionEvidence(
                domain="validation",
                receipt_id=item.evidence_id,
                subject_ids=(item.requirement_id,),
                semantic_roots=roots,
            )
            for item in admission.validation_results
            if item.evidence_id
        ),
        *additional,
    ]
    by_id: dict[str, ExecutionEvidence] = {}
    for item in records:
        previous = by_id.get(item.receipt_id)
        if previous is not None and previous != item:
            raise PermitIssuanceError(
                "the same receipt ID has contradictory permit evidence"
            )
        by_id[item.receipt_id] = item
    return tuple(
        sorted(by_id.values(), key=lambda item: (item.domain, item.receipt_id))
    )


def _reject_completion(decision: DecisionRequest) -> None:
    normalized_action = decision.action.action.lower().replace("-", "_")
    if (
        decision.decision_kind is DecisionKind.COMPLETE
        or decision.stage is DecisionStage.COMPLETION
        or normalized_action in _COMPLETION_WORDS
        or normalized_action.endswith("_complete")
    ):
        raise PermitIssuanceError(
            "execution permits never grant task-completion authority"
        )


def issue_execution_permit(
    admission_request: PlanAdmissionRequest,
    admission_receipt: PlanAdmissionReceipt,
    context_witness: ContextCompletenessWitness,
    *,
    caller: str,
    policy_id: str,
    policy_revision: str,
    issued_at_ms: int,
    expires_at_ms: int,
    allowed_use_count: int = 1,
    evidence_receipts: Sequence[ExecutionEvidence] = (),
    issuer_id: str = "agent-supervisor:execution-permit-issuer",
    max_ttl_ms: int = DEFAULT_MAX_PERMIT_TTL_MS,
) -> ExecutionPermit:
    """Issue an exact permit after independently rechecking all admission facts."""

    if not isinstance(admission_request, PlanAdmissionRequest):
        raise PermitIssuanceError(
            "admission_request must be a PlanAdmissionRequest"
        )
    if not isinstance(admission_receipt, PlanAdmissionReceipt):
        raise PermitIssuanceError(
            "admission_receipt must be a PlanAdmissionReceipt"
        )
    if not isinstance(context_witness, ContextCompletenessWitness):
        raise PermitIssuanceError(
            "context_witness must be a ContextCompletenessWitness"
        )
    decision = admission_request.decision_request
    if decision is None:
        raise PermitIssuanceError(
            "permit issuance requires the complete canonical DecisionRequest"
        )
    _reject_completion(decision)
    if decision.requested_authority is not DecisionAuthority.MUTATION:
        raise PermitIssuanceError(
            "permit issuance requires exact mutation authority"
        )
    recomputed = compile_plan_admission(admission_request)
    if (
        recomputed != admission_receipt
        or recomputed.receipt_id != admission_receipt.receipt_id
    ):
        raise PermitIssuanceError(
            "admission receipt is stale, forged, or detached from its request"
        )
    if (
        recomputed.verdict is not PlanAdmissionVerdict.ADMITTED
        or not recomputed.admitted
    ):
        raise PermitIssuanceError(
            "a rejected or unknown plan admission cannot issue a permit"
        )
    if not recomputed.security_decision_ids or not recomputed.security_grant_ids:
        raise PermitIssuanceError(
            "permit issuance requires exact SecurityIR authorization"
        )
    closure = admission_request.mandatory_closure
    if closure is None or not closure.complete:
        raise PermitIssuanceError(
            "permit issuance requires a complete mandatory dependency closure"
        )
    if (
        recomputed.closure_id != closure.closure_id
        or context_witness.closure_id != closure.closure_id
        or context_witness.mandatory_node_ids != closure.node_ids
        or context_witness.mandatory_edge_ids != closure.edge_ids
    ):
        raise PermitIssuanceError(
            "admission, dependency closure, and context witness differ"
        )
    if (
        context_witness.decision_request_id != decision.request_id
        or context_witness.roots_digest != _semantic_roots_digest(decision)
        or not context_witness.complete
        or context_witness.truncated
    ):
        raise PermitIssuanceError(
            "context completeness witness is stale, partial, or detached"
        )
    if admission_request.repository_tree_id not in _repository_tree_aliases(decision):
        raise PermitIssuanceError(
            "admission repository tree does not match the DecisionRequest roots"
        )
    action, candidate_effects = _candidate_records(admission_request)
    _check_candidate_binding(decision, action)
    issued = _integer(issued_at_ms, "issued_at_ms")
    expires = _integer(expires_at_ms, "expires_at_ms")
    ttl = _integer(
        max_ttl_ms,
        "max_ttl_ms",
        minimum=1,
        maximum=DEFAULT_MAX_PERMIT_TTL_MS,
    )
    if expires <= issued or expires - issued > ttl:
        raise PermitIssuanceError(
            "permit expiry must be positive and within the short-lived TTL bound"
        )
    evidence = _derived_evidence(
        admission_request,
        recomputed,
        _coerce_evidence(tuple(evidence_receipts)),
    )
    roots = dict(_semantic_root_tokens(decision))
    for item in evidence:
        if not item.current_and_authoritative:
            raise PermitIssuanceError(
                f"mandatory {item.domain} state is {item.state.value}"
            )
        if item.expires_at_ms is not None and item.expires_at_ms <= issued:
            raise PermitIssuanceError(
                f"mandatory {item.domain} receipt is expired"
            )
        if item.semantic_roots != roots:
            raise PermitIssuanceError(
                f"mandatory {item.domain} receipt is not bound to all current roots"
            )
    monitor_subjects = {
        subject
        for item in evidence
        if item.domain == "monitor"
        for subject in item.subject_ids
    }
    required_monitors = {
        entry.node_id
        for entry in context_witness.entries
        if entry.node_kind == "monitor"
    } | {
        entry.node_content_id
        for entry in context_witness.entries
        if entry.node_kind == "monitor"
    }
    if required_monitors and not all(
        entry.node_id in monitor_subjects
        or entry.node_content_id in monitor_subjects
        for entry in context_witness.entries
        if entry.node_kind == "monitor"
    ):
        raise PermitIssuanceError(
            "a mandatory runtime monitor lacks a current authoritative receipt"
        )
    return ExecutionPermit(
        decision_request=decision,
        candidate_plan_id=admission_request.candidate_plan_id,
        candidate_graph_id=admission_request.candidate_graph_id,
        candidate_action=action,
        candidate_effects=candidate_effects,
        admission_request_id=admission_request.request_id,
        admission_receipt_id=recomputed.receipt_id,
        repository_tree_id=admission_request.repository_tree_id,
        semantic_roots=roots,
        mandatory_closure=closure,
        context_witness=context_witness,
        evidence_receipts=evidence,
        validation_plan=admission_request.validation_requirements,
        caller=_text(caller, "caller"),
        policy_id=_text(policy_id, "policy_id"),
        policy_revision=_text(policy_revision, "policy_revision"),
        issued_at_ms=issued,
        expires_at_ms=expires,
        allowed_use_count=allowed_use_count,
        issuer_id=_text(issuer_id, "issuer_id"),
    )


@dataclass(frozen=True)
class ExecutionAttempt:
    """Current, immediately pre-effect facts checked against one permit."""

    decision_request: DecisionRequest
    candidate_plan_id: str
    candidate_graph_id: str
    candidate_action: Mapping[str, Any]
    candidate_effects: tuple[Mapping[str, Any], ...]
    repository_tree_id: str
    semantic_roots: Mapping[str, str]
    mandatory_closure: MandatoryClosure
    context_witness: ContextCompletenessWitness
    evidence_receipts: tuple[ExecutionEvidence, ...]
    validation_plan: tuple[ValidationRequirement, ...]
    caller: str
    policy_id: str
    policy_revision: str
    active_lease_id: str
    current_fencing_epoch: int
    idempotency_key: str
    actual_paths: tuple[str, ...]
    now_ms: int
    use_sequence: int = 1
    completion_requested: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.decision_request, DecisionRequest):
            raise ExecutionPermitError(
                "attempt decision_request must be a DecisionRequest"
            )
        for name in (
            "candidate_plan_id",
            "candidate_graph_id",
            "repository_tree_id",
            "caller",
            "policy_id",
            "policy_revision",
            "active_lease_id",
            "idempotency_key",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "candidate_action", _freeze(self.candidate_action))
        object.__setattr__(
            self,
            "candidate_effects",
            tuple(_freeze(item) for item in self.candidate_effects),
        )
        roots = {
            _text(key, "attempt semantic root kind"): _text(
                value, "attempt semantic root"
            )
            for key, value in self.semantic_roots.items()
        }
        object.__setattr__(
            self,
            "semantic_roots",
            MappingProxyType(dict(sorted(roots.items()))),
        )
        if not isinstance(self.mandatory_closure, MandatoryClosure):
            raise ExecutionPermitError(
                "attempt mandatory_closure must be a MandatoryClosure"
            )
        if not isinstance(self.context_witness, ContextCompletenessWitness):
            raise ExecutionPermitError(
                "attempt context_witness must be a ContextCompletenessWitness"
            )
        object.__setattr__(
            self, "evidence_receipts", _coerce_evidence(self.evidence_receipts)
        )
        validations = tuple(
            item
            if isinstance(item, ValidationRequirement)
            else ValidationRequirement.from_dict(item)
            for item in self.validation_plan
        )
        object.__setattr__(
            self,
            "validation_plan",
            tuple(sorted(validations, key=lambda item: item.requirement_id)),
        )
        object.__setattr__(
            self,
            "actual_paths",
            _strings(self.actual_paths, "actual_paths"),
        )
        object.__setattr__(
            self,
            "current_fencing_epoch",
            _integer(self.current_fencing_epoch, "current_fencing_epoch"),
        )
        object.__setattr__(self, "now_ms", _integer(self.now_ms, "now_ms"))
        object.__setattr__(
            self,
            "use_sequence",
            _integer(self.use_sequence, "use_sequence", minimum=1),
        )
        if not isinstance(self.completion_requested, bool):
            raise ExecutionPermitError("completion_requested must be boolean")

    @classmethod
    def from_permit(
        cls,
        permit: ExecutionPermit,
        *,
        now_ms: int,
        use_sequence: int = 1,
        **changes: Any,
    ) -> "ExecutionAttempt":
        values: dict[str, Any] = {
            "decision_request": permit.decision_request,
            "candidate_plan_id": permit.candidate_plan_id,
            "candidate_graph_id": permit.candidate_graph_id,
            "candidate_action": permit.candidate_action,
            "candidate_effects": permit.candidate_effects,
            "repository_tree_id": permit.repository_tree_id,
            "semantic_roots": permit.semantic_roots,
            "mandatory_closure": permit.mandatory_closure,
            "context_witness": permit.context_witness,
            "evidence_receipts": permit.evidence_receipts,
            "validation_plan": permit.validation_plan,
            "caller": permit.caller,
            "policy_id": permit.policy_id,
            "policy_revision": permit.policy_revision,
            "active_lease_id": permit.lease_id,
            "current_fencing_epoch": permit.fencing_epoch,
            "idempotency_key": permit.idempotency_key,
            "actual_paths": permit.declared_paths,
            "now_ms": now_ms,
            "use_sequence": use_sequence,
            "completion_requested": False,
        }
        values.update(changes)
        return cls(**values)

    @property
    def operation_fingerprint(self) -> str:
        return content_identity(
            {
                "decision_request": self.decision_request.to_dict(),
                "candidate_plan_id": self.candidate_plan_id,
                "candidate_graph_id": self.candidate_graph_id,
                "candidate_action": _plain(self.candidate_action),
                "candidate_effects": _plain(self.candidate_effects),
                "repository_tree_id": self.repository_tree_id,
                "semantic_roots": dict(self.semantic_roots),
                "closure_id": self.mandatory_closure.closure_id,
                "context_witness_id": self.context_witness.content_id,
                "evidence_receipt_ids": [
                    item.content_id for item in self.evidence_receipts
                ],
                "validation_plan": [
                    item.to_dict() for item in self.validation_plan
                ],
                "caller": self.caller,
                "policy_id": self.policy_id,
                "policy_revision": self.policy_revision,
                "lease_id": self.active_lease_id,
                "fencing_epoch": self.current_fencing_epoch,
                "idempotency_key": self.idempotency_key,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXECUTION_ATTEMPT_SCHEMA,
            "version": EXECUTION_PERMIT_VERSION,
            "operation_fingerprint": self.operation_fingerprint,
            "decision_request_id": self.decision_request.request_id,
            "caller": self.caller,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "active_lease_id": self.active_lease_id,
            "current_fencing_epoch": self.current_fencing_epoch,
            "idempotency_key": self.idempotency_key,
            "actual_paths": list(self.actual_paths),
            "now_ms": self.now_ms,
            "use_sequence": self.use_sequence,
            "completion_requested": self.completion_requested,
        }


@dataclass(frozen=True)
class PermitUseReceipt:
    permit_id: str
    operation_fingerprint: str
    idempotency_key: str
    use_sequence: int
    verified_at_ms: int
    remaining_uses: int

    @property
    def authorizes_effect(self) -> bool:
        return True

    @property
    def authorizes_completion(self) -> bool:
        return False

    @property
    def receipt_id(self) -> str:
        return content_identity(self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PERMIT_USE_RECEIPT_SCHEMA,
            "version": EXECUTION_PERMIT_VERSION,
            "permit_id": self.permit_id,
            "operation_fingerprint": self.operation_fingerprint,
            "idempotency_key": self.idempotency_key,
            "use_sequence": self.use_sequence,
            "verified_at_ms": self.verified_at_ms,
            "remaining_uses": self.remaining_uses,
            "authorizes_effect": True,
            "authorizes_completion": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}


@dataclass(frozen=True)
class PermitVerificationResult:
    permitted: bool
    code: PermitVerificationCode
    reason: str
    receipt: PermitUseReceipt | None = None

    @property
    def authorizes_completion(self) -> bool:
        return False


class PermitUseLedger:
    """Thread-safe permit and idempotency consumption ledger."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._uses: dict[str, set[int]] = {}
        self._idempotency: dict[tuple[str, str, str], tuple[str, str]] = {}

    def consume(
        self,
        permit: ExecutionPermit,
        attempt: ExecutionAttempt,
    ) -> PermitUseReceipt:
        key = (
            permit.principal_id,
            permit.objective_id,
            permit.idempotency_key,
        )
        with self._lock:
            prior = self._idempotency.get(key)
            binding = (permit.permit_id, permit.operation_fingerprint)
            if prior is not None and prior != binding:
                raise PermitReplayError(
                    PermitVerificationCode.REPLAYED,
                    "idempotency key was already bound to a different operation",
                )
            uses = self._uses.setdefault(permit.permit_id, set())
            if (
                attempt.use_sequence > permit.allowed_use_count
                or attempt.use_sequence in uses
                or len(uses) >= permit.allowed_use_count
            ):
                raise PermitReplayError(
                    PermitVerificationCode.REPLAYED,
                    "permit use was replayed or exceeds its allowed use count",
                )
            self._idempotency[key] = binding
            uses.add(attempt.use_sequence)
            return PermitUseReceipt(
                permit_id=permit.permit_id,
                operation_fingerprint=permit.operation_fingerprint,
                idempotency_key=permit.idempotency_key,
                use_sequence=attempt.use_sequence,
                verified_at_ms=attempt.now_ms,
                remaining_uses=permit.allowed_use_count - len(uses),
            )

    def used_count(self, permit: ExecutionPermit | str) -> int:
        permit_id = (
            permit.permit_id if isinstance(permit, ExecutionPermit) else permit
        )
        with self._lock:
            return len(self._uses.get(permit_id, ()))


class ExecutionPermitVerifier:
    """Validate and atomically consume exact permit uses before an effect.

    Callers must supply issued permit IDs or an issuer-backed trust resolver.
    A verifier with neither rejects every permit.
    """

    def __init__(
        self,
        *,
        ledger: PermitUseLedger | None = None,
        trusted_permit_ids: Sequence[str] | None = None,
        trust_resolver: Callable[[ExecutionPermit], bool] | None = None,
    ) -> None:
        self.ledger = ledger or PermitUseLedger()
        self._trusted = (
            None
            if trusted_permit_ids is None
            else frozenset(_strings(tuple(trusted_permit_ids), "trusted permit IDs"))
        )
        self._trust_resolver = trust_resolver

    def _reject(
        self, code: PermitVerificationCode, reason: str
    ) -> None:
        raise PermitVerificationError(code, reason)

    def verify(
        self,
        permit: ExecutionPermit,
        attempt: ExecutionAttempt,
    ) -> PermitUseReceipt:
        if not isinstance(permit, ExecutionPermit):
            self._reject(
                PermitVerificationCode.INVALID_PERMIT,
                "permit must be an ExecutionPermit",
            )
        if not isinstance(attempt, ExecutionAttempt):
            self._reject(
                PermitVerificationCode.INVALID_PERMIT,
                "attempt must be an ExecutionAttempt",
            )
        if self._trusted is not None and permit.permit_id not in self._trusted:
            self._reject(
                PermitVerificationCode.UNTRUSTED,
                "permit was not issued by the active permit authority",
            )
        if self._trusted is None and self._trust_resolver is None:
            self._reject(
                PermitVerificationCode.UNTRUSTED,
                "permit verification requires an explicit trust authority",
            )
        if (
            self._trust_resolver is not None
            and not self._trust_resolver(permit)
        ):
            self._reject(
                PermitVerificationCode.UNTRUSTED,
                "permit was rejected by the active permit authority",
            )
        if attempt.now_ms < permit.issued_at_ms:
            self._reject(
                PermitVerificationCode.NOT_YET_VALID,
                "permit is not yet valid",
            )
        if attempt.now_ms >= permit.expires_at_ms:
            self._reject(
                PermitVerificationCode.EXPIRED,
                "permit has expired",
            )
        if attempt.completion_requested:
            self._reject(
                PermitVerificationCode.COMPLETION_AUTHORITY_FORBIDDEN,
                "execution permit never authorizes task completion",
            )
        if attempt.caller != permit.caller:
            self._reject(
                PermitVerificationCode.CALLER_MISMATCH,
                "permit caller differs from the effect caller",
            )
        decision = attempt.decision_request
        if decision.objective_id != permit.objective_id:
            self._reject(
                PermitVerificationCode.TASK_MISMATCH,
                "permit cannot be used for another task",
            )
        if decision.principal != permit.principal_id:
            self._reject(
                PermitVerificationCode.PRINCIPAL_MISMATCH,
                "permit cannot be used by another principal",
            )
        if (
            attempt.policy_id != permit.policy_id
            or attempt.policy_revision != permit.policy_revision
        ):
            self._reject(
                PermitVerificationCode.POLICY_MISMATCH,
                "permit policy is no longer current",
            )
        if attempt.active_lease_id != permit.lease_id:
            self._reject(
                PermitVerificationCode.LEASE_LOST,
                "permit lease is inactive or changed",
            )
        if attempt.current_fencing_epoch != permit.fencing_epoch:
            self._reject(
                PermitVerificationCode.FENCE_LOST,
                "permit fencing epoch is no longer current",
            )
        if attempt.idempotency_key != permit.idempotency_key:
            self._reject(
                PermitVerificationCode.CHANGED_OPERATION,
                "effect idempotency key differs from the permit",
            )
        if attempt.repository_tree_id != permit.repository_tree_id:
            self._reject(
                PermitVerificationCode.STALE_ROOT,
                "repository tree changed after permit issuance",
            )
        if attempt.semantic_roots != permit.semantic_roots:
            self._reject(
                PermitVerificationCode.STALE_ROOT,
                "one or more semantic roots changed after permit issuance",
            )
        if (
            attempt.mandatory_closure.closure_id
            != permit.mandatory_closure.closure_id
            or attempt.mandatory_closure != permit.mandatory_closure
            or attempt.context_witness.content_id
            != permit.context_witness.content_id
            or attempt.context_witness != permit.context_witness
        ):
            self._reject(
                PermitVerificationCode.STALE_ROOT,
                "dependency closure or context witness changed",
            )
        if attempt.candidate_plan_id != permit.candidate_plan_id:
            self._reject(
                PermitVerificationCode.CHANGED_OPERATION,
                "candidate plan changed after permit issuance",
            )
        if (
            attempt.candidate_graph_id != permit.candidate_graph_id
            or attempt.candidate_action != permit.candidate_action
        ):
            self._reject(
                PermitVerificationCode.CHANGED_OPERATION,
                "candidate action, tool, or arguments changed",
            )
        if (
            attempt.decision_request.action.targets
            != permit.decision_request.action.targets
        ):
            self._reject(
                PermitVerificationCode.CHANGED_TARGET,
                "operation targets changed after permit issuance",
            )
        if (
            attempt.candidate_effects != permit.candidate_effects
            or attempt.decision_request.expected_effects
            != permit.decision_request.expected_effects
        ):
            self._reject(
                PermitVerificationCode.CHANGED_EFFECT,
                "expected effects changed after permit issuance",
            )
        if attempt.decision_request != permit.decision_request:
            self._reject(
                PermitVerificationCode.CHANGED_OPERATION,
                "complete DecisionRequest changed after permit issuance",
            )
        if attempt.actual_paths != permit.declared_paths:
            code = (
                PermitVerificationCode.PATH_BROADENING
                if not set(attempt.actual_paths).issubset(permit.declared_paths)
                else PermitVerificationCode.PARTIAL_AUTHORITY
            )
            self._reject(
                code,
                "effect paths do not exactly match the declared operation",
            )
        if attempt.validation_plan != permit.validation_plan:
            self._reject(
                PermitVerificationCode.STALE_RECEIPT,
                "validation plan changed after permit issuance",
            )
        current_roots = dict(permit.semantic_roots)
        for item in attempt.evidence_receipts:
            if item.state is MandatoryEvidenceState.UNKNOWN:
                self._reject(
                    PermitVerificationCode.MANDATORY_STATE_UNKNOWN,
                    f"mandatory {item.domain} state is unknown",
                )
            if item.state is MandatoryEvidenceState.CONTRADICTORY:
                self._reject(
                    PermitVerificationCode.MANDATORY_STATE_CONTRADICTORY,
                    f"mandatory {item.domain} state is contradictory",
                )
            if (
                item.state is MandatoryEvidenceState.STALE
                or item.authority not in _ACCEPTED_EVIDENCE_AUTHORITIES
                or (
                    item.expires_at_ms is not None
                    and attempt.now_ms >= item.expires_at_ms
                )
                or any(
                    current_roots.get(kind) != value
                    for kind, value in item.semantic_roots.items()
                )
            ):
                self._reject(
                    PermitVerificationCode.STALE_RECEIPT,
                    f"mandatory {item.domain} receipt is stale or unauthorized",
                )
        if attempt.evidence_receipts != permit.evidence_receipts:
            self._reject(
                PermitVerificationCode.STALE_RECEIPT,
                "domain receipt set changed after permit issuance",
            )
        if attempt.operation_fingerprint != permit.operation_fingerprint:
            self._reject(
                PermitVerificationCode.CHANGED_OPERATION,
                "effect operation fingerprint differs from the permit",
            )
        return self.ledger.consume(permit, attempt)

    authorize = verify
    check = verify

    def evaluate(
        self,
        permit: ExecutionPermit,
        attempt: ExecutionAttempt,
    ) -> PermitVerificationResult:
        try:
            receipt = self.verify(permit, attempt)
        except PermitVerificationError as exc:
            return PermitVerificationResult(False, exc.code, str(exc))
        return PermitVerificationResult(
            True,
            PermitVerificationCode.VALID,
            "exact permit verified and use consumed",
            receipt,
        )


class ExecutionPermitIssuer:
    """Stateful issuer with stable idempotency and trusted-permit provenance."""

    def __init__(
        self,
        *,
        issuer_id: str = "agent-supervisor:execution-permit-issuer",
        max_ttl_ms: int = DEFAULT_MAX_PERMIT_TTL_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.issuer_id = _text(issuer_id, "issuer_id")
        self.max_ttl_ms = _integer(
            max_ttl_ms,
            "max_ttl_ms",
            minimum=1,
            maximum=DEFAULT_MAX_PERMIT_TTL_MS,
        )
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = threading.Lock()
        self._permits: dict[str, ExecutionPermit] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}

    @property
    def permit_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._permits))

    def issued(self, permit: ExecutionPermit) -> bool:
        with self._lock:
            return self._permits.get(permit.permit_id) == permit

    def issue(
        self,
        admission_request: PlanAdmissionRequest,
        admission_receipt: PlanAdmissionReceipt,
        context_witness: ContextCompletenessWitness,
        *,
        caller: str,
        policy_id: str,
        policy_revision: str,
        expires_at_ms: int,
        allowed_use_count: int = 1,
        evidence_receipts: Sequence[ExecutionEvidence] = (),
        issued_at_ms: int | None = None,
    ) -> ExecutionPermit:
        issued = self._clock_ms() if issued_at_ms is None else issued_at_ms
        permit = issue_execution_permit(
            admission_request,
            admission_receipt,
            context_witness,
            caller=caller,
            policy_id=policy_id,
            policy_revision=policy_revision,
            issued_at_ms=issued,
            expires_at_ms=expires_at_ms,
            allowed_use_count=allowed_use_count,
            evidence_receipts=evidence_receipts,
            issuer_id=self.issuer_id,
            max_ttl_ms=self.max_ttl_ms,
        )
        key = (
            permit.principal_id,
            permit.objective_id,
            permit.idempotency_key,
        )
        with self._lock:
            previous_id = self._idempotency.get(key)
            if previous_id is not None:
                previous = self._permits[previous_id]
                if previous.operation_fingerprint != permit.operation_fingerprint:
                    raise PermitIssuanceError(
                        "idempotency key is already bound to different arguments, "
                        "targets, effects, roots, receipts, or authority"
                    )
                return previous
            self._permits[permit.permit_id] = permit
            self._idempotency[key] = permit.permit_id
        return permit

    def verifier(
        self, *, ledger: PermitUseLedger | None = None
    ) -> ExecutionPermitVerifier:
        return ExecutionPermitVerifier(
            ledger=ledger,
            trust_resolver=self.issued,
        )


def verify_execution_permit(
    permit: ExecutionPermit,
    attempt: ExecutionAttempt,
    *,
    ledger: PermitUseLedger | None = None,
    trusted_permit_ids: Sequence[str] | None = None,
) -> PermitUseReceipt:
    """Verify and consume one immediate permit use.

    ``trusted_permit_ids`` is required for a successful stateless verification.
    Reuse ``ledger`` across calls when replay protection spans multiple effects;
    an :class:`ExecutionPermitIssuer`-created verifier supplies provenance and
    is the preferred stateful boundary.
    """

    return ExecutionPermitVerifier(
        ledger=ledger,
        trusted_permit_ids=trusted_permit_ids,
    ).verify(permit, attempt)


# Readable compatibility names for callers that use authorization terminology.
ExactExecutionPermit = ExecutionPermit
ExecutionPermitRequest = ExecutionAttempt
ExecutionPermitUse = PermitUseReceipt
ExecutionPermitLedger = PermitUseLedger
ExecutionPermitAuthorizer = ExecutionPermitVerifier
issue_permit = issue_execution_permit
verify_permit = verify_execution_permit


__all__ = [
    "DEFAULT_MAX_PERMIT_TTL_MS",
    "EXECUTION_ATTEMPT_SCHEMA",
    "EXECUTION_EVIDENCE_SCHEMA",
    "EXECUTION_PERMIT_SCHEMA",
    "EXECUTION_PERMIT_VERSION",
    "MAX_ALLOWED_USES",
    "ExactExecutionPermit",
    "ExecutionAttempt",
    "ExecutionEvidence",
    "ExecutionPermit",
    "ExecutionPermitAuthorizer",
    "ExecutionPermitError",
    "ExecutionPermitIssuer",
    "ExecutionPermitLedger",
    "ExecutionPermitRequest",
    "ExecutionPermitUse",
    "ExecutionPermitVerifier",
    "MandatoryEvidenceState",
    "PermitIssuanceError",
    "PermitReplayError",
    "PermitUseLedger",
    "PermitUseReceipt",
    "PermitVerificationCode",
    "PermitVerificationError",
    "PermitVerificationResult",
    "issue_execution_permit",
    "issue_permit",
    "verify_execution_permit",
    "verify_permit",
]
