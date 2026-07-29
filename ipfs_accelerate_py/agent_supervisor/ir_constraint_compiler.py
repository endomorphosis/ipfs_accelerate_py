"""Canonical hard-constrained admission across supervisor IR domains.

The three IR adapters deliberately remain independent.  This module is the
single join boundary: it binds their typed results to one exact candidate
action/effect graph, current semantic roots, program dependencies, authority,
proof receipts, and validation results.  It emits no score.  A caller must
prune a rejected receipt before any quality or cost evaluation.

Intent conformance and legal permission are constraints, not grants.  A
generated formula is an obligation statement, not a proof.  Even an admitted
plan does not authorize execution; the exact short-lived execution permit is a
separate downstream contract.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .cve_security_gate import (
    CVESecurityGateResult,
    correlate_security_requests,
)
from .decision_contracts import DecisionRequest
from .formal_verification_contracts import (
    AssuranceLevel,
    EvidenceFreshness,
    ProofReceipt,
    ProofVerdict,
    content_identity,
)
from .intent_constraint_adapter import (
    IntentConformanceRequest,
    evaluate_intent_conformance,
)
from .legal_constraint_adapter import LegalCompilationResult
from .security_constraint_adapter import (
    SecurityAuthorizationRequest,
    SecurityDecisionOutcome,
    SecurityPolicyReceipt,
    evaluate_security_authorization,
    revalidate_security_authorization,
)
from .semantic_dependency_graph import MandatoryClosure


IR_CONSTRAINT_COMPILER_VERSION: Final[int] = 1
IR_CONFORMANCE_REQUIREMENT_ID: Final[str] = (
    "287667496524558776121661391058779883318"
)
PLAN_ADMISSION_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-request@1"
)
PLAN_ADMISSION_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-receipt@1"
)
PLAN_ADMISSION_REJECTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-rejection@1"
)
PLAN_ADMISSION_COUNTEREXAMPLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-counterexample@1"
)
CVE_SECURITY_ENFORCEMENT_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cve-security-enforcement-evidence@1"
)


class IRConstraintCompilerError(ValueError):
    """A plan-admission record is malformed or detached from its evidence."""


class PlanAdmissionVerdict(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"


class AdmissionDomain(str, Enum):
    GRAPH = "candidate_graph"
    INTENT = "intent"
    LEGAL = "legal"
    SECURITY = "security"
    PROGRAM = "program"
    ASSUMPTION = "assumption"
    PROOF = "proof"
    VALIDATION = "validation"
    ROOT = "root"
    AUTHORITY = "authority"


class AdmissionRejectionCode(str, Enum):
    INVALID_GRAPH = "invalid_candidate_graph"
    INCOMPLETE_GRAPH = "incomplete_candidate_graph"
    UNDECLARED_EFFECT = "undeclared_effect"
    DOMAIN_BINDING_MISSING = "domain_binding_missing"
    DOMAIN_BINDING_MISMATCH = "domain_binding_mismatch"
    INTENT_VIOLATION = "intent_violation"
    LEGAL_INCOMPLETE = "legal_incomplete"
    LEGAL_PROHIBITION = "legal_prohibition"
    LEGAL_OBLIGATION = "unresolved_legal_obligation"
    SECURITY_DENY = "security_deny"
    SECURITY_UNKNOWN = "security_unknown"
    SECURITY_CONFLICT = "security_conflict"
    CVE_SECURITY_GATE_MISSING = "cve_security_gate_missing"
    CVE_SECURITY_GATE_REJECTED = "cve_security_gate_rejected"
    CVE_SECURITY_GATE_STALE = "cve_security_gate_stale"
    CVE_SECURITY_GATE_DETACHED = "cve_security_gate_detached"
    DEPENDENCY_UNSATISFIED = "program_dependency_unsatisfied"
    ASSUMPTION_UNRESOLVED = "assumption_unresolved"
    MISSING_PROOF = "missing_proof"
    INVALID_PROOF = "invalid_proof"
    STALE_ROOT = "stale_root"
    AUTHORITY_MISMATCH = "authority_mismatch"
    VALIDATION_MISSING = "validation_missing"
    VALIDATION_FAILED = "validation_failed"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CVESecurityEnforcementStage(str, Enum):
    """Ordered CVE gate boundaries from proposal through merged-tree use."""

    PLAN_ADMISSION = "plan_admission"
    PRE_EXECUTION = "pre_execution"
    POST_GENERATION = "post_generation"
    MERGE_ADMISSION = "merge_admission"
    MERGED_TREE_REVALIDATION = "merged_tree_revalidation"


_CVE_SECURITY_STAGE_ORDER: Final[tuple[CVESecurityEnforcementStage, ...]] = (
    CVESecurityEnforcementStage.PLAN_ADMISSION,
    CVESecurityEnforcementStage.PRE_EXECUTION,
    CVESecurityEnforcementStage.POST_GENERATION,
    CVESecurityEnforcementStage.MERGE_ADMISSION,
    CVESecurityEnforcementStage.MERGED_TREE_REVALIDATION,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise IRConstraintCompilerError(
            "floating point values are not canonical admission data"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise IRConstraintCompilerError("admission mapping keys must be strings")
        return {key: _plain(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _plain(converter())
    raise IRConstraintCompilerError(
        f"unsupported admission value: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise IRConstraintCompilerError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise IRConstraintCompilerError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise IRConstraintCompilerError(f"{name} is required")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise IRConstraintCompilerError(f"{name} must be a sequence")
    result = tuple(sorted({_text(item, name) for item in value}))
    return result


def _identity(namespace: str, value: Any) -> str:
    return content_identity({"namespace": namespace, "value": _plain(value)})


def _record_id(value: Any) -> str:
    for name in ("content_id", "result_id", "request_id", "receipt_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return _identity("typed-admission-record", _plain(value))


def _root_token(artifact: Any) -> str:
    return ":".join(
        (
            str(getattr(artifact, "artifact_id", "") or ""),
            str(getattr(artifact, "cid_v1", "") or ""),
            str(getattr(artifact, "supervisor_digest", "") or ""),
        )
    )


@dataclass(frozen=True)
class CVESecurityEnforcementEvidence:
    """One independently replayable CVE gate result at a runtime boundary.

    The CVE gate result deliberately does not grant authority.  This wrapper
    binds it to the repository tree and to the preceding stage so admission,
    execution, generation, and merge callers cannot reuse a passing result
    after the tree changes or skip an earlier boundary.
    """

    stage: CVESecurityEnforcementStage
    repository_tree_id: str
    gate_result: CVESecurityGateResult
    parent_evidence_id: str = ""
    authority: str = "authoritative"
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stage", CVESecurityEnforcementStage(self.stage)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self,
            "parent_evidence_id",
            _text(
                self.parent_evidence_id,
                "parent_evidence_id",
                required=False,
            ),
        )
        object.__setattr__(self, "authority", _text(self.authority, "authority"))
        if not isinstance(self.gate_result, CVESecurityGateResult):
            raise IRConstraintCompilerError(
                "gate_result must be a CVESecurityGateResult"
            )
        if self.expires_at_ms is not None and (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms < 0
        ):
            raise IRConstraintCompilerError(
                "expires_at_ms must be a non-negative integer"
            )

    @property
    def current_and_authoritative(self) -> bool:
        return (
            self.gate_result.passed
            and self.authority in {"authoritative", "verified", "verified_input"}
        )

    @property
    def evidence_id(self) -> str:
        return _identity("cve-security-enforcement-evidence", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": CVE_SECURITY_ENFORCEMENT_EVIDENCE_SCHEMA,
            "compiler_version": IR_CONSTRAINT_COMPILER_VERSION,
            "stage": self.stage.value,
            "repository_tree_id": self.repository_tree_id,
            "gate_result": self.gate_result.to_dict(),
            "parent_evidence_id": self.parent_evidence_id,
            "authority": self.authority,
            "expires_at_ms": self.expires_at_ms,
            "grants_execution_authority": False,
            "authorizes_completion": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "evidence_id": self.evidence_id}


@dataclass(frozen=True)
class RootBinding:
    kind: str
    expected: str
    observed: str
    authority: str = "authoritative"

    def __post_init__(self) -> None:
        for name in ("kind", "expected", "observed", "authority"):
            object.__setattr__(self, name, _text(getattr(self, name), name))

    @property
    def current(self) -> bool:
        return self.expected == self.observed

    @property
    def authority_accepted(self) -> bool:
        return self.authority in {"authoritative", "verified", "verified_input"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "expected": self.expected,
            "observed": self.observed,
            "authority": self.authority,
            "current": self.current,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RootBinding":
        return cls(
            kind=value.get("kind", ""),
            expected=value.get("expected", ""),
            observed=value.get("observed", ""),
            authority=value.get("authority", ""),
        )


@dataclass(frozen=True)
class AdmissionAuthority:
    principal: str
    requested_authority: str
    grant_principal: str
    granted_authorities: tuple[str, ...]
    grant_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("principal", "requested_authority", "grant_principal"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "granted_authorities",
            _strings(self.granted_authorities, "granted_authorities"),
        )
        object.__setattr__(
            self, "grant_source_ids", _strings(self.grant_source_ids, "grant_source_ids")
        )

    @property
    def matched(self) -> bool:
        return (
            self.principal == self.grant_principal
            and self.requested_authority in self.granted_authorities
            and bool(self.grant_source_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "requested_authority": self.requested_authority,
            "grant_principal": self.grant_principal,
            "granted_authorities": list(self.granted_authorities),
            "grant_source_ids": list(self.grant_source_ids),
            "matched": self.matched,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdmissionAuthority":
        return cls(
            principal=value.get("principal", ""),
            requested_authority=value.get("requested_authority", ""),
            grant_principal=value.get("grant_principal", ""),
            granted_authorities=tuple(value.get("granted_authorities") or ()),
            grant_source_ids=tuple(value.get("grant_source_ids") or ()),
        )


@dataclass(frozen=True)
class ActionDomainBinding:
    action_id: str
    legal_result_ids: tuple[str, ...]
    security_request_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        for name in ("legal_result_ids", "security_request_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "legal_result_ids": list(self.legal_result_ids),
            "security_request_ids": list(self.security_request_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionDomainBinding":
        return cls(
            action_id=value.get("action_id", ""),
            legal_result_ids=tuple(value.get("legal_result_ids") or ()),
            security_request_ids=tuple(value.get("security_request_ids") or ()),
        )


@dataclass(frozen=True)
class ProgramDependency:
    dependency_id: str
    action_id: str
    depends_on_action_ids: tuple[str, ...] = ()
    required: bool = True
    satisfied: bool = True
    expected_root: str = ""
    observed_root: str = ""
    evidence_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dependency_id", _text(self.dependency_id, "dependency_id")
        )
        object.__setattr__(self, "action_id", _text(self.action_id, "action_id"))
        for name in (
            "depends_on_action_ids",
            "evidence_ids",
            "reason_codes",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        for name in ("expected_root", "observed_root"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if not isinstance(self.required, bool) or not isinstance(self.satisfied, bool):
            raise IRConstraintCompilerError(
                "dependency required/satisfied values must be booleans"
            )

    @property
    def current(self) -> bool:
        return (
            not self.expected_root
            or (
                bool(self.observed_root)
                and self.expected_root == self.observed_root
            )
        )

    @property
    def passed(self) -> bool:
        return not self.required or (self.satisfied and self.current and bool(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_id": self.dependency_id,
            "action_id": self.action_id,
            "depends_on_action_ids": list(self.depends_on_action_ids),
            "required": self.required,
            "satisfied": self.satisfied,
            "expected_root": self.expected_root,
            "observed_root": self.observed_root,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProgramDependency":
        return cls(
            dependency_id=value.get("dependency_id", ""),
            action_id=value.get("action_id", ""),
            depends_on_action_ids=tuple(value.get("depends_on_action_ids") or ()),
            required=value.get("required", True),
            satisfied=value.get("satisfied", False),
            expected_root=value.get("expected_root", ""),
            observed_root=value.get("observed_root", ""),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class AdmissionAssumption:
    assumption_id: str
    action_ids: tuple[str, ...] = ()
    required: bool = True
    satisfied: bool = False
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assumption_id", _text(self.assumption_id, "assumption_id")
        )
        object.__setattr__(self, "action_ids", _strings(self.action_ids, "action_ids"))
        object.__setattr__(
            self, "evidence_ids", _strings(self.evidence_ids, "evidence_ids")
        )
        if not isinstance(self.required, bool) or not isinstance(self.satisfied, bool):
            raise IRConstraintCompilerError(
                "assumption required/satisfied values must be booleans"
            )

    @property
    def passed(self) -> bool:
        return not self.required or (self.satisfied and bool(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "action_ids": list(self.action_ids),
            "required": self.required,
            "satisfied": self.satisfied,
            "evidence_ids": list(self.evidence_ids),
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdmissionAssumption":
        return cls(
            assumption_id=value.get("assumption_id", ""),
            action_ids=tuple(value.get("action_ids") or ()),
            required=value.get("required", True),
            satisfied=value.get("satisfied", False),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
        )


@dataclass(frozen=True)
class ValidationRequirement:
    requirement_id: str
    action_ids: tuple[str, ...] = ()
    command: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        object.__setattr__(self, "action_ids", _strings(self.action_ids, "action_ids"))
        object.__setattr__(
            self, "command", _text(self.command, "command", required=False)
        )
        if not isinstance(self.required, bool):
            raise IRConstraintCompilerError("validation required must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "action_ids": list(self.action_ids),
            "command": self.command,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationRequirement":
        return cls(
            requirement_id=value.get("requirement_id", ""),
            action_ids=tuple(value.get("action_ids") or ()),
            command=value.get("command", ""),
            required=value.get("required", True),
        )


@dataclass(frozen=True)
class ValidationResult:
    requirement_id: str
    status: ValidationStatus
    repository_tree_id: str
    evidence_id: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_id", _text(self.requirement_id, "requirement_id")
        )
        object.__setattr__(self, "status", ValidationStatus(self.status))
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id", required=False)
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status.value,
            "repository_tree_id": self.repository_tree_id,
            "evidence_id": self.evidence_id,
            "reason_codes": list(self.reason_codes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ValidationResult":
        return cls(
            requirement_id=value.get("requirement_id", ""),
            status=value.get("status", ValidationStatus.UNKNOWN),
            repository_tree_id=value.get("repository_tree_id", ""),
            evidence_id=value.get("evidence_id", ""),
            reason_codes=tuple(value.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class AdmissionRejection:
    code: AdmissionRejectionCode
    domain: AdmissionDomain
    message: str
    action_id: str = ""
    effect_id: str = ""
    dependency_id: str = ""
    obligation_id: str = ""
    source_ids: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", AdmissionRejectionCode(self.code))
        object.__setattr__(self, "domain", AdmissionDomain(self.domain))
        for name in (
            "message",
            "action_id",
            "effect_id",
            "dependency_id",
            "obligation_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=name == "message"),
            )
        object.__setattr__(self, "source_ids", _strings(self.source_ids, "source_ids"))
        object.__setattr__(self, "details", _freeze(self.details))

    @property
    def rejection_id(self) -> str:
        return _identity("plan-admission-rejection", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_REJECTION_SCHEMA,
            "code": self.code.value,
            "domain": self.domain.value,
            "message": self.message,
            "action_id": self.action_id,
            "effect_id": self.effect_id,
            "dependency_id": self.dependency_id,
            "obligation_id": self.obligation_id,
            "source_ids": list(self.source_ids),
            "details": _plain(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdmissionRejection":
        return cls(
            code=value.get("code", AdmissionRejectionCode.INVALID_GRAPH),
            domain=value.get("domain", AdmissionDomain.GRAPH),
            message=value.get("message", ""),
            action_id=value.get("action_id", ""),
            effect_id=value.get("effect_id", ""),
            dependency_id=value.get("dependency_id", ""),
            obligation_id=value.get("obligation_id", ""),
            source_ids=tuple(value.get("source_ids") or ()),
            details=value.get("details") or {},
        )


@dataclass(frozen=True)
class AdmissionCounterexample:
    rejection_id: str
    candidate_plan_id: str
    failing_action_ids: tuple[str, ...]
    affected_action_ids: tuple[str, ...]
    fixed_action_ids: tuple[str, ...]
    witness: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("rejection_id", "candidate_plan_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "failing_action_ids",
            "affected_action_ids",
            "fixed_action_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        object.__setattr__(self, "witness", _freeze(self.witness))

    @property
    def counterexample_id(self) -> str:
        return _identity("plan-admission-counterexample", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_COUNTEREXAMPLE_SCHEMA,
            "rejection_id": self.rejection_id,
            "candidate_plan_id": self.candidate_plan_id,
            "failing_action_ids": list(self.failing_action_ids),
            "affected_action_ids": list(self.affected_action_ids),
            "fixed_action_ids": list(self.fixed_action_ids),
            "witness": _plain(self.witness),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdmissionCounterexample":
        return cls(
            rejection_id=value.get("rejection_id", ""),
            candidate_plan_id=value.get("candidate_plan_id", ""),
            failing_action_ids=tuple(value.get("failing_action_ids") or ()),
            affected_action_ids=tuple(value.get("affected_action_ids") or ()),
            fixed_action_ids=tuple(value.get("fixed_action_ids") or ()),
            witness=value.get("witness") or {},
        )


def _coerce_records(values: Any, kind: type, name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise IRConstraintCompilerError(f"{name} must be a sequence")
    result = []
    decoder = getattr(kind, "from_dict", None)
    for item in values:
        if isinstance(item, kind):
            result.append(item)
        elif isinstance(item, Mapping) and callable(decoder):
            result.append(decoder(item))
        else:
            raise IRConstraintCompilerError(
                f"{name} must contain {kind.__name__} records"
            )
    return tuple(result)


def _candidate_graph(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw_actions = candidate.get("actions", candidate.get("tasks", ()))
    raw_effects = candidate.get("effects", ())
    if isinstance(raw_actions, Mapping):
        raw_actions = tuple(raw_actions.values())
    if isinstance(raw_actions, str) or not isinstance(raw_actions, Sequence):
        raise IRConstraintCompilerError("candidate actions must be a sequence")
    if isinstance(raw_effects, Mapping):
        raw_effects = tuple(raw_effects.values())
    if isinstance(raw_effects, str) or not isinstance(raw_effects, Sequence):
        raise IRConstraintCompilerError("candidate effects must be a sequence")

    actions: dict[str, dict[str, Any]] = {}
    effects: dict[str, dict[str, Any]] = {}
    for raw in raw_actions:
        if not isinstance(raw, Mapping):
            raise IRConstraintCompilerError("candidate actions must be objects")
        item = _plain(raw)
        action_id = item.get("action_id", item.get("task_id", item.get("id", "")))
        action_id = _text(action_id, "candidate action_id")
        if action_id in actions:
            raise IRConstraintCompilerError("candidate action IDs must be unique")
        item["action_id"] = action_id
        item.pop("task_id", None)
        embedded = item.get("effects", ())
        if isinstance(embedded, Mapping):
            embedded = (embedded,)
        if embedded and (
            isinstance(embedded, str) or not isinstance(embedded, Sequence)
        ):
            raise IRConstraintCompilerError("embedded effects must be a sequence")
        for index, raw_effect in enumerate(embedded):
            if not isinstance(raw_effect, Mapping):
                raise IRConstraintCompilerError("candidate effects must be objects")
            effect = _plain(raw_effect)
            effect_id = effect.get("effect_id") or _identity(
                "candidate-effect",
                {"action_id": action_id, "index": index, "effect": effect},
            )
            effect_id = _text(effect_id, "effect_id")
            effect["effect_id"] = effect_id
            effect["action_id"] = action_id
            effect.pop("task_id", None)
            if effect_id in effects and effects[effect_id] != effect:
                raise IRConstraintCompilerError("candidate effect IDs must be unique")
            effects[effect_id] = effect
        item.pop("effects", None)
        actions[action_id] = item
    for raw in raw_effects:
        if not isinstance(raw, Mapping):
            raise IRConstraintCompilerError("candidate effects must be objects")
        effect = _plain(raw)
        effect_id = _text(effect.get("effect_id", ""), "effect_id")
        action_id = _text(
            effect.get("action_id", effect.get("task_id", "")), "effect action_id"
        )
        if action_id not in actions:
            raise IRConstraintCompilerError(
                "candidate effect references an unknown action"
            )
        effect["effect_id"] = effect_id
        effect["action_id"] = action_id
        effect.pop("task_id", None)
        if effect_id in effects and effects[effect_id] != effect:
            raise IRConstraintCompilerError("candidate effect IDs must be unique")
        effects[effect_id] = effect

    for action_id, action in actions.items():
        dependencies = action.get("depends_on", ())
        if isinstance(dependencies, str):
            raise IRConstraintCompilerError("action depends_on must be a sequence")
        dependencies = _strings(tuple(dependencies or ()), "depends_on")
        if action_id in dependencies or not set(dependencies).issubset(actions):
            raise IRConstraintCompilerError(
                "candidate dependency references self or an unknown action"
            )
        action["depends_on"] = list(dependencies)
        action["effect_ids"] = sorted(
            effect_id
            for effect_id, effect in effects.items()
            if effect["action_id"] == action_id
        )

    return {
        "actions": [actions[key] for key in sorted(actions)],
        "effects": [effects[key] for key in sorted(effects)],
        "dependencies": [
            {"action_id": action_id, "depends_on_action_id": dependency}
            for action_id, action in sorted(actions.items())
            for dependency in action["depends_on"]
        ],
    }


def _graph_id(candidate: Mapping[str, Any]) -> str:
    return _identity("candidate-action-effect-graph", _candidate_graph(candidate))


def _candidate_plan_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("plan_id", candidate.get("candidate_plan_id", ""))
    return _text(value, "candidate plan_id") if value else _graph_id(candidate)


@dataclass(frozen=True)
class PlanAdmissionRequest:
    candidate_plan: Mapping[str, Any]
    repository_tree_id: str
    intent_request: IntentConformanceRequest
    legal_results: tuple[LegalCompilationResult, ...]
    security_policy: SecurityPolicyReceipt
    security_requests: tuple[SecurityAuthorizationRequest, ...]
    action_bindings: tuple[ActionDomainBinding, ...]
    authority: AdmissionAuthority
    root_bindings: tuple[RootBinding, ...]
    program_dependencies: tuple[ProgramDependency, ...] = ()
    assumptions: tuple[AdmissionAssumption, ...] = ()
    proof_results: tuple[ProofReceipt, ...] = ()
    validation_requirements: tuple[ValidationRequirement, ...] = ()
    validation_results: tuple[ValidationResult, ...] = ()
    cve_security_evidence: tuple[CVESecurityEnforcementEvidence, ...] = ()
    required_cve_security_stage: CVESecurityEnforcementStage | None = None
    generated_formula_ids: tuple[str, ...] = ()
    decision_request: DecisionRequest | None = None
    mandatory_closure: MandatoryClosure | None = None
    graph_complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_plan, Mapping):
            raise IRConstraintCompilerError("candidate_plan must be a mapping")
        candidate = _freeze(self.candidate_plan)
        _candidate_graph(candidate)
        object.__setattr__(self, "candidate_plan", candidate)
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        if not isinstance(self.intent_request, IntentConformanceRequest):
            raise IRConstraintCompilerError(
                "intent_request must be IntentConformanceRequest"
            )
        if any(not isinstance(item, LegalCompilationResult) for item in self.legal_results):
            raise IRConstraintCompilerError(
                "legal_results must contain LegalCompilationResult records"
            )
        if not isinstance(self.security_policy, SecurityPolicyReceipt):
            raise IRConstraintCompilerError(
                "security_policy must be SecurityPolicyReceipt"
            )
        if any(
            not isinstance(item, SecurityAuthorizationRequest)
            for item in self.security_requests
        ):
            raise IRConstraintCompilerError(
                "security_requests must contain SecurityAuthorizationRequest records"
            )
        if not isinstance(self.authority, AdmissionAuthority):
            if not isinstance(self.authority, Mapping):
                raise IRConstraintCompilerError("authority is malformed")
            object.__setattr__(
                self, "authority", AdmissionAuthority.from_dict(self.authority)
            )
        coercions = (
            ("action_bindings", ActionDomainBinding),
            ("root_bindings", RootBinding),
            ("program_dependencies", ProgramDependency),
            ("assumptions", AdmissionAssumption),
            ("validation_requirements", ValidationRequirement),
            ("validation_results", ValidationResult),
        )
        for name, kind in coercions:
            values = _coerce_records(getattr(self, name), kind, name)
            key_name = next(
                candidate
                for candidate in (
                    "action_id",
                    "kind",
                    "dependency_id",
                    "assumption_id",
                    "requirement_id",
                )
                if hasattr(values[0], candidate)
            ) if values else ""
            if key_name:
                values = tuple(sorted(values, key=lambda item: getattr(item, key_name)))
                keys = [getattr(item, key_name) for item in values]
                if len(keys) != len(set(keys)):
                    raise IRConstraintCompilerError(f"{name} IDs must be unique")
            object.__setattr__(self, name, values)
        if any(not isinstance(item, ProofReceipt) for item in self.proof_results):
            raise IRConstraintCompilerError(
                "proof_results must contain ProofReceipt records"
            )
        object.__setattr__(
            self,
            "proof_results",
            tuple(sorted(self.proof_results, key=lambda item: item.receipt_id)),
        )
        object.__setattr__(
            self,
            "legal_results",
            tuple(sorted(self.legal_results, key=_record_id)),
        )
        object.__setattr__(
            self,
            "security_requests",
            tuple(sorted(self.security_requests, key=_record_id)),
        )
        if any(
            not isinstance(item, CVESecurityEnforcementEvidence)
            for item in self.cve_security_evidence
        ):
            raise IRConstraintCompilerError(
                "cve_security_evidence must contain "
                "CVESecurityEnforcementEvidence records"
            )
        object.__setattr__(
            self,
            "cve_security_evidence",
            tuple(self.cve_security_evidence),
        )
        if self.required_cve_security_stage is not None:
            object.__setattr__(
                self,
                "required_cve_security_stage",
                CVESecurityEnforcementStage(self.required_cve_security_stage),
            )
        object.__setattr__(
            self,
            "generated_formula_ids",
            _strings(self.generated_formula_ids, "generated_formula_ids"),
        )
        if self.decision_request is not None and not isinstance(
            self.decision_request, DecisionRequest
        ):
            raise IRConstraintCompilerError(
                "decision_request must be a DecisionRequest"
            )
        if self.mandatory_closure is not None and not isinstance(
            self.mandatory_closure, MandatoryClosure
        ):
            raise IRConstraintCompilerError(
                "mandatory_closure must be MandatoryClosure"
            )
        if not isinstance(self.graph_complete, bool):
            raise IRConstraintCompilerError("graph_complete must be boolean")

    @property
    def candidate_plan_id(self) -> str:
        return _candidate_plan_id(self.candidate_plan)

    @property
    def candidate_graph_id(self) -> str:
        return _graph_id(self.candidate_plan)

    @property
    def semantic_roots(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.kind: item.expected for item in self.root_bindings}
        )

    @property
    def request_id(self) -> str:
        return _identity("plan-admission-request", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_REQUEST_SCHEMA,
            "compiler_version": IR_CONSTRAINT_COMPILER_VERSION,
            "requirement_id": IR_CONFORMANCE_REQUIREMENT_ID,
            "candidate_plan_id": self.candidate_plan_id,
            "candidate_graph_id": self.candidate_graph_id,
            "candidate_plan": _plain(self.candidate_plan),
            "repository_tree_id": self.repository_tree_id,
            "intent_request": self.intent_request.to_dict(),
            "legal_results": [item.to_dict() for item in self.legal_results],
            "security_policy": self.security_policy.to_dict(),
            "security_requests": [item.to_dict() for item in self.security_requests],
            "action_bindings": [item.to_dict() for item in self.action_bindings],
            "authority": self.authority.to_dict(),
            "root_bindings": [item.to_dict() for item in self.root_bindings],
            "program_dependencies": [
                item.to_dict() for item in self.program_dependencies
            ],
            "assumptions": [item.to_dict() for item in self.assumptions],
            "proof_results": [item.to_dict() for item in self.proof_results],
            "validation_requirements": [
                item.to_dict() for item in self.validation_requirements
            ],
            "validation_results": [
                item.to_dict() for item in self.validation_results
            ],
            "cve_security_evidence": [
                item.to_dict() for item in self.cve_security_evidence
            ],
            "required_cve_security_stage": (
                self.required_cve_security_stage.value
                if self.required_cve_security_stage is not None
                else None
            ),
            "generated_formula_ids": list(self.generated_formula_ids),
            "decision_request": (
                self.decision_request.to_dict()
                if self.decision_request is not None
                else None
            ),
            "mandatory_closure": (
                self.mandatory_closure.to_dict()
                if self.mandatory_closure is not None
                else None
            ),
            "graph_complete": self.graph_complete,
            "permissions_are_grants": False,
            "generated_formulas_are_proofs": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "request_id": self.request_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanAdmissionRequest":
        """Construct from a live typed mapping.

        Adapter compilation receipts intentionally do not all expose decoders;
        serialized request restoration therefore belongs to the producer that
        can re-verify pinned artifacts.  This helper accepts typed nested
        records and decodes the admission-owned records only.
        """

        result = cls(
            candidate_plan=value.get("candidate_plan") or {},
            repository_tree_id=value.get("repository_tree_id", ""),
            intent_request=value.get("intent_request"),
            legal_results=tuple(value.get("legal_results") or ()),
            security_policy=value.get("security_policy"),
            security_requests=tuple(value.get("security_requests") or ()),
            action_bindings=tuple(value.get("action_bindings") or ()),
            authority=value.get("authority") or {},
            root_bindings=tuple(value.get("root_bindings") or ()),
            program_dependencies=tuple(
                value.get("program_dependencies") or ()
            ),
            assumptions=tuple(value.get("assumptions") or ()),
            proof_results=tuple(value.get("proof_results") or ()),
            validation_requirements=tuple(
                value.get("validation_requirements") or ()
            ),
            validation_results=tuple(value.get("validation_results") or ()),
            cve_security_evidence=tuple(
                value.get("cve_security_evidence") or ()
            ),
            required_cve_security_stage=value.get(
                "required_cve_security_stage"
            ),
            generated_formula_ids=tuple(
                value.get("generated_formula_ids") or ()
            ),
            decision_request=value.get("decision_request"),
            mandatory_closure=value.get("mandatory_closure"),
            graph_complete=value.get("graph_complete", True),
        )
        claimed = str(value.get("request_id") or "")
        if claimed and claimed != result.request_id:
            raise IRConstraintCompilerError(
                "plan-admission request identity does not match content"
            )
        return result

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


@dataclass(frozen=True)
class PlanAdmissionReceipt:
    request_id: str
    candidate_plan_id: str
    candidate_graph_id: str
    repository_tree_id: str
    verdict: PlanAdmissionVerdict
    semantic_roots: Mapping[str, str]
    intent_result_id: str
    legal_result_ids: tuple[str, ...]
    legal_permission_ids: tuple[str, ...]
    security_decision_ids: tuple[str, ...]
    security_grant_ids: tuple[str, ...]
    checked_dependency_ids: tuple[str, ...]
    checked_assumption_ids: tuple[str, ...]
    generated_formula_ids: tuple[str, ...]
    proof_result_ids: tuple[str, ...]
    checked_validation_ids: tuple[str, ...]
    cve_security_evidence_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[AdmissionRejection, ...] = ()
    counterexamples: tuple[AdmissionCounterexample, ...] = ()
    local_replan_action_ids: tuple[str, ...] = ()
    closure_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", PlanAdmissionVerdict(self.verdict))
        for name in (
            "request_id",
            "candidate_plan_id",
            "candidate_graph_id",
            "repository_tree_id",
            "intent_result_id",
            "closure_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    name,
                    required=name not in {"intent_result_id", "closure_id"},
                ),
            )
        roots = {
            _text(key, "semantic root kind"): _text(value, "semantic root value")
            for key, value in self.semantic_roots.items()
        }
        object.__setattr__(self, "semantic_roots", MappingProxyType(dict(sorted(roots.items()))))
        for name in (
            "legal_result_ids",
            "legal_permission_ids",
            "security_decision_ids",
            "security_grant_ids",
            "checked_dependency_ids",
            "checked_assumption_ids",
            "generated_formula_ids",
            "proof_result_ids",
            "checked_validation_ids",
            "cve_security_evidence_ids",
            "local_replan_action_ids",
        ):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        rejections = _coerce_records(
            self.rejection_reasons, AdmissionRejection, "rejection_reasons"
        )
        rejections = tuple(
            sorted(
                {item.rejection_id: item for item in rejections}.values(),
                key=lambda item: item.rejection_id,
            )
        )
        object.__setattr__(self, "rejection_reasons", rejections)
        examples = _coerce_records(
            self.counterexamples, AdmissionCounterexample, "counterexamples"
        )
        examples = tuple(
            sorted(
                {item.counterexample_id: item for item in examples}.values(),
                key=lambda item: item.counterexample_id,
            )
        )
        object.__setattr__(self, "counterexamples", examples)
        if self.verdict is PlanAdmissionVerdict.ADMITTED and rejections:
            raise IRConstraintCompilerError(
                "admitted receipt cannot carry rejection reasons"
            )
        if self.verdict is PlanAdmissionVerdict.REJECTED and not rejections:
            raise IRConstraintCompilerError(
                "rejected receipt requires rejection reasons"
            )

    @property
    def admitted(self) -> bool:
        return self.verdict is PlanAdmissionVerdict.ADMITTED

    @property
    def authorizes_execution(self) -> bool:
        return False

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(sorted({item.code.value for item in self.rejection_reasons}))

    @property
    def replan_action_ids(self) -> tuple[str, ...]:
        return self.local_replan_action_ids

    @property
    def receipt_id(self) -> str:
        return _identity("plan-admission-receipt", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_RECEIPT_SCHEMA,
            "compiler_version": IR_CONSTRAINT_COMPILER_VERSION,
            "requirement_id": IR_CONFORMANCE_REQUIREMENT_ID,
            "request_id": self.request_id,
            "candidate_plan_id": self.candidate_plan_id,
            "candidate_graph_id": self.candidate_graph_id,
            "repository_tree_id": self.repository_tree_id,
            "verdict": self.verdict.value,
            "admitted": self.admitted,
            "semantic_roots": dict(self.semantic_roots),
            "intent_result_id": self.intent_result_id,
            "legal_result_ids": list(self.legal_result_ids),
            "legal_permission_ids": list(self.legal_permission_ids),
            "security_decision_ids": list(self.security_decision_ids),
            "security_grant_ids": list(self.security_grant_ids),
            "checked_dependency_ids": list(self.checked_dependency_ids),
            "checked_assumption_ids": list(self.checked_assumption_ids),
            "generated_formula_ids": list(self.generated_formula_ids),
            "proof_result_ids": list(self.proof_result_ids),
            "checked_validation_ids": list(self.checked_validation_ids),
            "cve_security_evidence_ids": list(
                self.cve_security_evidence_ids
            ),
            "rejection_reasons": [
                {**item.to_dict(), "rejection_id": item.rejection_id}
                for item in self.rejection_reasons
            ],
            "reason_codes": list(self.reason_codes),
            "counterexamples": [
                {**item.to_dict(), "counterexample_id": item.counterexample_id}
                for item in self.counterexamples
            ],
            "local_replan_action_ids": list(self.local_replan_action_ids),
            "closure_id": self.closure_id,
            "permissions_are_grants": False,
            "generated_formulas_are_proofs": False,
            "authorizes_execution": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "receipt_id": self.receipt_id}

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanAdmissionReceipt":
        if value.get("schema") != PLAN_ADMISSION_RECEIPT_SCHEMA:
            raise IRConstraintCompilerError("unsupported plan-admission receipt schema")
        result = cls(
            request_id=value.get("request_id", ""),
            candidate_plan_id=value.get("candidate_plan_id", ""),
            candidate_graph_id=value.get("candidate_graph_id", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            verdict=value.get("verdict", PlanAdmissionVerdict.REJECTED),
            semantic_roots=value.get("semantic_roots") or {},
            intent_result_id=value.get("intent_result_id", ""),
            legal_result_ids=tuple(value.get("legal_result_ids") or ()),
            legal_permission_ids=tuple(value.get("legal_permission_ids") or ()),
            security_decision_ids=tuple(value.get("security_decision_ids") or ()),
            security_grant_ids=tuple(value.get("security_grant_ids") or ()),
            checked_dependency_ids=tuple(value.get("checked_dependency_ids") or ()),
            checked_assumption_ids=tuple(value.get("checked_assumption_ids") or ()),
            generated_formula_ids=tuple(value.get("generated_formula_ids") or ()),
            proof_result_ids=tuple(value.get("proof_result_ids") or ()),
            checked_validation_ids=tuple(value.get("checked_validation_ids") or ()),
            cve_security_evidence_ids=tuple(
                value.get("cve_security_evidence_ids") or ()
            ),
            rejection_reasons=tuple(
                AdmissionRejection.from_dict(item)
                for item in value.get("rejection_reasons") or ()
            ),
            counterexamples=tuple(
                AdmissionCounterexample.from_dict(item)
                for item in value.get("counterexamples") or ()
            ),
            local_replan_action_ids=tuple(
                value.get("local_replan_action_ids") or ()
            ),
            closure_id=value.get("closure_id", ""),
        )
        if value.get("receipt_id") != result.receipt_id:
            raise IRConstraintCompilerError(
                "plan-admission receipt identity does not match content"
            )
        if value.get("admitted") is not result.admitted:
            raise IRConstraintCompilerError(
                "plan-admission admitted projection does not match verdict"
            )
        if tuple(value.get("reason_codes") or ()) != result.reason_codes:
            raise IRConstraintCompilerError(
                "plan-admission reason-code projection does not match findings"
            )
        for raw, rejection in zip(
            value.get("rejection_reasons") or (), result.rejection_reasons
        ):
            claimed = str(raw.get("rejection_id") or "")
            if claimed and claimed != rejection.rejection_id:
                raise IRConstraintCompilerError(
                    "plan-admission rejection identity does not match content"
                )
        for raw, counterexample in zip(
            value.get("counterexamples") or (), result.counterexamples
        ):
            claimed = str(raw.get("counterexample_id") or "")
            if claimed and claimed != counterexample.counterexample_id:
                raise IRConstraintCompilerError(
                    "plan-admission counterexample identity does not match content"
                )
        if bool(value.get("authorizes_execution", False)):
            raise IRConstraintCompilerError(
                "plan admission cannot authorize execution"
            )
        if bool(value.get("permissions_are_grants", False)):
            raise IRConstraintCompilerError(
                "legal permissions cannot be promoted to authority grants"
            )
        if bool(value.get("generated_formulas_are_proofs", False)):
            raise IRConstraintCompilerError(
                "generated formulas cannot be promoted to proofs"
            )
        return result


def _effect_projection(effect: Mapping[str, Any]) -> Any:
    return {
        key: value
        for key, value in _plain(effect).items()
        if key not in {"effect_id", "action_id", "task_id", "metadata"}
    }


class IRConstraintCompiler:
    """Compile one exact cross-domain request into a hard admission receipt."""

    def compile(self, request: PlanAdmissionRequest) -> PlanAdmissionReceipt:
        if not isinstance(request, PlanAdmissionRequest):
            raise IRConstraintCompilerError(
                "request must be a PlanAdmissionRequest"
            )
        rejections: list[AdmissionRejection] = []

        def reject(
            code: AdmissionRejectionCode,
            domain: AdmissionDomain,
            message: str,
            *,
            action_id: str = "",
            effect_id: str = "",
            dependency_id: str = "",
            obligation_id: str = "",
            source_ids: Sequence[str] = (),
            details: Mapping[str, Any] | None = None,
        ) -> None:
            rejections.append(
                AdmissionRejection(
                    code=code,
                    domain=domain,
                    message=message,
                    action_id=action_id,
                    effect_id=effect_id,
                    dependency_id=dependency_id,
                    obligation_id=obligation_id,
                    source_ids=tuple(source_ids),
                    details=details or {},
                )
            )

        graph = _candidate_graph(request.candidate_plan)
        actions = {item["action_id"]: item for item in graph["actions"]}
        effects = {item["effect_id"]: item for item in graph["effects"]}
        if not actions or not request.graph_complete:
            reject(
                AdmissionRejectionCode.INCOMPLETE_GRAPH,
                AdmissionDomain.GRAPH,
                "admission requires a complete non-empty action/effect graph",
            )
        if request.mandatory_closure is not None and not request.mandatory_closure.complete:
            reject(
                AdmissionRejectionCode.INCOMPLETE_GRAPH,
                AdmissionDomain.GRAPH,
                "mandatory dependency closure is incomplete",
            )

        cve_evidence = request.cve_security_evidence
        if (
            request.required_cve_security_stage is not None
            and (
                not cve_evidence
                or cve_evidence[-1].stage
                is not request.required_cve_security_stage
            )
        ):
            reject(
                AdmissionRejectionCode.CVE_SECURITY_GATE_MISSING,
                AdmissionDomain.SECURITY,
                "the required CVE security enforcement stage is absent",
                source_ids=tuple(
                    item.evidence_id for item in cve_evidence
                ),
                details={
                    "required_stage": (
                        request.required_cve_security_stage.value
                    ),
                    "terminal_stage": (
                        cve_evidence[-1].stage.value
                        if cve_evidence
                        else None
                    ),
                },
            )
        if cve_evidence:
            stages = tuple(item.stage for item in cve_evidence)
            expected_stages = _CVE_SECURITY_STAGE_ORDER[: len(stages)]
            if stages != expected_stages:
                reject(
                    AdmissionRejectionCode.CVE_SECURITY_GATE_MISSING,
                    AdmissionDomain.SECURITY,
                    "CVE security enforcement stages must form an exact "
                    "plan-to-current prefix",
                    source_ids=tuple(item.evidence_id for item in cve_evidence),
                    details={
                        "observed_stages": [item.value for item in stages],
                        "expected_stages": [
                            item.value for item in expected_stages
                        ],
                    },
                )
            for index, evidence in enumerate(cve_evidence):
                expected_parent = (
                    "" if index == 0 else cve_evidence[index - 1].evidence_id
                )
                if evidence.parent_evidence_id != expected_parent:
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_DETACHED,
                        AdmissionDomain.SECURITY,
                        "CVE security enforcement evidence is detached from "
                        "the preceding stage",
                        source_ids=(
                            evidence.evidence_id,
                            evidence.parent_evidence_id,
                            expected_parent,
                        ),
                    )
            if cve_evidence[-1].repository_tree_id != request.repository_tree_id:
                reject(
                    AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                    AdmissionDomain.ROOT,
                    "the terminal CVE security gate is bound to a different tree",
                    source_ids=(
                        cve_evidence[-1].repository_tree_id,
                        request.repository_tree_id,
                    ),
                )
            pre_merge_trees = {
                item.repository_tree_id
                for item in cve_evidence
                if item.stage
                in {
                    CVESecurityEnforcementStage.PLAN_ADMISSION,
                    CVESecurityEnforcementStage.PRE_EXECUTION,
                    CVESecurityEnforcementStage.POST_GENERATION,
                }
            }
            merge_trees = {
                item.repository_tree_id
                for item in cve_evidence
                if item.stage
                in {
                    CVESecurityEnforcementStage.MERGE_ADMISSION,
                    CVESecurityEnforcementStage.MERGED_TREE_REVALIDATION,
                }
            }
            if len(pre_merge_trees) > 1 or len(merge_trees) > 1:
                reject(
                    AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                    AdmissionDomain.ROOT,
                    "CVE security evidence changed tree within one enforcement "
                    "phase",
                    source_ids=tuple(
                        item.evidence_id for item in cve_evidence
                    ),
                )

            prior_evaluated_at = -1
            admission_security_request_ids = {
                item.content_id for item in request.security_requests
            }
            policy_root = (
                request.security_policy.security_root_artifact_id,
                request.security_policy.security_root_cid_v1,
                request.security_policy.security_root_supervisor_digest,
            )
            for evidence in cve_evidence:
                gate = evidence.gate_result
                gate_root = (
                    gate.context.security_root_artifact_id,
                    gate.context.security_root_cid_v1,
                    gate.context.security_root_supervisor_digest,
                )
                if (
                    not evidence.current_and_authoritative
                    or gate.findings
                    or gate.policy_receipt_id
                    != request.security_policy.content_id
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_REJECTED,
                        AdmissionDomain.SECURITY,
                        "CVE security gate is rejected, unknown, or "
                        "non-authoritative",
                        source_ids=(
                            evidence.evidence_id,
                            gate.gate_id,
                            *(
                                item.finding_id
                                for item in gate.findings
                            ),
                        ),
                    )
                if gate_root != policy_root:
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                        AdmissionDomain.ROOT,
                        "CVE security gate binds a stale Security IR root",
                        source_ids=(evidence.evidence_id, gate.gate_id),
                    )
                if (
                    gate.context.principal != request.authority.principal
                    or gate.context.requested_authority
                    != request.authority.requested_authority
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_DETACHED,
                        AdmissionDomain.AUTHORITY,
                        "CVE security gate principal or authority differs "
                        "from plan admission",
                        source_ids=(evidence.evidence_id, gate.gate_id),
                    )
                if gate.context.evaluated_at_ms < prior_evaluated_at:
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                        AdmissionDomain.SECURITY,
                        "CVE security stages are not evaluated monotonically",
                        source_ids=(evidence.evidence_id,),
                    )
                prior_evaluated_at = gate.context.evaluated_at_ms
                if (
                    evidence.expires_at_ms is not None
                    and evidence.expires_at_ms <= gate.context.evaluated_at_ms
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                        AdmissionDomain.SECURITY,
                        "CVE security enforcement evidence is already expired",
                        source_ids=(evidence.evidence_id,),
                    )

                mappings = (*gate.intent_mappings, *gate.code_mappings)
                if (
                    not gate.intent_mappings
                    or not gate.code_mappings
                    or any(not item.exact for item in mappings)
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_REJECTED,
                        AdmissionDomain.SECURITY,
                        "CVE security gate requires non-empty exact intent and "
                        "generated-code mappings",
                        source_ids=(evidence.evidence_id, gate.gate_id),
                    )
                mapped_requests = {
                    item.request.content_id
                    for item in mappings
                    if item.request is not None
                }
                if mapped_requests != admission_security_request_ids:
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_DETACHED,
                        AdmissionDomain.SECURITY,
                        "CVE gate requests differ from the exact plan security "
                        "request population",
                        source_ids=tuple(
                            sorted(
                                mapped_requests
                                ^ admission_security_request_ids
                            )
                        ),
                    )
                if correlate_security_requests(
                    gate.intent_mappings, gate.code_mappings
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_REJECTED,
                        AdmissionDomain.SECURITY,
                        "intent and generated-code security mappings conflict",
                        source_ids=(evidence.evidence_id, gate.gate_id),
                    )

                mapping_by_id = {
                    item.mapping_id: item for item in mappings
                }
                decisions_by_mapping = {
                    item.mapping_id: item for item in gate.decisions
                }
                if (
                    len(mapping_by_id) != len(mappings)
                    or len(decisions_by_mapping) != len(gate.decisions)
                    or set(decisions_by_mapping) != set(mapping_by_id)
                ):
                    reject(
                        AdmissionRejectionCode.CVE_SECURITY_GATE_DETACHED,
                        AdmissionDomain.SECURITY,
                        "CVE security decisions do not cover each exact mapping "
                        "once",
                        source_ids=(evidence.evidence_id, gate.gate_id),
                    )
                for mapping_id, mapping in mapping_by_id.items():
                    mapped = decisions_by_mapping.get(mapping_id)
                    if mapped is None or mapping.request is None:
                        continue
                    if mapped.stream is not mapping.stream:
                        reject(
                            AdmissionRejectionCode.CVE_SECURITY_GATE_DETACHED,
                            AdmissionDomain.SECURITY,
                            "CVE security decision stream differs from its "
                            "mapping",
                            source_ids=(evidence.evidence_id, mapping_id),
                        )
                        continue
                    try:
                        current = revalidate_security_authorization(
                            request.security_policy,
                            mapping.request,
                            mapped.decision,
                        )
                    except ValueError:
                        reject(
                            AdmissionRejectionCode.CVE_SECURITY_GATE_STALE,
                            AdmissionDomain.SECURITY,
                            "CVE security decision is stale, forged, or "
                            "detached",
                            source_ids=(evidence.evidence_id, mapping_id),
                        )
                        continue
                    if current.outcome is not SecurityDecisionOutcome.PERMIT:
                        reject(
                            {
                                SecurityDecisionOutcome.DENY: (
                                    AdmissionRejectionCode.SECURITY_DENY
                                ),
                                SecurityDecisionOutcome.UNKNOWN: (
                                    AdmissionRejectionCode.SECURITY_UNKNOWN
                                ),
                                SecurityDecisionOutcome.CONFLICT: (
                                    AdmissionRejectionCode.SECURITY_CONFLICT
                                ),
                            }.get(
                                current.outcome,
                                AdmissionRejectionCode.CVE_SECURITY_GATE_REJECTED,
                            ),
                            AdmissionDomain.SECURITY,
                            "CVE security decision does not permit the exact "
                            "mapping",
                            source_ids=(
                                evidence.evidence_id,
                                mapping_id,
                                current.content_id,
                            ),
                        )

        binding_by_action = {item.action_id: item for item in request.action_bindings}
        if set(binding_by_action) != set(actions):
            for action_id in sorted(set(actions) - set(binding_by_action)):
                reject(
                    AdmissionRejectionCode.DOMAIN_BINDING_MISSING,
                    AdmissionDomain.GRAPH,
                    "candidate action lacks exact legal/security bindings",
                    action_id=action_id,
                )
            for action_id in sorted(set(binding_by_action) - set(actions)):
                reject(
                    AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH,
                    AdmissionDomain.GRAPH,
                    "domain binding names an undeclared action",
                    action_id=action_id,
                )

        legal_by_id = {_record_id(item): item for item in request.legal_results}
        security_request_by_id = {
            item.content_id: item for item in request.security_requests
        }
        used_legal: set[str] = set()
        used_security: set[str] = set()
        for action_id, binding in sorted(binding_by_action.items()):
            if not binding.legal_result_ids or not binding.security_request_ids:
                reject(
                    AdmissionRejectionCode.DOMAIN_BINDING_MISSING,
                    AdmissionDomain.GRAPH,
                    "every action requires independent legal and security checks",
                    action_id=action_id,
                )
            unknown_legal = set(binding.legal_result_ids) - set(legal_by_id)
            unknown_security = set(binding.security_request_ids) - set(
                security_request_by_id
            )
            if unknown_legal or unknown_security:
                reject(
                    AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH,
                    AdmissionDomain.GRAPH,
                    "domain binding names an unknown typed result/request",
                    action_id=action_id,
                    source_ids=tuple(sorted(unknown_legal | unknown_security)),
                )
            used_legal.update(binding.legal_result_ids)
            used_security.update(binding.security_request_ids)
        for result_id in sorted(set(legal_by_id) - used_legal):
            reject(
                AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH,
                AdmissionDomain.LEGAL,
                "unbound legal result is not part of the exact candidate graph",
                source_ids=(result_id,),
            )
        for request_id in sorted(set(security_request_by_id) - used_security):
            reject(
                AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH,
                AdmissionDomain.SECURITY,
                "unbound security request is not part of the exact candidate graph",
                source_ids=(request_id,),
            )

        intent_result = evaluate_intent_conformance(request.intent_request)
        try:
            intent_graph_id = _graph_id(request.intent_request.candidate_plan)
        except IRConstraintCompilerError:
            intent_graph_id = ""
        if intent_graph_id != request.candidate_graph_id:
            reject(
                AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH,
                AdmissionDomain.INTENT,
                "IntentIR conformance request binds a different candidate graph",
                source_ids=(intent_result.result_id,),
            )
        if not intent_result.conformant:
            for finding in intent_result.findings:
                reject(
                    AdmissionRejectionCode.INTENT_VIOLATION,
                    AdmissionDomain.INTENT,
                    finding.message,
                    action_id=finding.action_id,
                    source_ids=tuple(
                        item
                        for item in (
                            finding.constraint_id,
                            finding.source_node_id,
                            finding.finding_id,
                        )
                        if item
                    ),
                    details={
                        "finding_code": finding.code.value,
                        **_plain(finding.details),
                    },
                )

        legal_permissions: list[str] = []
        required_proof_ids: set[str] = {
            item.obligation_id
            for item in request.intent_request.constraint_set.proof_obligations
        }
        for result_id, result in sorted(legal_by_id.items()):
            legal_permissions.extend(
                item.provision_id for item in result.permissions
            )
            if result.fail_closed or not result.accepted:
                reject(
                    AdmissionRejectionCode.LEGAL_INCOMPLETE,
                    AdmissionDomain.LEGAL,
                    "LegalIR applicability is incomplete, contradictory, or unresolved",
                    source_ids=(result_id, *result.reason_codes),
                )
            for prohibition in result.prohibitions:
                reject(
                    AdmissionRejectionCode.LEGAL_PROHIBITION,
                    AdmissionDomain.LEGAL,
                    "an applicable active legal prohibition blocks the candidate",
                    source_ids=(result_id, prohibition.provision_id),
                )
            for obligation in result.proof_obligations:
                if obligation.required:
                    required_proof_ids.add(obligation.obligation_id)
                if obligation.required and not obligation.discharged:
                    reject(
                        AdmissionRejectionCode.LEGAL_OBLIGATION,
                        AdmissionDomain.LEGAL,
                        "an applicable legal proof obligation is unresolved",
                        obligation_id=obligation.obligation_id,
                        source_ids=(result_id, *obligation.provision_ids),
                    )
            for obligation in result.obligations:
                if not obligation.proof_obligation_ids:
                    reject(
                        AdmissionRejectionCode.LEGAL_OBLIGATION,
                        AdmissionDomain.LEGAL,
                        "an applicable legal obligation has no verified discharge",
                        source_ids=(result_id, obligation.provision_id),
                    )

        security_decisions = []
        security_grants: list[str] = []
        for security_request in request.security_requests:
            matching_actions = [
                action_id
                for action_id, binding in binding_by_action.items()
                if security_request.content_id in binding.security_request_ids
            ]
            action_id = matching_actions[0] if len(matching_actions) == 1 else ""
            candidate_effects = [
                _effect_projection(effect)
                for effect in effects.values()
                if effect["action_id"] == action_id
            ]
            if _plain(security_request.expected_effect) not in candidate_effects:
                reject(
                    AdmissionRejectionCode.UNDECLARED_EFFECT,
                    AdmissionDomain.GRAPH,
                    "SecurityIR request names an effect absent from the candidate graph",
                    action_id=action_id,
                    source_ids=(security_request.content_id,),
                    details={"expected_effect": _plain(security_request.expected_effect)},
                )
            candidate_action = actions.get(action_id, {})
            comparisons = {
                "principal": security_request.principal,
                "action": security_request.action,
                "tool": security_request.tool,
                "target": security_request.target,
                "requested_authority": security_request.requested_authority,
            }
            mismatches = {
                name: {"candidate": candidate_action.get(name), "security": value}
                for name, value in comparisons.items()
                if name in candidate_action and candidate_action[name] != value
            }
            if mismatches:
                reject(
                    AdmissionRejectionCode.AUTHORITY_MISMATCH,
                    AdmissionDomain.AUTHORITY,
                    "candidate action and SecurityIR authorization inputs differ",
                    action_id=action_id,
                    source_ids=(security_request.content_id,),
                    details=mismatches,
                )
            decision = evaluate_security_authorization(
                request.security_policy, security_request
            )
            security_decisions.append(decision)
            if decision.outcome is SecurityDecisionOutcome.PERMIT:
                security_grants.append(decision.content_id)
            else:
                code = {
                    SecurityDecisionOutcome.DENY: AdmissionRejectionCode.SECURITY_DENY,
                    SecurityDecisionOutcome.UNKNOWN: AdmissionRejectionCode.SECURITY_UNKNOWN,
                    SecurityDecisionOutcome.CONFLICT: AdmissionRejectionCode.SECURITY_CONFLICT,
                }.get(decision.outcome, AdmissionRejectionCode.SECURITY_DENY)
                reject(
                    code,
                    AdmissionDomain.SECURITY,
                    f"SecurityIR authorization outcome is {decision.outcome.value}",
                    action_id=action_id,
                    source_ids=(
                        decision.content_id,
                        *decision.matched_policy_ids,
                        *decision.reason_codes,
                    ),
                    details={
                        "checks": [item.to_dict() for item in decision.checks]
                    },
                )

        if not request.authority.matched:
            reject(
                AdmissionRejectionCode.AUTHORITY_MISMATCH,
                AdmissionDomain.AUTHORITY,
                "requested principal/authority is not covered by an explicit grant",
                source_ids=request.authority.grant_source_ids,
            )
        for security_request in request.security_requests:
            if (
                security_request.principal != request.authority.principal
                or security_request.requested_authority
                != request.authority.requested_authority
            ):
                reject(
                    AdmissionRejectionCode.AUTHORITY_MISMATCH,
                    AdmissionDomain.AUTHORITY,
                    "SecurityIR request does not match admission authority",
                    source_ids=(security_request.content_id,),
                )

        for root in request.root_bindings:
            if not root.current:
                reject(
                    AdmissionRejectionCode.STALE_ROOT,
                    AdmissionDomain.ROOT,
                    f"{root.kind} semantic root is stale",
                    source_ids=(root.expected, root.observed),
                    details={"kind": root.kind},
                )
            if not root.authority_accepted:
                reject(
                    AdmissionRejectionCode.AUTHORITY_MISMATCH,
                    AdmissionDomain.ROOT,
                    f"{root.kind} semantic root has insufficient authority",
                    source_ids=(root.expected, root.authority),
                )
        if request.decision_request is not None:
            if _root_token(request.decision_request.repository_root) != request.repository_tree_id:
                repository_aliases = {
                    request.decision_request.repository_root.artifact_id,
                    request.decision_request.repository_root.cid_v1,
                    request.decision_request.repository_root.supervisor_digest,
                    _root_token(request.decision_request.repository_root),
                }
                if request.repository_tree_id not in repository_aliases:
                    reject(
                        AdmissionRejectionCode.STALE_ROOT,
                        AdmissionDomain.ROOT,
                        "repository tree does not match the canonical DecisionRequest",
                    )
            expected_decision_roots = {
                root.kind.value: _root_token(root.artifact)
                for root in request.decision_request.semantic_roots
            }
            observed_decision_roots = dict(request.semantic_roots)
            for kind, expected in expected_decision_roots.items():
                if kind in observed_decision_roots and observed_decision_roots[kind] != expected:
                    reject(
                        AdmissionRejectionCode.STALE_ROOT,
                        AdmissionDomain.ROOT,
                        "root binding differs from the canonical DecisionRequest",
                        source_ids=(kind, expected, observed_decision_roots[kind]),
                    )

        for dependency in request.program_dependencies:
            if not dependency.passed:
                reject(
                    (
                        AdmissionRejectionCode.STALE_ROOT
                        if not dependency.current
                        else AdmissionRejectionCode.DEPENDENCY_UNSATISFIED
                    ),
                    AdmissionDomain.PROGRAM,
                    "required program dependency is stale or unsatisfied",
                    action_id=dependency.action_id,
                    dependency_id=dependency.dependency_id,
                    source_ids=dependency.reason_codes,
                )
        for assumption in request.assumptions:
            if not assumption.passed:
                for action_id in assumption.action_ids or ("",):
                    reject(
                        AdmissionRejectionCode.ASSUMPTION_UNRESOLVED,
                        AdmissionDomain.ASSUMPTION,
                        "required assumption lacks current authoritative evidence",
                        action_id=action_id,
                        source_ids=(assumption.assumption_id,),
                    )

        proof_by_obligation: dict[str, list[ProofReceipt]] = defaultdict(list)
        for proof in request.proof_results:
            proof_by_obligation[proof.obligation_id].append(proof)
        required_assurance_by_id = {
            item.obligation_id: AssuranceLevel.KERNEL_VERIFIED
            for item in request.intent_request.constraint_set.proof_obligations
        }
        for obligation_id in required_proof_ids:
            receipts = proof_by_obligation.get(obligation_id, ())
            valid = [
                item
                for item in receipts
                if item.repository_tree_id == request.repository_tree_id
                and item.freshness is EvidenceFreshness.CURRENT
                and item.authoritative_verdict is ProofVerdict.PROVED
                and item.satisfies(
                    required_assurance_by_id.get(
                        obligation_id, AssuranceLevel.KERNEL_VERIFIED
                    )
                )
            ]
            if not receipts:
                reject(
                    AdmissionRejectionCode.MISSING_PROOF,
                    AdmissionDomain.PROOF,
                    "required proof obligation has no typed ProofReceipt",
                    obligation_id=obligation_id,
                    details={
                        "generated_formula_present": (
                            obligation_id in request.generated_formula_ids
                        )
                    },
                )
            elif not valid:
                reject(
                    AdmissionRejectionCode.INVALID_PROOF,
                    AdmissionDomain.PROOF,
                    "proof receipt is stale, mismatched, or not authoritatively proved",
                    obligation_id=obligation_id,
                    source_ids=tuple(item.receipt_id for item in receipts),
                )

        validation_by_id = {
            item.requirement_id: item for item in request.validation_results
        }
        for requirement in request.validation_requirements:
            if not requirement.required:
                continue
            result = validation_by_id.get(requirement.requirement_id)
            if result is None:
                for action_id in requirement.action_ids or ("",):
                    reject(
                        AdmissionRejectionCode.VALIDATION_MISSING,
                        AdmissionDomain.VALIDATION,
                        "required validation has no result",
                        action_id=action_id,
                        source_ids=(requirement.requirement_id,),
                    )
            elif (
                result.status is not ValidationStatus.PASSED
                or result.repository_tree_id != request.repository_tree_id
                or not result.evidence_id
            ):
                for action_id in requirement.action_ids or ("",):
                    reject(
                        (
                            AdmissionRejectionCode.STALE_ROOT
                            if result.repository_tree_id != request.repository_tree_id
                            else AdmissionRejectionCode.VALIDATION_FAILED
                        ),
                        AdmissionDomain.VALIDATION,
                        "validation failed, is unknown, or is stale",
                        action_id=action_id,
                        source_ids=(requirement.requirement_id, *result.reason_codes),
                    )

        unique = {item.rejection_id: item for item in rejections}
        rejections = [unique[key] for key in sorted(unique)]
        dependants: dict[str, set[str]] = defaultdict(set)
        for action in actions.values():
            for dependency in action.get("depends_on", ()):
                dependants[dependency].add(action["action_id"])

        def affected(seeds: set[str]) -> tuple[str, ...]:
            found = set(seed for seed in seeds if seed in actions)
            queue = deque(sorted(found))
            while queue:
                for item in sorted(dependants.get(queue.popleft(), ())):
                    if item not in found:
                        found.add(item)
                        queue.append(item)
            return tuple(sorted(found))

        examples: list[AdmissionCounterexample] = []
        replan: set[str] = set()
        all_actions = set(actions)
        for rejection in rejections:
            seeds = {rejection.action_id} if rejection.action_id else set()
            scope = affected(seeds) if seeds else tuple(sorted(all_actions))
            replan.update(scope)
            examples.append(
                AdmissionCounterexample(
                    rejection_id=rejection.rejection_id,
                    candidate_plan_id=request.candidate_plan_id,
                    failing_action_ids=tuple(sorted(seeds)),
                    affected_action_ids=scope,
                    fixed_action_ids=tuple(sorted(all_actions - set(scope))),
                    witness={
                        "code": rejection.code.value,
                        "domain": rejection.domain.value,
                        "effect_id": rejection.effect_id,
                        "dependency_id": rejection.dependency_id,
                        "obligation_id": rejection.obligation_id,
                        "source_ids": list(rejection.source_ids),
                        "details": _plain(rejection.details),
                    },
                )
            )

        return PlanAdmissionReceipt(
            request_id=request.request_id,
            candidate_plan_id=request.candidate_plan_id,
            candidate_graph_id=request.candidate_graph_id,
            repository_tree_id=request.repository_tree_id,
            verdict=(
                PlanAdmissionVerdict.REJECTED
                if rejections
                else PlanAdmissionVerdict.ADMITTED
            ),
            semantic_roots=request.semantic_roots,
            intent_result_id=intent_result.result_id,
            legal_result_ids=tuple(legal_by_id),
            legal_permission_ids=tuple(legal_permissions),
            security_decision_ids=tuple(
                item.content_id for item in security_decisions
            ),
            security_grant_ids=(
                request.authority.grant_source_ids
                if security_grants
                and len(security_grants) == len(request.security_requests)
                else ()
            ),
            checked_dependency_ids=tuple(
                item.dependency_id for item in request.program_dependencies
            ),
            checked_assumption_ids=tuple(
                item.assumption_id for item in request.assumptions
            ),
            generated_formula_ids=request.generated_formula_ids,
            proof_result_ids=tuple(
                item.receipt_id for item in request.proof_results
            ),
            checked_validation_ids=tuple(
                item.requirement_id for item in request.validation_requirements
            ),
            cve_security_evidence_ids=tuple(
                item.evidence_id for item in request.cve_security_evidence
            ),
            rejection_reasons=tuple(rejections),
            counterexamples=tuple(examples),
            local_replan_action_ids=tuple(sorted(replan)),
            closure_id=(
                request.mandatory_closure.closure_id
                if request.mandatory_closure is not None
                else ""
            ),
        )

    admit = compile
    evaluate = compile


def compile_plan_admission(
    request: PlanAdmissionRequest | Mapping[str, Any],
) -> PlanAdmissionReceipt:
    if isinstance(request, Mapping):
        request = PlanAdmissionRequest.from_dict(request)
    return IRConstraintCompiler().compile(request)


def _compile_cve_stage_admission(
    request: PlanAdmissionRequest,
    stage: CVESecurityEnforcementStage,
) -> PlanAdmissionReceipt:
    if not isinstance(request, PlanAdmissionRequest):
        raise IRConstraintCompilerError(
            "CVE security admission requires a PlanAdmissionRequest"
        )
    if request.required_cve_security_stage is not stage:
        raise IRConstraintCompilerError(
            f"request must require the {stage.value} CVE security stage"
        )
    return compile_plan_admission(request)


def compile_cve_plan_admission(
    request: PlanAdmissionRequest,
) -> PlanAdmissionReceipt:
    """Require and replay the plan-admission CVE security gate."""

    return _compile_cve_stage_admission(
        request, CVESecurityEnforcementStage.PLAN_ADMISSION
    )


def compile_cve_pre_execution_admission(
    request: PlanAdmissionRequest,
) -> PlanAdmissionReceipt:
    """Require an unbroken gate chain through the pre-execution boundary."""

    return _compile_cve_stage_admission(
        request, CVESecurityEnforcementStage.PRE_EXECUTION
    )


def compile_cve_post_generation_validation(
    request: PlanAdmissionRequest,
) -> PlanAdmissionReceipt:
    """Re-admit generated code only after its exact CVE gate passes."""

    return _compile_cve_stage_admission(
        request, CVESecurityEnforcementStage.POST_GENERATION
    )


def compile_cve_merge_admission(
    request: PlanAdmissionRequest,
) -> PlanAdmissionReceipt:
    """Admit a merge only after rebuilding the CVE gate on its tree."""

    return _compile_cve_stage_admission(
        request, CVESecurityEnforcementStage.MERGE_ADMISSION
    )


def revalidate_cve_merged_tree(
    request: PlanAdmissionRequest,
) -> PlanAdmissionReceipt:
    """Replay the complete enforcement chain on the synthesized merged tree."""

    return _compile_cve_stage_admission(
        request, CVESecurityEnforcementStage.MERGED_TREE_REVALIDATION
    )


compile_ir_constraints = compile_plan_admission
admit_plan = compile_plan_admission
PlanAdmissionDecision = PlanAdmissionReceipt
PlanAdmissionRejection = AdmissionRejection
PlanAdmissionCounterexample = AdmissionCounterexample


__all__ = [
    "IR_CONFORMANCE_REQUIREMENT_ID",
    "IR_CONSTRAINT_COMPILER_VERSION",
    "PLAN_ADMISSION_COUNTEREXAMPLE_SCHEMA",
    "PLAN_ADMISSION_RECEIPT_SCHEMA",
    "PLAN_ADMISSION_REJECTION_SCHEMA",
    "PLAN_ADMISSION_REQUEST_SCHEMA",
    "CVE_SECURITY_ENFORCEMENT_EVIDENCE_SCHEMA",
    "ActionDomainBinding",
    "AdmissionAssumption",
    "AdmissionAuthority",
    "AdmissionCounterexample",
    "AdmissionDomain",
    "AdmissionRejection",
    "AdmissionRejectionCode",
    "CVESecurityEnforcementEvidence",
    "CVESecurityEnforcementStage",
    "IRConstraintCompiler",
    "IRConstraintCompilerError",
    "PlanAdmissionCounterexample",
    "PlanAdmissionDecision",
    "PlanAdmissionReceipt",
    "PlanAdmissionRejection",
    "PlanAdmissionRequest",
    "PlanAdmissionVerdict",
    "ProgramDependency",
    "RootBinding",
    "ValidationRequirement",
    "ValidationResult",
    "ValidationStatus",
    "admit_plan",
    "compile_ir_constraints",
    "compile_cve_merge_admission",
    "compile_cve_plan_admission",
    "compile_cve_post_generation_validation",
    "compile_cve_pre_execution_admission",
    "compile_plan_admission",
    "revalidate_cve_merged_tree",
]
