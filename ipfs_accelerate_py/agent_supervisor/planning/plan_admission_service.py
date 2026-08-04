"""Independent multi-gate plan admission (PDR-027 / ``PlanAdmissionService@1``).

This service is the planner-side join over formal compilation, cross-IR
admission, security forbidden-logic checks, proof obligations, authority
grants, and parallel execution/policy bindings.  It deliberately:

* constructs its own exact :class:`PlanAdmissionRequest` from primitive domain
  materials and never treats provider ``admitted`` / ``valid`` / ``passed``
  claims as authority;
* runs every hard stage in a fixed order;
* fails closed on unknown mandatory applicability, security, authority,
  effect, or proof state; and
* binds every admitted receipt to the candidate, evidence bundle, formal plan,
  IR roots, proof obligations, execution plan, policies, and current tree so
  tampering or replay against a different binding fails.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    EvidenceFreshness,
    ProofReceipt,
    ProofVerdict,
    content_identity,
)
from ..proof.intent_constraint_adapter import IntentConformanceRequest
from ..proof.ir_constraint_compiler import (
    ActionDomainBinding,
    AdmissionAssumption,
    AdmissionAuthority,
    AdmissionDomain,
    AdmissionRejection,
    AdmissionRejectionCode,
    IRConstraintCompiler,
    IRConstraintCompilerError,
    PlanAdmissionReceipt,
    PlanAdmissionRequest,
    PlanAdmissionVerdict,
    ProgramDependency,
    RootBinding,
    ValidationRequirement,
    ValidationResult,
    ValidationStatus,
    compile_plan_admission,
)
from ..proof.legal_constraint_adapter import LegalCompilationResult
from ..proof.security_constraint_adapter import (
    SecurityAuthorizationRequest,
    SecurityDecisionOutcome,
    SecurityPolicyReceipt,
    SecurityRuleEffect,
    evaluate_security_authorization,
)
from ..analysis.semantic_dependency_graph import MandatoryClosure


PLAN_ADMISSION_SERVICE_INTERFACE: Final[str] = "PlanAdmissionService@1"
PLAN_ADMISSION_SERVICE_VERSION: Final[int] = 1
PLAN_ADMISSION_SERVICE_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-service-request@1"
)
PLAN_ADMISSION_SERVICE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-service-receipt@1"
)
PLAN_ADMISSION_STAGE_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/plan-admission-stage-result@1"
)

# Provider claim keys that must never influence admission authority.
_PROVIDER_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "admitted",
        "accepted",
        "allowed",
        "approved",
        "authority_granted",
        "authorizes_execution",
        "authorizes_completion",
        "completed",
        "passed",
        "permitted",
        "provider_admitted",
        "provider_verdict",
        "score",
        "valid",
        "verdict",
    }
)

_FORBIDDEN_SECURITY_OUTCOMES: Final[frozenset[SecurityDecisionOutcome]] = frozenset(
    {
        SecurityDecisionOutcome.DENY,
        SecurityDecisionOutcome.UNKNOWN,
        SecurityDecisionOutcome.CONFLICT,
    }
)


class PlanAdmissionServiceError(ValueError):
    """Materials or a receipt cannot be admitted without weakening authority."""


class PlanAdmissionStage(str, Enum):
    """Fixed-order hard gates.  Order is authority, not presentation."""

    FORMAL = "formal"
    CANDIDATE_GRAPH = "candidate_graph"
    ROOTS = "roots"
    INTENT = "intent"
    APPLICABILITY = "applicability"
    SECURITY = "security"
    AUTHORITY = "authority"
    EFFECTS = "effects"
    PROOF = "proof"
    VALIDATION = "validation"
    EXECUTION_PLAN = "execution_plan"
    POLICY = "policy"
    IR_JOIN = "ir_join"


ADMISSION_STAGE_ORDER: Final[tuple[PlanAdmissionStage, ...]] = (
    PlanAdmissionStage.FORMAL,
    PlanAdmissionStage.CANDIDATE_GRAPH,
    PlanAdmissionStage.ROOTS,
    PlanAdmissionStage.INTENT,
    PlanAdmissionStage.APPLICABILITY,
    PlanAdmissionStage.SECURITY,
    PlanAdmissionStage.AUTHORITY,
    PlanAdmissionStage.EFFECTS,
    PlanAdmissionStage.PROOF,
    PlanAdmissionStage.VALIDATION,
    PlanAdmissionStage.EXECUTION_PLAN,
    PlanAdmissionStage.POLICY,
    PlanAdmissionStage.IR_JOIN,
)


class PlanAdmissionServiceCode(str, Enum):
    PROVIDER_CLAIM_IGNORED = "provider_claim_ignored"
    MISSING_FORMAL_PLAN = "missing_formal_plan"
    FORMAL_IDENTITY_MISMATCH = "formal_identity_mismatch"
    INCOMPLETE_CANDIDATE = "incomplete_candidate"
    UNKNOWN_MANDATORY_ROOT = "unknown_mandatory_root"
    STALE_ROOT = "stale_root"
    UNKNOWN_MANDATORY_INTENT = "unknown_mandatory_intent"
    UNKNOWN_MANDATORY_APPLICABILITY = "unknown_mandatory_applicability"
    APPLICABILITY_REJECTED = "applicability_rejected"
    UNKNOWN_MANDATORY_SECURITY = "unknown_mandatory_security"
    SECURITY_FORBIDDEN = "security_forbidden"
    SECURITY_STREAM_GAP = "security_stream_gap"
    UNKNOWN_MANDATORY_AUTHORITY = "unknown_mandatory_authority"
    AUTHORITY_MISMATCH = "authority_mismatch"
    UNKNOWN_MANDATORY_EFFECT = "unknown_mandatory_effect"
    UNDECLARED_EFFECT = "undeclared_effect"
    UNKNOWN_MANDATORY_PROOF = "unknown_mandatory_proof"
    MISSING_PROOF = "missing_proof"
    UNKNOWN_MANDATORY_VALIDATION = "unknown_mandatory_validation"
    VALIDATION_FAILED = "validation_failed"
    MISSING_EXECUTION_PLAN = "missing_execution_plan"
    MISSING_POLICY = "missing_policy"
    IR_REJECTED = "ir_rejected"
    REQUEST_CONSTRUCTION_FAILED = "request_construction_failed"
    RECEIPT_TAMPERED = "receipt_tampered"
    REPLAY_MISMATCH = "replay_mismatch"


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 16:
        raise PlanAdmissionServiceError("admission value exceeds depth bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise PlanAdmissionServiceError(
            "floating point values are not canonical admission data"
        )
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise PlanAdmissionServiceError("admission mapping keys must be strings")
        return {
            key: _plain(item, depth=depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
            if key not in _PROVIDER_CLAIM_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(item, depth=depth + 1) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _plain(converter(), depth=depth + 1)
    raise PlanAdmissionServiceError(
        f"unsupported admission value: {type(value).__name__}"
    )


def _freeze(value: Any) -> Any:
    value = _plain(value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise PlanAdmissionServiceError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise PlanAdmissionServiceError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise PlanAdmissionServiceError(f"{name} is required")
    return value


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise PlanAdmissionServiceError(f"{name} must be a sequence")
    return tuple(sorted({_text(item, name) for item in value}))


def _identity(namespace: str, value: Any) -> str:
    return content_identity({"namespace": namespace, "value": _plain(value)})


def _effect_projection(effect: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in _plain(effect).items()
        if key not in {"effect_id", "action_id", "task_id", "metadata"}
    }


def _record_id(value: Any) -> str:
    for name in ("content_id", "result_id", "request_id", "receipt_id"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return _identity("typed-admission-record", _plain(value))


def _candidate_actions(candidate: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = candidate.get("actions", candidate.get("tasks", ()))
    if isinstance(raw, Mapping):
        raw = tuple(raw.values())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise PlanAdmissionServiceError("candidate actions must be a sequence")
    actions: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise PlanAdmissionServiceError("candidate actions must be objects")
        plain = _plain(item)
        action_id = plain.get("action_id", plain.get("task_id", plain.get("id", "")))
        action_id = _text(action_id, "candidate action_id")
        plain["action_id"] = action_id
        actions.append(plain)
    return tuple(actions)


def _candidate_effects(candidate: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = _candidate_actions(candidate)
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw = candidate.get("effects", ())
    if isinstance(raw, Mapping):
        raw = tuple(raw.values())
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise PlanAdmissionServiceError("candidate effects must be a sequence")
    for item in raw:
        if not isinstance(item, Mapping):
            raise PlanAdmissionServiceError("candidate effects must be objects")
        plain = _plain(item)
        effect_id = _text(plain.get("effect_id", ""), "effect_id")
        action_id = _text(
            plain.get("action_id", plain.get("task_id", "")), "effect action_id"
        )
        plain["effect_id"] = effect_id
        plain["action_id"] = action_id
        if effect_id in seen:
            raise PlanAdmissionServiceError("candidate effect IDs must be unique")
        seen.add(effect_id)
        collected.append(plain)
    for action in actions:
        embedded = action.get("effects", ())
        if isinstance(embedded, Mapping):
            embedded = (embedded,)
        if not embedded:
            continue
        if isinstance(embedded, str) or not isinstance(embedded, Sequence):
            raise PlanAdmissionServiceError("embedded effects must be a sequence")
        action_id = action["action_id"]
        for index, item in enumerate(embedded):
            if not isinstance(item, Mapping):
                raise PlanAdmissionServiceError("candidate effects must be objects")
            plain = _plain(item)
            effect_id = plain.get("effect_id") or _identity(
                "candidate-effect",
                {"action_id": action_id, "index": index, "effect": plain},
            )
            effect_id = _text(effect_id, "effect_id")
            plain["effect_id"] = effect_id
            plain["action_id"] = action_id
            if effect_id in seen:
                continue
            seen.add(effect_id)
            collected.append(plain)
    return tuple(collected)


def _candidate_plan_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("plan_id", candidate.get("candidate_plan_id", ""))
    if value:
        return _text(value, "candidate plan_id")
    return _identity("candidate-plan", candidate)


def _candidate_graph_id(candidate: Mapping[str, Any]) -> str:
    return _identity(
        "candidate-action-effect-graph",
        {
            "actions": list(_candidate_actions(candidate)),
            "effects": list(_candidate_effects(candidate)),
        },
    )


def _strip_provider_claims(value: Any) -> Any:
    """Drop provider admission claims so they cannot become authority."""

    if isinstance(value, Mapping):
        return {
            key: _strip_provider_claims(item)
            for key, item in value.items()
            if key not in _PROVIDER_CLAIM_KEYS
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_strip_provider_claims(item) for item in value]
    return value


@dataclass(frozen=True)
class PlanAdmissionStageResult:
    """One fixed-order gate outcome."""

    stage: PlanAdmissionStage
    passed: bool
    reason_codes: tuple[str, ...] = ()
    detail_ids: tuple[str, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", PlanAdmissionStage(self.stage))
        if not isinstance(self.passed, bool):
            raise PlanAdmissionServiceError("stage passed must be boolean")
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )
        object.__setattr__(
            self, "detail_ids", _strings(self.detail_ids, "detail_ids")
        )
        object.__setattr__(
            self, "message", _text(self.message, "message", required=False)
        )

    @property
    def stage_result_id(self) -> str:
        return _identity("plan-admission-stage-result", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_STAGE_RESULT_SCHEMA,
            "stage": self.stage.value,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "detail_ids": list(self.detail_ids),
            "message": self.message,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "stage_result_id": self.stage_result_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanAdmissionStageResult":
        result = cls(
            stage=value.get("stage", PlanAdmissionStage.FORMAL),
            passed=bool(value.get("passed", False)),
            reason_codes=tuple(value.get("reason_codes") or ()),
            detail_ids=tuple(value.get("detail_ids") or ()),
            message=str(value.get("message") or ""),
        )
        claimed = str(value.get("stage_result_id") or "")
        if claimed and claimed != result.stage_result_id:
            raise PlanAdmissionServiceError(
                "stage result identity does not match content"
            )
        return result


@dataclass(frozen=True)
class PlanAdmissionServiceRequest:
    """Primitive materials used to construct an exact PlanAdmissionRequest.

    Provider admission claims attached to any nested mapping are stripped and
    never become authority.  Missing mandatory domain materials fail closed at
    the corresponding fixed stage.
    """

    candidate_plan: Mapping[str, Any]
    repository_tree_id: str
    formal_plan_id: str
    intent_request: IntentConformanceRequest
    legal_results: tuple[LegalCompilationResult, ...]
    security_policy: SecurityPolicyReceipt
    security_requests: tuple[SecurityAuthorizationRequest, ...]
    action_bindings: tuple[ActionDomainBinding, ...]
    authority: AdmissionAuthority
    root_bindings: tuple[RootBinding, ...]
    evidence_bundle_id: str
    execution_plan_id: str
    policy_ids: tuple[str, ...]
    formal_source_identity: str = ""
    program_dependencies: tuple[ProgramDependency, ...] = ()
    assumptions: tuple[AdmissionAssumption, ...] = ()
    proof_results: tuple[ProofReceipt, ...] = ()
    validation_requirements: tuple[ValidationRequirement, ...] = ()
    validation_results: tuple[ValidationResult, ...] = ()
    intent_effects: tuple[Mapping[str, Any], ...] = ()
    code_effects: tuple[Mapping[str, Any], ...] = ()
    generated_formula_ids: tuple[str, ...] = ()
    mandatory_closure: MandatoryClosure | None = None
    graph_complete: bool = True
    # Optional opaque provider claim envelope — stripped, never trusted.
    provider_admission_claim: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_plan, Mapping):
            raise PlanAdmissionServiceError("candidate_plan must be a mapping")
        candidate = _freeze(_strip_provider_claims(self.candidate_plan))
        # Touch the graph once so construction fails early on malformed plans.
        _candidate_actions(candidate)
        _candidate_effects(candidate)
        object.__setattr__(self, "candidate_plan", candidate)
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        # formal/evidence/execution may be empty so the corresponding fixed
        # stage can fail closed with a typed code rather than construction error.
        object.__setattr__(
            self,
            "formal_plan_id",
            _text(self.formal_plan_id, "formal_plan_id", required=False),
        )
        object.__setattr__(
            self,
            "formal_source_identity",
            _text(
                self.formal_source_identity,
                "formal_source_identity",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "evidence_bundle_id",
            _text(self.evidence_bundle_id, "evidence_bundle_id", required=False),
        )
        object.__setattr__(
            self,
            "execution_plan_id",
            _text(self.execution_plan_id, "execution_plan_id", required=False),
        )
        object.__setattr__(
            self, "policy_ids", _strings(self.policy_ids, "policy_ids")
        )
        if not isinstance(self.intent_request, IntentConformanceRequest):
            raise PlanAdmissionServiceError(
                "intent_request must be IntentConformanceRequest"
            )
        if any(
            not isinstance(item, LegalCompilationResult)
            for item in self.legal_results
        ):
            raise PlanAdmissionServiceError(
                "legal_results must contain LegalCompilationResult records"
            )
        if not isinstance(self.security_policy, SecurityPolicyReceipt):
            raise PlanAdmissionServiceError(
                "security_policy must be SecurityPolicyReceipt"
            )
        if any(
            not isinstance(item, SecurityAuthorizationRequest)
            for item in self.security_requests
        ):
            raise PlanAdmissionServiceError(
                "security_requests must contain SecurityAuthorizationRequest records"
            )
        if not isinstance(self.authority, AdmissionAuthority):
            if isinstance(self.authority, Mapping):
                object.__setattr__(
                    self,
                    "authority",
                    AdmissionAuthority.from_dict(
                        _strip_provider_claims(self.authority)
                    ),
                )
            else:
                raise PlanAdmissionServiceError("authority is malformed")
        for name, kind in (
            ("action_bindings", ActionDomainBinding),
            ("root_bindings", RootBinding),
            ("program_dependencies", ProgramDependency),
            ("assumptions", AdmissionAssumption),
            ("validation_requirements", ValidationRequirement),
            ("validation_results", ValidationResult),
        ):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise PlanAdmissionServiceError(f"{name} must be a sequence")
            coerced: list[Any] = []
            for item in values:
                if isinstance(item, kind):
                    coerced.append(item)
                elif isinstance(item, Mapping):
                    coerced.append(kind.from_dict(_strip_provider_claims(item)))
                else:
                    raise PlanAdmissionServiceError(
                        f"{name} must contain {kind.__name__} records"
                    )
            object.__setattr__(self, name, tuple(coerced))
        if any(not isinstance(item, ProofReceipt) for item in self.proof_results):
            raise PlanAdmissionServiceError(
                "proof_results must contain ProofReceipt records"
            )
        object.__setattr__(
            self,
            "proof_results",
            tuple(sorted(self.proof_results, key=lambda item: item.receipt_id)),
        )
        intent_effects = []
        for item in self.intent_effects:
            if not isinstance(item, Mapping):
                raise PlanAdmissionServiceError("intent_effects must be objects")
            intent_effects.append(_freeze(_strip_provider_claims(item)))
        object.__setattr__(self, "intent_effects", tuple(intent_effects))
        code_effects = []
        for item in self.code_effects:
            if not isinstance(item, Mapping):
                raise PlanAdmissionServiceError("code_effects must be objects")
            code_effects.append(_freeze(_strip_provider_claims(item)))
        object.__setattr__(self, "code_effects", tuple(code_effects))
        object.__setattr__(
            self,
            "generated_formula_ids",
            _strings(self.generated_formula_ids, "generated_formula_ids"),
        )
        if self.mandatory_closure is not None and not isinstance(
            self.mandatory_closure, MandatoryClosure
        ):
            raise PlanAdmissionServiceError(
                "mandatory_closure must be MandatoryClosure"
            )
        if not isinstance(self.graph_complete, bool):
            raise PlanAdmissionServiceError("graph_complete must be boolean")
        if self.provider_admission_claim is not None:
            if not isinstance(self.provider_admission_claim, Mapping):
                raise PlanAdmissionServiceError(
                    "provider_admission_claim must be a mapping when present"
                )
            # Keep a stripped witness only — claims never become authority.
            object.__setattr__(
                self,
                "provider_admission_claim",
                _freeze(_strip_provider_claims(self.provider_admission_claim)),
            )

    @property
    def candidate_plan_id(self) -> str:
        return _candidate_plan_id(self.candidate_plan)

    @property
    def candidate_graph_id(self) -> str:
        return _candidate_graph_id(self.candidate_plan)

    @property
    def semantic_roots(self) -> Mapping[str, str]:
        return MappingProxyType(
            {item.kind: item.expected for item in self.root_bindings}
        )

    @property
    def request_id(self) -> str:
        return _identity("plan-admission-service-request", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_SERVICE_REQUEST_SCHEMA,
            "service_version": PLAN_ADMISSION_SERVICE_VERSION,
            "interface": PLAN_ADMISSION_SERVICE_INTERFACE,
            "candidate_plan_id": self.candidate_plan_id,
            "candidate_graph_id": self.candidate_graph_id,
            "candidate_plan": _plain(self.candidate_plan),
            "repository_tree_id": self.repository_tree_id,
            "formal_plan_id": self.formal_plan_id,
            "formal_source_identity": self.formal_source_identity,
            "evidence_bundle_id": self.evidence_bundle_id,
            "execution_plan_id": self.execution_plan_id,
            "policy_ids": list(self.policy_ids),
            "intent_request": self.intent_request.to_dict(),
            "legal_results": [item.to_dict() for item in self.legal_results],
            "security_policy": self.security_policy.to_dict(),
            "security_requests": [
                item.to_dict() for item in self.security_requests
            ],
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
            "intent_effects": [_plain(item) for item in self.intent_effects],
            "code_effects": [_plain(item) for item in self.code_effects],
            "generated_formula_ids": list(self.generated_formula_ids),
            "mandatory_closure": (
                self.mandatory_closure.to_dict()
                if self.mandatory_closure is not None
                and hasattr(self.mandatory_closure, "to_dict")
                else (
                    {
                        "closure_id": self.mandatory_closure.closure_id,
                        "complete": self.mandatory_closure.complete,
                    }
                    if self.mandatory_closure is not None
                    else None
                )
            ),
            "graph_complete": self.graph_complete,
            # Provider claims are never serialized as authority inputs.
            "provider_admission_claim": None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "request_id": self.request_id}


@dataclass(frozen=True)
class PlanAdmissionServiceReceipt:
    """Independent service receipt with full multi-artifact bindings.

    An admitted receipt always binds the candidate, evidence bundle, formal
    plan, IR roots, proof obligations, execution plan, policies, and current
    tree.  ``from_dict`` recomputes content identity and rejects tampering.
    """

    request_id: str
    candidate_plan_id: str
    candidate_graph_id: str
    repository_tree_id: str
    formal_plan_id: str
    formal_source_identity: str
    evidence_bundle_id: str
    execution_plan_id: str
    policy_ids: tuple[str, ...]
    ir_request_id: str
    ir_receipt_id: str
    proof_obligation_ids: tuple[str, ...]
    proof_result_ids: tuple[str, ...]
    semantic_roots: Mapping[str, str]
    verdict: PlanAdmissionVerdict
    stage_results: tuple[PlanAdmissionStageResult, ...]
    rejection_reasons: tuple[AdmissionRejection, ...] = ()
    ir_receipt: PlanAdmissionReceipt | None = None
    constructed_request_id: str = ""
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "verdict", PlanAdmissionVerdict(self.verdict))
        for name in (
            "request_id",
            "candidate_plan_id",
            "candidate_graph_id",
            "repository_tree_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name)
            )
        for name in (
            "formal_plan_id",
            "formal_source_identity",
            "evidence_bundle_id",
            "execution_plan_id",
            "ir_request_id",
            "ir_receipt_id",
            "constructed_request_id",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self, "policy_ids", _strings(self.policy_ids, "policy_ids")
        )
        object.__setattr__(
            self,
            "proof_obligation_ids",
            _strings(self.proof_obligation_ids, "proof_obligation_ids"),
        )
        object.__setattr__(
            self,
            "proof_result_ids",
            _strings(self.proof_result_ids, "proof_result_ids"),
        )
        roots = {
            _text(key, "semantic root kind"): _text(
                value, "semantic root value"
            )
            for key, value in dict(self.semantic_roots).items()
        }
        object.__setattr__(
            self, "semantic_roots", MappingProxyType(dict(sorted(roots.items())))
        )
        stages = []
        for item in self.stage_results:
            if isinstance(item, PlanAdmissionStageResult):
                stages.append(item)
            elif isinstance(item, Mapping):
                stages.append(PlanAdmissionStageResult.from_dict(item))
            else:
                raise PlanAdmissionServiceError(
                    "stage_results must contain PlanAdmissionStageResult records"
                )
        # Stage order is fixed and authoritative.
        stages = tuple(
            sorted(stages, key=lambda item: ADMISSION_STAGE_ORDER.index(item.stage))
        )
        object.__setattr__(self, "stage_results", stages)
        rejections: list[AdmissionRejection] = []
        for item in self.rejection_reasons:
            if isinstance(item, AdmissionRejection):
                rejections.append(item)
            elif isinstance(item, Mapping):
                rejections.append(AdmissionRejection.from_dict(item))
            else:
                raise PlanAdmissionServiceError(
                    "rejection_reasons must contain AdmissionRejection records"
                )
        rejections = sorted(
            {item.rejection_id: item for item in rejections}.values(),
            key=lambda item: item.rejection_id,
        )
        object.__setattr__(self, "rejection_reasons", tuple(rejections))
        if self.ir_receipt is not None and not isinstance(
            self.ir_receipt, PlanAdmissionReceipt
        ):
            raise PlanAdmissionServiceError(
                "ir_receipt must be PlanAdmissionReceipt when present"
            )
        derived_codes = tuple(
            sorted(
                {
                    *{item.code.value for item in rejections},
                    *(
                        code
                        for stage in stages
                        for code in stage.reason_codes
                        if not stage.passed
                    ),
                }
            )
        )
        object.__setattr__(self, "reason_codes", derived_codes)
        if self.verdict is PlanAdmissionVerdict.ADMITTED and rejections:
            raise PlanAdmissionServiceError(
                "admitted receipt cannot carry rejection reasons"
            )
        if self.verdict is PlanAdmissionVerdict.ADMITTED:
            for name in (
                "formal_plan_id",
                "evidence_bundle_id",
                "execution_plan_id",
                "ir_request_id",
                "ir_receipt_id",
            ):
                if not getattr(self, name):
                    raise PlanAdmissionServiceError(
                        f"admitted receipt must bind {name}"
                    )
            if not self.policy_ids:
                raise PlanAdmissionServiceError(
                    "admitted receipt must bind policies"
                )
            if not self.stage_results or any(
                not item.passed for item in self.stage_results
            ):
                raise PlanAdmissionServiceError(
                    "admitted receipt requires every stage to pass"
                )
            observed_stages = tuple(item.stage for item in self.stage_results)
            if observed_stages != ADMISSION_STAGE_ORDER:
                raise PlanAdmissionServiceError(
                    "admitted receipt must record every stage in fixed order"
                )

    @property
    def admitted(self) -> bool:
        return self.verdict is PlanAdmissionVerdict.ADMITTED

    @property
    def authorizes_execution(self) -> bool:
        return False

    @property
    def receipt_id(self) -> str:
        return _identity("plan-admission-service-receipt", self._payload())

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": PLAN_ADMISSION_SERVICE_RECEIPT_SCHEMA,
            "service_version": PLAN_ADMISSION_SERVICE_VERSION,
            "interface": PLAN_ADMISSION_SERVICE_INTERFACE,
            "request_id": self.request_id,
            "candidate_plan_id": self.candidate_plan_id,
            "candidate_graph_id": self.candidate_graph_id,
            "repository_tree_id": self.repository_tree_id,
            "formal_plan_id": self.formal_plan_id,
            "formal_source_identity": self.formal_source_identity,
            "evidence_bundle_id": self.evidence_bundle_id,
            "execution_plan_id": self.execution_plan_id,
            "policy_ids": list(self.policy_ids),
            "ir_request_id": self.ir_request_id,
            "ir_receipt_id": self.ir_receipt_id,
            "proof_obligation_ids": list(self.proof_obligation_ids),
            "proof_result_ids": list(self.proof_result_ids),
            "semantic_roots": dict(self.semantic_roots),
            "verdict": self.verdict.value,
            "admitted": self.admitted,
            "stage_results": [item.to_dict() for item in self.stage_results],
            "rejection_reasons": [
                {**item.to_dict(), "rejection_id": item.rejection_id}
                for item in self.rejection_reasons
            ],
            "reason_codes": list(self.reason_codes),
            "ir_receipt": (
                self.ir_receipt.to_dict() if self.ir_receipt is not None else None
            ),
            "constructed_request_id": self.constructed_request_id,
            "authorizes_execution": False,
            "provider_claims_are_authority": False,
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
    def from_dict(cls, value: Mapping[str, Any]) -> "PlanAdmissionServiceReceipt":
        if value.get("schema") != PLAN_ADMISSION_SERVICE_RECEIPT_SCHEMA:
            raise PlanAdmissionServiceError(
                "unsupported plan-admission service receipt schema"
            )
        if bool(value.get("authorizes_execution", False)):
            raise PlanAdmissionServiceError(
                "plan admission cannot authorize execution"
            )
        if bool(value.get("provider_claims_are_authority", False)):
            raise PlanAdmissionServiceError(
                "provider admission claims cannot become authority"
            )
        ir_receipt = None
        raw_ir = value.get("ir_receipt")
        if raw_ir is not None:
            if not isinstance(raw_ir, Mapping):
                raise PlanAdmissionServiceError("ir_receipt must be a mapping")
            ir_receipt = PlanAdmissionReceipt.from_dict(raw_ir)
        result = cls(
            request_id=value.get("request_id", ""),
            candidate_plan_id=value.get("candidate_plan_id", ""),
            candidate_graph_id=value.get("candidate_graph_id", ""),
            repository_tree_id=value.get("repository_tree_id", ""),
            formal_plan_id=value.get("formal_plan_id", ""),
            formal_source_identity=value.get("formal_source_identity", ""),
            evidence_bundle_id=value.get("evidence_bundle_id", ""),
            execution_plan_id=value.get("execution_plan_id", ""),
            policy_ids=tuple(value.get("policy_ids") or ()),
            ir_request_id=value.get("ir_request_id", ""),
            ir_receipt_id=value.get("ir_receipt_id", ""),
            proof_obligation_ids=tuple(value.get("proof_obligation_ids") or ()),
            proof_result_ids=tuple(value.get("proof_result_ids") or ()),
            semantic_roots=value.get("semantic_roots") or {},
            verdict=value.get("verdict", PlanAdmissionVerdict.REJECTED),
            stage_results=tuple(
                PlanAdmissionStageResult.from_dict(item)
                for item in value.get("stage_results") or ()
            ),
            rejection_reasons=tuple(
                AdmissionRejection.from_dict(item)
                for item in value.get("rejection_reasons") or ()
            ),
            ir_receipt=ir_receipt,
            constructed_request_id=value.get("constructed_request_id", ""),
        )
        if value.get("receipt_id") != result.receipt_id:
            raise PlanAdmissionServiceError(
                "plan-admission service receipt identity does not match content"
            )
        if value.get("admitted") is not result.admitted:
            raise PlanAdmissionServiceError(
                "plan-admission admitted projection does not match verdict"
            )
        if tuple(value.get("reason_codes") or ()) != result.reason_codes:
            raise PlanAdmissionServiceError(
                "plan-admission reason-code projection does not match findings"
            )
        return result


class PlanAdmissionService:
    """Construct an exact PlanAdmissionRequest and admit through fixed stages.

    Interface: ``PlanAdmissionService@1``
    """

    INTERFACE: Final[str] = PLAN_ADMISSION_SERVICE_INTERFACE
    VERSION: Final[int] = PLAN_ADMISSION_SERVICE_VERSION

    def __init__(
        self,
        *,
        ir_compiler: IRConstraintCompiler | None = None,
    ) -> None:
        self._ir_compiler = ir_compiler or IRConstraintCompiler()

    def construct_request(
        self,
        materials: PlanAdmissionServiceRequest | Mapping[str, Any],
    ) -> PlanAdmissionRequest:
        """Build an exact PlanAdmissionRequest, ignoring provider claims."""

        request = self._coerce_materials(materials)
        try:
            return PlanAdmissionRequest(
                candidate_plan=_plain(request.candidate_plan),
                repository_tree_id=request.repository_tree_id,
                intent_request=request.intent_request,
                legal_results=request.legal_results,
                security_policy=request.security_policy,
                security_requests=request.security_requests,
                action_bindings=request.action_bindings,
                authority=request.authority,
                root_bindings=request.root_bindings,
                program_dependencies=request.program_dependencies,
                assumptions=request.assumptions,
                proof_results=request.proof_results,
                validation_requirements=request.validation_requirements,
                validation_results=request.validation_results,
                generated_formula_ids=request.generated_formula_ids,
                mandatory_closure=request.mandatory_closure,
                graph_complete=request.graph_complete,
            )
        except (IRConstraintCompilerError, TypeError, ValueError) as exc:
            raise PlanAdmissionServiceError(
                f"failed to construct PlanAdmissionRequest: {exc}"
            ) from exc

    def admit(
        self,
        materials: PlanAdmissionServiceRequest | Mapping[str, Any],
    ) -> PlanAdmissionServiceReceipt:
        """Run every hard stage in fixed order and emit a bound receipt."""

        request = self._coerce_materials(materials)
        stage_results: list[PlanAdmissionStageResult] = []
        rejections: list[AdmissionRejection] = []
        constructed: PlanAdmissionRequest | None = None
        ir_receipt: PlanAdmissionReceipt | None = None

        for stage in ADMISSION_STAGE_ORDER:
            if stage is PlanAdmissionStage.FORMAL:
                result = self._stage_formal(request)
            elif stage is PlanAdmissionStage.CANDIDATE_GRAPH:
                result = self._stage_candidate_graph(request)
            elif stage is PlanAdmissionStage.ROOTS:
                result = self._stage_roots(request)
            elif stage is PlanAdmissionStage.INTENT:
                result = self._stage_intent(request)
            elif stage is PlanAdmissionStage.APPLICABILITY:
                result = self._stage_applicability(request)
            elif stage is PlanAdmissionStage.SECURITY:
                result = self._stage_security(request)
            elif stage is PlanAdmissionStage.AUTHORITY:
                result = self._stage_authority(request)
            elif stage is PlanAdmissionStage.EFFECTS:
                result = self._stage_effects(request)
            elif stage is PlanAdmissionStage.PROOF:
                result = self._stage_proof(request)
            elif stage is PlanAdmissionStage.VALIDATION:
                result = self._stage_validation(request)
            elif stage is PlanAdmissionStage.EXECUTION_PLAN:
                result = self._stage_execution_plan(request)
            elif stage is PlanAdmissionStage.POLICY:
                result = self._stage_policy(request)
            else:
                # IR join always runs last so earlier stage failures still leave
                # an independent IR counterexample trail when construction works.
                try:
                    constructed = self.construct_request(request)
                    ir_receipt = self._ir_compiler.compile(constructed)
                    if ir_receipt.admitted:
                        result = PlanAdmissionStageResult(
                            stage=stage,
                            passed=True,
                            detail_ids=(ir_receipt.receipt_id,),
                            message="IR join admitted the constructed request",
                        )
                    else:
                        for rejection in ir_receipt.rejection_reasons:
                            rejections.append(rejection)
                        result = PlanAdmissionStageResult(
                            stage=stage,
                            passed=False,
                            reason_codes=(
                                PlanAdmissionServiceCode.IR_REJECTED.value,
                                *ir_receipt.reason_codes,
                            ),
                            detail_ids=(ir_receipt.receipt_id,),
                            message="IR constraint compiler rejected the request",
                        )
                except PlanAdmissionServiceError as exc:
                    rejections.append(
                        AdmissionRejection(
                            code=AdmissionRejectionCode.INVALID_GRAPH,
                            domain=AdmissionDomain.GRAPH,
                            message=str(exc),
                        )
                    )
                    result = PlanAdmissionStageResult(
                        stage=stage,
                        passed=False,
                        reason_codes=(
                            PlanAdmissionServiceCode.REQUEST_CONSTRUCTION_FAILED.value,
                        ),
                        message=str(exc),
                    )
            stage_results.append(result)
            if not result.passed:
                for code in result.reason_codes:
                    if code in {
                        item.code.value for item in rejections
                    }:
                        continue
                    rejections.append(
                        self._service_rejection(
                            code,
                            stage,
                            result.message or f"stage {stage.value} failed",
                            source_ids=result.detail_ids,
                        )
                    )

        # Provider claims never admit a plan; if every stage passed we still
        # require a constructed IR admission.
        admitted = (
            all(item.passed for item in stage_results)
            and ir_receipt is not None
            and ir_receipt.admitted
            and not rejections
        )
        if admitted and rejections:
            admitted = False

        proof_obligation_ids = self._required_proof_ids(request)
        return PlanAdmissionServiceReceipt(
            request_id=request.request_id,
            candidate_plan_id=request.candidate_plan_id,
            candidate_graph_id=request.candidate_graph_id,
            repository_tree_id=request.repository_tree_id,
            formal_plan_id=request.formal_plan_id,
            formal_source_identity=request.formal_source_identity,
            evidence_bundle_id=request.evidence_bundle_id,
            execution_plan_id=request.execution_plan_id,
            policy_ids=request.policy_ids,
            ir_request_id=constructed.request_id if constructed is not None else "",
            ir_receipt_id=ir_receipt.receipt_id if ir_receipt is not None else "",
            proof_obligation_ids=tuple(sorted(proof_obligation_ids)),
            proof_result_ids=tuple(
                item.receipt_id for item in request.proof_results
            ),
            semantic_roots=request.semantic_roots,
            verdict=(
                PlanAdmissionVerdict.ADMITTED
                if admitted
                else PlanAdmissionVerdict.REJECTED
            ),
            stage_results=tuple(stage_results),
            rejection_reasons=tuple(rejections) if not admitted else (),
            ir_receipt=ir_receipt,
            constructed_request_id=(
                constructed.request_id if constructed is not None else ""
            ),
        )

    def replay(
        self,
        materials: PlanAdmissionServiceRequest | Mapping[str, Any],
        receipt: PlanAdmissionServiceReceipt | Mapping[str, Any],
    ) -> PlanAdmissionServiceReceipt:
        """Re-admit materials and reject tampered or stale receipts."""

        if isinstance(receipt, Mapping):
            receipt = PlanAdmissionServiceReceipt.from_dict(receipt)
        elif not isinstance(receipt, PlanAdmissionServiceReceipt):
            raise PlanAdmissionServiceError(
                "receipt must be PlanAdmissionServiceReceipt or mapping"
            )
        fresh = self.admit(materials)
        if fresh.receipt_id != receipt.receipt_id:
            raise PlanAdmissionServiceError(
                f"{PlanAdmissionServiceCode.REPLAY_MISMATCH.value}: "
                "replayed admission does not match the presented receipt"
            )
        if (
            fresh.candidate_plan_id != receipt.candidate_plan_id
            or fresh.repository_tree_id != receipt.repository_tree_id
            or fresh.formal_plan_id != receipt.formal_plan_id
            or fresh.evidence_bundle_id != receipt.evidence_bundle_id
            or fresh.execution_plan_id != receipt.execution_plan_id
            or fresh.policy_ids != receipt.policy_ids
            or dict(fresh.semantic_roots) != dict(receipt.semantic_roots)
        ):
            raise PlanAdmissionServiceError(
                f"{PlanAdmissionServiceCode.RECEIPT_TAMPERED.value}: "
                "receipt bindings do not match the independent admission"
            )
        return fresh

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_formal(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.formal_plan_id:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.FORMAL,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.MISSING_FORMAL_PLAN.value,
                ),
                message="formal plan identity is required",
            )
        if request.formal_plan_id != request.candidate_plan_id:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.FORMAL,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.FORMAL_IDENTITY_MISMATCH.value,
                ),
                detail_ids=(request.formal_plan_id, request.candidate_plan_id),
                message="formal plan identity differs from the candidate plan",
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.FORMAL,
            passed=True,
            detail_ids=(request.formal_plan_id,),
        )

    def _stage_candidate_graph(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        try:
            actions = _candidate_actions(request.candidate_plan)
            effects = _candidate_effects(request.candidate_plan)
        except PlanAdmissionServiceError as exc:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.CANDIDATE_GRAPH,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.INCOMPLETE_CANDIDATE.value,
                ),
                message=str(exc),
            )
        if not actions or not request.graph_complete:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.CANDIDATE_GRAPH,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.INCOMPLETE_CANDIDATE.value,
                ),
                message="admission requires a complete non-empty candidate graph",
            )
        if request.mandatory_closure is not None and not request.mandatory_closure.complete:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.CANDIDATE_GRAPH,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.INCOMPLETE_CANDIDATE.value,
                ),
                message="mandatory dependency closure is incomplete",
            )
        action_ids = {item["action_id"] for item in actions}
        binding_ids = {item.action_id for item in request.action_bindings}
        if binding_ids != action_ids:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.CANDIDATE_GRAPH,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.INCOMPLETE_CANDIDATE.value,
                ),
                detail_ids=tuple(sorted(action_ids ^ binding_ids)),
                message="every action requires exact domain bindings",
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.CANDIDATE_GRAPH,
            passed=True,
            detail_ids=tuple(sorted(action_ids)),
            message=f"{len(actions)} actions / {len(effects)} effects",
        )

    def _stage_roots(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.root_bindings:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.ROOTS,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_ROOT.value,
                ),
                message="semantic root bindings are unknown",
            )
        for root in request.root_bindings:
            if not root.current:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.ROOTS,
                    passed=False,
                    reason_codes=(PlanAdmissionServiceCode.STALE_ROOT.value,),
                    detail_ids=(root.kind, root.expected, root.observed),
                    message=f"{root.kind} semantic root is stale",
                )
            if not root.authority_accepted:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.ROOTS,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.AUTHORITY_MISMATCH.value,
                    ),
                    detail_ids=(root.kind, root.authority),
                    message=f"{root.kind} semantic root has insufficient authority",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.ROOTS,
            passed=True,
            detail_ids=tuple(sorted(request.semantic_roots)),
        )

    def _stage_intent(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if request.intent_request is None:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.INTENT,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_INTENT.value,
                ),
                message="intent conformance materials are unknown",
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.INTENT,
            passed=True,
            detail_ids=(_record_id(request.intent_request),),
        )

    def _stage_applicability(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.legal_results:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.APPLICABILITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_APPLICABILITY.value,
                ),
                message="legal applicability materials are unknown",
            )
        for result in request.legal_results:
            if getattr(result, "fail_closed", False) or not getattr(
                result, "accepted", False
            ):
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.APPLICABILITY,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.APPLICABILITY_REJECTED.value,
                    ),
                    detail_ids=(_record_id(result),),
                    message="legal applicability is incomplete or rejected",
                )
            prohibitions = getattr(result, "prohibitions", ()) or ()
            if prohibitions:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.APPLICABILITY,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.APPLICABILITY_REJECTED.value,
                    ),
                    detail_ids=(_record_id(result),),
                    message="an applicable legal prohibition blocks the candidate",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.APPLICABILITY,
            passed=True,
            detail_ids=tuple(_record_id(item) for item in request.legal_results),
        )

    def _stage_security(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.security_requests:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.SECURITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_SECURITY.value,
                ),
                message="security authorization materials are unknown",
            )

        # Security forbidden logic is checked against *both* intent and code
        # effect streams.  A single-stream evaluation is not sufficient.
        intent_effects = [
            json.dumps(
                _effect_projection(item)
                if "operation" in item or "target" in item
                else _plain(item),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in (
                request.intent_effects
                or tuple(_candidate_effects(request.candidate_plan))
            )
        ]
        code_effects = [
            json.dumps(
                _effect_projection(item)
                if "operation" in item or "target" in item
                else _plain(item),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in (
                request.code_effects
                or tuple(_candidate_effects(request.candidate_plan))
            )
        ]
        if not intent_effects or not code_effects:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.SECURITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.SECURITY_STREAM_GAP.value,
                ),
                message=(
                    "security forbidden logic requires non-empty intent and "
                    "code effect streams"
                ),
            )

        covered_intent: set[str] = set()
        covered_code: set[str] = set()
        for security_request in request.security_requests:
            expected = json.dumps(
                _plain(security_request.expected_effect),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if expected in intent_effects:
                covered_intent.add(expected)
            if expected in code_effects:
                covered_code.add(expected)
            decision = evaluate_security_authorization(
                request.security_policy, security_request
            )
            if decision.outcome in _FORBIDDEN_SECURITY_OUTCOMES:
                # Forbidden / deny / unknown / conflict must be reported with
                # both streams considered — not offset by the other stream.
                if expected not in intent_effects or expected not in code_effects:
                    return PlanAdmissionStageResult(
                        stage=PlanAdmissionStage.SECURITY,
                        passed=False,
                        reason_codes=(
                            PlanAdmissionServiceCode.SECURITY_STREAM_GAP.value,
                            PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value,
                        ),
                        detail_ids=(security_request.content_id, decision.content_id),
                        message=(
                            "security forbidden outcome lacks dual intent/code "
                            f"effect coverage ({decision.outcome.value})"
                        ),
                    )
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.SECURITY,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value,
                    ),
                    detail_ids=(security_request.content_id, decision.content_id),
                    message=(
                        "security authorization forbids the candidate against "
                        f"intent and code effects ({decision.outcome.value})"
                    ),
                )
            # Also treat explicit DENY rule effects on matched policies as
            # forbidden even if the aggregate outcome is somehow permit.
            for check in getattr(decision, "checks", ()) or ():
                effect = getattr(check, "effect", None)
                if effect is SecurityRuleEffect.DENY:
                    return PlanAdmissionStageResult(
                        stage=PlanAdmissionStage.SECURITY,
                        passed=False,
                        reason_codes=(
                            PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value,
                        ),
                        detail_ids=(security_request.content_id,),
                        message=(
                            "security forbidden rule matched intent/code effects"
                        ),
                    )

        # Dual-stream coverage: every intent and code effect must be named by
        # at least one security request when effects are present.
        if set(intent_effects) - covered_intent or set(code_effects) - covered_code:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.SECURITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.SECURITY_STREAM_GAP.value,
                ),
                message=(
                    "security requests must cover every intent and code effect"
                ),
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.SECURITY,
            passed=True,
            detail_ids=tuple(
                item.content_id for item in request.security_requests
            ),
        )

    def _stage_authority(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        authority = request.authority
        if not authority.grant_source_ids:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.AUTHORITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_AUTHORITY.value,
                ),
                message="authority grant sources are unknown",
            )
        if not authority.matched:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.AUTHORITY,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.AUTHORITY_MISMATCH.value,
                ),
                detail_ids=authority.grant_source_ids,
                message="requested principal/authority is not covered by an explicit grant",
            )
        for security_request in request.security_requests:
            if (
                security_request.principal != authority.principal
                or security_request.requested_authority
                != authority.requested_authority
            ):
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.AUTHORITY,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.AUTHORITY_MISMATCH.value,
                    ),
                    detail_ids=(security_request.content_id,),
                    message="security request does not match admission authority",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.AUTHORITY,
            passed=True,
            detail_ids=authority.grant_source_ids,
        )

    def _stage_effects(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        try:
            actions = _candidate_actions(request.candidate_plan)
            effects = _candidate_effects(request.candidate_plan)
        except PlanAdmissionServiceError as exc:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.EFFECTS,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_EFFECT.value,
                ),
                message=str(exc),
            )
        if not effects:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.EFFECTS,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_EFFECT.value,
                ),
                message="candidate effects are unknown",
            )
        action_ids = {item["action_id"] for item in actions}
        dependency_ids = {
            item.action_id for item in request.program_dependencies
        }
        if dependency_ids != action_ids:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.EFFECTS,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.UNKNOWN_MANDATORY_EFFECT.value,
                ),
                detail_ids=tuple(sorted(action_ids ^ dependency_ids)),
                message="every action requires a declared program dependency state",
            )
        for dependency in request.program_dependencies:
            if not dependency.passed:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.EFFECTS,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.UNDECLARED_EFFECT.value
                        if dependency.current
                        else PlanAdmissionServiceCode.STALE_ROOT.value
                    ),
                    detail_ids=(dependency.dependency_id, dependency.action_id),
                    message="required program dependency is stale or unsatisfied",
                )
        security_effects = {
            json.dumps(
                _plain(item.expected_effect),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            for item in request.security_requests
        }
        for effect in effects:
            projected = json.dumps(
                _effect_projection(effect),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if projected not in security_effects:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.EFFECTS,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.UNDECLARED_EFFECT.value,
                    ),
                    detail_ids=(effect.get("effect_id", ""),),
                    message="candidate effect is not covered by a security request",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.EFFECTS,
            passed=True,
            detail_ids=tuple(sorted(action_ids)),
        )

    def _stage_proof(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        required = self._required_proof_ids(request)
        if not required and not request.proof_results:
            # Unknown mandatory proof surface: neither formal obligations nor
            # independent receipts were supplied.
            proof_ids = {
                item
                for action in _candidate_actions(request.candidate_plan)
                for item in action.get("proof_obligation_ids", ()) or ()
            }
            if proof_ids:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.PROOF,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.UNKNOWN_MANDATORY_PROOF.value,
                    ),
                    detail_ids=tuple(sorted(proof_ids)),
                    message="mandatory proof obligations lack independent receipts",
                )
        present = {item.obligation_id for item in request.proof_results}
        missing = required - present
        if missing:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.PROOF,
                passed=False,
                reason_codes=(PlanAdmissionServiceCode.MISSING_PROOF.value,),
                detail_ids=tuple(sorted(missing)),
                message="required proof obligation has no typed ProofReceipt",
            )
        for proof in request.proof_results:
            if proof.obligation_id not in required:
                continue
            if (
                proof.repository_tree_id != request.repository_tree_id
                or proof.freshness is not EvidenceFreshness.CURRENT
                or proof.authoritative_verdict is not ProofVerdict.PROVED
            ):
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.PROOF,
                    passed=False,
                    reason_codes=(PlanAdmissionServiceCode.MISSING_PROOF.value,),
                    detail_ids=(proof.receipt_id, proof.obligation_id),
                    message="proof receipt is stale, mismatched, or unproved",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.PROOF,
            passed=True,
            detail_ids=tuple(item.receipt_id for item in request.proof_results),
        )

    def _stage_validation(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        required = {
            item.requirement_id
            for item in request.validation_requirements
            if item.required
        }
        if not required:
            # Formal candidates always declare validation requirements per action.
            declared = {
                item
                for action in _candidate_actions(request.candidate_plan)
                for item in action.get("validation_requirement_ids", ()) or ()
            }
            if declared:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.VALIDATION,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.UNKNOWN_MANDATORY_VALIDATION.value,
                    ),
                    detail_ids=tuple(sorted(declared)),
                    message="mandatory validation requirements are unknown",
                )
        results = {
            item.requirement_id: item for item in request.validation_results
        }
        for requirement_id in sorted(required):
            result = results.get(requirement_id)
            if result is None:
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.VALIDATION,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.UNKNOWN_MANDATORY_VALIDATION.value,
                    ),
                    detail_ids=(requirement_id,),
                    message="required validation has no result",
                )
            if (
                result.status is not ValidationStatus.PASSED
                or result.repository_tree_id != request.repository_tree_id
                or not result.evidence_id
            ):
                return PlanAdmissionStageResult(
                    stage=PlanAdmissionStage.VALIDATION,
                    passed=False,
                    reason_codes=(
                        PlanAdmissionServiceCode.VALIDATION_FAILED.value,
                    ),
                    detail_ids=(requirement_id, *result.reason_codes),
                    message="validation failed, is unknown, or is stale",
                )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.VALIDATION,
            passed=True,
            detail_ids=tuple(sorted(required)),
        )

    def _stage_execution_plan(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.execution_plan_id:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.EXECUTION_PLAN,
                passed=False,
                reason_codes=(
                    PlanAdmissionServiceCode.MISSING_EXECUTION_PLAN.value,
                ),
                message="parallel execution plan binding is required",
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.EXECUTION_PLAN,
            passed=True,
            detail_ids=(request.execution_plan_id,),
        )

    def _stage_policy(
        self, request: PlanAdmissionServiceRequest
    ) -> PlanAdmissionStageResult:
        if not request.policy_ids:
            return PlanAdmissionStageResult(
                stage=PlanAdmissionStage.POLICY,
                passed=False,
                reason_codes=(PlanAdmissionServiceCode.MISSING_POLICY.value,),
                message="policy bindings are required",
            )
        return PlanAdmissionStageResult(
            stage=PlanAdmissionStage.POLICY,
            passed=True,
            detail_ids=request.policy_ids,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _coerce_materials(
        self,
        materials: PlanAdmissionServiceRequest | Mapping[str, Any],
    ) -> PlanAdmissionServiceRequest:
        if isinstance(materials, PlanAdmissionServiceRequest):
            return materials
        if not isinstance(materials, Mapping):
            raise PlanAdmissionServiceError(
                "materials must be PlanAdmissionServiceRequest or mapping"
            )
        stripped = _strip_provider_claims(materials)
        if not isinstance(stripped, Mapping):
            raise PlanAdmissionServiceError("materials mapping is malformed")
        return PlanAdmissionServiceRequest(
            candidate_plan=stripped.get("candidate_plan") or {},
            repository_tree_id=str(stripped.get("repository_tree_id") or ""),
            formal_plan_id=str(stripped.get("formal_plan_id") or ""),
            formal_source_identity=str(
                stripped.get("formal_source_identity") or ""
            ),
            evidence_bundle_id=str(stripped.get("evidence_bundle_id") or ""),
            execution_plan_id=str(stripped.get("execution_plan_id") or ""),
            policy_ids=tuple(stripped.get("policy_ids") or ()),
            intent_request=stripped["intent_request"],
            legal_results=tuple(stripped.get("legal_results") or ()),
            security_policy=stripped["security_policy"],
            security_requests=tuple(stripped.get("security_requests") or ()),
            action_bindings=tuple(stripped.get("action_bindings") or ()),
            authority=stripped["authority"],
            root_bindings=tuple(stripped.get("root_bindings") or ()),
            program_dependencies=tuple(
                stripped.get("program_dependencies") or ()
            ),
            assumptions=tuple(stripped.get("assumptions") or ()),
            proof_results=tuple(stripped.get("proof_results") or ()),
            validation_requirements=tuple(
                stripped.get("validation_requirements") or ()
            ),
            validation_results=tuple(stripped.get("validation_results") or ()),
            intent_effects=tuple(stripped.get("intent_effects") or ()),
            code_effects=tuple(stripped.get("code_effects") or ()),
            generated_formula_ids=tuple(
                stripped.get("generated_formula_ids") or ()
            ),
            mandatory_closure=stripped.get("mandatory_closure"),
            graph_complete=bool(stripped.get("graph_complete", True)),
            provider_admission_claim=stripped.get("provider_admission_claim"),
        )

    def _required_proof_ids(
        self, request: PlanAdmissionServiceRequest
    ) -> set[str]:
        required: set[str] = set()
        for action in _candidate_actions(request.candidate_plan):
            for obligation_id in action.get("proof_obligation_ids", ()) or ():
                text = str(obligation_id).strip()
                if text:
                    required.add(text)
        constraint_set = getattr(
            request.intent_request, "constraint_set", None
        )
        if constraint_set is not None:
            for item in getattr(constraint_set, "proof_obligations", ()) or ():
                obligation_id = str(getattr(item, "obligation_id", "") or "")
                if obligation_id:
                    required.add(obligation_id)
        for legal in request.legal_results:
            for item in getattr(legal, "proof_obligations", ()) or ():
                if getattr(item, "required", False):
                    obligation_id = str(
                        getattr(item, "obligation_id", "") or ""
                    )
                    if obligation_id:
                        required.add(obligation_id)
        return required

    def _service_rejection(
        self,
        code: str,
        stage: PlanAdmissionStage,
        message: str,
        *,
        source_ids: Sequence[str] = (),
    ) -> AdmissionRejection:
        domain = {
            PlanAdmissionStage.FORMAL: AdmissionDomain.GRAPH,
            PlanAdmissionStage.CANDIDATE_GRAPH: AdmissionDomain.GRAPH,
            PlanAdmissionStage.ROOTS: AdmissionDomain.ROOT,
            PlanAdmissionStage.INTENT: AdmissionDomain.INTENT,
            PlanAdmissionStage.APPLICABILITY: AdmissionDomain.LEGAL,
            PlanAdmissionStage.SECURITY: AdmissionDomain.SECURITY,
            PlanAdmissionStage.AUTHORITY: AdmissionDomain.AUTHORITY,
            PlanAdmissionStage.EFFECTS: AdmissionDomain.PROGRAM,
            PlanAdmissionStage.PROOF: AdmissionDomain.PROOF,
            PlanAdmissionStage.VALIDATION: AdmissionDomain.VALIDATION,
            PlanAdmissionStage.EXECUTION_PLAN: AdmissionDomain.GRAPH,
            PlanAdmissionStage.POLICY: AdmissionDomain.AUTHORITY,
            PlanAdmissionStage.IR_JOIN: AdmissionDomain.GRAPH,
        }.get(stage, AdmissionDomain.GRAPH)
        rejection_code = {
            PlanAdmissionServiceCode.INCOMPLETE_CANDIDATE.value: (
                AdmissionRejectionCode.INCOMPLETE_GRAPH
            ),
            PlanAdmissionServiceCode.STALE_ROOT.value: (
                AdmissionRejectionCode.STALE_ROOT
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_ROOT.value: (
                AdmissionRejectionCode.STALE_ROOT
            ),
            PlanAdmissionServiceCode.APPLICABILITY_REJECTED.value: (
                AdmissionRejectionCode.LEGAL_INCOMPLETE
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_APPLICABILITY.value: (
                AdmissionRejectionCode.LEGAL_INCOMPLETE
            ),
            PlanAdmissionServiceCode.SECURITY_FORBIDDEN.value: (
                AdmissionRejectionCode.SECURITY_DENY
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_SECURITY.value: (
                AdmissionRejectionCode.SECURITY_UNKNOWN
            ),
            PlanAdmissionServiceCode.SECURITY_STREAM_GAP.value: (
                AdmissionRejectionCode.SECURITY_UNKNOWN
            ),
            PlanAdmissionServiceCode.AUTHORITY_MISMATCH.value: (
                AdmissionRejectionCode.AUTHORITY_MISMATCH
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_AUTHORITY.value: (
                AdmissionRejectionCode.AUTHORITY_MISMATCH
            ),
            PlanAdmissionServiceCode.UNDECLARED_EFFECT.value: (
                AdmissionRejectionCode.UNDECLARED_EFFECT
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_EFFECT.value: (
                AdmissionRejectionCode.UNDECLARED_EFFECT
            ),
            PlanAdmissionServiceCode.MISSING_PROOF.value: (
                AdmissionRejectionCode.MISSING_PROOF
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_PROOF.value: (
                AdmissionRejectionCode.MISSING_PROOF
            ),
            PlanAdmissionServiceCode.VALIDATION_FAILED.value: (
                AdmissionRejectionCode.VALIDATION_FAILED
            ),
            PlanAdmissionServiceCode.UNKNOWN_MANDATORY_VALIDATION.value: (
                AdmissionRejectionCode.VALIDATION_MISSING
            ),
            PlanAdmissionServiceCode.MISSING_FORMAL_PLAN.value: (
                AdmissionRejectionCode.INCOMPLETE_GRAPH
            ),
            PlanAdmissionServiceCode.FORMAL_IDENTITY_MISMATCH.value: (
                AdmissionRejectionCode.DOMAIN_BINDING_MISMATCH
            ),
            PlanAdmissionServiceCode.MISSING_EXECUTION_PLAN.value: (
                AdmissionRejectionCode.INCOMPLETE_GRAPH
            ),
            PlanAdmissionServiceCode.MISSING_POLICY.value: (
                AdmissionRejectionCode.AUTHORITY_MISMATCH
            ),
            PlanAdmissionServiceCode.IR_REJECTED.value: (
                AdmissionRejectionCode.INVALID_GRAPH
            ),
            PlanAdmissionServiceCode.REQUEST_CONSTRUCTION_FAILED.value: (
                AdmissionRejectionCode.INVALID_GRAPH
            ),
        }.get(code, AdmissionRejectionCode.INVALID_GRAPH)
        return AdmissionRejection(
            code=rejection_code,
            domain=domain,
            message=message,
            source_ids=tuple(source_ids),
            details={"service_code": code, "stage": stage.value},
        )


def admit_plan_through_service(
    materials: PlanAdmissionServiceRequest | Mapping[str, Any],
) -> PlanAdmissionServiceReceipt:
    """Module-level convenience for independent multi-gate plan admission."""

    return PlanAdmissionService().admit(materials)


def construct_plan_admission_request(
    materials: PlanAdmissionServiceRequest | Mapping[str, Any],
) -> PlanAdmissionRequest:
    """Construct an exact PlanAdmissionRequest from independent materials."""

    return PlanAdmissionService().construct_request(materials)


# Re-export IR join contracts named by the task interface list.
admit_plan = compile_plan_admission


__all__ = [
    "ADMISSION_STAGE_ORDER",
    "PLAN_ADMISSION_SERVICE_INTERFACE",
    "PLAN_ADMISSION_SERVICE_RECEIPT_SCHEMA",
    "PLAN_ADMISSION_SERVICE_REQUEST_SCHEMA",
    "PLAN_ADMISSION_SERVICE_VERSION",
    "PLAN_ADMISSION_STAGE_RESULT_SCHEMA",
    "PlanAdmissionReceipt",
    "PlanAdmissionRequest",
    "PlanAdmissionService",
    "PlanAdmissionServiceCode",
    "PlanAdmissionServiceError",
    "PlanAdmissionServiceReceipt",
    "PlanAdmissionServiceRequest",
    "PlanAdmissionStage",
    "PlanAdmissionStageResult",
    "PlanAdmissionVerdict",
    "admit_plan",
    "admit_plan_through_service",
    "construct_plan_admission_request",
]
